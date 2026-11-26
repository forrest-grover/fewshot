import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { useUploads, type SendBatch, type UploadStop } from "./useUploads";
import { taskDetail } from "./test-data";
import { BATCH, batchLimits, newUndoGroup } from "./upload";

afterEach(cleanup);

const files = (n: number) => Array.from({ length: n }, (_, i) => new File([new Uint8Array(10)], `${String(i)}.jpg`));
const limits = { files: 2, bytes: 1000 };

/** A ``send`` whose requests wait until ``release`` is called, recording each batch's undo group. */
function gatedSend() {
  const groups: string[] = [];
  let release: () => void = () => { /* set below */ };
  const send = vi.fn<SendBatch>((batch, signal, group) => {
    groups.push(group);
    return new Promise((res, rej) => {
      signal.addEventListener("abort", () => { rej(new DOMException("aborted", "AbortError")); });
      release = () => { res({ added: batch, errors: [], task: taskDetail() }); };
    });
  });
  return { send, groups, release: () => { act(() => { release(); }); } };
}

function setup(task: string) {
  const stops: UploadStop[] = [];
  const onTask = vi.fn();
  const hook = renderHook(({ t }) => useUploads(t, { onTask, onStop: (s) => { stops.push(s); }, limits, prepare: (fs) => Promise.resolve(fs) }),
    { initialProps: { t: task as string | null } });
  return { hook, stops, onTask };
}

describe("useUploads", () => {
  it("sends every batch of one drop with the same undo group, and reports progress per slot", async () => {
    const { hook, stops, onTask } = setup("pets");
    const { send, groups, release } = gatedSend();
    let done: Promise<unknown> = Promise.resolve();
    act(() => { done = hook.result.current.start("cat", files(5), send); });
    for (let i = 1; i <= 3; i++) {
      await waitFor(() => { expect(send).toHaveBeenCalledTimes(i); });
      expect(hook.result.current.slots.cat?.progress?.done).toBe(Math.min(5, (i - 1) * 2));
      release();
    }
    await act(() => done);
    expect(new Set(groups).size).toBe(1);
    expect(groups[0].length).toBeLessThanOrEqual(64);
    expect(onTask).toHaveBeenCalledTimes(3);
    expect(hook.result.current.slots.cat).toEqual({ skipped: [] });
    expect(stops).toEqual([]);
  });

  it("goes on while the task stays open, and stops when another opens, counting the batch in flight", async () => {
    const { hook, stops } = setup("pets");
    const { send, release } = gatedSend();
    act(() => { void hook.result.current.start("cat", files(5), send); });
    await waitFor(() => { expect(send).toHaveBeenCalledTimes(1); });
    release();
    await waitFor(() => { expect(send).toHaveBeenCalledTimes(2); });
    hook.rerender({ t: "pets" }); // the App re-rendering, e.g. for a tab switch: nothing stops
    expect(stops).toEqual([]);
    hook.rerender({ t: "birds" });
    expect(hook.result.current.slots).toEqual({});
    expect(send.mock.calls[1][1].aborted).toBe(false); // the server may store it: wait and count it
    release();
    await waitFor(() => { expect(stops).toEqual([{ task: "pets", added: 4, total: 5, error: null }]); });
    expect(send).toHaveBeenCalledTimes(2);
  });

  it("keeps going when the task is renamed, sending the rest to its new name", async () => {
    const { hook, stops } = setup("pets");
    const { send, release } = gatedSend();
    let done: Promise<unknown> = Promise.resolve();
    act(() => { done = hook.result.current.start("cat", files(5), send); });
    await waitFor(() => { expect(send).toHaveBeenCalledTimes(1); });
    act(() => { hook.result.current.rename("cats"); });
    expect(hook.result.current.slots.cat?.progress).toBeDefined(); // shown while the route catches up
    hook.rerender({ t: "cats" });
    expect(hook.result.current.slots.cat?.progress).toBeDefined();
    release();
    await waitFor(() => { expect(send).toHaveBeenCalledTimes(2); });
    release();
    await waitFor(() => { expect(send).toHaveBeenCalledTimes(3); });
    release();
    await act(() => done);
    expect(send.mock.calls.map((c) => c[3])).toEqual(["pets", "cats", "cats"]);
    expect(stops).toEqual([]);
  });

  it("stop() stops after the batch in flight and says how many got in", async () => {
    const { hook, stops } = setup("pets");
    const { send, release } = gatedSend();
    act(() => { void hook.result.current.start("cat", files(5), send); });
    await waitFor(() => { expect(send).toHaveBeenCalledTimes(1); });
    expect(hook.result.current.busy).toBe(true);
    let added: number | null = null;
    act(() => { void hook.result.current.stop().then((n) => { added = n; }); });
    expect(hook.result.current.slots.cat?.progress?.stopping).toBe(true);
    expect(added).toBeNull();
    release();
    await waitFor(() => { expect(added).toBe(2); });
    expect(send).toHaveBeenCalledTimes(1);
    expect(stops).toEqual([{ task: "pets", added: 2, total: 5, error: null }]);
    expect(hook.result.current.busy).toBe(false);
  });

  it("keeps how many images were moved from other classes", async () => {
    const { hook } = setup("pets");
    const moving: SendBatch = (batch) => Promise.resolve({ added: batch, errors: [], task: taskDetail(), moved: [{ id: "a", previous_label: "dog" }] });
    await act(async () => { await hook.result.current.start("cat", files(3), moving); });
    expect(hook.result.current.slots.cat).toEqual({ skipped: [], moved: 2 });
  });

  it("Cancel stops the rest; a failure part-way is reported with its error", async () => {
    const { hook, stops } = setup("pets");
    const gated = gatedSend();
    act(() => { void hook.result.current.start("cat", files(3), gated.send); });
    await waitFor(() => { expect(hook.result.current.slots.cat?.progress).toBeDefined(); });
    act(() => { hook.result.current.slots.cat?.progress?.cancel(); });
    gated.release(); // the batch in flight lands, and is counted
    await waitFor(() => { expect(stops).toEqual([{ task: "pets", added: 2, total: 3, error: null }]); });
    expect(gated.send).toHaveBeenCalledTimes(1);

    let calls = 0;
    const failing: SendBatch = (batch) => (++calls === 1
      ? Promise.resolve({ added: batch, errors: [{ file: "1.jpg", error: "not an image" }], task: taskDetail() })
      : Promise.reject(new Error("413 Payload Too Large")));
    let skips: unknown = undefined;
    await act(async () => { skips = await hook.result.current.start("dog", files(4), failing); });
    expect(skips).toBeNull();
    expect(stops[1]).toMatchObject({ added: 2, total: 4, error: new Error("413 Payload Too Large") });
    expect(hook.result.current.slots.dog?.skipped).toEqual([{ file: "1.jpg", error: "not an image" }]);
  });
});

describe("batchLimits", () => {
  it("keeps batches to half the server's request limit, at most 32 MB", () => {
    expect(batchLimits(1024 * 1024 * 1024)).toEqual(BATCH);
    expect(batchLimits(20 * 1024 * 1024)).toEqual({ files: 100, bytes: 10 * 1024 * 1024 });
    expect(batchLimits(undefined)).toEqual(BATCH);
    expect(batchLimits(0)).toEqual(BATCH);
  });
});

describe("newUndoGroup", () => {
  it("is new each time and fits the server's 64 characters, with or without randomUUID", () => {
    expect(newUndoGroup()).not.toBe(newUndoGroup());
    const c = crypto as { randomUUID?: unknown };
    const saved = c.randomUUID;
    c.randomUUID = undefined;
    try {
      expect(newUndoGroup()).toMatch(/^[0-9a-f]{32}$/);
    } finally {
      c.randomUUID = saved;
    }
  });
});

describe("upload requests", () => {
  it("carry undo_group when given, beside the files and label", async () => {
    const answer = () => new Response(JSON.stringify({ added: [], errors: [], task: taskDetail() }), { status: 201 });
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(() => Promise.resolve(answer()));
    const body = (i: number) => fetch.mock.calls[i][1]?.body as FormData;
    await api.addExamples("pets", "cat", files(2), undefined, "g-1");
    expect(body(0).get("undo_group")).toBe("g-1");
    expect(body(0).get("label")).toBe("cat");
    expect(body(0).getAll("files")).toHaveLength(2);
    await api.addToInbox("pets", files(1), undefined, "g-2");
    expect(body(1).get("undo_group")).toBe("g-2");
    await api.addToInbox("pets", files(1));
    expect(body(2).has("undo_group")).toBe(false);
  });
});
