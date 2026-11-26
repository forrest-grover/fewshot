import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api, type InboxEntry, type TaskDetail } from "./api";
import type { Uploads } from "./useUploads";
import Inbox from "./Inbox";
import { fetchInbox, stillGone } from "./inboxPages";
import { taskDetail } from "./test-data";

afterEach(cleanup);

const entry = (i: number): InboxEntry => ({
  id: `i${String(i)}`, source: `${String(i)}.jpg`, file: `${String(i)}.jpg`, added: "2026-01-01T00:00:00", margin: null, prediction: null,
});
/** A server holding ``n`` inbox images. */
const server = (n: number) => vi.fn((offset: number, limit: number) => Promise.resolve({
  items: Array.from({ length: Math.max(0, Math.min(limit, n - offset)) }, (_, k) => entry(offset + k)), total: n, offset,
}));

describe("fetchInbox", () => {
  it("fetches pages until it has as many as wanted", async () => {
    const page = server(2500);
    const r = await fetchInbox(page, 1500, 1000);
    expect(page.mock.calls).toEqual([[0, 1000], [1000, 1000]]);
    expect(r?.items).toHaveLength(2000);
    expect(r?.total).toBe(2500);
  });

  it("fetches one page when fewer are wanted, and stops at the end", async () => {
    const one = server(2500);
    await fetchInbox(one, 0, 1000);
    expect(one).toHaveBeenCalledTimes(1);
    const short = server(1200);
    expect((await fetchInbox(short, 5000, 1000))?.items).toHaveLength(1200);
    expect(short).toHaveBeenCalledTimes(2);
  });

  it("gives up if a request fails", async () => {
    expect(await fetchInbox(() => Promise.resolve(undefined), 10, 5)).toBeUndefined();
  });
});

describe("Inbox", () => {
  it("keeps the pages loaded with Load more when the task changes", async () => {
    const page = server(1500);
    const inbox = vi.spyOn(api, "inbox").mockImplementation((_n, _s, offset = 0, limit = 1000) => page(offset, limit));
    const uploads: Uploads = { slots: {}, start: () => Promise.resolve([]), dismiss: vi.fn() };
    const props = { setTask: vi.fn(), run: <T,>(p: Promise<T>) => p, uploads };
    const { rerender } = render(<Inbox task={taskDetail({ inbox_count: 1500 })} {...props} />);
    await screen.findByText(/1000 loaded/);
    // Show every card loaded, then the next page.
    for (let i = 0; i < 4; i++) fireEvent.click(screen.getByText(/Show more/));
    fireEvent.click(await screen.findByText(/Load more/));
    await waitFor(() => { expect(screen.queryByText(/loaded;/)).toBeNull(); }); // all 1500 loaded
    inbox.mockClear();
    rerender(<Inbox task={taskDetail({ inbox_count: 1500, revision: 4 })} {...props} />);
    await waitFor(() => { expect(inbox).toHaveBeenCalledTimes(2); });
    expect(inbox.mock.calls.map((c) => c[2])).toEqual([0, 1000]);
    await new Promise((r) => setTimeout(r, 10));
    expect(screen.queryByText(/Load more/)).toBeNull();
  }, 20000); // renders 1500 cards: slow on a busy machine
});

describe("stillGone", () => {
  it("keeps changes being saved and those newer than the list, and forgets the rest", () => {
    const gone = new Map<string, number | null>([["a", null], ["b", 4], ["c", 5], ["d", 6]]);
    expect([...stillGone(gone, 5).keys()]).toEqual(["a", "d"]);
  });
});

const suggest = (i: number): InboxEntry => ({
  ...entry(i), prediction: { label: "cat", labels: ["cat"], top_label: "cat", rejected: false, confidence: 0.8, similarity: 0.5,
    scores: { cat: 0.8, dog: 0.2 }, evidence: [], error: null },
});
const classes = [{ name: "cat", description: "", count: 1, reject_below: null }, { name: "dog", description: "", count: 0, reject_below: null }];
const later = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** A server whose inbox holds ``n`` images, answering each request after ``ms``, and the Inbox
 * showing its task as the App would (each answer's task shown, so the list reloads each time). */
function labelling(n: number, ms: number, uploads: Uploads) {
  let waiting = Array.from({ length: n }, (_, i) => suggest(i));
  let revision = 1;
  const accepted: Record<string, string>[] = [];
  const task = () => taskDetail({ revision, inbox_count: waiting.length, classes });
  vi.spyOn(api, "inbox").mockImplementation(async () => {
    const items = waiting; // as it is when asked: the reply may arrive after later changes
    await later(ms);
    return { items, total: items.length, offset: 0 };
  });
  vi.spyOn(api, "acceptInbox").mockImplementation(async (_n, labels) => {
    await later(ms);
    accepted.push(labels);
    waiting = waiting.filter((i) => !(i.id in labels));
    revision += 1;
    return task();
  });
  function Harness() {
    const [t, setT] = useState<TaskDetail>(task);
    return <Inbox task={t} setTask={setT} run={(p) => p} uploads={uploads} />;
  }
  render(<Harness />);
  return { accepted };
}

describe("Inbox keys", () => {
  const idle: Uploads = { slots: {}, start: () => Promise.resolve([]), dismiss: vi.fn() };

  it.each([[0, 0], [5, 30], [40, 10]])("label one image per key press, in order (%i ms apart, server %i ms)", async (gap, ms) => {
    const { accepted } = labelling(12, ms, idle);
    await screen.findByAltText("0.jpg");
    for (let k = 0; k < 10; k++) {
      fireEvent.keyDown(document.body, { key: k % 2 ? "2" : "Enter" });
      if (gap) await act(() => later(gap));
    }
    await waitFor(() => { expect(accepted).toHaveLength(10); });
    expect(accepted).toEqual(Array.from({ length: 10 }, (_, k) => ({ [`i${String(k)}`]: k % 2 ? "dog" : "cat" })));
    await screen.findByText(/^2 waiting/);
    await waitFor(() => { expect(document.querySelectorAll(".inbox-card")).toHaveLength(2); });
  });

  it("keep labelling while images are being uploaded to the inbox", async () => {
    const uploading: Uploads = { ...idle, slots: { inbox: { skipped: [], progress: { done: 100, total: 300, cancel: vi.fn() } } } };
    const { accepted } = labelling(3, 0, uploading);
    await screen.findByAltText("0.jpg");
    fireEvent.keyDown(document.body, { key: "Enter" });
    fireEvent.keyDown(document.body, { key: "Enter" });
    await waitFor(() => { expect(accepted).toEqual([{ i0: "cat" }, { i1: "cat" }]); });
    expect(screen.getByText(/Uploading 100 \/ 300/)).toBeDefined();
  });
});
