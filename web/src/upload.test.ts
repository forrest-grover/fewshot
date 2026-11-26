import { describe, expect, it } from "vitest";
import { isAbort, sendInBatches, splitBytes } from "./upload";

const file = (name: string, size: number) => new File([new Uint8Array(size)], name);
const names = (fs: File[]) => fs.map((f) => f.name).join("");

describe("splitBytes", () => {
  it("keeps order and starts a new run before the limit is passed", () => {
    const fs = [file("a", 4), file("b", 4), file("c", 4), file("d", 20), file("e", 1)];
    expect(splitBytes(fs, 10).map(names)).toEqual(["ab", "c", "d", "e"]); // d is larger than the limit: alone
    expect(splitBytes([], 10)).toEqual([]);
  });
});

describe("sendInBatches", () => {
  const files = Array.from({ length: 7 }, (_, i) => file(String(i), 10));
  const limits = { files: 3, bytes: 25 };

  it("sends every file once, in order, within both limits, reporting progress", async () => {
    const sent: string[] = [];
    const progress: string[] = [];
    const prepared: number[] = [];
    await sendInBatches(files, (b) => Promise.resolve(names(b)), {
      signal: new AbortController().signal, limits,
      prepare: (fs) => { prepared.push(fs.length); return Promise.resolve(fs); },
      onBatch: (r) => { sent.push(r); },
      onProgress: (done, total) => { progress.push(`${String(done)}/${String(total)}`); },
    });
    expect(prepared).toEqual([3, 3, 1]); // a chunk at a time, not everything up front
    expect(sent).toEqual(["01", "2", "34", "5", "6"]); // 3 files of 10 bytes exceed 25 bytes
    expect(progress).toEqual(["0/7", "2/7", "3/7", "5/7", "6/7", "7/7"]);
  });

  it("stops between batches once aborted, keeping what was sent", async () => {
    const ctl = new AbortController();
    const sent: string[] = [];
    const p = sendInBatches(files, (b) => Promise.resolve(names(b)), {
      signal: ctl.signal, limits,
      onBatch: (r) => { sent.push(r); if (sent.length === 2) ctl.abort(); },
    });
    await expect(p).rejects.toSatisfy(isAbort);
    expect(sent).toEqual(["01", "2"]);
  });

  it("passes the signal to the request and stops at the first failure", async () => {
    const ctl = new AbortController();
    let seen: AbortSignal | null = null;
    let calls = 0;
    const p = sendInBatches(files, (_b, signal) => {
      seen = signal;
      calls += 1;
      return Promise.reject(new Error("413 Payload Too Large"));
    }, { signal: ctl.signal, limits, onBatch: () => { /* none */ } });
    await expect(p).rejects.toThrow("413");
    expect(seen).toBe(ctl.signal);
    expect(calls).toBe(1);
  });
});
