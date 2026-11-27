/** Most files and bytes sent in one request. The server refuses more than 1000 files and (by
 * default) 1024 MB per request; smaller batches also show progress and can be cancelled between. */
export interface BatchLimits { files: number; bytes: number }
export const BATCH: BatchLimits = { files: 100, bytes: 32 * 1024 * 1024 };

/** BATCH for a server that takes at most ``maxRequestBytes`` per request (``/api/health`` says):
 * batches of at most half that, leaving room for the multipart framing, so a server configured
 * below 32 MB still takes them. Unknown (an old server): BATCH. */
export function batchLimits(maxRequestBytes?: number | null): BatchLimits {
  if (!maxRequestBytes || !(maxRequestBytes > 0)) return BATCH;
  return { files: BATCH.files, bytes: Math.max(1, Math.min(BATCH.bytes, Math.floor(maxRequestBytes / 2))) };
}

/** A new id for the undo group of one drop: the server makes its batches one undo step. A UUID
 * where the browser has ``crypto.randomUUID`` (secure contexts only: https or localhost), else
 * 32 random hex digits. */
export function newUndoGroup(): string {
  const c = crypto as Partial<Crypto> & Pick<Crypto, "getRandomValues">;
  if (c.randomUUID) return c.randomUUID();
  return Array.from(c.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** How many undo steps the batches of some uploads made, from the task revision each batch made,
 * by undo group: the server joins a batch to the step before when that step is the same group's
 * and nothing was saved since (no other change, nor a rename), so each batch whose revision
 * doesn't directly follow one of its own group's starts a step. */
export function uploadSteps(batches: Map<string, number[]>): number {
  const groupAt = new Map<number, string>();
  batches.forEach((revs, g) => { revs.forEach((r) => groupAt.set(r, g)); });
  let steps = 0;
  groupAt.forEach((g, r) => { if (groupAt.get(r - 1) !== g) steps++; });
  return steps;
}

/** ``files`` split, in order, into runs of at most ``maxBytes`` bytes (a larger file goes alone). */
export function splitBytes<T extends { size: number }>(files: T[], maxBytes: number): T[][] {
  const out: T[][] = [];
  let cur: T[] = [], bytes = 0;
  for (const f of files) {
    if (cur.length && bytes + f.size > maxBytes) { out.push(cur); cur = []; bytes = 0; }
    cur.push(f);
    bytes += f.size;
  }
  if (cur.length) out.push(cur);
  return out;
}

export interface BatchOptions<R> {
  signal: AbortSignal; // stops before the next step (and aborts a request in flight, if ``send`` passes it on)
  stop?: AbortSignal; // stops before the next step, letting a request in flight finish (and count)
  prepare?: (files: File[]) => Promise<File[]>; // e.g. scale images down, a batch at a time to bound memory
  onBatch: (result: R, files: File[]) => void; // each batch's result as it arrives
  onProgress?: (done: number, total: number) => void; // files sent so far
  limits?: BatchLimits;
}

/** Send ``files`` with ``send`` in batches (see BATCH), one request at a time. Rejects with the
 * first failure, or an AbortError once ``signal`` or ``stop`` is aborted; batches already sent stay sent. */
export async function sendInBatches<R>(
  files: File[], send: (batch: File[], signal: AbortSignal) => Promise<R>,
  { signal, stop, prepare, onBatch, onProgress, limits = BATCH }: BatchOptions<R>,
): Promise<void> {
  let done = 0;
  onProgress?.(0, files.length);
  for (let i = 0; i < files.length; i += limits.files) {
    signal.throwIfAborted();
    stop?.throwIfAborted();
    const chunk = files.slice(i, i + limits.files);
    const ready = prepare ? await prepare(chunk) : chunk;
    for (const batch of splitBytes(ready, limits.bytes)) {
      signal.throwIfAborted();
      stop?.throwIfAborted();
      const r = await send(batch, signal);
      done += batch.length;
      onBatch(r, batch);
      onProgress?.(done, files.length);
    }
  }
}

/** True for the error a fetch (or ``sendInBatches``) rejects with once it is aborted. */
export const isAbort = (e: unknown) => e instanceof DOMException && e.name === "AbortError";

/** "120 / 500" while a batched upload runs. */
export const progressText = (p: { done: number; total: number }) => `${String(p.done)} / ${String(p.total)}`;
