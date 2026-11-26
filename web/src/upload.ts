/** Most files and bytes sent in one request. The server refuses more than 1000 files and (by
 * default) 1024 MB per request; smaller batches also show progress and can be cancelled between. */
export interface BatchLimits { files: number; bytes: number }
export const BATCH: BatchLimits = { files: 100, bytes: 32 * 1024 * 1024 };

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
  prepare?: (files: File[]) => Promise<File[]>; // e.g. scale images down, a batch at a time to bound memory
  onBatch: (result: R, files: File[]) => void; // each batch's result as it arrives
  onProgress?: (done: number, total: number) => void; // files sent so far
  limits?: BatchLimits;
}

/** Send ``files`` with ``send`` in batches (see BATCH), one request at a time. Rejects with the
 * first failure, or an AbortError once ``signal`` is aborted; batches already sent stay sent. */
export async function sendInBatches<R>(
  files: File[], send: (batch: File[], signal: AbortSignal) => Promise<R>,
  { signal, prepare, onBatch, onProgress, limits = BATCH }: BatchOptions<R>,
): Promise<void> {
  let done = 0;
  onProgress?.(0, files.length);
  for (let i = 0; i < files.length; i += limits.files) {
    signal.throwIfAborted();
    const chunk = files.slice(i, i + limits.files);
    const ready = prepare ? await prepare(chunk) : chunk;
    for (const batch of splitBytes(ready, limits.bytes)) {
      signal.throwIfAborted();
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
