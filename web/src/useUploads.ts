import { useCallback, useEffect, useRef, useState } from "react";
import { HttpError, type TaskDetail } from "./api";
import { shrinkImages } from "./images";
import { type BatchLimits, isAbort, newUndoGroup, sendInBatches } from "./upload";

/** A file an upload couldn't use (or, with no ``file``, a whole batch of them) and why. */
export interface Skip { file: string | null; error: string }
/** An upload in progress: images sent so far, and how to stop it. */
export interface UploadProgress { done: number; total: number; cancel: () => void }
/** One place uploads go to (a class, the inbox): the upload in progress, if any, and the files
 * the last one skipped. */
export interface UploadSlot { progress?: UploadProgress; skipped: Skip[] }
/** An upload that stopped before the end: cancelled (``error`` null), or failed. ``added`` counts
 * the images the server confirmed adding. */
export interface UploadStop { task: string; added: number; total: number; error: Error | null }
/** What the server answers each upload batch with. */
interface Uploaded { added: unknown[]; errors: Skip[]; task: TaskDetail }
/** Sends one batch, all of whose requests carry ``undoGroup``, so the drop is one undo step. */
export type SendBatch = (batch: File[], signal: AbortSignal, undoGroup: string) => Promise<Uploaded>;

interface UploadOptions {
  onTask: (t: TaskDetail) => void; // each batch's updated task
  onStop: (s: UploadStop) => void; // an upload was cancelled or failed part-way
  limits?: BatchLimits;
  prepare?: (files: File[]) => Promise<File[]>; // default: scale images down
}

/** Uploads for the open task, kept by the App so they go on while its tabs are switched: one at
 * a time per slot (a class, the inbox), in batches (see sendInBatches), with progress and
 * skipped files per slot. They stop when the user cancels one, or when another task is opened
 * (or the App unmounts); a stop is reported to ``onStop``. */
export function useUploads(task: string | null, options: UploadOptions) {
  const name = task ?? "";
  const opts = useRef(options);
  opts.current = options;
  const current = useRef(name);
  current.current = name;
  const [state, setState] = useState<{ task: string; slots: Partial<Record<string, UploadSlot>> }>({ task: name, slots: {} });
  if (state.task !== name) setState({ task: name, slots: {} }); // another task: adjust during render
  const live = useRef(new Set<AbortController>());
  useEffect(() => {
    const set = live.current;
    return () => { set.forEach((c) => { c.abort(); }); set.clear(); };
  }, [name]);

  const setSlot = useCallback((t: string, slot: string, f: (s: UploadSlot | undefined) => UploadSlot) => {
    setState((s) => (s.task === t ? { ...s, slots: { ...s.slots, [slot]: f(s.slots[slot]) } } : s));
  }, []);

  /** Upload ``files`` to ``slot`` with ``send``. Resolves to the files skipped, or null if it
   * stopped before the end (cancelled or failed; reported to ``onStop``). A batch the server
   * refused as a whole (400: no usable image) is skipped, not the end of the upload. */
  const start = useCallback(async (slot: string, files: File[], send: SendBatch): Promise<Skip[] | null> => {
    const t = current.current;
    const c = new AbortController();
    live.current.add(c);
    const group = newUndoGroup();
    const skips: Skip[] = [];
    let added = 0;
    setSlot(t, slot, () => ({ skipped: [] }));
    const one = async (fs: File[], signal: AbortSignal): Promise<Uploaded | null> => {
      try {
        return await send(fs, signal, group);
      } catch (e) {
        if (!(e instanceof HttpError && e.status === 400)) throw e;
        skips.push({ file: null, error: e.message });
        return null;
      }
    };
    try {
      await sendInBatches(files, one, {
        signal: c.signal, prepare: opts.current.prepare ?? shrinkImages, limits: opts.current.limits,
        onProgress: (done, total) => {
          setSlot(t, slot, (s) => ({ skipped: s?.skipped ?? [], progress: { done, total, cancel: () => { c.abort(); } } }));
        },
        onBatch: (r) => {
          if (!r) return;
          added += r.added.length;
          skips.push(...r.errors);
          opts.current.onTask(r.task);
        },
      });
      return skips;
    } catch (e) {
      const error = isAbort(e) ? null : e instanceof Error ? e : new Error(String(e));
      opts.current.onStop({ task: t, added, total: files.length, error });
      return null;
    } finally {
      live.current.delete(c);
      setSlot(t, slot, () => ({ skipped: skips }));
    }
  }, [setSlot]);

  /** Forget the files ``slot``'s last upload skipped. */
  const dismiss = useCallback((slot: string) => { setSlot(current.current, slot, () => ({ skipped: [] })); }, [setSlot]);
  const slots = state.task === name ? state.slots : {};
  return { slots, start, dismiss };
}
export type Uploads = ReturnType<typeof useUploads>;
