import { useCallback, useEffect, useRef, useState } from "react";
import { HttpError, type TaskDetail } from "./api";
import { shrinkImages } from "./images";
import { type BatchLimits, isAbort, newUndoGroup, sendInBatches } from "./upload";

/** A file an upload couldn't use (or, with no ``file``, a whole batch of them) and why. */
export interface Skip { file: string | null; error: string }
/** An upload in progress: images sent so far, and how to stop it (after the batch being sent,
 * so that every image the server stores is counted: ``stopping`` until then). */
export interface UploadProgress { done: number; total: number; cancel: () => void; stopping?: boolean }
/** One place uploads go to (a class, the inbox): the upload in progress, if any, the files the
 * last one skipped, and how many of its images were moved from another class (examples only). */
export interface UploadSlot { progress?: UploadProgress; skipped: Skip[]; moved?: number }
/** An upload that stopped before the end: cancelled (``error`` null), or failed. ``added`` counts
 * the images the server confirmed adding. */
export interface UploadStop { task: string; added: number; total: number; error: Error | null }
/** What the server answers each upload batch with: the images new to the task (``added``), there
 * already (``unchanged``), and already examples of another class, now moved to this one
 * (``moved``: example uploads only; the inbox has none). */
interface Uploaded { added: unknown[]; unchanged?: unknown[]; errors: Skip[]; task: TaskDetail; moved?: readonly unknown[] }
/** Sends one batch to task ``task`` (its current name), all of whose requests carry ``undoGroup``,
 * so the drop is one undo step (with the other uploads to the task running at the same time). */
export type SendBatch = (batch: File[], signal: AbortSignal, undoGroup: string, task: string) => Promise<Uploaded>;

interface UploadOptions {
  onTask: (t: TaskDetail) => void; // each batch's updated task
  onStop: (s: UploadStop) => void; // an upload was cancelled or failed part-way
  limits?: BatchLimits;
  prepare?: (files: File[]) => Promise<File[]>; // default: scale images down
}

/** The undo groups of the uploads ``stop`` stopped, each with the task revisions its batches
 * made (see ``uploadSteps``); none if no batch of theirs landed. */
export type Stopped = Map<string, number[]>;

/** An upload being sent. */
interface Running {
  task: string; // its task's name (updated if the task is renamed meanwhile)
  group: string; // its undo group: shared by the uploads to the task running at the same time
  added: number; // images the server confirmed so far
  stop: () => void; // after the batch in flight
  stopping: AbortSignal; // aborted once it has been asked to stop
  abort: () => void; // at once
  done: Promise<void>; // once it has ended, however
}

/** Uploads for the open task, kept by the App so they go on while its tabs are switched: one at
 * a time per slot (a class, the inbox), in batches (see sendInBatches), with progress and
 * skipped files per slot. They stop when the user cancels one, or when another task is opened
 * (not when this one is renamed, see ``rename``): after the batch being sent, so the report to
 * ``onStop`` counts every image that got in. Unmounting the App aborts them at once. */
export function useUploads(task: string | null, options: UploadOptions) {
  const name = task ?? "";
  const opts = useRef(options);
  opts.current = options;
  const current = useRef(name);
  current.current = name;
  const [state, setState] = useState<{ task: string; slots: Partial<Record<string, UploadSlot>> }>({ task: name, slots: {} });
  // Slots already moved to the task's new name, while the route still shows the old one, stay
  // (as in usePredictResults).
  const renamed = useRef<{ from: string; to: string } | null>(null);
  if (renamed.current?.to === name) renamed.current = null;
  const moving = renamed.current?.from === name && renamed.current.to === state.task;
  const [epoch, setEpoch] = useState(0); // counts the times another task was opened
  if (state.task !== name && !moving) { // another task: adjust during render
    setState({ task: name, slots: {} });
    setEpoch((e) => e + 1);
  }
  const live = useRef(new Set<Running>());
  // The task revisions made by the batches of each undo group a running upload uses.
  const revisions = useRef(new Map<string, number[]>());
  useEffect(() => {
    const set = live.current;
    return () => { set.forEach((r) => { r.stop(); }); };
  }, [epoch]);
  useEffect(() => {
    const set = live.current;
    return () => { set.forEach((r) => { r.abort(); }); };
  }, []);

  const setSlot = useCallback((t: string, slot: string, f: (s: UploadSlot | undefined) => UploadSlot) => {
    setState((s) => (s.task === t ? { ...s, slots: { ...s.slots, [slot]: f(s.slots[slot]) } } : s));
  }, []);

  /** Upload ``files`` to ``slot`` with ``send``. Resolves to the files skipped, or null if it
   * stopped before the end (cancelled or failed; reported to ``onStop``). A batch the server
   * refused as a whole (400: no usable image) is skipped, not the end of the upload. */
  const start = useCallback(async (slot: string, files: File[], send: SendBatch): Promise<Skip[] | null> => {
    const hard = new AbortController(), soft = new AbortController();
    let finished: () => void = () => { /* set below */ };
    const task = current.current;
    // Uploads running at the same time share one undo group, so that batches to two classes,
    // sent in turn, still make one undo step. Not one that is stopping: its last batch may still
    // land, and a drop made now would join the part of it the user was told got in, so that
    // undoing the new drop would undo that too.
    const group = [...live.current].find((x) => x.task === task && !x.stopping.aborted)?.group ?? newUndoGroup();
    const r: Running = {
      task, group, added: 0, stopping: soft.signal,
      stop: () => {
        if (soft.signal.aborted) return;
        soft.abort();
        setSlot(r.task, slot, (s) => (s?.progress ? { ...s, progress: { ...s.progress, stopping: true } } : { skipped: s?.skipped ?? [] }));
      },
      abort: () => { hard.abort(); },
      done: new Promise((res) => { finished = res; }),
    };
    live.current.add(r);
    if (!revisions.current.has(group)) revisions.current.set(group, []);
    const skips: Skip[] = [];
    let moved = 0;
    setSlot(r.task, slot, () => ({ skipped: [] }));
    const one = async (fs: File[], signal: AbortSignal): Promise<Uploaded | null> => {
      try {
        return await send(fs, signal, group, r.task);
      } catch (e) {
        if (!(e instanceof HttpError && e.status === 400)) throw e;
        skips.push({ file: null, error: e.message });
        return null;
      }
    };
    try {
      await sendInBatches(files, one, {
        signal: hard.signal, stop: soft.signal, prepare: opts.current.prepare ?? shrinkImages, limits: opts.current.limits,
        onProgress: (done, total) => {
          setSlot(r.task, slot, (s) => ({
            skipped: s?.skipped ?? [], progress: { done, total, cancel: r.stop, stopping: soft.signal.aborted },
          }));
        },
        onBatch: (b) => {
          if (!b) return;
          r.added += b.added.length + (b.unchanged?.length ?? 0) + (b.moved?.length ?? 0); // each is there now
          revisions.current.get(group)?.push(b.task.revision);
          moved += b.moved?.length ?? 0;
          skips.push(...b.errors);
          opts.current.onTask(b.task);
        },
      });
      return skips;
    } catch (e) {
      const error = isAbort(e) ? null : e instanceof Error ? e : new Error(String(e));
      if (!hard.signal.aborted) opts.current.onStop({ task: r.task, added: r.added, total: files.length, error });
      return null;
    } finally {
      live.current.delete(r);
      if (![...live.current].some((x) => x.group === group)) revisions.current.delete(group);
      setSlot(r.task, slot, () => ({ skipped: skips, ...(moved ? { moved } : {}) }));
      finished();
    }
  }, [setSlot]);

  /** Forget what ``slot``'s last upload skipped or moved. */
  const dismiss = useCallback((slot: string) => { setSlot(current.current, slot, () => ({ skipped: [] })); }, [setSlot]);
  /** Stop every upload of the open task after its batch in flight; resolves, once they have
   * ended, to their undo groups and the revisions their batches made (those of uploads in the
   * same group that ended earlier too, since they are part of the same undo steps). */
  const stop = useCallback(async (): Promise<Stopped> => {
    const mine = [...live.current].filter((r) => r.task === current.current);
    const groups = new Map(mine.map((r) => [r.group, revisions.current.get(r.group) ?? []]));
    mine.forEach((r) => { r.stop(); });
    await Promise.all(mine.map((r) => r.done));
    return new Map([...groups].filter(([, revs]) => revs.length > 0));
  }, []);
  /** Keep uploads (and their slots) going when their task is renamed to ``to``. */
  const rename = useCallback((to: string) => {
    const from = current.current;
    renamed.current = { from, to };
    live.current.forEach((r) => { if (r.task === from) r.task = to; });
    setState((s) => (s.task === from ? { ...s, task: to } : s));
  }, []);
  const slots = state.task === name || moving ? state.slots : {};
  /** Whether an upload is being sent to the open task. */
  const busy = Object.values(slots).some((s) => s?.progress);
  return { slots, start, dismiss, stop, rename, busy };
}
export type Uploads = ReturnType<typeof useUploads>;
