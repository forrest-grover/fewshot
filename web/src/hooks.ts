import { useEffect, useRef, useState } from "react";
import { api, HttpError, type Job, type Prediction, type TaskDetail } from "./api";

const IMAGE_RE = /\.(jpe?g|png|webp|bmp|gif|tiff?)$/i;

/** Drag type carrying example ids (JSON array) when a thumbnail is dragged to another class. */
export const EXAMPLE_DRAG = "application/x-fewshot-examples";

export const isImage = (f: File) => f.type.startsWith("image/") || IMAGE_RE.test(f.name);

/** Elements that hold typed text (and their own Ctrl+Z); a focused checkbox or button doesn't. */
const TEXT_ENTRY = "textarea, [contenteditable=''], [contenteditable='true'], "
  + "input:not([type=checkbox]):not([type=radio]):not([type=button]):not([type=submit]):not([type=reset]):not([type=file])";
/** Elements that take typed keys: text entry, and a select (typing picks an option). */
const TYPING = `select, ${TEXT_ENTRY}`;

/** True when keys typed at ``target`` belong to a form field rather than to page shortcuts. */
export function typingAt(target: EventTarget | null): boolean {
  // The target is the window or document when nothing is focused.
  return target instanceof Element && !!target.closest(TYPING);
}
const typing = (e: Event) => typingAt(e.target);

/** The dialog in front, if one is open: the last modal in the document (a nested or later one
 * opened over an earlier one). While one is open, page shortcuts must not act behind it. */
export function topModal(): Element | null {
  const all = document.querySelectorAll("[aria-modal='true']");
  return all.length ? all[all.length - 1] : null;
}

/** Marks a dialog (the image viewer) in which Ctrl+Z still undoes the task's last change. */
export const ALLOWS_UNDO = "data-allows-undo";

/** What Ctrl/Cmd+Z (undo), Ctrl/Cmd+Shift+Z or Ctrl+Y (redo) should do to the task: null when it
 * isn't one of those, when a text field has focus (where it undoes the typing; a select, checkbox
 * or button left focused after using it doesn't count), or when a dialog other than the image
 * viewer is in front. */
export function taskUndoKey(e: KeyboardEvent): "undo" | "redo" | null {
  const key = e.key.toLowerCase();
  if (!(e.ctrlKey || e.metaKey) || e.altKey || (key !== "z" && key !== "y")) return null;
  const top = topModal();
  if (top && !top.hasAttribute(ALLOWS_UNDO)) return null;
  if (e.target instanceof Element && e.target.closest(TEXT_ENTRY)) return null;
  return key === "y" || e.shiftKey ? "redo" : "undo";
}

/** The toast after an undo or redo that returned ``t``: what was undone is now what redo would do
 * (and the reverse), so the label comes from the response, not from the task as shown before it
 * (``fallback``), which may be out of date. ``then`` offers the reverse if it is possible. */
export function stepToast(which: "undo" | "redo", t: TaskDetail, fallback: string) {
  const what = (which === "undo" ? t.redo_label : t.undo_label) || fallback || "the last change";
  return {
    text: `${which === "undo" ? "Undid" : "Redid"} ${what}`,
    then: which === "undo" ? (t.can_redo ? "redo" as const : null) : (t.can_undo ? "undo" as const : null),
  };
}

/** Lets one async action run at a time: ``run`` ignores calls (resolving to undefined) while an
 * earlier one is still in flight, e.g. a held-down Ctrl+Z. */
export class Gate {
  private busy = false;
  get running(): boolean { return this.busy; }
  async run<T>(f: () => Promise<T>): Promise<T | undefined> {
    if (this.busy) return undefined;
    this.busy = true;
    try {
      return await f();
    } finally {
      this.busy = false;
    }
  }
}

const FOCUSABLE = "button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), "
  + "textarea:not([disabled]), [tabindex]:not([tabindex='-1'])";

/** Keep Tab and Shift+Tab inside dialog ``root`` while it is the dialog in front, so keyboard
 * focus can't wander to the page behind it. */
export function useFocusTrap(root: React.RefObject<HTMLElement | null>) {
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const box = root.current;
      if (e.key !== "Tab" || !box || topModal() !== box) return;
      const els = Array.from(box.querySelectorAll<HTMLElement>(FOCUSABLE));
      const active = document.activeElement;
      const first = els[0] as HTMLElement | undefined, last = els[els.length - 1] as HTMLElement | undefined;
      if (!first || !last) { e.preventDefault(); box.focus(); return; }
      const outside = !(active instanceof Node && box.contains(active)) || active === box;
      if (e.shiftKey && (outside || active === first)) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && (outside || active === last)) { e.preventDefault(); first.focus(); }
    };
    window.addEventListener("keydown", on, true);
    return () => { window.removeEventListener("keydown", on, true); };
  }, [root]);
}

/** Controls whose own keys (Enter, Space, arrows) must keep working. A checkbox only takes Enter and Space. */
const CONTROL = "button, a[href], summary, input[type=radio], [role=tab], [role=button], [role=option], [role=menuitem]";
const CONTROL_KEYS = new Set(["Enter", " ", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]);

/** True when a focused button, link, checkbox or tab should handle ``e`` itself. */
function forControl(e: KeyboardEvent): boolean {
  if (!(e.target instanceof Element)) return false;
  return (CONTROL_KEYS.has(e.key) && !!e.target.closest(CONTROL))
    || ((e.key === "Enter" || e.key === " ") && e.target.matches("input[type=checkbox]"));
}

/** Call ``onKey`` for key presses without Ctrl/Alt/Meta that aren't typed into a form field, nor
 * Enter/Space/arrows meant for a focused button, link or tab, while no dialog is open. */
export function useShortcuts(onKey: (e: KeyboardEvent) => void) {
  const latest = useRef(onKey);
  latest.current = onKey;
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (!e.ctrlKey && !e.altKey && !e.metaKey && !e.defaultPrevented && !typing(e) && !forControl(e) && !topModal()) {
        latest.current(e);
      }
    };
    window.addEventListener("keydown", on);
    return () => { window.removeEventListener("keydown", on); };
  }, []);
}

/** Call ``onFiles`` with the images in a paste (Ctrl/Cmd+V) that isn't into a form field. */
export function usePasteImages(onFiles: (files: File[]) => void) {
  const latest = useRef(onFiles);
  latest.current = onFiles;
  useEffect(() => {
    const on = (e: ClipboardEvent) => {
      if (typing(e)) return;
      const files = Array.from(e.clipboardData?.files ?? []).filter(isImage);
      if (!files.length) return;
      e.preventDefault();
      // Pasted screenshots are all called "image.png"; give each a distinct, dated name.
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      latest.current(files.map((f, i) => (f.name === "image.png" || !f.name
        ? new File([f], `pasted-${stamp}-${String(i)}.${f.type.split("/")[1] || "png"}`, { type: f.type })
        : f)));
    };
    window.addEventListener("paste", on);
    return () => { window.removeEventListener("paste", on); };
  }, []);
}

/** One result. ``file`` is what was classified (scaled down), and is also what "Add as" uploads,
 * so adding it reuses the embedding computed for the prediction. A result for an image URL has no
 * ``file``: its ``url`` is the remote image, which "Add as" has the server fetch again. */
export interface Item {
  key: number; name: string; file?: File; url: string; pred: Prediction; adding?: boolean;
  added?: { label: string; id: string }; // set once added as an example (shown while that example exists)
}

/** Prediction results for one task, kept by the App so they survive switching tabs. */
export interface Results {
  task: string;
  items: Item[];
  batch: number | null; // the classification in progress, if any (it lands here even if the task is renamed meanwhile)
  current: number | null; // key of the result that keyboard shortcuts act on
}

/** State for the Predict tab, held above it. Results (and their preview URLs) are dropped when
 * another task is opened (not when ``rename`` moved them to the task's new name) and when the App
 * unmounts. */
export function usePredictResults(task: string | null) {
  const [results, setResults] = useState<Results>({ task: task ?? "", items: [], batch: null, current: null });
  const urls = useRef(new Set<string>()); // preview object URLs still in use
  const name = task ?? "";
  const renamed = useRef<{ from: string; to: string } | null>(null);
  // Results already moved to the task's new name, while the route still shows the old one, stay.
  // Once the route reaches the new name the rename is over, so a later task reusing the old name
  // (say, a duplicate) starts with fresh results.
  if (renamed.current?.to === name) renamed.current = null;
  const moving = renamed.current?.from === name && renamed.current.to === results.task;
  if (results.task !== name && !moving) { // another task: adjust during render, as React recommends
    revoke(urls.current);
    setResults({ task: name, items: [], batch: null, current: null });
  }
  useEffect(() => {
    const live = urls.current;
    return () => { revoke(live); };
  }, []);
  /** Keep the results when their task is renamed. */
  const rename = (to: string) => {
    renamed.current = { from: name, to };
    setResults((r) => (r.task === name ? { ...r, task: to } : r));
  };
  return { results, setResults, urls, rename };
}

export function revoke(urls: Set<string>) {
  urls.forEach((u) => { URL.revokeObjectURL(u); });
  urls.clear();
}

/** Collect image files from a drop, including files inside dropped folders. */
export async function filesFromDrop(dt: DataTransfer): Promise<File[]> {
  const entries = Array.from(dt.items)
    .map((i) => i.webkitGetAsEntry())
    .filter((e): e is FileSystemEntry => e !== null);
  if (!entries.length) return Array.from(dt.files);
  const out: File[] = [];
  const walk = async (e: FileSystemEntry): Promise<void> => {
    if (e.isFile) {
      out.push(await new Promise<File>((res, rej) => { (e as FileSystemFileEntry).file(res, rej); }));
    } else if (e.isDirectory) {
      const reader = (e as FileSystemDirectoryEntry).createReader();
      let batch: FileSystemEntry[];
      do {
        batch = await new Promise((res, rej) => { reader.readEntries(res, rej); });
        for (const c of batch) await walk(c);
      } while (batch.length);
    }
  };
  for (const e of entries) await walk(e);
  return out;
}

const POLL_MS = 500;
const POLL_TRIES = 5; // failed requests in a row before giving up on a job

/** Poll job ``jobId`` (if any) every half second, passing each state to ``onJob`` (the last one is
 * finished). A failed request is retried (with growing pauses) up to POLL_TRIES times in a row,
 * since the job keeps running on the server; then, or at once if the server no longer knows the
 * job, it goes to ``onError`` and polling stops. Each request waits for the one before. Unmounting
 * stops polling too. */
export function usePollJob<T>(jobId: string | undefined, onJob: (j: Job<T>) => void, onError: (e: Error) => void) {
  const latest = useRef({ onJob, onError });
  latest.current = { onJob, onError };
  useEffect(() => {
    if (!jobId) return;
    let live = true;
    let failures = 0;
    let timer: ReturnType<typeof setTimeout>;
    const poll = () => {
      api.job<T>(jobId).then((j) => {
        if (!live) return;
        failures = 0;
        latest.current.onJob(j);
        if (j.status === "running") timer = setTimeout(poll, POLL_MS);
      }, (e: unknown) => {
        if (!live) return;
        failures += 1;
        if (failures < POLL_TRIES && !(e instanceof HttpError && e.status === 404)) {
          timer = setTimeout(poll, POLL_MS * 2 ** failures);
          return;
        }
        latest.current.onError(e instanceof Error ? e : new Error(String(e)));
      });
    };
    timer = setTimeout(poll, POLL_MS);
    return () => { live = false; clearTimeout(timer); };
  }, [jobId]);
}
