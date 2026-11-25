import { useEffect, useRef, useState } from "react";
import type { Prediction } from "./api";

const IMAGE_RE = /\.(jpe?g|png|webp|bmp|gif|tiff?)$/i;

/** Drag type carrying example ids (JSON array) when a thumbnail is dragged to another class. */
export const EXAMPLE_DRAG = "application/x-fewshot-examples";

export const isImage = (f: File) => f.type.startsWith("image/") || IMAGE_RE.test(f.name);

/** Elements that take typed keys; a focused checkbox or button doesn't. */
const TYPING = "textarea, select, [contenteditable=''], [contenteditable='true'], "
  + "input:not([type=checkbox]):not([type=radio]):not([type=button]):not([type=submit]):not([type=reset]):not([type=file])";

/** True when keys typed now belong to a form field or dialog rather than to page shortcuts. */
function typing(e: Event): boolean {
  // The target is the window or document when nothing is focused.
  return e.target instanceof Element
    && !!e.target.closest(TYPING);
}

/** Call ``onKey`` for key presses without Ctrl/Alt/Meta that aren't typed into a form field. */
export function useShortcuts(onKey: (e: KeyboardEvent) => void) {
  const latest = useRef(onKey);
  latest.current = onKey;
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (!e.ctrlKey && !e.altKey && !e.metaKey && !e.defaultPrevented && !typing(e)) latest.current(e);
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
 * so adding it reuses the embedding computed for the prediction. */
export interface Item {
  key: number; name: string; file: File; url: string; pred: Prediction; adding?: boolean; addedAs?: string;
}

/** Prediction results for one task, kept by the App so they survive switching tabs. */
export interface Results {
  task: string;
  items: Item[];
  busy: boolean;
  current: number | null; // key of the result that keyboard shortcuts act on
}

/** State for the Predict tab, held above it. Results (and their preview URLs) are dropped when
 * ``task`` changes and when the App unmounts. */
export function usePredictResults(task: string | null) {
  const [results, setResults] = useState<Results>({ task: task ?? "", items: [], busy: false, current: null });
  const urls = useRef(new Set<string>()); // preview object URLs still in use
  useEffect(() => {
    const live = urls.current;
    setResults({ task: task ?? "", items: [], busy: false, current: null });
    return () => { revoke(live); };
  }, [task]);
  return { results, setResults, urls };
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
