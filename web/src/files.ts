import { useEffect, useRef } from "react";
import { typingAt } from "./keys";

// Image files from drops (with folders) and pastes, and thumbnails dragged between classes.

const IMAGE_RE = /\.(jpe?g|png|webp|bmp|gif|tiff?)$/i;

/** Drag type carrying example ids (JSON array) when a thumbnail is dragged to another class. */
export const EXAMPLE_DRAG = "application/x-fewshot-examples";

export const isImage = (f: File) => f.type.startsWith("image/") || IMAGE_RE.test(f.name);

/** Call ``onFiles`` with the images in a paste (Ctrl/Cmd+V) that isn't into a form field. */
export function usePasteImages(onFiles: (files: File[]) => void) {
  const latest = useRef(onFiles);
  latest.current = onFiles;
  useEffect(() => {
    const on = (e: ClipboardEvent) => {
      if (typingAt(e.target)) return;
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
