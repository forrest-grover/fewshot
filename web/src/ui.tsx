import { useRef, useState, type ReactNode } from "react";

const IMAGE_RE = /\.(jpe?g|png|webp|bmp|gif|tiff?)$/i;

/** Collect image files from a drop, including files inside dropped folders. */
async function filesFromDrop(dt: DataTransfer): Promise<File[]> {
  const entries = Array.from(dt.items).map((i) => i.webkitGetAsEntry?.()).filter(Boolean) as FileSystemEntry[];
  if (!entries.length) return Array.from(dt.files);
  const out: File[] = [];
  const walk = async (e: FileSystemEntry): Promise<void> => {
    if (e.isFile) {
      out.push(await new Promise<File>((res, rej) => (e as FileSystemFileEntry).file(res, rej)));
    } else if (e.isDirectory) {
      const reader = (e as FileSystemDirectoryEntry).createReader();
      let batch: FileSystemEntry[];
      do {
        batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        for (const c of batch) await walk(c);
      } while (batch.length);
    }
  };
  for (const e of entries) await walk(e);
  return out;
}

export function DropZone(props: {
  onFiles: (files: File[]) => void;
  children?: ReactNode;
  compact?: boolean;
  disabled?: boolean;
}) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const accept = (fs: File[]) => {
    const imgs = fs.filter((f) => f.type.startsWith("image/") || IMAGE_RE.test(f.name));
    if (imgs.length) props.onFiles(imgs);
  };
  return (
    <div
      className={`drop ${over ? "over" : ""} ${props.compact ? "compact" : ""} ${props.disabled ? "disabled" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={async (e) => {
        e.preventDefault();
        setOver(false);
        if (!props.disabled) accept(await filesFromDrop(e.dataTransfer));
      }}
      onClick={() => !props.disabled && input.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
    >
      <input
        ref={input}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(e) => { accept(Array.from(e.target.files ?? [])); e.target.value = ""; }}
      />
      {props.children ?? "Drop images or folders here, or click to choose"}
    </div>
  );
}

export function Bar({ value, label, strong }: { value: number; label: string; strong?: boolean }) {
  return (
    <div className={`bar ${strong ? "strong" : ""}`}>
      <span className="bar-label" title={label}>{label}</span>
      <span className="bar-track"><span className="bar-fill" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} /></span>
      <span className="bar-value">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

export function ErrorBox({ error, onClose }: { error: string | null; onClose: () => void }) {
  if (!error) return null;
  return (
    <div className="error" role="alert">
      <span>{error}</span>
      <button className="ghost" onClick={onClose} aria-label="Dismiss">×</button>
    </div>
  );
}

export const pct = (v: number | null | undefined) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
