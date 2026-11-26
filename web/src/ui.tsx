import { useRef, useState, type ReactNode } from "react";
import type { Job } from "./api";
import { filesFromDrop, isImage, usePollJob } from "./hooks";

export function DropZone(props: {
  onFiles: (files: File[]) => void;
  children?: ReactNode;
  compact?: boolean;
  disabled?: boolean;
}) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const accept = (fs: File[]) => {
    const imgs = fs.filter(isImage);
    if (imgs.length) props.onFiles(imgs);
  };
  const hasFiles = (dt: DataTransfer) => dt.types.includes("Files"); // not a dragged thumbnail
  return (
    <div
      className={`drop ${over ? "over" : ""} ${props.compact ? "compact" : ""} ${props.disabled ? "disabled" : ""}`}
      onDragOver={(e) => { if (hasFiles(e.dataTransfer)) { e.preventDefault(); setOver(true); } }}
      onDragLeave={() => { setOver(false); }}
      onDrop={(e) => {
        setOver(false);
        if (!hasFiles(e.dataTransfer)) return;
        e.preventDefault();
        e.stopPropagation(); // handled here, not again by an enclosing drop target
        if (!props.disabled) void filesFromDrop(e.dataTransfer).then(accept);
      }}
      onClick={() => { if (!props.disabled) input.current?.click(); }}
      role="button"
      tabIndex={0}
      aria-disabled={props.disabled ?? false}
      onKeyDown={(e) => {
        if (e.key !== "Enter" && e.key !== " ") return;
        e.preventDefault(); // Space would otherwise scroll the page
        if (!props.disabled) input.current?.click();
      }}
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
      <span className="bar-track"><span className="bar-fill" style={{ width: `${String(Math.max(0, Math.min(1, value)) * 100)}%` }} /></span>
      <span className="bar-value">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

/** Polls job ``id`` while mounted (see usePollJob); renders nothing. */
export function JobPoller<T>({ id, onJob, onError }: { id: string; onJob: (j: Job<T>) => void; onError: (e: Error) => void }) {
  usePollJob<T>(id, onJob, onError);
  return null;
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
