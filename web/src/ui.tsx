import { memo, useRef, useState, type ReactNode } from "react";
import type { Job } from "./api";
import { filesFromDrop, isImage } from "./files";
import { usePollJob } from "./pollJob";

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
      onDragLeave={(e) => {
        // Moving onto a child (the text, a button) fires dragleave too: only leaving the zone counts.
        if (!(e.relatedTarget instanceof Node && e.currentTarget.contains(e.relatedTarget))) setOver(false);
      }}
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

/** A menu of class names (numbered 1–9 if ``numbered``) whose ``onPick`` gets the class chosen.
 * Until it is pressed or focused it holds only the current option (and ``placeholder``), so a page
 * of hundreds of them doesn't build hundreds of full class lists. */
export const ClassSelect = memo(function ClassSelect({ value = "", classes, placeholder, numbered, exclude, onPick, ...rest }: {
  value?: string; classes: string[]; placeholder?: string; numbered?: boolean; exclude?: string;
  onPick: (name: string) => void; disabled?: boolean; className?: string; "aria-label": string;
}) {
  const [full, setFull] = useState(false);
  const open = () => { if (!full) setFull(true); };
  return (
    <select {...rest} value={value} onMouseDown={open} onFocus={open} onTouchStart={open}
      onChange={(e) => { if (e.target.value) onPick(e.target.value); }}>
      {placeholder !== undefined && <option value="">{placeholder}</option>}
      {classes.map((c, i) => (c === exclude || (!full && c !== value) ? null : (
        <option key={c} value={c}>{numbered && i < 9 ? `${String(i + 1)} · ` : ""}{c}</option>
      )))}
    </select>
  );
});

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
