import { useEffect, useId, useRef } from "react";
import { topModal } from "./hooks";

export interface LightboxItem {
  url: string; // the image (large)
  title: string;
  details?: string[]; // lines under the title: source, date, tags, note...
}

interface Props {
  items: LightboxItem[];
  index: number;
  onIndex: (i: number) => void;
  onClose: () => void;
  /** Extra keys while open, e.g. 1-9 to relabel; return true if handled. */
  onKey?: (e: KeyboardEvent, index: number) => boolean;
  footer?: React.ReactNode;
}

/** A full-size image viewer: ←/→ step through ``items``, Esc closes. */
export default function Lightbox({ items, index, onIndex, onClose, onKey, footer }: Props) {
  const id = useId();
  const box = useRef<HTMLDivElement>(null);
  const item = items[index] as LightboxItem | undefined;
  const latest = useRef({ index, items, onIndex, onClose, onKey });
  latest.current = { index, items, onIndex, onClose, onKey };

  useEffect(() => {
    const prev = document.activeElement;
    box.current?.focus();
    // Captured before the page's own shortcuts, so they don't act on what is behind the viewer.
    const on = (e: KeyboardEvent) => {
      const { index: i, items: list } = latest.current;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const top = topModal();
      if (top && top !== box.current) return; // a dialog opened over the viewer has the keys
      const t = e.target instanceof Element ? e.target : null;
      if (e.key !== "Escape" && t?.closest("input, select, textarea")) return; // typing
      if ((e.key === "Enter" || e.key === " ") && t?.closest("button")) return; // pressing the button
      let handled = true;
      if (e.key === "Escape") latest.current.onClose();
      else if (e.key === "ArrowRight" || e.key === "j") latest.current.onIndex(Math.min(list.length - 1, i + 1));
      else if (e.key === "ArrowLeft" || e.key === "k") latest.current.onIndex(Math.max(0, i - 1));
      else handled = latest.current.onKey?.(e, i) ?? false;
      if (handled) { e.preventDefault(); e.stopPropagation(); }
    };
    window.addEventListener("keydown", on, true);
    return () => {
      window.removeEventListener("keydown", on, true);
      if (prev instanceof HTMLElement) prev.focus();
    };
  }, []);

  useEffect(() => { if (!item) onClose(); }, [item, onClose]);
  if (!item) return null;
  return (
    <div className="modal-backdrop lightbox" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div ref={box} className="lightbox-box" role="dialog" aria-modal="true" aria-labelledby={`${id}-title`} tabIndex={-1}>
        <div className="row lightbox-head">
          <strong id={`${id}-title`}>{item.title}</strong>
          <span className="muted">{index + 1} / {items.length}</span>
          <span className="spacer" />
          <button className="ghost small" aria-label="Previous" disabled={index === 0} onClick={() => { onIndex(index - 1); }}>←</button>
          <button className="ghost small" aria-label="Next" disabled={index === items.length - 1} onClick={() => { onIndex(index + 1); }}>→</button>
          <button className="ghost small" aria-label="Close" onClick={onClose}>×</button>
        </div>
        <img className="lightbox-img" src={item.url} alt={item.title} />
        {item.details && item.details.length > 0 && (
          <div className="muted lightbox-details">{item.details.map((d, i) => <div key={i}>{d}</div>)}</div>
        )}
        {footer}
      </div>
    </div>
  );
}
