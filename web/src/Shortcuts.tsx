import { useEffect, useRef } from "react";
import { useFocusTrap } from "./hooks";

const KEYS: [string, [string, string][]][] = [
  ["Everywhere", [
    ["Ctrl+Z", "Undo the task's last change"],
    ["Ctrl+Shift+Z / Ctrl+Y", "Redo"],
    ["←  →  (on the tabs)", "Switch tab"],
    ["?", "Show these keys"],
  ]],
  ["Examples", [
    ["1–9", "Move ticked examples to that class"],
    ["Delete", "Delete ticked examples"],
    ["Esc", "Untick all"],
    ["Ctrl+V", "Paste images into the class under the pointer"],
    ["Enter (on a thumbnail)", "Open it full size"],
  ]],
  ["Inbox", [
    ["Enter", "Accept the highlighted suggestion"],
    ["1–9", "Label the highlighted image as that class"],
    ["Delete", "Dismiss it"],
    ["←  →  /  j  k", "Move"],
    ["Space", "View it full size"],
  ]],
  ["Predict", [
    ["Enter", "Add the highlighted image as its prediction"],
    ["1–9", "Add it as that class"],
    ["↑  ↓  /  j  k", "Move"],
    ["Ctrl+V", "Paste images to classify"],
  ]],
  ["Full-size viewer", [
    ["←  →", "Previous / next image"],
    ["1–9", "Move the example to that class"],
    ["Delete", "Delete it"],
    ["Esc", "Close"],
  ]],
];

/** The keyboard shortcuts, opened with "?". */
export default function Shortcuts({ onClose }: { onClose: () => void }) {
  const box = useRef<HTMLDivElement>(null);
  useFocusTrap(box);
  useEffect(() => {
    const on = (e: KeyboardEvent) => { if (e.key === "Escape" || e.key === "?") { e.preventDefault(); e.stopPropagation(); onClose(); } };
    window.addEventListener("keydown", on, true);
    return () => { window.removeEventListener("keydown", on, true); };
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div ref={box} className="modal shortcuts" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
        <div className="row"><h3 className="modal-title">Keyboard shortcuts</h3><span className="spacer" />
          <button className="ghost small" aria-label="Close" onClick={onClose} autoFocus>×</button></div>
        {KEYS.map(([group, keys]) => (
          <section key={group}>
            <h4>{group}</h4>
            <dl>{keys.map(([k, what]) => <div key={k} className="row"><dt><kbd>{k}</kbd></dt><dd>{what}</dd></div>)}</dl>
          </section>
        ))}
      </div>
    </div>
  );
}
