import { useEffect, useRef } from "react";

// Keyboard handling shared by the views: which keys belong to a form field, a focused control or
// a dialog rather than to page shortcuts, and keeping Tab inside a dialog.

/** Form fields that take keys (typing, or arrows for a slider or date); a checkbox or button doesn't. */
const FIELD = "textarea, [contenteditable=''], [contenteditable='true'], "
  + "input:not([type=checkbox]):not([type=radio]):not([type=button]):not([type=submit]):not([type=reset]):not([type=file])";
/** Fields that hold typed text, and so their own Ctrl+Z: not a slider, colour or date picker. */
const TEXT_ENTRY = FIELD.split(", ").map((s) => (s.startsWith("input")
  ? s + ":not([type=range]):not([type=color]):not([type=date]):not([type=datetime-local]):not([type=month]):not([type=week]):not([type=time])"
  : s)).join(", ");
/** Elements that take typed keys: those fields, and a select (typing picks an option). */
const TYPING = `select, ${FIELD}`;

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
