import { useEffect, useId, useRef, useState } from "react";
import { useFocusTrap } from "./hooks";

/** What the editor saves: one example's tags and note, or tags added to and removed from several. */
export type TagEdit = { tags: string[]; note: string } | { add: string[]; remove: string[] };

interface Props {
  title: string;
  allTags: string[]; // suggestions
  /** One example: its tags and note. Several: the tags any of them has (offered for removal). */
  tags: string[];
  note?: string; // given for one example only
  onSave: (edit: TagEdit) => void;
  onClose: () => void;
}

/** A small dialog to edit tags as chips (with suggestions from the task's tags) and a note. */
export default function TagEditor({ title, allTags, tags, note, onSave, onClose }: Props) {
  const single = note !== undefined;
  const [kept, setKept] = useState(tags); // one example: its tags; several: tags not marked for removal
  const [added, setAdded] = useState<string[]>([]);
  const [text, setText] = useState("");
  const [noteText, setNoteText] = useState(note ?? "");
  const input = useRef<HTMLInputElement>(null);
  const form = useRef<HTMLFormElement>(null);
  useFocusTrap(form);
  const id = useId();

  useEffect(() => {
    input.current?.focus();
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") { e.stopPropagation(); onClose(); } };
    window.addEventListener("keydown", esc, true);
    return () => { window.removeEventListener("keydown", esc, true); };
  }, [onClose]);

  const add = (raw: string) => {
    const words = raw.split(",").map((w) => w.trim()).filter(Boolean);
    if (!words.length) return;
    if (single) setKept((k) => [...new Set([...k, ...words])]);
    else setAdded((a) => [...new Set([...a, ...words])]);
    setText("");
  };
  const save = () => {
    const pending = text.split(",").map((w) => w.trim()).filter(Boolean); // typed but not yet added
    if (single) onSave({ tags: [...new Set([...kept, ...pending])], note: noteText });
    else onSave({ add: [...new Set([...added, ...pending])], remove: tags.filter((t) => !kept.includes(t)) });
  };
  const chips = single ? kept : [...kept, ...added.filter((a) => !kept.includes(a))];
  const removed = single ? [] : tags.filter((t) => !kept.includes(t));

  return (
    <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <form ref={form} className="modal" role="dialog" aria-modal="true" aria-labelledby={`${id}-title`}
        onSubmit={(e) => { e.preventDefault(); save(); }}>
        <h3 className="modal-title" id={`${id}-title`}>{title}</h3>
        <div className="chips" aria-label="Tags">
          {chips.map((t) => (
            <span key={t} className={`chip ${added.includes(t) ? "new" : ""}`}>
              {t}
              <button type="button" className="ghost small" aria-label={`Remove tag ${t}`} onClick={() => {
                setKept((k) => k.filter((x) => x !== t));
                setAdded((a) => a.filter((x) => x !== t));
              }}>×</button>
            </span>
          ))}
          {removed.map((t) => (
            <span key={t} className="chip removed" title="Will be removed">
              {t}
              <button type="button" className="ghost small" aria-label={`Keep tag ${t}`} onClick={() => { setKept((k) => [...k, t]); }}>↺</button>
            </span>
          ))}
          {!chips.length && !removed.length && <span className="muted">No tags</span>}
        </div>
        <label>
          <span>{single ? "Add tags" : "Add tags to all"}</span>
          <input ref={input} value={text} list={`${id}-tags`} placeholder="e.g. blurry, hard (Enter to add)"
            onChange={(e) => {
              if (e.target.value.endsWith(",")) add(e.target.value);
              else setText(e.target.value);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && text.trim() && !e.nativeEvent.isComposing) { e.preventDefault(); add(text); }
            }} />
          <datalist id={`${id}-tags`}>{allTags.filter((t) => !chips.includes(t)).map((t) => <option key={t} value={t} />)}</datalist>
        </label>
        {single && (
          <label>
            <span>Note</span>
            <textarea rows={3} value={noteText} onChange={(e) => { setNoteText(e.target.value); }} maxLength={2000} />
          </label>
        )}
        <div className="row">
          <button type="submit">Save</button>
          <button type="button" className="ghost" onClick={onClose}>Cancel</button>
        </div>
      </form>
    </div>
  );
}
