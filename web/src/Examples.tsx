import { useRef, useState, type DragEvent } from "react";
import { api, type Example, type TaskDetail } from "./api";
import { EXAMPLE_DRAG, filesFromDrop, isImage, usePasteImages, useShortcuts } from "./hooks";
import { shrinkImages, THUMB_SIZE } from "./images";
import { DropZone } from "./ui";

interface Props { task: TaskDetail; setTask: (t: TaskDetail) => void; run: <T>(p: Promise<T>) => Promise<T | undefined> }
interface Skipped { file: string; error: string }

/** Thumbnails shown per class at first, and added by each "Show more", so a class with thousands of
 * examples doesn't render thousands of images and class menus at once. */
const PAGE = 60;

export default function Examples({ task, setTask, run }: Props) {
  const [newClass, setNewClass] = useState("");
  const [uploading, setUploading] = useState<Partial<Record<string, number>>>({}); // class -> uploads in flight
  const [skipped, setSkipped] = useState<Partial<Record<string, Skipped[]>>>({}); // class -> files the last upload skipped
  const [shown, setShown] = useState<Partial<Record<string, number>>>({}); // class -> thumbnails shown
  const [picked, setPicked] = useState<ReadonlySet<string>>(new Set()); // selected example ids
  const [dropOn, setDropOn] = useState<string | null>(null); // class card under a drag
  const pointed = useRef<string | null>(null); // class card last hovered or focused: where a paste goes
  const anchor = useRef<string | null>(null); // last ticked example, for Shift+click ranges

  const busy = (label: string, delta: number) => {
    setUploading((u) => ({ ...u, [label]: (u[label] ?? 0) + delta }));
  };
  const upload = async (label: string, files: File[]) => {
    busy(label, 1);
    try {
      const r = await run(shrinkImages(files).then((fs) => api.addExamples(task.name, label, fs)));
      if (r) {
        setTask(r.task);
        setSkipped((s) => ({ ...s, [label]: r.errors }));
      }
    } finally {
      busy(label, -1);
    }
  };
  /** Run a mutation and show its result; resolves to whether it succeeded. */
  const save = async (p: Promise<TaskDetail>) => {
    const t = await run(p);
    if (t) setTask(t);
    return t !== undefined;
  };
  const fail = (message: string) => { void run(Promise.reject(new Error(message))); };

  const byClass = new Map<string, Example[]>(task.classes.map((c) => [c.name, []]));
  for (const e of task.examples) byClass.get(e.label)?.push(e);
  const labelOf = new Map(task.examples.map((e) => [e.id, e.label]));
  const selection = [...picked].filter((id) => labelOf.has(id)); // ids still in the task

  const move = async (ids: string[], label: string) => {
    const moving = ids.filter((id) => labelOf.get(id) !== label);
    if (moving.length && (await save(api.relabelMany(task.name, moving, label)))) setPicked(new Set());
  };
  const remove = async (ids: string[]) => {
    if (!confirm(`Delete ${String(ids.length)} example${ids.length === 1 ? "" : "s"}?`)) return;
    if (await save(api.deleteExamples(task.name, ids))) setPicked(new Set());
  };
  /** Tick or untick one example; with Shift, everything between it and the last ticked one in the same class. */
  const toggle = (ex: Example, shift: boolean) => {
    const list = byClass.get(ex.label) ?? [];
    const from = shift && anchor.current ? list.findIndex((e) => e.id === anchor.current) : -1;
    const to = list.findIndex((e) => e.id === ex.id);
    const ids = from >= 0 ? list.slice(Math.min(from, to), Math.max(from, to) + 1).map((e) => e.id) : [ex.id];
    const on = !picked.has(ex.id);
    setPicked((p) => {
      const next = new Set(p);
      ids.forEach((id) => { if (on) next.add(id); else next.delete(id); });
      return next;
    });
    anchor.current = ex.id;
  };

  // Pasted images go to the class under the pointer (or the only class).
  usePasteImages((fs) => {
    const target = pointed.current && byClass.has(pointed.current)
      ? pointed.current
      : task.classes.length === 1 ? task.classes[0].name : null;
    if (target) void upload(target, fs);
    else fail("To paste images, first point at (or click in) the class they belong to.");
  });
  // With examples ticked: 1-9 move them to that class, Delete removes them, Escape unticks.
  useShortcuts((e) => {
    if (!selection.length) return;
    if (/^[1-9]$/.test(e.key)) {
      const c = task.classes[Number(e.key) - 1] as TaskDetail["classes"][number] | undefined;
      if (c) void move(selection, c.name);
    } else if (e.key === "Delete" || e.key === "Backspace") void remove(selection);
    else if (e.key === "Escape") setPicked(new Set());
    else return;
    e.preventDefault();
  });

  const cardDrag = (c: string) => ({
    onDragOver: (e: DragEvent) => {
      const types = e.dataTransfer.types;
      if (!types.includes(EXAMPLE_DRAG) && !types.includes("Files")) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = types.includes(EXAMPLE_DRAG) ? "move" : "copy";
      setDropOn(c);
    },
    onDragLeave: (e: DragEvent) => {
      if (!(e.relatedTarget instanceof Node && e.currentTarget.contains(e.relatedTarget))) setDropOn(null);
    },
    onDrop: (e: DragEvent) => {
      setDropOn(null);
      const ids = e.dataTransfer.getData(EXAMPLE_DRAG);
      if (ids) {
        e.preventDefault();
        void move(JSON.parse(ids) as string[], c);
      } else if (e.dataTransfer.types.includes("Files")) {
        e.preventDefault(); // files dropped anywhere on the card, not only on its drop zone
        void filesFromDrop(e.dataTransfer).then((fs) => {
          const imgs = fs.filter(isImage);
          if (imgs.length) void upload(c, imgs);
        });
      }
    },
  });

  return (
    <div className="examples">
      <form
        className="row add-class"
        onSubmit={(e) => {
          e.preventDefault();
          if (newClass.trim()) void save(api.addClass(task.name, newClass.trim())).then((ok) => { if (ok) setNewClass(""); });
        }}
      >
        <input value={newClass} onChange={(e) => { setNewClass(e.target.value); }} placeholder="New class name" />
        <button type="submit" disabled={!newClass.trim()}>Add class</button>
        {task.supports_text && task.classes.length > 0 && (
          <button
            type="button"
            className="secondary"
            title="Fill empty descriptions from the class names. CLIP uses them as text prompts."
            onClick={() => {
              const tpl = prompt("Description template ({} = class name)", "a photo of a {}");
              if (tpl) void save(api.describe(task.name, tpl, false));
            }}
          >
            Describe from names…
          </button>
        )}
      </form>

      {task.classes.length === 0 && (
        <p className="hint">
          Add at least two classes, then drop a few example images on each. One to five images per class is often enough.
          {task.supports_text && " With CLIP, a class can also work from its text description alone."}
        </p>
      )}
      {task.examples.length > 0 && selection.length === 0 && (
        <p className="muted keys">
          Drag a thumbnail onto another class to move it. Tick several (Shift+click for a range) to move or delete them
          together; <kbd>1</kbd>–<kbd>9</kbd> then move them to that class. Paste (<kbd>Ctrl</kbd>+<kbd>V</kbd>) adds images to the class under the pointer.
        </p>
      )}
      {selection.length > 0 && (
        <div className="bulk-bar" role="toolbar" aria-label="Selected examples">
          <strong>{selection.length} selected</strong>
          <select value="" aria-label="Move selected to class" onChange={(e) => { if (e.target.value) void move(selection, e.target.value); }}>
            <option value="">Move to…</option>
            {task.classes.map((c, i) => <option key={c.name} value={c.name}>{i < 9 ? `${String(i + 1)} · ` : ""}{c.name}</option>)}
          </select>
          <button className="small danger" onClick={() => { void remove(selection); }}>Delete</button>
          <button className="ghost small" onClick={() => { setPicked(new Set()); }}>Clear selection <kbd>Esc</kbd></button>
        </div>
      )}

      {task.classes.map((c, ci) => {
        const limit = shown[c.name] ?? PAGE;
        const skip = skipped[c.name] ?? [];
        const list = byClass.get(c.name) ?? [];
        const allPicked = list.length > 0 && list.every((e) => picked.has(e.id));
        return (
          <section
            key={c.name}
            className={`class-card ${dropOn === c.name ? "drop-target" : ""}`}
            onMouseEnter={() => { pointed.current = c.name; }}
            onFocus={() => { pointed.current = c.name; }}
            {...cardDrag(c.name)}
          >
            <header>
              {ci < 9 && <kbd title={`Press ${String(ci + 1)} to move ticked examples here`}>{ci + 1}</kbd>}
              <h3>{c.name}</h3>
              <span className="muted">{c.count} example{c.count === 1 ? "" : "s"}</span>
              <span className="spacer" />
              {list.length > 0 && (
                <button className="ghost small" onClick={() => {
                  setPicked((p) => {
                    const next = new Set(p);
                    list.forEach((e) => { if (allPicked) next.delete(e.id); else next.add(e.id); });
                    return next;
                  });
                }}>{allPicked ? "Untick all" : "Tick all"}</button>
              )}
              <button className="ghost small" onClick={() => {
                const n = prompt("Rename class", c.name);
                if (n && n !== c.name) void save(api.patchClass(task.name, c.name, { name: n }));
              }}>Rename</button>
              <button className="ghost small danger" onClick={() => {
                if (confirm(`Delete class "${c.name}" and its ${String(c.count)} example(s)?`)) void save(api.deleteClass(task.name, c.name));
              }}>Delete</button>
            </header>
            {task.supports_text && (
              <input
                className="desc"
                key={c.description}
                defaultValue={c.description}
                placeholder="Text description used as a CLIP prompt, e.g. 'a photo of a cracked weld' (optional)"
                onBlur={(e) => {
                  if (e.target.value !== c.description) void save(api.patchClass(task.name, c.name, { description: e.target.value }));
                }}
                onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }}
              />
            )}
            <div className="thumbs">
              {list.slice(0, limit).map((ex) => (
                <figure
                  key={ex.id}
                  className={`thumb ${picked.has(ex.id) ? "picked" : ""}`}
                  draggable
                  onDragStart={(e) => {
                    // Dragging a ticked thumbnail moves every ticked example.
                    const ids = picked.has(ex.id) ? selection : [ex.id];
                    e.dataTransfer.setData(EXAMPLE_DRAG, JSON.stringify(ids));
                    e.dataTransfer.effectAllowed = "move";
                  }}
                >
                  <label className="pick" title="Tick to select (Shift+click for a range)">
                    <input
                      type="checkbox"
                      checked={picked.has(ex.id)}
                      aria-label={`Select ${ex.source || ex.id}`}
                      onClick={(e) => { toggle(ex, e.shiftKey); }}
                      onChange={() => { /* handled on click, which knows about Shift */ }}
                    />
                  </label>
                  <img
                    src={api.imageUrl(task.name, ex.id, THUMB_SIZE)}
                    alt={ex.source || ex.id}
                    loading="lazy"
                    draggable={false}
                    onClick={(e) => { if (selection.length) toggle(ex, e.shiftKey); }}
                  />
                  <figcaption>
                    <select
                      value={ex.label}
                      aria-label="Move to class"
                      onChange={(e) => { void save(api.relabel(task.name, ex.id, e.target.value)); }}
                    >
                      {task.classes.map((k) => <option key={k.name}>{k.name}</option>)}
                    </select>
                    <button className="ghost small danger" aria-label="Delete example" onClick={() => { void save(api.deleteExample(task.name, ex.id)); }}>×</button>
                  </figcaption>
                </figure>
              ))}
              {c.count > limit && (
                <button
                  className="secondary more"
                  onClick={() => { setShown((s) => ({ ...s, [c.name]: limit + PAGE * 4 })); }}
                >
                  Show more ({c.count - limit} hidden)
                </button>
              )}
              <DropZone compact onFiles={(fs) => { void upload(c.name, fs); }}>
                {uploading[c.name] ? "Uploading…" : "+ Add images"}
              </DropZone>
            </div>
            {skip.length > 0 && (
              <div className="error skipped" role="alert">
                <div>
                  Skipped {skip.length} file{skip.length === 1 ? "" : "s"}:
                  <ul>{skip.map((e, i) => <li key={i}><b>{e.file}</b>: {e.error}</li>)}</ul>
                </div>
                <button className="ghost" aria-label="Dismiss" onClick={() => { setSkipped((s) => ({ ...s, [c.name]: [] })); }}>×</button>
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}
