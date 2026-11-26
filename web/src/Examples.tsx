import { memo, useMemo, useRef, useState, type DragEvent } from "react";
import { api, type Example, type TaskDetail } from "./api";
import { EXAMPLE_DRAG, filesFromDrop, isImage, usePasteImages } from "./files";
import { useShortcuts } from "./keys";
import type { Uploads } from "./useUploads";
import { THUMB_SIZE } from "./images";
import Lightbox from "./Lightbox";
import TagEditor, { type TagEdit } from "./TagEditor";
import { ClassSelect, DropZone } from "./ui";
import { progressText } from "./upload";

interface Props {
  task: TaskDetail; setTask: (t: TaskDetail) => void; run: <T>(p: Promise<T>) => Promise<T | undefined>;
  uploads: Uploads; // uploads to classes (by class name), kept by the App so they go on in other tabs
}

/** Thumbnails shown per class at first, and added by each "Show more", so a class with thousands of
 * examples doesn't render thousands of images and class menus at once. */
const PAGE = 60;

export default function Examples({ task, setTask, run, uploads }: Props) {
  const [newClass, setNewClass] = useState("");
  const [shown, setShown] = useState<Partial<Record<string, number>>>({}); // class -> thumbnails shown
  const [picked, setPicked] = useState<ReadonlySet<string>>(new Set()); // selected example ids
  const [dropOn, setDropOn] = useState<string | null>(null); // class card under a drag
  const [tagFilter, setTagFilter] = useState(""); // show only examples with this tag
  const [query, setQuery] = useState(""); // show only examples whose source, tags, note or id contain this
  const [order, setOrder] = useState<"task" | "newest" | "oldest" | "source">("task");
  const [folded, setFolded] = useState<ReadonlySet<string>>(() => loadFolded(task.name)); // collapsed classes
  const [viewer, setViewer] = useState<{ ids: string[]; index: number } | null>(null); // full-size viewer
  const [editing, setEditing] = useState<{ one: Example } | { ids: string[] } | null>(null); // tag editor
  const pointed = useRef<string | null>(null); // class card last hovered or focused: where a paste goes
  const anchor = useRef<string | null>(null); // last ticked example, for Shift+click ranges

  /** Add images to class ``label``, in batches (see useUploads), as one undo step: each batch
   * shows as it lands, and Cancel (or opening another task) stops the rest. A class takes one
   * upload at a time. */
  const upload = async (label: string, files: File[]) => {
    if (uploads.slots[label]?.progress) { fail(`Still adding images to ${label}: wait for it or cancel it first.`); return; }
    await uploads.start(label, files, (fs, signal, group, t) => api.addExamples(t, label, fs, signal, group));
  };
  /** Run a mutation and show its result; resolves to whether it succeeded. */
  const save = async (p: Promise<TaskDetail>) => {
    const t = await run(p);
    if (t) setTask(t);
    return t !== undefined;
  };
  const fail = (message: string) => { void run(Promise.reject(new Error(message))); };

  const allTags = [...new Set(task.examples.flatMap((e) => e.tags))].sort();
  if (tagFilter && !allTags.includes(tagFilter)) setTagFilter(""); // its tag is gone: don't revive it later
  const filter = allTags.includes(tagFilter) ? tagFilter : "";
  const byClass = new Map<string, Example[]>(task.classes.map((c) => [c.name, []]));
  const q = query.trim().toLowerCase();
  const matches = (e: Example) => !q || [e.source, e.note, e.id, ...e.tags].some((v) => v.toLowerCase().includes(q));
  for (const e of task.examples) if ((!filter || e.tags.includes(filter)) && matches(e)) byClass.get(e.label)?.push(e);
  if (order !== "task") {
    const key = (e: Example) => (order === "source" ? e.source.toLowerCase() : e.added);
    for (const list of byClass.values()) {
      list.sort((a, b) => (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0));
      if (order === "newest") list.reverse();
    }
  }
  const fold = (c: string) => {
    setFolded((f) => {
      const next = new Set(f);
      if (next.has(c)) next.delete(c); else next.add(c);
      saveFolded(task.name, next);
      return next;
    });
  };
  const byId = new Map(task.examples.map((e) => [e.id, e]));
  const open = (list: Example[], ex: Example) => { setViewer({ ids: list.map((e) => e.id), index: list.indexOf(ex) }); };
  const labelOf = new Map(task.examples.map((e) => [e.id, e.label]));
  const selection = [...picked].filter((id) => labelOf.has(id)); // ids still in the task

  const move = async (ids: string[], label: string) => {
    const moving = ids.filter((id) => labelOf.get(id) !== label);
    if (moving.length && (await save(api.relabelMany(task.name, moving, label)))) setPicked(new Set());
  };
  const tagSelection = (ids: string[]) => { setEditing({ ids }); };
  const annotate = (ex: Example) => { setEditing({ one: ex }); };
  /** Save what the tag editor returns, and close it if that worked. */
  const saveTags = async (edit: TagEdit) => {
    if (!editing) return;
    let ok = true;
    if ("tags" in edit && "one" in editing) ok = await save(api.annotate(task.name, editing.one.id, edit));
    else if ("add" in edit && "ids" in editing && (edit.add.length || edit.remove.length)) {
      ok = await save(api.tagMany(task.name, editing.ids, edit.add, edit.remove));
    }
    if (ok) setEditing(null);
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
        // From this page, or another tab (maybe another task): only this task's examples move.
        const mine = draggedIds(ids).filter((id) => labelOf.has(id));
        if (mine.length) void move(mine, c);
        else fail("Only this task's examples can be dragged here.");
      } else if (e.dataTransfer.types.includes("Files")) {
        e.preventDefault(); // files dropped anywhere on the card, not only on its drop zone
        void filesFromDrop(e.dataTransfer).then((fs) => {
          const imgs = fs.filter(isImage);
          if (imgs.length) void upload(c, imgs);
        });
      }
    },
  });

  // What a thumbnail does, as one object that never changes (calling the latest handlers), so
  // memoised thumbnails only re-render when their own example or tick changes.
  const thumbHandlers = {
    dragStart: (ex: Example, e: DragEvent) => {
      // Dragging a ticked thumbnail moves every ticked example.
      const ids = picked.has(ex.id) ? selection : [ex.id];
      e.dataTransfer.setData(EXAMPLE_DRAG, JSON.stringify(ids));
      e.dataTransfer.effectAllowed = "move";
    },
    tick: (ex: Example, shift: boolean) => { toggle(ex, shift); },
    open: (ex: Example, shift: boolean) => {
      if (selection.length) toggle(ex, shift); else open(byClass.get(ex.label) ?? [ex], ex);
    },
    view: (ex: Example) => { open(byClass.get(ex.label) ?? [ex], ex); },
    relabel: (ex: Example, label: string) => { void save(api.relabel(task.name, ex.id, label)); },
    annotate: (ex: Example) => { annotate(ex); },
    remove: (ex: Example) => { void save(api.deleteExample(task.name, ex.id)); },
  };
  const latestThumb = useRef(thumbHandlers);
  latestThumb.current = thumbHandlers;
  const [thumbActions] = useState<ThumbActions>(() => ({
    dragStart: (ex, e) => { latestThumb.current.dragStart(ex, e); },
    tick: (ex, shift) => { latestThumb.current.tick(ex, shift); },
    open: (ex, shift) => { latestThumb.current.open(ex, shift); },
    view: (ex) => { latestThumb.current.view(ex); },
    relabel: (ex, label) => { latestThumb.current.relabel(ex, label); },
    annotate: (ex) => { latestThumb.current.annotate(ex); },
    remove: (ex) => { latestThumb.current.remove(ex); },
  }));
  const classNames = useMemo(() => task.classes.map((c) => c.name), [task.classes]);

  const nFolded = task.classes.filter((c) => folded.has(c.name)).length; // ``folded`` may name deleted classes
  const editedIds = editing ? ("one" in editing ? [editing.one.id] : editing.ids) : [];
  const editedTags = [...new Set(task.examples.filter((e) => editedIds.includes(e.id)).flatMap((e) => e.tags))];

  const viewed = viewer ? viewer.ids.map((id) => byId.get(id)).filter((e): e is Example => e !== undefined) : [];
  const shownEx = viewer ? (viewed[Math.min(viewer.index, viewed.length - 1)] as Example | undefined) : undefined;
  if (viewer && viewed.length === 0) setViewer(null); // emptied: don't reopen when its examples come back (undo)

  return (
    <div className="examples">
      {viewer && viewed.length > 0 && (
        <Lightbox
          items={viewed.map((e) => ({
            url: api.imageUrl(task.name, e.id, 1024), title: `${e.label}: ${e.source || e.id.slice(0, 12)}`,
            details: [`Added ${e.added.slice(0, 16).replace("T", " ")}`, e.tags.length ? `Tags: ${e.tags.join(", ")}` : "",
              e.note ? `Note: ${e.note}` : ""].filter(Boolean),
          }))}
          index={Math.min(viewer.index, viewed.length - 1)}
          onIndex={(i) => { setViewer((v) => (v ? { ...v, index: i } : v)); }}
          onClose={() => { setViewer(null); }}
          onKey={(e, i) => {
            const ex = viewed[i];
            if (/^[1-9]$/.test(e.key)) {
              const c = task.classes[Number(e.key) - 1] as TaskDetail["classes"][number] | undefined;
              if (c && c.name !== ex.label) void save(api.relabel(task.name, ex.id, c.name));
              return true;
            }
            if (e.key === "Delete" || e.key === "Backspace") {
              void save(api.deleteExample(task.name, ex.id));
              return true;
            }
            return false;
          }}
          footer={shownEx && (
            <div className="row">
              <select value={shownEx.label} aria-label="Move to class"
                onChange={(e) => { void save(api.relabel(task.name, shownEx.id, e.target.value)); }}>
                {task.classes.map((k, i) => <option key={k.name} value={k.name}>{i < 9 ? `${String(i + 1)} · ` : ""}{k.name}</option>)}
              </select>
              <button className="small ghost" onClick={() => { annotate(shownEx); }}>Tags and note</button>
              <button className="small ghost danger" onClick={() => { void save(api.deleteExample(task.name, shownEx.id)); }}>Delete <kbd>Del</kbd></button>
            </div>
          )}
        />
      )}
      {editing && (
        <TagEditor
          title={"one" in editing ? `Tags and note: ${editing.one.source || editing.one.id.slice(0, 12)}`
            : `Tag ${String(editing.ids.length)} examples`}
          allTags={allTags} tags={editedTags} note={"one" in editing ? editing.one.note : undefined}
          onSave={(edit) => { void saveTags(edit); }} onClose={() => { setEditing(null); }} />
      )}
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
              // Several templates, separated by "|", become several prompts per class.
              const tpl = prompt("Description template ({} = class name; separate several with |). Naming the kind of thing helps, e.g. \"a photo of a {}, a type of bird\".", "a photo of a {} | a close-up photo of a {}")
                ?.split("|").map((x) => x.trim()).filter(Boolean).join("\n");
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
      {task.examples.length > 0 && (
        <div className="row tag-filter">
          <input type="search" className="search" value={query} placeholder="Search file names, tags, notes…" aria-label="Search examples"
            onChange={(e) => { setQuery(e.target.value); setPicked(new Set()); }} />
          {allTags.length > 0 && (
            <label className="inline">
              <span>Show</span>
              <select value={filter} onChange={(e) => {
                setTagFilter(e.target.value);
                setPicked(new Set()); // bulk actions must not reach examples the filter hides
              }}>
                <option value="">all examples</option>
                {allTags.map((t) => <option key={t} value={t}>tagged “{t}”</option>)}
              </select>
            </label>
          )}
          <label className="inline">
            <span>Order</span>
            <select value={order} onChange={(e) => { setOrder(e.target.value as typeof order); }}>
              <option value="task">as added</option>
              <option value="newest">newest first</option>
              <option value="oldest">oldest first</option>
              <option value="source">by file name</option>
            </select>
          </label>
          {task.classes.length > 1 && (
            <button className="ghost small" onClick={() => {
              const all = nFolded < task.classes.length;
              const next = new Set(all ? task.classes.map((c) => c.name) : []);
              saveFolded(task.name, next);
              setFolded(next);
            }}>{nFolded < task.classes.length ? "Collapse all" : "Expand all"}</button>
          )}
        </div>
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
          <button className="small secondary" onClick={() => { tagSelection(selection); }}>Tag…</button>
          <button className="small danger" onClick={() => { void remove(selection); }}>Delete</button>
          <button className="ghost small" onClick={() => { setPicked(new Set()); }}>Clear selection <kbd>Esc</kbd></button>
        </div>
      )}

      {task.classes.map((c, ci) => {
        const limit = shown[c.name] ?? PAGE;
        const skip = uploads.slots[c.name]?.skipped ?? [];
        const moved = uploads.slots[c.name]?.moved ?? 0; // of the last upload: images that were examples of other classes
        const up = uploads.slots[c.name]?.progress; // an upload in progress
        const list = byClass.get(c.name) ?? [];
        const allPicked = list.length > 0 && list.every((e) => picked.has(e.id));
        return (
          <section
            key={c.name}
            className={`class-card ${dropOn === c.name ? "drop-target" : ""}`}
            onMouseEnter={() => { pointed.current = c.name; }}
            onMouseLeave={(e) => {
              // A paste then goes to the class with keyboard focus, if any, not the last one hovered.
              if (!e.currentTarget.contains(document.activeElement)) pointed.current = null;
            }}
            onFocus={() => { pointed.current = c.name; }}
            {...cardDrag(c.name)}
          >
            <header>
              <button className="ghost small fold" aria-expanded={!folded.has(c.name)} aria-label={`${folded.has(c.name) ? "Expand" : "Collapse"} ${c.name}`}
                onClick={() => { fold(c.name); }}>{folded.has(c.name) ? "▸" : "▾"}</button>
              {ci < 9 && <kbd title={`Press ${String(ci + 1)} to move ticked examples here`}>{ci + 1}</kbd>}
              <h3>{c.name}</h3>
              <span className="muted">{c.count} example{c.count === 1 ? "" : "s"}{filter || q ? ` (${String(list.length)} shown)` : ""}</span>
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
            {!folded.has(c.name) && task.supports_text && (
              <textarea
                className="desc"
                key={c.description}
                defaultValue={c.description}
                rows={Math.max(1, c.description.split("\n").length)}
                placeholder="Text description used as a CLIP prompt, e.g. 'a photo of a cracked weld' (optional; one prompt per line, Shift+Enter for a new line)"
                onBlur={(e) => {
                  if (e.target.value !== c.description) void save(api.patchClass(task.name, c.name, { description: e.target.value }));
                }}
                onKeyDown={(e) => {
                  // Enter that confirms an input-method composition (Japanese, Chinese, Korean) isn't "save".
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); e.currentTarget.blur(); }
                }}
              />
            )}
            {!folded.has(c.name) && <div className="thumbs">
              {list.slice(0, limit).map((ex) => (
                <Thumb key={ex.id} ex={ex} task={task.name} picked={picked.has(ex.id)} classes={classNames} on={thumbActions} />
              ))}
              {list.length > limit && (
                <button
                  className="secondary more"
                  onClick={() => { setShown((s) => ({ ...s, [c.name]: limit + PAGE * 4 })); }}
                >
                  Show more ({list.length - limit} hidden)
                </button>
              )}
              <DropZone compact disabled={!!up} onFiles={(fs) => { void upload(c.name, fs); }}>
                {up ? `${up.stopping ? "Stopping" : "Uploading"} ${progressText(up)}…` : "+ Add images"}
              </DropZone>
              {up && <button className="ghost small" disabled={up.stopping} onClick={up.cancel}>Cancel upload</button>}
            </div>}
            {moved > 0 && (
              <div className="row note-moved" role="status">
                <span>{moved} moved from other classes</span>
                <button className="ghost small" aria-label="Dismiss" onClick={() => { uploads.dismiss(c.name); }}>×</button>
              </div>
            )}
            {skip.length > 0 && (
              <div className="error skipped" role="alert">
                <div>
                  Skipped {skip.length} file{skip.length === 1 ? "" : "s"}:
                  <ul>{skip.map((e, i) => <li key={i}>{e.file && <><b>{e.file}</b>: </>}{e.error}</li>)}</ul>
                </div>
                <button className="ghost" aria-label="Dismiss" onClick={() => { uploads.dismiss(c.name); }}>×</button>
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}

/** The example ids in an EXAMPLE_DRAG payload; none if it is malformed. */
function draggedIds(data: string): string[] {
  try {
    const v: unknown = JSON.parse(data);
    return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

interface ThumbActions {
  dragStart: (ex: Example, e: DragEvent) => void; tick: (ex: Example, shift: boolean) => void;
  open: (ex: Example, shift: boolean) => void; view: (ex: Example) => void; relabel: (ex: Example, label: string) => void;
  annotate: (ex: Example) => void; remove: (ex: Example) => void;
}

interface ThumbProps { ex: Example; task: string; picked: boolean; classes: string[]; on: ThumbActions }
const sameList = (a: readonly string[], b: readonly string[]) => a.length === b.length && a.every((x, i) => x === b[i]);
/** Whether a thumbnail would show the same: every response brings new example objects and class
 * lists, so they are compared by what is shown, not by identity. */
const sameThumb = (a: ThumbProps, b: ThumbProps) => a.task === b.task && a.picked === b.picked && a.on === b.on
  && a.ex.id === b.ex.id && a.ex.label === b.ex.label && a.ex.source === b.ex.source && a.ex.note === b.ex.note
  && sameList(a.ex.tags, b.ex.tags) && sameList(a.classes, b.classes);

/** One example's thumbnail: tick box, image (click, Enter or Space opens it), class menu, edit
 * and delete. Memoised (see sameThumb): a class can show hundreds. */
const Thumb = memo(function Thumb({ ex, task, picked, classes, on }: ThumbProps) {
  return (
    <figure className={`thumb ${picked ? "picked" : ""}`} draggable onDragStart={(e) => { on.dragStart(ex, e); }}>
      <label className="pick" title="Tick to select (Shift+click for a range)">
        <input
          type="checkbox"
          checked={picked}
          aria-label={`Select ${ex.source || ex.id}`}
          onClick={(e) => { on.tick(ex, e.shiftKey); }}
          onChange={() => { /* handled on click, which knows about Shift */ }}
        />
      </label>
      {(ex.tags.length > 0 || ex.note) && (
        <span className="tags" title={[ex.tags.join(", "), ex.note].filter(Boolean).join("\n")}>
          {ex.tags.length > 0 ? ex.tags.join(", ") : "note"}
        </span>
      )}
      <img
        src={api.imageUrl(task, ex.id, THUMB_SIZE)}
        title={[ex.source, ex.note].filter(Boolean).join(" · ") || undefined}
        alt={ex.source || ex.id}
        loading="lazy"
        draggable={false}
        tabIndex={0}
        role="button"
        onClick={(e) => { on.open(ex, e.shiftKey); }}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); on.view(ex); } }}
      />
      <figcaption>
        <ClassSelect value={ex.label} classes={classes} aria-label="Move to class" onPick={(label) => { on.relabel(ex, label); }} />
        <button className="ghost small" aria-label="Tags and note" title="Tags and note" onClick={() => { on.annotate(ex); }}>✎</button>
        <button className="ghost small danger" aria-label="Delete example" onClick={() => { on.remove(ex); }}>×</button>
      </figcaption>
    </figure>
  );
}, sameThumb);

const FOLD_KEY = "fewshot:folded:";

/** Collapsed classes of a task, remembered in this browser (if storage works). */
function loadFolded(task: string): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(FOLD_KEY + task) ?? "[]") as string[]);
  } catch {
    return new Set();
  }
}

function saveFolded(task: string, folded: ReadonlySet<string>) {
  try {
    localStorage.setItem(FOLD_KEY + task, JSON.stringify([...folded]));
  } catch { /* private mode: not remembered */ }
}
