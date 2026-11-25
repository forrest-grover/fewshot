import { useState } from "react";
import { api, type Example, type TaskDetail } from "./api";
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

  const byClass = new Map<string, Example[]>(task.classes.map((c) => [c.name, []]));
  for (const e of task.examples) byClass.get(e.label)?.push(e);

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

      {task.classes.map((c) => {
        const limit = shown[c.name] ?? PAGE;
        const skip = skipped[c.name] ?? [];
        return (
          <section key={c.name} className="class-card">
            <header>
              <h3>{c.name}</h3>
              <span className="muted">{c.count} example{c.count === 1 ? "" : "s"}</span>
              <span className="spacer" />
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
              {(byClass.get(c.name) ?? []).slice(0, limit).map((ex) => (
                <figure key={ex.id} className="thumb">
                  <img src={api.imageUrl(task.name, ex.id, THUMB_SIZE)} alt={ex.source || ex.id} loading="lazy" />
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
