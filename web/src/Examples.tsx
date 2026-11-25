import { useState } from "react";
import { api, type TaskDetail } from "./api";
import { DropZone } from "./ui";

interface Props { task: TaskDetail; setTask: (t: TaskDetail) => void; run: <T>(p: Promise<T>) => Promise<T | undefined> }

export default function Examples({ task, setTask, run }: Props) {
  const [newClass, setNewClass] = useState("");
  const [uploading, setUploading] = useState<string | null>(null);

  const upload = async (label: string, files: File[]) => {
    setUploading(label);
    const r = await run(api.addExamples(task.name, label, files));
    setUploading(null);
    if (r) {
      setTask(r.task);
      if (r.errors.length) alert(`Skipped ${String(r.errors.length)} file(s):\n` + r.errors.map((e) => `${e.file}: ${e.error}`).join("\n"));
    }
  };
  /** Run a mutation and show its result; resolves to whether it succeeded. */
  const save = async (p: Promise<TaskDetail>) => {
    const t = await run(p);
    if (t) setTask(t);
    return t !== undefined;
  };

  const byClass = new Map(task.classes.map((c) => [c.name, task.examples.filter((e) => e.label === c.name)]));

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

      {task.classes.map((c) => (
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
            {(byClass.get(c.name) ?? []).map((ex) => (
              <figure key={ex.id} className="thumb">
                <img src={api.imageUrl(task.name, ex.id)} alt={ex.source || ex.id} loading="lazy" />
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
            <DropZone compact disabled={uploading === c.name} onFiles={(fs) => { void upload(c.name, fs); }}>
              {uploading === c.name ? "Uploading…" : "+ Add images"}
            </DropZone>
          </div>
        </section>
      ))}
    </div>
  );
}
