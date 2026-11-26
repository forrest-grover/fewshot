import { useRef, useState } from "react";
import { api, type ModelInfo, type TaskDetail } from "./api";

/** The form for a new task, and importing one from a zip. */
export default function NewTask({ models, run, onCreated }: {
  models: ModelInfo | null; run: <T>(p: Promise<T>) => Promise<T | undefined>; onCreated: (t: TaskDetail) => void;
}) {
  const [name, setName] = useState("");
  const [embedder, setEmbedder] = useState("");
  const [busy, setBusy] = useState(false);
  const spec = embedder || (models?.embedders[0]?.spec ?? "");
  const importZip = useRef<HTMLInputElement>(null);
  return (
    <>
    <form
      className="new-task"
      onSubmit={(e) => {
        e.preventDefault();
        setBusy(true);
        void run(api.createTask(name.trim(), spec || undefined, "gda")).then((t) => {
          setBusy(false);
          if (t) onCreated(t);
        });
      }}
    >
      <h1>New task</h1>
      <p className="muted">
        A task is a set of classes plus a few labeled images of each. New images are embedded with a pretrained vision model and
        assigned to the closest class. No training run is needed: add, remove, or relabel examples and predictions update immediately.
      </p>
      <label>
        <span>Name</span>
        <input value={name} onChange={(e) => { setName(e.target.value); }} placeholder="e.g. weld-defects" pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}" required autoFocus />
      </label>
      <label>
        <span>Embedder</span>
        <select value={spec} onChange={(e) => { setEmbedder(e.target.value); }}>
          {models?.embedders.map((m) => <option key={m.spec} value={m.spec}>{m.label}</option>)}
        </select>
        <small className="muted">{models?.embedders.find((m) => m.spec === spec)?.notes}</small>
      </label>
      <button type="submit" disabled={busy || !name.trim()}>{busy ? "Creating… (loads the model on first use)" : "Create task"}</button>
    </form>
    <div className="new-task import">
      <h3>Or import a task</h3>
      <p className="muted">From a zip made with “Export” on a task's Settings tab, or <code>fewshot export</code>.</p>
      <input ref={importZip} type="file" accept=".zip,application/zip" hidden onChange={(e) => {
        const f = e.target.files?.[0];
        e.target.value = "";
        if (!f) return;
        setBusy(true);
        void run(api.importTask(f, name.trim() || undefined)).then((t) => {
          setBusy(false);
          if (t) onCreated(t);
        });
      }} />
      <button className="secondary" disabled={busy} onClick={() => { importZip.current?.click(); }}>
        Import zip{name.trim() ? ` as “${name.trim()}”` : ""}…
      </button>
    </div>
    </>
  );
}
