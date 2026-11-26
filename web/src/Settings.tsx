import { useState } from "react";
import { api, type ModelInfo, type TaskDetail } from "./api";
import { afterSave, formOf, pickClassifier, settingsPatch, type Edits, type SettingsForm } from "./settingsForm";

interface Props {
  task: TaskDetail; models: ModelInfo | null; setTask: (t: TaskDetail) => void;
  run: <T>(p: Promise<T>) => Promise<T | undefined>; onDeleted: () => void; onRenamed: (t: TaskDetail) => void;
  onDuplicated: (t: TaskDetail) => void;
}

const PARAMS: Record<string, { key: string; label: string; help: string; def: number }[]> = {
  prototype: [{ key: "scale", label: "Scale", help: "Sharpness of image-similarity scores (default 30)", def: 30 }],
  knn: [
    { key: "k", label: "k", help: "Neighbours that vote (default 5)", def: 5 },
    { key: "scale", label: "Scale", help: "Sharpness of neighbour weights (default 30)", def: 30 },
  ],
  logreg: [{ key: "C", label: "C", help: "Inverse regularisation strength (default 10)", def: 10 }],
};

export default function Settings({ task, models, setTask, run, onDeleted, onRenamed, onDuplicated }: Props) {
  // Only fields the user changed are held here; the rest show the saved task as it is now.
  const [edits, setEdits] = useState<Edits>({});
  const [newName, setNewName] = useState(task.name);
  const [copyName, setCopyName] = useState(`${task.name}-copy`);
  const [copyExamples, setCopyExamples] = useState(true);
  const [saving, setSaving] = useState(false);

  // Another task (or this one renamed): start over. Done during render rather than in an effect:
  // https://react.dev/learn/you-might-not-need-an-effect#adjusting-some-state-when-a-prop-changes
  const [formTask, setFormTask] = useState(task.name);
  if (formTask !== task.name) {
    setFormTask(task.name);
    setEdits({}); setNewName(task.name); setCopyName(`${task.name}-copy`);
  }
  const form: SettingsForm = { ...formOf(task), ...edits };
  const edit = (e: Edits) => { setEdits((old) => ({ ...old, ...e })); };
  const { embedder, classifier, params, textWeight, reject, rejectZero, flip, names, temperature } = form;

  const presets = models?.embedders ?? [];
  const isPreset = presets.some((p) => p.spec === embedder);
  // The server knows about custom registered embedders; fall back to the spec prefix for a typed-in spec.
  const supportsText = embedder === task.embedder
    ? task.supports_text
    : presets.find((p) => p.spec === embedder)?.supports_text ?? embedder.startsWith("openclip:");

  const { patch, problems } = settingsPatch(task, form);
  const dirty = Object.keys(patch).length > 0;

  const save = async () => {
    const sent = edits;
    setSaving(true);
    const t = await run(api.updateTask(task.name, patch));
    setSaving(false);
    if (t) {
      setEdits((cur) => afterSave(cur, sent)); // keep anything typed while it was saving
      setTask(t);
    }
  };

  return (
    <div className="settings">
      <label>
        <span>Embedder</span>
        <select value={isPreset ? embedder : "__custom"} onChange={(e) => { edit({ embedder: e.target.value === "__custom" ? "" : e.target.value }); }}>
          {presets.map((p) => <option key={p.spec} value={p.spec}>{p.label}</option>)}
          <option value="__custom">Custom spec…</option>
        </select>
        {!isPreset && <input value={embedder} onChange={(e) => { edit({ embedder: e.target.value }); }} placeholder="openclip:ViT-B-16@laion2b_s34b_b88k or timm:<model>" />}
        <small className="muted">{presets.find((p) => p.spec === embedder)?.notes ?? "Any open_clip (openclip:<arch>@<pretrained>) or timm (timm:<model>) model."} Switching embedders re-embeds every example once; results are cached.</small>
      </label>

      <label>
        <span>Classifier</span>
        <select value={classifier} onChange={(e) => { const c = e.target.value; setEdits((old) => pickClassifier(old, task, c)); }}>
          <option value="gda">GDA: class means and shared spread, tuned per task</option>
          <option value="prototype">Prototype: nearest class centroid</option>
          <option value="knn">kNN: vote of nearest examples</option>
          <option value="logreg">Logistic regression (linear probe)</option>
        </select>
        <small className="muted">GDA is the most accurate from 1 to 16+ examples per class (by about 10 points at 16 on held-out benchmarks), with or without descriptions; the others are kept for comparison and older tasks.</small>
      </label>

      {(PARAMS[classifier] ?? []).map((p) => (
        <label key={p.key}>
          <span>{p.label}</span>
          <input type="text" inputMode="decimal" value={params[p.key] ?? ""} placeholder={String(p.def)}
            onChange={(e) => { edit({ params: { ...params, [p.key]: e.target.value } }); }} />
          <small className="muted">{p.help}</small>
        </label>
      ))}

      <label className={supportsText ? "" : "disabled"}>
        <span>Text weight: {textWeight.toFixed(2)}</span>
        <input type="range" min={0} max={3} step={0.05} value={textWeight} disabled={!supportsText}
          onChange={(e) => { edit({ textWeight: Number(e.target.value) }); }} />
        <small className="muted">{supportsText
          ? "How much class descriptions count relative to example images. 0 ignores descriptions."
          : "Only CLIP embedders use text descriptions."}</small>
      </label>

      <label>
        <span>Reject below similarity</span>
        <input type="text" inputMode="decimal" value={reject} placeholder="off" onChange={(e) => { edit({ reject: e.target.value }); }} />
        <small className="muted">
          Images whose best similarity falls below this value are returned as "unknown". Start near 0.7 for image examples, and tune it
          with the similarity values shown on the Predict tab, or use the value the Evaluate tab suggests. Several classes above it are
          all listed (multi-label). With a single class, this is what separates members from everything else.
          {task.classes.some((c) => c.reject_below !== null) && (
            <> Classes with their own threshold (set on the Evaluate tab) use that instead: {task.classes
              .filter((c) => c.reject_below !== null).map((c) => `${c.name} ${String(c.reject_below)}`).join(", ")}.</>
          )}
        </small>
      </label>

      <label className={supportsText ? "" : "disabled"}>
        <span>Reject below similarity, zero-shot</span>
        <input type="text" inputMode="decimal" value={rejectZero} placeholder="off" disabled={!supportsText}
          onChange={(e) => { edit({ rejectZero: e.target.value }); }} />
        <small className="muted">
          The same, for a task with descriptions but no example images. Image-to-text similarities are much lower (about 0.2–0.35
          for CLIP), so this needs its own value, e.g. 0.22.
        </small>
      </label>

      <label>
        <span>Confidence temperature</span>
        <input type="text" inputMode="decimal" value={temperature} placeholder="none (1)" onChange={(e) => { edit({ temperature: e.target.value }); }} />
        <small className="muted">Calibrates confidence so that 0.9 means right about 90% of the time; the Evaluate tab suggests a value. Predictions don't change.</small>
      </label>

      <label className="check">
        <input type="checkbox" checked={flip} onChange={(e) => { edit({ flip: e.target.checked }); }} />
        <span>Average each image with its mirror image</span>
        <small className="muted check-help">Often a little more accurate when left and right don't matter (most photos). Embedding takes twice as long.</small>
      </label>

      {supportsText && (
        <label className="check">
          <input type="checkbox" checked={names} onChange={(e) => { edit({ names: e.target.checked }); }} />
          <span>Use class names as descriptions</span>
          <small className="muted check-help">A class without a description is described as "a photo of a &lt;name&gt;.", which helps a lot with few examples when names are real words (about 20 points at one example per class). Names that don't match the examples, such as codes, are ignored automatically.</small>
        </label>
      )}

      {problems.length > 0 && <div className="error" role="alert"><span>{problems.join(". ")}.</span></div>}
      <div className="row">
        <button onClick={() => { void save(); }} disabled={saving || !dirty || !embedder || problems.length > 0}>{saving ? "Saving…" : "Save settings"}</button>
        {dirty && !saving && (
          <>
            <span className="muted">Unsaved changes</span>
            <button className="ghost small" onClick={() => { setEdits({}); }}>Discard</button>
          </>
        )}
      </div>

      <hr />
      <form className="row" onSubmit={(e) => {
        e.preventDefault();
        void run(api.rename(task.name, newName.trim())).then((t) => { if (t) onRenamed(t); });
      }}>
        <label className="grow">
          <span>Task name</span>
          <input value={newName} onChange={(e) => { setNewName(e.target.value); }} pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}" required />
        </label>
        <button type="submit" className="secondary" disabled={!newName.trim() || newName.trim() === task.name}>Rename</button>
      </form>
      <form className="row" onSubmit={(e) => {
        e.preventDefault();
        void run(api.duplicate(task.name, copyName.trim(), copyExamples)).then((t) => { if (t) onDuplicated(t); });
      }}>
        <label className="grow">
          <span>Copy as</span>
          <input value={copyName} onChange={(e) => { setCopyName(e.target.value); }} pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}" required />
        </label>
        <label className="check inline">
          <input type="checkbox" checked={copyExamples} onChange={(e) => { setCopyExamples(e.target.checked); }} />
          <span>with examples</span>
        </label>
        <button type="submit" className="secondary" disabled={!copyName.trim()}>Duplicate</button>
      </form>
      <p className="muted">A copy is a safe place to try another embedder or classifier; images are shared, not stored twice.</p>
      <div className="row">
        <a className="button secondary" href={api.exportUrl(task.name)} download={`${task.name}.zip`}>Export as zip</a>
        <span className="muted">Settings, classes, labels and images, for backup or to import into another workspace.</span>
      </div>
      <hr />
      <div className="row">
        <button className="danger" onClick={() => {
          if (confirm(`Delete task "${task.name}" and all ${String(task.examples.length)} images? This cannot be undone (export it first to keep a copy).`)) {
            void run(api.deleteTask(task.name).then(() => true)).then((ok) => { if (ok) onDeleted(); });
          }
        }}>Delete task</button>
      </div>
    </div>
  );
}
