import { useState } from "react";
import { api, type ModelInfo, type TaskDetail } from "./api";

interface Props {
  task: TaskDetail; models: ModelInfo | null; setTask: (t: TaskDetail) => void;
  run: <T>(p: Promise<T>) => Promise<T | undefined>; onDeleted: () => void; onRenamed: (t: TaskDetail) => void;
}

const PARAMS: Record<string, { key: string; label: string; help: string; def: number }[]> = {
  prototype: [{ key: "scale", label: "Scale", help: "Sharpness of image-similarity scores (default 30)", def: 30 }],
  knn: [
    { key: "k", label: "k", help: "Neighbours that vote (default 5)", def: 5 },
    { key: "scale", label: "Scale", help: "Sharpness of neighbour weights (default 30)", def: 30 },
  ],
  logreg: [{ key: "C", label: "C", help: "Inverse regularisation strength (default 10)", def: 10 }],
};

/** Classifier params from the server as the text shown in their inputs. */
function paramTexts(p: Record<string, unknown>): Record<string, string> {
  return Object.fromEntries(Object.entries(p).map(([k, v]) => [k, typeof v === "number" || typeof v === "string" ? String(v) : ""]));
}

/** A typed number, or null for blank (use the default) or NaN for anything unparseable. */
function parseNumber(s: string): number | null {
  const t = s.trim();
  return t === "" ? null : Number(t);
}

const optional = (v: number | null) => (v == null ? "" : String(v));

export default function Settings({ task, models, setTask, run, onDeleted, onRenamed }: Props) {
  const [embedder, setEmbedder] = useState(task.embedder);
  const [classifier, setClassifier] = useState(task.classifier);
  // Inputs keep exactly what was typed; it is parsed and checked on save.
  const [params, setParams] = useState<Record<string, string>>(paramTexts(task.classifier_params));
  const [textWeight, setTextWeight] = useState(task.text_weight);
  const [reject, setReject] = useState(optional(task.reject_below));
  const [rejectZero, setRejectZero] = useState(optional(task.reject_below_zero_shot));
  const [flip, setFlip] = useState(task.flip);
  const [newName, setNewName] = useState(task.name);
  const [saving, setSaving] = useState(false);

  // Reset the form whenever the saved task changes (another task, or a new revision after a save).
  // Done during render rather than in an effect: https://react.dev/learn/you-might-not-need-an-effect#adjusting-some-state-when-a-prop-changes
  const taskKey = `${task.name}\u0000${String(task.revision)}`;
  const [formKey, setFormKey] = useState(taskKey);
  if (formKey !== taskKey) {
    setFormKey(taskKey);
    setEmbedder(task.embedder); setClassifier(task.classifier); setParams(paramTexts(task.classifier_params));
    setTextWeight(task.text_weight); setReject(optional(task.reject_below));
    setRejectZero(optional(task.reject_below_zero_shot)); setFlip(task.flip); setNewName(task.name);
  }

  const presets = models?.embedders ?? [];
  const isPreset = presets.some((p) => p.spec === embedder);
  // The server knows about custom registered embedders; fall back to the spec prefix for a typed-in spec.
  const supportsText = embedder === task.embedder
    ? task.supports_text
    : presets.find((p) => p.spec === embedder)?.supports_text ?? embedder.startsWith("openclip:");

  // Switching classifier clears ``params`` (see its select), so they always belong to ``classifier``.
  const parsedParams: Record<string, number> = {};
  const problems: string[] = [];
  for (const [k, text] of Object.entries(params)) {
    const n = parseNumber(text);
    if (n === null) continue; // blank: the server uses the default
    if (Number.isFinite(n)) parsedParams[k] = n;
    else problems.push(`${k} must be a number`);
  }
  const rejectValue = parseNumber(reject);
  if (rejectValue !== null && !Number.isFinite(rejectValue)) problems.push("Reject below must be a number, or blank for off");
  const rejectZeroValue = parseNumber(rejectZero);
  if (rejectZeroValue !== null && !Number.isFinite(rejectZeroValue)) {
    problems.push("Zero-shot reject below must be a number, or blank for off");
  }

  const save = async () => {
    setSaving(true);
    const t = await run(api.updateTask(task.name, {
      embedder, classifier, classifier_params: parsedParams,
      text_weight: textWeight,
      ...(rejectValue === null ? { clear_reject: true } : { reject_below: rejectValue }),
      ...(rejectZeroValue === null ? { clear_reject_zero_shot: true } : { reject_below_zero_shot: rejectZeroValue }),
      flip,
    }));
    setSaving(false);
    if (t) setTask(t);
  };

  return (
    <div className="settings">
      <label>
        <span>Embedder</span>
        <select value={isPreset ? embedder : "__custom"} onChange={(e) => { setEmbedder(e.target.value === "__custom" ? "" : e.target.value); }}>
          {presets.map((p) => <option key={p.spec} value={p.spec}>{p.label}</option>)}
          <option value="__custom">Custom spec…</option>
        </select>
        {!isPreset && <input value={embedder} onChange={(e) => { setEmbedder(e.target.value); }} placeholder="openclip:ViT-B-16@laion2b_s34b_b88k or timm:<model>" />}
        <small className="muted">{presets.find((p) => p.spec === embedder)?.notes ?? "Any open_clip (openclip:<arch>@<pretrained>) or timm (timm:<model>) model."} Switching embedders re-embeds every example once; results are cached.</small>
      </label>

      <label>
        <span>Classifier</span>
        <select value={classifier} onChange={(e) => { setClassifier(e.target.value); setParams({}); }}>
          <option value="prototype">Prototype: nearest class centroid</option>
          <option value="knn">kNN: vote of nearest examples</option>
          <option value="logreg">Logistic regression (linear probe)</option>
        </select>
        <small className="muted">At 1 to 5 shots these perform within about a point of each other. Logistic regression tends to pull ahead as examples grow.</small>
      </label>

      {(PARAMS[classifier] ?? []).map((p) => (
        <label key={p.key}>
          <span>{p.label}</span>
          <input type="text" inputMode="decimal" value={params[p.key] ?? ""} placeholder={String(p.def)}
            onChange={(e) => { setParams({ ...params, [p.key]: e.target.value }); }} />
          <small className="muted">{p.help}</small>
        </label>
      ))}

      <label className={supportsText ? "" : "disabled"}>
        <span>Text weight: {textWeight.toFixed(2)}</span>
        <input type="range" min={0} max={3} step={0.05} value={textWeight} disabled={!supportsText}
          onChange={(e) => { setTextWeight(Number(e.target.value)); }} />
        <small className="muted">{supportsText
          ? "How much class descriptions count relative to example images. 0 ignores descriptions."
          : "Only CLIP embedders use text descriptions."}</small>
      </label>

      <label>
        <span>Reject below similarity</span>
        <input type="text" inputMode="decimal" value={reject} placeholder="off" onChange={(e) => { setReject(e.target.value); }} />
        <small className="muted">
          Images whose best similarity falls below this value are returned as "unknown". Start near 0.7 for image examples, and tune it
          with the similarity values shown on the Predict tab, or use the value the Evaluate tab suggests. Several classes above it are
          all listed (multi-label). With a single class, this is what separates members from everything else.
        </small>
      </label>

      <label className={supportsText ? "" : "disabled"}>
        <span>Reject below similarity, zero-shot</span>
        <input type="text" inputMode="decimal" value={rejectZero} placeholder="off" disabled={!supportsText}
          onChange={(e) => { setRejectZero(e.target.value); }} />
        <small className="muted">
          The same, for a task with descriptions but no example images. Image-to-text similarities are much lower (about 0.2–0.35
          for CLIP), so this needs its own value, e.g. 0.22.
        </small>
      </label>

      <label className="check">
        <input type="checkbox" checked={flip} onChange={(e) => { setFlip(e.target.checked); }} />
        <span>Average each image with its mirror image</span>
        <small className="muted check-help">Often a little more accurate when left and right don't matter (most photos). Embedding takes twice as long.</small>
      </label>

      {problems.length > 0 && <div className="error" role="alert"><span>{problems.join(". ")}.</span></div>}
      <div className="row">
        <button onClick={() => { void save(); }} disabled={saving || !embedder || problems.length > 0}>{saving ? "Saving…" : "Save settings"}</button>
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
