import { useEffect, useState } from "react";
import { api, type ModelInfo, type TaskDetail } from "./api";

type Props = {
  task: TaskDetail; models: ModelInfo | null; setTask: (t: TaskDetail) => void;
  run: <T>(p: Promise<T>) => Promise<T | undefined>; onDeleted: () => void;
};

const PARAMS: Record<string, { key: string; label: string; help: string; def: number }[]> = {
  prototype: [{ key: "scale", label: "Scale", help: "Sharpness of image-similarity scores (default 30)", def: 30 }],
  knn: [
    { key: "k", label: "k", help: "Neighbours that vote (default 5)", def: 5 },
    { key: "scale", label: "Scale", help: "Sharpness of neighbour weights (default 30)", def: 30 },
  ],
  logreg: [{ key: "C", label: "C", help: "Inverse regularisation strength (default 10)", def: 10 }],
};

export default function Settings({ task, models, setTask, run, onDeleted }: Props) {
  const [embedder, setEmbedder] = useState(task.embedder);
  const [classifier, setClassifier] = useState(task.classifier);
  const [params, setParams] = useState<Record<string, unknown>>(task.classifier_params);
  const [textWeight, setTextWeight] = useState(task.text_weight);
  const [reject, setReject] = useState(task.reject_below == null ? "" : String(task.reject_below));
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setEmbedder(task.embedder); setClassifier(task.classifier); setParams(task.classifier_params);
    setTextWeight(task.text_weight); setReject(task.reject_below == null ? "" : String(task.reject_below));
  }, [task.name, task.revision]); // eslint-disable-line react-hooks/exhaustive-deps

  const presets = models?.embedders ?? [];
  const isPreset = presets.some((p) => p.spec === embedder);
  const supportsText = embedder.startsWith("openclip:");

  const save = async () => {
    setSaving(true);
    const r = reject.trim();
    const t = await run(api.updateTask(task.name, {
      embedder, classifier, classifier_params: classifier === task.classifier ? params : {},
      text_weight: textWeight,
      ...(r === "" ? { clear_reject: true } : { reject_below: Number(r) }),
    }));
    setSaving(false);
    if (t) setTask(t);
  };

  return (
    <div className="settings">
      <label>
        <span>Embedder</span>
        <select value={isPreset ? embedder : "__custom"} onChange={(e) => setEmbedder(e.target.value === "__custom" ? "" : e.target.value)}>
          {presets.map((p) => <option key={p.spec} value={p.spec}>{p.label}</option>)}
          <option value="__custom">Custom spec…</option>
        </select>
        {!isPreset && <input value={embedder} onChange={(e) => setEmbedder(e.target.value)} placeholder="openclip:ViT-B-16@laion2b_s34b_b88k or timm:<model>" />}
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
          <input type="number" step="any" value={String(params[p.key] ?? p.def)}
            onChange={(e) => setParams({ ...params, [p.key]: e.target.value === "" ? p.def : Number(e.target.value) })} />
          <small className="muted">{p.help}</small>
        </label>
      ))}

      <label className={supportsText ? "" : "disabled"}>
        <span>Text weight: {textWeight.toFixed(2)}</span>
        <input type="range" min={0} max={3} step={0.05} value={textWeight} disabled={!supportsText}
          onChange={(e) => setTextWeight(Number(e.target.value))} />
        <small className="muted">{supportsText
          ? "How much class descriptions count relative to example images. 0 ignores descriptions."
          : "Only CLIP embedders use text descriptions."}</small>
      </label>

      <label>
        <span>Reject below similarity</span>
        <input type="number" step="0.01" min={0} max={1} value={reject} placeholder="off" onChange={(e) => setReject(e.target.value)} />
        <small className="muted">
          Images whose best similarity falls below this value are returned as "unknown". Start near 0.7 for image examples, and tune it
          with the similarity values shown on the Predict tab. Tasks that only have text descriptions score lower (about 0.2–0.35).
        </small>
      </label>

      <div className="row">
        <button onClick={save} disabled={saving || !embedder}>{saving ? "Saving…" : "Save settings"}</button>
      </div>

      <hr />
      <div className="row">
        <button className="danger" onClick={async () => {
          if (confirm(`Delete task "${task.name}" and all ${task.examples.length} images? This cannot be undone.`)) {
            const ok = await run(api.deleteTask(task.name).then(() => true));
            if (ok) onDeleted();
          }
        }}>Delete task</button>
      </div>
    </div>
  );
}
