import { useEffect, useRef, useState } from "react";
import { api, type Prediction, type TaskDetail } from "./api";
import { Bar, DropZone } from "./ui";

interface Props { task: TaskDetail; setTask: (t: TaskDetail) => void; run: <T>(p: Promise<T>) => Promise<T | undefined> }
interface Item { file: File; url: string; pred: Prediction; addedAs?: string }

export default function Predict({ task, setTask, run }: Props) {
  const [items, setItems] = useState<Item[]>([]);
  const [busy, setBusy] = useState(false);

  // Every preview object URL created by this tab, revoked when the tab unmounts.
  const objectUrls = useRef(new Set<string>());
  useEffect(() => {
    const urls = objectUrls.current;
    return () => {
      urls.forEach((u) => { URL.revokeObjectURL(u); });
      urls.clear();
    };
  }, []);

  const classify = async (files: File[]) => {
    setBusy(true);
    const preds = await run(api.predict(task.name, files));
    setBusy(false);
    if (preds) {
      const added = files.map((f, i) => ({ file: f, url: URL.createObjectURL(f), pred: preds[i] }));
      added.forEach((it) => objectUrls.current.add(it.url));
      setItems((old) => [...added, ...old]);
    }
  };

  const addAs = async (idx: number, label: string) => {
    const r = await run(api.addExamples(task.name, label, [items[idx].file]));
    if (r) {
      setTask(r.task);
      setItems((old) => old.map((it, i) => (i === idx ? { ...it, addedAs: label } : it)));
    }
  };

  const ready = task.classes.length >= 2;
  return (
    <div>
      {!ready && <p className="hint">Add at least two classes on the Examples tab first.</p>}
      <DropZone disabled={!ready || busy} onFiles={(fs) => { void classify(fs); }}>
        {busy ? "Classifying… (the first run loads the model, which can take several seconds)" : "Drop images to classify, or click to choose"}
      </DropZone>
      {items.length > 0 && (
        <div className="row" style={{ justifyContent: "space-between", margin: "12px 0" }}>
          <span className="muted">{items.length} result{items.length === 1 ? "" : "s"}. Add correct or corrected images as examples to improve the task.</span>
          <button className="ghost small" onClick={() => { setItems([]); }}>Clear</button>
        </div>
      )}
      <div className="results">
        {items.map((it, idx) => {
          if (it.pred.error) {
            return (
              <article key={it.url} className="result">
                <img src={it.url} alt={it.file.name} />
                <div className="result-body">
                  <strong className="unknown">Could not read image</strong>
                  <div className="muted file">{it.file.name}</div>
                  <div className="muted">{it.pred.error}</div>
                </div>
              </article>
            );
          }
          const top = Object.entries(it.pred.scores).sort((a, b) => b[1] - a[1]).slice(0, 3);
          return (
            <article key={it.url} className="result">
              <img src={it.url} alt={it.file.name} />
              <div className="result-body">
                <div className="result-head">
                  <strong className={it.pred.rejected ? "unknown" : ""}>
                    {it.pred.rejected ? `Unknown (closest: ${it.pred.top_label})` : it.pred.label}
                  </strong>
                  <span className="muted" title="Cosine similarity to the closest prototype or example. Low values suggest the image fits none of the classes.">
                    sim {it.pred.similarity.toFixed(3)}
                  </span>
                </div>
                <div className="muted file">{it.file.name}</div>
                {top.map(([c, v], i) => <Bar key={c} label={c} value={v} strong={i === 0} />)}
                <div className="row add-as">
                  {it.addedAs ? (
                    <span className="ok">Added as {it.addedAs}</span>
                  ) : (
                    <>
                      <button className="small" onClick={() => { void addAs(idx, it.pred.top_label); }}>Add as {it.pred.top_label}</button>
                      <select className="small" value="" onChange={(e) => { if (e.target.value) void addAs(idx, e.target.value); }} aria-label="Add as another class">
                        <option value="">Add as other…</option>
                        {task.classes.filter((c) => c.name !== it.pred.top_label).map((c) => <option key={c.name}>{c.name}</option>)}
                      </select>
                    </>
                  )}
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}
