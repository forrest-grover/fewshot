import { useRef } from "react";
import { api, type TaskDetail } from "./api";
import { revoke, usePasteImages, usePredictResults, useShortcuts, type Item, type Results } from "./hooks";
import { shrinkImages } from "./images";
import { Bar, DropZone } from "./ui";

type ResultsState = ReturnType<typeof usePredictResults>;

interface Props {
  task: TaskDetail; setTask: (t: TaskDetail) => void; run: <T>(p: Promise<T>) => Promise<T | undefined>;
  state: ResultsState;
}

let nextKey = 0;

export default function Predict({ task, setTask, run, state }: Props) {
  const { setResults, urls } = state;
  // Results for another task (e.g. a classify that finished after switching) are never shown.
  const results = state.results.task === task.name ? state.results : { items: [], busy: false, current: null };
  const { items, busy } = results;
  const update = (f: (r: Results) => Results) => {
    setResults((r) => (r.task === task.name ? f(r) : r));
  };
  const cardRefs = useRef(new Map<number, HTMLElement>());
  const actionable = (it: Item) => !it.pred.error && !it.addedAs;
  const current = items.find((it) => it.key === results.current && actionable(it)) ?? items.find(actionable);

  const classify = async (originals: File[]) => {
    update((r) => ({ ...r, busy: true }));
    try {
      const files = await shrinkImages(originals);
      const preds = await run(api.predict(task.name, files));
      if (preds) {
        const added = files.map((f, i) => ({
          key: nextKey++, name: originals[i].name, file: f, url: URL.createObjectURL(f), pred: preds[i],
        }));
        added.forEach((it) => urls.current.add(it.url));
        update((r) => ({ ...r, items: [...added, ...r.items], current: null }));
        // Focus the new top result, so Enter and the number keys act on it rather than on the drop zone.
        const top = added[0].key;
        requestAnimationFrame(() => { cardRefs.current.get(top)?.focus(); });
      }
    } finally {
      update((r) => ({ ...r, busy: false }));
    }
  };

  const patch = (key: number, p: Partial<Item>) => {
    update((r) => ({ ...r, items: r.items.map((it) => (it.key === key ? { ...it, ...p } : it)) }));
  };
  const addAs = async (it: Item, label: string) => {
    if (it.adding || it.addedAs) return;
    patch(it.key, { adding: true });
    const r = await run(api.addExamples(task.name, label, [it.file]));
    patch(it.key, r ? { adding: false, addedAs: label } : { adding: false }); // a failure shows in the error box
    if (r) setTask(r.task);
  };
  const clear = () => {
    revoke(urls.current);
    update((r) => ({ ...r, items: [], current: null }));
  };

  const ready = task.classes.length >= 2;
  usePasteImages((fs) => { if (ready && !busy) void classify(fs); });

  // 1-9: add the highlighted result as that class. Enter: as its prediction. Up/Down (or k/j): move.
  useShortcuts((e) => {
    if (!current) return;
    const open = items.filter(actionable);
    const at = open.indexOf(current);
    let move = 0;
    if (/^[1-9]$/.test(e.key)) {
      const c = task.classes[Number(e.key) - 1] as TaskDetail["classes"][number] | undefined;
      if (!c) return;
      void addAs(current, c.name);
      move = 1;
    } else if (e.key === "Enter") {
      void addAs(current, current.pred.top_label);
      move = 1;
    } else if (e.key === "ArrowDown" || e.key === "j") move = 1;
    else if (e.key === "ArrowUp" || e.key === "k") move = -1;
    else return;
    e.preventDefault();
    const next = open[Math.max(0, Math.min(open.length - 1, at + move))];
    update((r) => ({ ...r, current: next.key }));
    cardRefs.current.get(next.key)?.focus();
  });

  return (
    <div>
      {!ready && <p className="hint">Add at least two classes on the Examples tab first.</p>}
      <DropZone disabled={!ready || busy} onFiles={(fs) => { void classify(fs); }}>
        {busy ? "Classifying… (the first run loads the model, which can take several seconds)" : "Drop images to classify, click to choose, or paste"}
      </DropZone>
      {items.length > 0 && (
        <>
          <div className="row" style={{ justifyContent: "space-between", margin: "12px 0 4px" }}>
            <span className="muted">{items.length} result{items.length === 1 ? "" : "s"}. Add correct or corrected images as examples to improve the task.</span>
            <button className="ghost small" onClick={clear}>Clear</button>
          </div>
          {current && (
            <p className="muted keys">
              Keys for the highlighted result: <kbd>1</kbd>–<kbd>{Math.min(9, task.classes.length)}</kbd> add it as that class,{" "}
              <kbd>Enter</kbd> accepts the prediction, <kbd>↑</kbd>/<kbd>↓</kbd> move.
            </p>
          )}
        </>
      )}
      <div className="results">
        {items.map((it) => {
          if (it.pred.error) {
            return (
              <article key={it.key} className="result">
                <img src={it.url} alt={it.name} />
                <div className="result-body">
                  <strong className="unknown">Could not read image</strong>
                  <div className="muted file">{it.name}</div>
                  <div className="muted">{it.pred.error}</div>
                </div>
              </article>
            );
          }
          const top = Object.entries(it.pred.scores).sort((a, b) => b[1] - a[1]).slice(0, 3);
          return (
            <article
              key={it.key}
              ref={(el) => { if (el) cardRefs.current.set(it.key, el); else cardRefs.current.delete(it.key); }}
              className={`result ${it === current ? "current" : ""}`}
              tabIndex={-1}
              onClick={() => { if (actionable(it)) update((r) => ({ ...r, current: it.key })); }}
            >
              <img src={it.url} alt={it.name} />
              <div className="result-body">
                <div className="result-head">
                  <strong className={it.pred.rejected ? "unknown" : ""}>
                    {it.pred.rejected ? `Unknown (closest: ${it.pred.top_label})` : it.pred.label}
                  </strong>
                  <span className="muted" title="Cosine similarity to the closest prototype or example. Low values suggest the image fits none of the classes.">
                    sim {it.pred.similarity.toFixed(3)}
                  </span>
                </div>
                <div className="muted file">{it.name}</div>
                {top.map(([c, v], i) => <Bar key={c} label={c} value={v} strong={i === 0} />)}
                <div className="row add-as">
                  {it.addedAs ? (
                    <span className="ok">Added as {it.addedAs}</span>
                  ) : (
                    <>
                      <button className="small" disabled={it.adding} onClick={() => { void addAs(it, it.pred.top_label); }}>Add as {it.pred.top_label}</button>
                      <select className="small" value="" disabled={it.adding} onChange={(e) => { if (e.target.value) void addAs(it, e.target.value); }} aria-label="Add as another class">
                        <option value="">Add as other…</option>
                        {task.classes.map((c, i) => c.name === it.pred.top_label ? null : (
                          <option key={c.name} value={c.name}>{i < 9 ? `${String(i + 1)} · ` : ""}{c.name}</option>
                        ))}
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
