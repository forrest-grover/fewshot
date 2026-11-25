import { useRef, useState } from "react";
import { api, type Prediction, type TaskDetail } from "./api";
import { downloadCsv } from "./csv";
import { revoke, usePasteImages, usePredictResults, useShortcuts, type Item, type Results } from "./hooks";
import { shrinkImages } from "./images";
import { Bar, DropZone } from "./ui";

type ResultsState = ReturnType<typeof usePredictResults>;

interface Props {
  task: TaskDetail; setTask: (t: TaskDetail) => void; run: <T>(p: Promise<T>) => Promise<T | undefined>;
  state: ResultsState; urlFetch: boolean;
}

let nextKey = 0;

export default function Predict({ task, setTask, run, state, urlFetch }: Props) {
  const [urlText, setUrlText] = useState("");
  const { setResults, urls } = state;
  // Results for another task (e.g. a classify that finished after switching) are never shown.
  const results = state.results.task === task.name ? state.results : { items: [], busy: false, current: null };
  const { items, busy } = results;
  const update = (f: (r: Results) => Results) => {
    setResults((r) => (r.task === task.name ? f(r) : r));
  };
  const cardRefs = useRef(new Map<number, HTMLElement>());
  // "Added" only while that example is still in the task, so an undone add can be made again.
  const addedAs = (it: Item) => (it.added && task.examples.some((e) => e.id === it.added?.id) ? it.added.label : null);
  const actionable = (it: Item) => !it.pred.error && !addedAs(it);
  const current = items.find((it) => it.key === results.current && actionable(it)) ?? items.find(actionable);

  const show = (added: Item[]) => {
    update((r) => ({ ...r, items: [...added, ...r.items], current: null }));
    // Focus the new top result, so Enter and the number keys act on it rather than on the drop zone.
    const top = added[0].key;
    requestAnimationFrame(() => { cardRefs.current.get(top)?.focus(); });
  };
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
        show(added);
      }
    } finally {
      update((r) => ({ ...r, busy: false }));
    }
  };
  const classifyUrls = async () => {
    const list = urlText.split(/\s+/).filter((u) => /^https?:\/\//i.test(u));
    if (!list.length) return;
    update((r) => ({ ...r, busy: true }));
    try {
      const preds = await run(api.predictUrls(task.name, list));
      if (preds) {
        setUrlText("");
        show(list.map((u, i) => ({ key: nextKey++, name: u, url: u, pred: preds[i] })));
      }
    } finally {
      update((r) => ({ ...r, busy: false }));
    }
  };
  const exportCsv = () => {
    const row = (it: Item) => {
      const p: Prediction = it.pred;
      return [it.name, p.label, p.labels.join(";"), p.top_label, p.confidence.toFixed(4), p.similarity.toFixed(4), p.rejected, p.error];
    };
    downloadCsv(`${task.name}-predictions.csv`, [
      ["file", "label", "labels", "top_label", "confidence", "similarity", "rejected", "error"], ...items.map(row),
    ]);
  };

  const patch = (key: number, p: Partial<Item>) => {
    update((r) => ({ ...r, items: r.items.map((it) => (it.key === key ? { ...it, ...p } : it)) }));
  };
  const addAs = async (it: Item, label: string) => {
    if (it.adding || addedAs(it)) return;
    patch(it.key, { adding: true });
    const r = await run(it.file ? api.addExamples(task.name, label, [it.file]) : api.addExampleUrls(task.name, label, [it.url]));
    const added = r?.added[0];
    patch(it.key, added ? { adding: false, added: { label, id: added.id } } : { adding: false }); // a failure shows in the error box
    if (r) setTask(r.task);
  };
  const clear = () => {
    revoke(urls.current);
    update((r) => ({ ...r, items: [], current: null }));
  };

  const ready = task.classes.length >= 1;
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
      {!ready && <p className="hint">Add classes on the Examples tab first (one class works too, with a reject threshold, to tell members from outsiders).</p>}
      <DropZone disabled={!ready || busy} onFiles={(fs) => { void classify(fs); }}>
        {busy ? "Classifying… (the first run loads the model, which can take several seconds)" : "Drop images to classify, click to choose, or paste"}
      </DropZone>
      {urlFetch && (
        <form className="row url-form" onSubmit={(e) => { e.preventDefault(); void classifyUrls(); }}>
          <input value={urlText} onChange={(e) => { setUrlText(e.target.value); }} className="url-input" placeholder="Or image URLs (https://…), separated by spaces"
            aria-label="Image URLs to classify" disabled={!ready || busy} />
          <button type="submit" className="secondary" disabled={!ready || busy || !/https?:\/\//i.test(urlText)}>Classify URLs</button>
        </form>
      )}
      {items.length > 0 && (
        <>
          <div className="row" style={{ justifyContent: "space-between", margin: "12px 0 4px" }}>
            <span className="muted">{items.length} result{items.length === 1 ? "" : "s"}. Add correct or corrected images as examples to improve the task.</span>
            <span className="row">
              <button className="ghost small" onClick={exportCsv}>Export CSV</button>
              <button className="ghost small" onClick={clear}>Clear</button>
            </span>
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
          const also = it.pred.labels.slice(1);
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
                {also.length > 0 && (
                  <div className="also" title="Other classes whose similarity also reaches the reject threshold">
                    Also matches: {also.join(", ")}
                  </div>
                )}
                {top.map(([c, v], i) => <Bar key={c} label={c} value={v} strong={i === 0} />)}
                <div className="row add-as">
                  {addedAs(it) ? (
                    <span className="ok">Added as {addedAs(it)}</span>
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
