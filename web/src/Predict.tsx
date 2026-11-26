import { useMemo, useRef, useState } from "react";
import { api, type Prediction, type TaskDetail } from "./api";
import { downloadCsv } from "./csv";
import { usePasteImages, usePredictResults, useShortcuts, type Item, type Results } from "./hooks";
import { shrinkImages } from "./images";
import { Bar, ClassSelect, DropZone } from "./ui";
import { progressText, sendInBatches } from "./upload";

type ResultsState = ReturnType<typeof usePredictResults>;

interface Props {
  task: TaskDetail; setTask: (t: TaskDetail) => void; run: <T>(p: Promise<T>) => Promise<T | undefined>;
  state: ResultsState; urlFetch: boolean;
}

let nextKey = 0;

/** The http(s) URLs in text separated by whitespace. */
const urlsIn = (text: string) => text.split(/\s+/).filter((u) => /^https?:\/\/\S+/i.test(u));

export default function Predict({ task, setTask, run, state, urlFetch }: Props) {
  const [urlText, setUrlText] = useState("");
  const { setResults, urls } = state;
  // Results for another task (e.g. a classify that finished after switching) are never shown.
  const results = state.results.task === task.name ? state.results : { items: [], batch: null, current: null };
  const { items } = results;
  const busy = results.batch !== null;
  const update = (f: (r: Results) => Results) => {
    setResults((r) => (r.task === task.name ? f(r) : r));
  };
  /** Start a classification; its results go to ``finish`` with the token returned. */
  const begin = () => {
    const batch = nextKey++;
    update((r) => ({ ...r, batch }));
    return batch;
  };
  /** Change the results a classification started on, unless another task was opened since (which
   * drops them). They may have moved to the task's new name: a rename keeps ``batch``. */
  const finish = (batch: number, f: (r: Results) => Results) => {
    setResults((r) => (r.batch === batch ? f(r) : r));
  };
  const cardRefs = useRef(new Map<number, HTMLElement>());
  // "Added" only while that example is still in the task, so an undone add can be made again.
  const exampleIds = useMemo(() => new Set(task.examples.map((e) => e.id)), [task.examples]);
  const classNames = useMemo(() => task.classes.map((c) => c.name), [task.classes]);
  const addedAs = (it: Item) => (it.added && exampleIds.has(it.added.id) ? it.added.label : null);
  const actionable = (it: Item) => !it.pred.error && !addedAs(it);
  const current = items.find((it) => it.key === results.current && actionable(it)) ?? items.find(actionable);

  /** Show the results ``added`` on top, below earlier ones of the same classification (keys in
   * ``mine``, if it came in batches). */
  const show = (batch: number, added: Item[], mine = new Set<number>()) => {
    const first = mine.size === 0;
    finish(batch, (r) => {
      const at = r.items.findIndex((it) => !mine.has(it.key));
      const split = at < 0 ? r.items.length : at;
      return { ...r, items: [...r.items.slice(0, split), ...added, ...r.items.slice(split)], current: first ? null : r.current };
    });
    added.forEach((it) => mine.add(it.key));
    // Focus the new top result, so Enter and the number keys act on it rather than on the drop zone.
    const top = added[0].key;
    if (first) requestAnimationFrame(() => { cardRefs.current.get(top)?.focus(); });
  };
  /** Classify images, in batches (see sendInBatches) whose results show as they arrive; Cancel
   * or opening another task stops it. */
  const classify = async (originals: File[]) => {
    const batch = begin();
    const req = state.abortable();
    const mine = new Set<number>();
    let sent = 0;
    try {
      await run(sendInBatches(originals, (fs, signal) => api.predict(task.name, fs, signal), {
        signal: req.signal, prepare: shrinkImages,
        onProgress: (done, total) => { finish(batch, (r) => ({ ...r, progress: { done, total } })); },
        onBatch: (preds, files) => {
          const added = files.map((f, i) => ({
            key: nextKey++, name: originals[sent + i].name, file: f, url: URL.createObjectURL(f), pred: preds[i],
          }));
          sent += files.length;
          added.forEach((it) => urls.current.add(it.url));
          show(batch, added, mine);
        },
      }));
    } finally {
      req.done();
      finish(batch, (r) => ({ ...r, batch: null, progress: undefined }));
    }
  };
  const classifyUrls = async () => {
    const list = urlsIn(urlText);
    if (!list.length) return;
    const batch = begin();
    try {
      const preds = await run(api.predictUrls(task.name, list));
      if (preds) {
        setUrlText("");
        show(batch, list.map((u, i) => ({ key: nextKey++, name: u, url: u, pred: preds[i] })));
      }
    } finally {
      finish(batch, (r) => ({ ...r, batch: null }));
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

  // By key, which is unique across tasks: this reaches the item even if the task was renamed meanwhile.
  const patch = (key: number, p: Partial<Item>) => {
    setResults((r) => ({ ...r, items: r.items.map((it) => (it.key === key ? { ...it, ...p } : it)) }));
  };
  const addAs = async (it: Item, label: string) => {
    if (it.adding || addedAs(it)) return;
    patch(it.key, { adding: true });
    const r = await run(it.file ? api.addExamples(task.name, label, [it.file]) : api.addExampleUrls(task.name, label, [it.url]));
    const added = r?.added[0];
    patch(it.key, added ? { adding: false, added: { label, id: added.id } } : { adding: false }); // a failure shows in the error box
    if (r) setTask(r.task);
  };
  const clear = () => { update((r) => ({ ...r, items: [], current: null })); }; // their preview URLs are released

  const ready = task.classes.length >= 1;
  const fail = (message: string) => { void run(Promise.reject(new Error(message))); };
  usePasteImages((fs) => {
    if (!ready) fail("Add classes on the Examples tab before classifying images.");
    else if (busy) fail("Still classifying the last images: paste again when they are done.");
    else void classify(fs);
  });

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
      if (current.pred.rejected) return; // "unknown": a number key says which class it is
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
        {busy
          ? `Classifying${results.progress && results.progress.total > 1 ? ` ${progressText(results.progress)}` : ""}… (the first run loads the model, which can take several seconds)`
          : "Drop images to classify, click to choose, or paste"}
      </DropZone>
      {busy && <div className="row"><button className="ghost small" onClick={state.cancel}>Cancel</button></div>}
      {urlFetch && (
        <form className="row url-form" onSubmit={(e) => { e.preventDefault(); void classifyUrls(); }}>
          <input value={urlText} onChange={(e) => { setUrlText(e.target.value); }} className="url-input" placeholder="Or image URLs (https://…), separated by spaces"
            aria-label="Image URLs to classify" disabled={!ready || busy} />
          <button type="submit" className="secondary" disabled={!ready || busy || urlsIn(urlText).length === 0}>Classify URLs</button>
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
              {current.pred.rejected ? "" : <><kbd>Enter</kbd> accepts the prediction, </>}<kbd>↑</kbd>/<kbd>↓</kbd> move.
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
          const also = it.pred.rejected ? it.pred.labels : it.pred.labels.slice(1);
          return (
            <article
              key={it.key}
              ref={(el) => { if (el) cardRefs.current.set(it.key, el); else cardRefs.current.delete(it.key); }}
              className={`result ${it === current ? "current" : ""}`}
              tabIndex={it === current ? 0 : -1} // Tab reaches the highlighted result; arrows move between them
              onFocus={() => { if (actionable(it) && it !== current) update((r) => ({ ...r, current: it.key })); }}
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
                    {it.pred.rejected ? "Matches" : "Also matches"}: {also.join(", ")}
                  </div>
                )}
                {top.map(([c, v], i) => <Bar key={c} label={c} value={v} strong={i === 0} />)}
                {it.pred.evidence.length > 0 && (
                  <div className="evidence" title="The labelled examples most like this image">
                    <span className="muted">Most like</span>
                    {it.pred.evidence.map((e) => (
                      <figure key={e.id}>
                        <img className="evidence-img" src={api.imageUrl(task.name, e.id, 96)} alt={`${e.label} example`} loading="lazy" />
                        <figcaption className="evidence-caption">{e.label} · {e.similarity.toFixed(2)}</figcaption>
                      </figure>
                    ))}
                  </div>
                )}
                <div className="row add-as">
                  {addedAs(it) ? (
                    <span className="ok">Added as {addedAs(it)}</span>
                  ) : (
                    <>
                      {!it.pred.rejected && (
                        <button className="small" disabled={it.adding} onClick={() => { void addAs(it, it.pred.top_label); }}>Add as {it.pred.top_label}</button>
                      )}
                      <ClassSelect className="small" classes={classNames} numbered disabled={it.adding} aria-label="Add as another class"
                        placeholder={it.pred.rejected ? "Add as…" : "Add as other…"} exclude={it.pred.rejected ? undefined : it.pred.top_label}
                        onPick={(c) => { void addAs(it, c); }} />
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
