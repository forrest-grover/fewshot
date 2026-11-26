import { useEffect, useRef, useState } from "react";
import { api, type InboxEntry, type InboxSort, type TaskDetail } from "./api";
import { pct } from "./format";
import { useShortcuts } from "./hooks";
import { shrinkImages, THUMB_SIZE } from "./images";
import Lightbox from "./Lightbox";
import { DropZone } from "./ui";

interface Props { task: TaskDetail; setTask: (t: TaskDetail) => void; run: <T>(p: Promise<T>) => Promise<T | undefined> }

const SORTS: Record<InboxSort, string> = {
  uncertain: "Most useful to label first (the model is least sure)",
  confident: "Most confident first (to accept in bulk)",
  novel: "Least like any example first (new kinds of image)",
  added: "Oldest first",
};
const PAGE = 60;

/** Unlabelled images waiting in the task's inbox, with the model's suggestion for each. */
export default function Inbox({ task, setTask, run }: Props) {
  const [sort, setSort] = useState<InboxSort>("uncertain");
  const [items, setItems] = useState<{ key: string; list: InboxEntry[] } | null>(null);
  const [picked, setPicked] = useState<ReadonlySet<string>>(new Set());
  const [current, setCurrent] = useState<string | null>(null);
  const [minConf, setMinConf] = useState(0.9);
  const [shown, setShown] = useState(PAGE);
  const [busy, setBusy] = useState(false);
  const [viewing, setViewing] = useState<number | null>(null); // index into the list, full size
  const cards = useRef(new Map<string, HTMLElement>());

  // Suggestions depend on the task's examples and settings: reload whenever it changes.
  const key = `${task.name}\u0000${String(task.revision)}\u0000${sort}`;
  useEffect(() => {
    let live = true;
    void run(api.inbox(task.name, sort)).then((r) => { if (live && r) setItems({ key, list: r.items }); });
    return () => { live = false; };
  }, [key, task.name, sort, run]);
  const list = items?.list ?? [];
  const stale = items?.key !== key;

  const save = async (p: Promise<TaskDetail>) => {
    setBusy(true);
    const t = await run(p);
    setBusy(false);
    if (t) { setTask(t); setPicked(new Set()); }
  };
  const label = (labels: Record<string, string>) => save(api.acceptInbox(task.name, labels));
  const dismiss = (ids: string[]) => save(api.dismissInbox(task.name, ids));
  const suggested = (i: InboxEntry) => (i.prediction && !i.prediction.rejected && i.prediction.label ? i.prediction.label : null);
  const sure = list.filter((i) => suggested(i) && (i.prediction?.confidence ?? 0) >= minConf);
  const selection = list.filter((i) => picked.has(i.id));
  const cur = list.find((i) => i.id === current) ?? list.find((i) => !picked.has(i.id)) ?? null;

  const upload = async (files: File[]) => {
    setBusy(true);
    try {
      const r = await run(shrinkImages(files).then((fs) => api.addToInbox(task.name, fs)));
      if (r) {
        setTask(r.task);
        if (r.errors.length) void run(Promise.reject(new Error(`Skipped ${String(r.errors.length)}: ${r.errors.map((e) => `${e.file}: ${e.error}`).join("; ")}`)));
      }
    } finally {
      setBusy(false);
    }
  };

  // Keys for the highlighted image: Enter accepts the suggestion, 1-9 label it, Delete dismisses, arrows move.
  useShortcuts((e) => {
    if (!cur || busy) return;
    const at = list.indexOf(cur);
    let move = 0;
    if (e.key === "Enter") {
      const s = suggested(cur);
      if (!s) return;
      void label({ [cur.id]: s });
      move = 1;
    } else if (/^[1-9]$/.test(e.key)) {
      const c = task.classes[Number(e.key) - 1] as TaskDetail["classes"][number] | undefined;
      if (!c) return;
      void label({ [cur.id]: c.name });
      move = 1;
    } else if (e.key === "Delete" || e.key === "Backspace") {
      void dismiss([cur.id]);
      move = 1;
    } else if (e.key === "ArrowDown" || e.key === "j" || e.key === "ArrowRight") move = 1;
    else if (e.key === "ArrowUp" || e.key === "k" || e.key === "ArrowLeft") move = -1;
    else return;
    e.preventDefault();
    const to = Math.max(0, Math.min(list.length - 1, at + move));
    const next = list[to] as InboxEntry | undefined;
    if (next && next !== cur) {
      setCurrent(next.id);
      if (to >= shown) { // moving past the cards shown so far: show more, then focus it once rendered
        setShown((v) => Math.max(v, to + 1) + PAGE);
        requestAnimationFrame(() => { cards.current.get(next.id)?.focus(); });
      } else {
        cards.current.get(next.id)?.focus();
      }
    }
  });

  return (
    <div className="inbox">
      <DropZone disabled={busy} onFiles={(fs) => { void upload(fs); }}>
        {busy ? "Working…" : "Drop unlabelled images here (or click) to sort them with the model's suggestions"}
      </DropZone>
      {viewing !== null && list.length > 0 && (
        <Lightbox
          items={list.map((i) => ({
            url: api.imageUrl(task.name, i.id, 1024), title: i.source || i.id.slice(0, 12),
            details: [i.prediction ? `Suggestion: ${suggested(i) ?? `unknown (closest ${i.prediction.top_label})`}, ${pct(i.prediction.confidence)}` : ""].filter(Boolean),
          }))}
          index={Math.min(viewing, list.length - 1)} onIndex={setViewing} onClose={() => { setViewing(null); }}
          onKey={(e, k) => {
            const item = list[k];
            const s = suggested(item);
            if (e.key === "Enter" && s) { void label({ [item.id]: s }); return true; }
            if (/^[1-9]$/.test(e.key)) {
              const c = task.classes[Number(e.key) - 1] as TaskDetail["classes"][number] | undefined;
              if (c) void label({ [item.id]: c.name });
              return true;
            }
            if (e.key === "Delete" || e.key === "Backspace") { void dismiss([item.id]); return true; }
            return false;
          }} />
      )}
      {task.inbox_count === 0 ? (
        <p className="hint">
          The inbox holds images that aren't labelled yet. Add a batch, then label them from the model's suggestions: the
          ones it is least sure about teach it the most, and confident ones can be accepted all at once.
        </p>
      ) : (
        <>
          <div className="row inbox-bar">
            <label className="inline">
              <span>Order</span>
              <select value={sort} onChange={(e) => { setSort(e.target.value as InboxSort); setShown(PAGE); }}>
                {Object.entries(SORTS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </label>
            <span className="spacer" />
            <label className="inline">
              <span>Confidence at least</span>
              <input type="number" min={0} max={1} step={0.05} value={minConf} className="threshold-input"
                onChange={(e) => { setMinConf(Number(e.target.value)); }} />
            </label>
            <button disabled={busy || stale || sure.length === 0}
              onClick={() => { void label(Object.fromEntries(sure.map((i) => [i.id, suggested(i) ?? ""]))); }}>
              Accept {sure.length} suggestion{sure.length === 1 ? "" : "s"}
            </button>
          </div>
          <p className="muted keys">
            {task.inbox_count} waiting. Keys for the highlighted image: <kbd>Enter</kbd> accepts the suggestion,{" "}
            <kbd>1</kbd>–<kbd>{Math.min(9, Math.max(1, task.classes.length))}</kbd> label it as that class, <kbd>Delete</kbd> dismisses it,
            {" "}<kbd>←</kbd>/<kbd>→</kbd> move.
          </p>
          {selection.length > 0 && (
            <div className="bulk-bar" role="toolbar" aria-label="Selected inbox images">
              <strong>{selection.length} selected</strong>
              <button className="small" disabled={busy || !selection.some(suggested)}
                onClick={() => { void label(Object.fromEntries(selection.filter(suggested).map((i) => [i.id, suggested(i) ?? ""]))); }}>
                Accept suggestions
              </button>
              <select value="" aria-label="Label selected as" onChange={(e) => {
                if (e.target.value) void label(Object.fromEntries(selection.map((i) => [i.id, e.target.value])));
              }}>
                <option value="">Label as…</option>
                {task.classes.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
              <button className="small danger" onClick={() => { void dismiss(selection.map((i) => i.id)); }}>Dismiss</button>
              <button className="ghost small" onClick={() => { setPicked(new Set()); }}>Clear selection</button>
            </div>
          )}
          <div className={`inbox-grid ${stale ? "stale" : ""}`}>
            {list.slice(0, shown).map((i, n) => {
              const p = i.prediction;
              const s = suggested(i);
              return (
                <article key={i.id} className={`inbox-card ${cur?.id === i.id ? "current" : ""} ${picked.has(i.id) ? "picked" : ""}`}
                  tabIndex={-1} ref={(el) => { if (el) cards.current.set(i.id, el); else cards.current.delete(i.id); }}
                  onClick={() => { setCurrent(i.id); }}>
                  <label className="inbox-pick" title="Tick to select">
                    <input type="checkbox" checked={picked.has(i.id)} aria-label={`Select ${i.source || i.id}`}
                      onChange={() => {
                        setPicked((old) => {
                          const next = new Set(old);
                          if (next.has(i.id)) next.delete(i.id); else next.add(i.id);
                          return next;
                        });
                      }} />
                  </label>
                  <img className="inbox-img zoomable" src={api.imageUrl(task.name, i.id, THUMB_SIZE)} alt={i.source || i.id}
                    loading={n < 12 ? "eager" : "lazy"} onDoubleClick={() => { setViewing(n); }} title="Double-click to view full size" />
                  <div className="inbox-body">
                    {p ? (
                      <span className={s ? "" : "unknown"}>
                        {s ?? `Unknown (closest: ${p.top_label})`} <span className="muted">{pct(p.confidence)}</span>
                      </span>
                    ) : <span className="muted">No classes yet</span>}
                    <div className="row">
                      {s && <button className="small" disabled={busy} onClick={(e) => { e.stopPropagation(); void label({ [i.id]: s }); }}>✓ {s}</button>}
                      <select className="small" value="" disabled={busy} aria-label="Label as"
                        onChange={(e) => { if (e.target.value) void label({ [i.id]: e.target.value }); }}>
                        <option value="">Label as…</option>
                        {task.classes.map((c, k) => <option key={c.name} value={c.name}>{k < 9 ? `${String(k + 1)} · ` : ""}{c.name}</option>)}
                      </select>
                      <button className="ghost small danger" aria-label="Dismiss" title="Dismiss (Delete)" disabled={busy}
                        onClick={(e) => { e.stopPropagation(); void dismiss([i.id]); }}>×</button>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
          {list.length > shown && (
            <button className="secondary more" onClick={() => { setShown((v) => v + PAGE * 4); }}>Show more ({list.length - shown} hidden)</button>
          )}
        </>
      )}
    </div>
  );
}
