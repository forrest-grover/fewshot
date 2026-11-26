import { useEffect, useRef, useState } from "react";
import { api, HttpError, type InboxAdded, type InboxEntry, type InboxSort, type TaskDetail } from "./api";
import { pct } from "./format";
import { useAbortOnLeave, useShortcuts } from "./hooks";
import { shrinkImages, THUMB_SIZE } from "./images";
import Lightbox from "./Lightbox";
import { DropZone } from "./ui";
import { progressText, sendInBatches } from "./upload";

interface Props { task: TaskDetail; setTask: (t: TaskDetail) => void; run: <T>(p: Promise<T>) => Promise<T | undefined> }

const SORTS: Record<InboxSort, string> = {
  uncertain: "Most useful to label first (the model is least sure)",
  confident: "Most confident first (to accept in bulk)",
  novel: "Least like any example first (new kinds of image)",
  added: "Oldest first",
};
const PAGE = 60; // cards rendered at first, and added by each "Show more"
const INBOX_PAGE = 1000; // images fetched per request (the server's most)

/** Unlabelled images waiting in the task's inbox, with the model's suggestion for each. */
export default function Inbox({ task, setTask, run }: Props) {
  const [sort, setSort] = useState<InboxSort>("uncertain");
  // The images loaded so far (a page at a time, in the chosen order), of ``total`` waiting.
  const [items, setItems] = useState<{ key: string; list: InboxEntry[]; total: number } | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [picked, setPicked] = useState<ReadonlySet<string>>(new Set());
  const [current, setCurrent] = useState<string | null>(null);
  const [minConfText, setMinConfText] = useState("0.9");
  const minConf = minConfText.trim() === "" ? NaN : Number(minConfText); // NaN (empty or not a number): accept none
  const [shown, setShown] = useState(PAGE);
  const [busy, setBusy] = useState(false);
  const [viewing, setViewing] = useState<number | null>(null); // index into the list, full size
  const [gone, setGone] = useState<ReadonlySet<string>>(new Set()); // accepted or dismissed, until the list reloads
  const cards = useRef(new Map<string, HTMLElement>());

  // Suggestions depend on the task's examples and settings: reload whenever it changes.
  const key = `${task.name}\u0000${String(task.revision)}\u0000${sort}`;
  useEffect(() => {
    let live = true;
    void run(api.inbox(task.name, sort, 0, INBOX_PAGE)).then((r) => {
      if (live && r) { setItems({ key, list: r.items, total: r.total }); setGone(new Set()); }
    });
    return () => { live = false; };
  }, [key, task.name, sort, run]);
  const list = (items?.list ?? []).filter((i) => !gone.has(i.id));
  const stale = items?.key !== key;
  const unloaded = items ? Math.max(0, items.total - items.list.length) : 0; // waiting beyond the pages loaded
  /** Load the next page, unless the list has been reloaded (the task changed) meanwhile. */
  const loadMore = async () => {
    if (!items) return;
    setLoadingMore(true);
    const r = await run(api.inbox(task.name, sort, items.list.length, INBOX_PAGE));
    setLoadingMore(false);
    if (r) setItems((cur) => (cur?.key === key ? { ...cur, list: [...cur.list, ...r.items], total: r.total } : cur));
  };
  const locked = busy || stale; // a change is being saved, or the list is reloading after one
  if (viewing !== null && list.length === 0) setViewing(null); // emptied: don't reopen when the list refills

  const cur = list.find((i) => i.id === current) ?? list.find((i) => !picked.has(i.id)) ?? null;
  /** Save a change to inbox images ``ids``: they leave the list at once (and come back if it
   * fails), and the highlight moves on to the image after the highlighted one if it was among them. */
  const save = async (ids: string[], p: Promise<TaskDetail>) => {
    const rest = list.filter((i) => !ids.includes(i.id));
    if (cur && ids.includes(cur.id)) {
      const next = rest[Math.min(list.indexOf(cur), rest.length - 1)] as InboxEntry | undefined;
      setCurrent(next?.id ?? null);
    }
    setGone((g) => new Set([...g, ...ids]));
    setBusy(true);
    const t = await run(p);
    setBusy(false);
    if (t) { setTask(t); setPicked(new Set()); }
    else setGone((g) => new Set([...g].filter((id) => !ids.includes(id))));
  };
  const label = (labels: Record<string, string>) => save(Object.keys(labels), api.acceptInbox(task.name, labels));
  const dismiss = (ids: string[]) => save(ids, api.dismissInbox(task.name, ids));
  const suggested = (i: InboxEntry) => (i.prediction && !i.prediction.rejected && i.prediction.label ? i.prediction.label : null);
  const sure = list.filter((i) => suggested(i) && (i.prediction?.confidence ?? 0) >= minConf);
  const selection = list.filter((i) => picked.has(i.id));

  const [uploading, setUploading] = useState<{ done: number; total: number; cancel: () => void } | null>(null);
  const newUpload = useAbortOnLeave(task.name); // uploads stop when the task is closed
  /** Add images to the inbox in batches (see sendInBatches); Cancel (or leaving) stops the rest. */
  const upload = async (files: File[]) => {
    const req = newUpload();
    const skips: string[] = [];
    /** One batch; one whose images were all unusable (400) is skipped, not the end of the upload. */
    const send = async (fs: File[], signal: AbortSignal): Promise<InboxAdded | null> => {
      try {
        return await api.addToInbox(task.name, fs, signal);
      } catch (e) {
        if (!(e instanceof HttpError && e.status === 400)) throw e;
        skips.push(e.message);
        return null;
      }
    };
    setBusy(true);
    try {
      await run(sendInBatches(files, send, {
        signal: req.signal, prepare: shrinkImages,
        onProgress: (done, total) => { setUploading({ done, total, cancel: req.cancel }); },
        onBatch: (r) => { if (r) { setTask(r.task); skips.push(...r.errors.map((e) => (e.file ? `${e.file}: ${e.error}` : e.error))); } },
      }));
    } finally {
      req.done();
      setBusy(false);
      setUploading(null);
    }
    if (skips.length) void run(Promise.reject(new Error(`Skipped: ${skips.join("; ")}`)));
  };

  // Keys for the highlighted image: Enter accepts the suggestion, 1-9 label it, Delete dismisses, arrows move.
  useShortcuts((e) => {
    if (!cur) return;
    const at = list.indexOf(cur);
    let move = 0;
    const act = (done: () => void) => {
      e.preventDefault();
      if (locked) return; // the last change is still being saved: don't act on a stale list
      done();
      // The image leaves the list and the highlight moves to the next one (see ``save``): focus it.
      const rest = list.filter((i) => i !== cur);
      const next = rest[Math.min(at, rest.length - 1)] as InboxEntry | undefined;
      if (next) requestAnimationFrame(() => { cards.current.get(next.id)?.focus(); });
    };
    if (e.key === "Enter") {
      const s = suggested(cur);
      if (s) act(() => { void label({ [cur.id]: s }); });
      return;
    } else if (/^[1-9]$/.test(e.key)) {
      const c = task.classes[Number(e.key) - 1] as TaskDetail["classes"][number] | undefined;
      if (c) act(() => { void label({ [cur.id]: c.name }); });
      return;
    } else if (e.key === "Delete" || e.key === "Backspace") {
      act(() => { void dismiss([cur.id]); });
      return;
    } else if (e.key === " ") {
      e.preventDefault();
      setViewing(at);
      return;
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
        {uploading ? `Uploading ${progressText(uploading)}…` : busy ? "Working…" : "Drop unlabelled images here (or click) to sort them with the model's suggestions"}
      </DropZone>
      {uploading && <div className="row"><button className="ghost small" onClick={uploading.cancel}>Cancel upload</button></div>}
      {viewing !== null && list.length > 0 && (
        <Lightbox
          items={list.map((i) => ({
            url: api.imageUrl(task.name, i.id, 1024), title: i.source || i.id.slice(0, 12),
            details: [i.prediction ? `Suggestion: ${suggested(i) ?? `unknown (closest ${i.prediction.top_label})`}, ${pct(i.prediction.confidence)}` : ""].filter(Boolean),
          }))}
          index={Math.min(viewing, list.length - 1)} onIndex={setViewing} onClose={() => { setViewing(null); }}
          onKey={(e, k) => {
            // A labelled or dismissed image leaves the list, so the viewer moves on to the next one.
            const item = list[k];
            const s = suggested(item);
            if (e.key === "Enter" && s) { if (!locked) void label({ [item.id]: s }); return true; }
            if (/^[1-9]$/.test(e.key)) {
              const c = task.classes[Number(e.key) - 1] as TaskDetail["classes"][number] | undefined;
              if (c && !locked) void label({ [item.id]: c.name });
              return true;
            }
            if (e.key === "Delete" || e.key === "Backspace") { if (!locked) void dismiss([item.id]); return true; }
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
              <input type="number" min={0} max={1} step={0.05} value={minConfText} className="threshold-input"
                onChange={(e) => { setMinConfText(e.target.value); }} />
            </label>
            <button disabled={locked || sure.length === 0}
              onClick={() => { void label(Object.fromEntries(sure.map((i) => [i.id, suggested(i) ?? ""]))); }}>
              Accept {sure.length} suggestion{sure.length === 1 ? "" : "s"}{unloaded ? " loaded" : ""}
            </button>
          </div>
          <p className="muted keys">
            {task.inbox_count} waiting{unloaded ? ` (${String(list.length)} loaded; accepting only acts on those)` : ""}. Keys for the highlighted image: <kbd>Enter</kbd> accepts the suggestion,{" "}
            <kbd>1</kbd>–<kbd>{Math.min(9, Math.max(1, task.classes.length))}</kbd> label it as that class, <kbd>Delete</kbd> dismisses it,
            {" "}<kbd>←</kbd>/<kbd>→</kbd> move.
          </p>
          {selection.length > 0 && (
            <div className="bulk-bar" role="toolbar" aria-label="Selected inbox images">
              <strong>{selection.length} selected</strong>
              <button className="small" disabled={locked || !selection.some(suggested)}
                onClick={() => { void label(Object.fromEntries(selection.filter(suggested).map((i) => [i.id, suggested(i) ?? ""]))); }}>
                Accept suggestions
              </button>
              <select value="" aria-label="Label selected as" disabled={locked} onChange={(e) => {
                if (e.target.value) void label(Object.fromEntries(selection.map((i) => [i.id, e.target.value])));
              }}>
                <option value="">Label as…</option>
                {task.classes.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
              <button className="small danger" disabled={locked} onClick={() => { void dismiss(selection.map((i) => i.id)); }}>Dismiss</button>
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
                    loading={n < 12 ? "eager" : "lazy"} onDoubleClick={() => { setViewing(n); }} title="Double-click (or Space) to view full size" />
                  <div className="inbox-body">
                    {p ? (
                      <span className={s ? "" : "unknown"}>
                        {s ?? `Unknown (closest: ${p.top_label})`} <span className="muted">{pct(p.confidence)}</span>
                      </span>
                    ) : <span className="muted">No classes yet</span>}
                    <div className="row">
                      {s && <button className="small" disabled={locked} onClick={(e) => { e.stopPropagation(); void label({ [i.id]: s }); }}>✓ {s}</button>}
                      <select className="small" value="" disabled={locked} aria-label="Label as"
                        onChange={(e) => { if (e.target.value) void label({ [i.id]: e.target.value }); }}>
                        <option value="">Label as…</option>
                        {task.classes.map((c, k) => <option key={c.name} value={c.name}>{k < 9 ? `${String(k + 1)} · ` : ""}{c.name}</option>)}
                      </select>
                      <button className="ghost small danger" aria-label="Dismiss" title="Dismiss (Delete)" disabled={locked}
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
          {list.length <= shown && unloaded > 0 && (
            <button className="secondary more" disabled={loadingMore || stale} onClick={() => { void loadMore(); }}>
              {loadingMore ? "Loading…" : `Load more (${String(unloaded)} not loaded yet)`}
            </button>
          )}
        </>
      )}
    </div>
  );
}
