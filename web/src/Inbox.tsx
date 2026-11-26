import { useEffect, useMemo, useRef, useState } from "react";
import { api, type InboxEntry, type InboxSort, type TaskDetail } from "./api";
import { pct } from "./format";
import { fetchInbox, stillGone, type Gone } from "./inboxPages";
import { useShortcuts } from "./keys";
import type { Uploads } from "./useUploads";
import { THUMB_SIZE } from "./images";
import Lightbox from "./Lightbox";
import { ClassSelect, DropZone } from "./ui";
import { progressText } from "./upload";

interface Props {
  task: TaskDetail; setTask: (t: TaskDetail) => void; run: <T>(p: Promise<T>) => Promise<T | undefined>;
  uploads: Uploads; // uploads to the inbox (slot INBOX), kept by the App so they go on in other tabs
}

const SORTS: Record<InboxSort, string> = {
  uncertain: "Most useful to label first (the model is least sure)",
  confident: "Most confident first (to accept in bulk)",
  novel: "Least like any example first (new kinds of image)",
  added: "Oldest first",
};
const PAGE = 60; // cards rendered at first, and added by each "Show more"
const INBOX_PAGE = 1000; // images fetched per request (the server's most)
/** The upload slot (see useUploads) of the inbox. */
const INBOX = "inbox";
/** The images loaded so far (a page at a time, in the chosen order), of ``total`` waiting, as
 * fetched at task revision ``rev``. */
interface Loaded { key: string; of: string; rev: number; list: InboxEntry[]; total: number }

/** Unlabelled images waiting in the task's inbox, with the model's suggestion for each. */
export default function Inbox({ task, setTask, run, uploads }: Props) {
  const [sort, setSort] = useState<InboxSort>("uncertain");
  const [items, setItemsState] = useState<Loaded | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [picked, setPicked] = useState<ReadonlySet<string>>(new Set());
  const [current, setCurrent] = useState<string | null>(null);
  const [minConfText, setMinConfText] = useState("0.9");
  const minConf = minConfText.trim() === "" ? NaN : Number(minConfText); // NaN (empty or not a number): accept none
  const [shown, setShown] = useState(PAGE);
  const [saving, setSaving] = useState(0); // changes sent and not yet answered
  const [viewing, setViewing] = useState<number | null>(null); // index into the list, full size
  // Accepted or dismissed images leave the list at once (see Gone). The list, what left it and the
  // highlight are kept in refs as well as in state, so keys pressed faster than React renders each
  // act on the list as the last one left it.
  const itemsNow = useRef(items);
  const setItems = (f: (l: Loaded | null) => Loaded | null) => { itemsNow.current = f(itemsNow.current); setItemsState(itemsNow.current); };
  const [gone, setGoneState] = useState<Gone>(new Map());
  const goneNow = useRef<Gone>(gone);
  const setGone = (f: (g: Gone) => Gone) => { goneNow.current = f(goneNow.current); setGoneState(goneNow.current); };
  const currentNow = useRef(current);
  const setCurrentNow = (id: string | null) => { currentNow.current = id; setCurrent(id); };
  const queue = useRef<Promise<unknown>>(Promise.resolve()); // changes are sent one at a time, in order
  const cards = useRef(new Map<string, HTMLElement>());
  const classNames = useMemo(() => task.classes.map((c) => c.name), [task.classes]);

  // Suggestions depend on the task's examples and settings: reload whenever it changes.
  const key = `${task.name}\u0000${String(task.revision)}\u0000${sort}`;
  const listId = `${task.name}\u0000${sort}`; // the same list, at any revision
  // A reload after a change to the task fetches as many images as were loaded (with "Load more"),
  // so they stay, and so does the highlight; another task or order starts from one page.
  const loaded = useRef(items);
  loaded.current = items;
  useEffect(() => {
    let live = true;
    const rev = task.revision;
    const want = loaded.current?.of === listId ? loaded.current.list.length : 0;
    void fetchInbox((offset, limit) => run(api.inbox(task.name, sort, offset, limit)), want, INBOX_PAGE).then((r) => {
      if (!live || !r) return;
      setItems(() => ({ key, of: listId, rev, list: r.items, total: r.total }));
      goneNow.current = stillGone(goneNow.current, rev);
      setGoneState(goneNow.current);
    });
    return () => { live = false; };
  }, [key, listId, task.name, task.revision, sort, run]);
  /** The images to show, without those acted on: as rendered, or (``now``) as the last change left them. */
  const visible = (g: Gone, l = items) => (l?.list ?? []).filter((i) => !g.has(i.id));
  const list = visible(gone);
  /** The list as the last change left it, rendered or not. */
  const listNow = () => visible(goneNow.current, itemsNow.current);
  const reordering = items !== null && items.of !== listId; // another order (or task) is loading
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
  const uploading = uploads.slots[INBOX]?.progress ?? null;
  if (viewing !== null && list.length === 0) setViewing(null); // emptied: don't reopen when the list refills

  /** The highlighted image in ``l``: the one picked, else the first not ticked. */
  const highlighted = (l: InboxEntry[], id: string | null) => l.find((i) => i.id === id) ?? l.find((i) => !picked.has(i.id)) ?? null;
  const cur = highlighted(list, current);
  /** Save a change to inbox images ``ids`` (``send`` makes the request, once the changes before it
   * are answered): they leave the list at once (and come back if it fails), and the highlight moves
   * on to the image after the highlighted one if it was among them. Resolves to the image
   * highlighted next. Nothing waits for the list to reload, so keys pressed in a row each act on the
   * next image. */
  const save = (ids: string[], send: () => Promise<TaskDetail>): InboxEntry | null => {
    const before = listNow();
    const was = highlighted(before, currentNow.current);
    const rest = before.filter((i) => !ids.includes(i.id));
    let next = was && !ids.includes(was.id) ? was : null;
    if (was && ids.includes(was.id)) {
      next = rest[Math.min(before.indexOf(was), rest.length - 1)] as InboxEntry | undefined ?? null;
      setCurrentNow(next?.id ?? null);
    }
    setGone((g) => new Map([...g, ...ids.map((id) => [id, null] as const)]));
    setSaving((n) => n + 1);
    const done = queue.current.then(() => run(send())).then((t) => {
      setSaving((n) => n - 1);
      if (t) {
        setGone((g) => new Map([...g].map(([id, r]) => [id, ids.includes(id) ? t.revision : r])));
        setTask(t);
        setPicked((p) => new Set([...p].filter((id) => !ids.includes(id))));
      } else {
        setGone((g) => new Map([...g].filter(([id]) => !ids.includes(id))));
      }
    });
    queue.current = done;
    return next;
  };
  const label = (labels: Record<string, string>) => save(Object.keys(labels), () => api.acceptInbox(task.name, labels));
  const dismiss = (ids: string[]) => save(ids, () => api.dismissInbox(task.name, ids));
  const suggested = (i: InboxEntry) => (i.prediction && !i.prediction.rejected && i.prediction.label ? i.prediction.label : null);
  const sure = list.filter((i) => suggested(i) && (i.prediction?.confidence ?? 0) >= minConf);
  const selection = list.filter((i) => picked.has(i.id));

  /** Add images to the inbox in batches (see useUploads), as one undo step; Cancel (or opening
   * another task) stops the rest. Labelling goes on meanwhile. */
  const upload = async (files: File[]) => {
    const name = task.name;
    const skips = await uploads.start(INBOX, files, (fs, signal, group) => api.addToInbox(name, fs, signal, group));
    if (skips?.length) {
      void run(Promise.reject(new Error(`Skipped: ${skips.map((e) => (e.file ? `${e.file}: ${e.error}` : e.error)).join("; ")}`)));
    }
  };

  /** What key ``k`` does to inbox image ``item``: Enter accepts its suggestion, 1-9 label it as
   * that class, Delete dismisses it. Returns false for other keys; true if handled (even when there
   * was nothing to do, e.g. Enter without a suggestion), with the image highlighted next if acted on. */
  const keyAction = (k: string, item: InboxEntry): { handled: boolean; next?: InboxEntry | null } => {
    if (k === "Enter") {
      const s = suggested(item);
      return { handled: true, next: s ? label({ [item.id]: s }) : undefined };
    }
    if (/^[1-9]$/.test(k)) {
      const c = task.classes[Number(k) - 1] as TaskDetail["classes"][number] | undefined;
      return { handled: true, next: c ? label({ [item.id]: c.name }) : undefined };
    }
    if (k === "Delete" || k === "Backspace") return { handled: true, next: dismiss([item.id]) };
    return { handled: false };
  };

  // Keys for the highlighted image (see keyAction); Space views it, arrows move.
  useShortcuts((e) => {
    const now = listNow(); // as the last key left it, even if not rendered yet
    const at = highlighted(now, currentNow.current);
    if (!at) return;
    const done = keyAction(e.key, at);
    if (done.handled) {
      e.preventDefault();
      // The image left the list and the highlight moved on (see ``save``): focus it once rendered
      // (where it is then, if more keys moved it on meanwhile).
      if (done.next) requestAnimationFrame(() => { if (currentNow.current) cards.current.get(currentNow.current)?.focus(); });
      return;
    }
    let move = 0;
    const i = now.indexOf(at);
    if (e.key === " ") {
      e.preventDefault();
      setViewing(i);
      return;
    } else if (e.key === "ArrowDown" || e.key === "j" || e.key === "ArrowRight") move = 1;
    else if (e.key === "ArrowUp" || e.key === "k" || e.key === "ArrowLeft") move = -1;
    else return;
    e.preventDefault();
    const to = Math.max(0, Math.min(now.length - 1, i + move));
    const next = now[to] as InboxEntry | undefined;
    if (next && next !== at) {
      setCurrentNow(next.id);
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
      <DropZone disabled={uploading !== null} onFiles={(fs) => { void upload(fs); }}>
        {uploading ? `Uploading ${progressText(uploading)}…` : saving ? "Saving…" : "Drop unlabelled images here (or click) to sort them with the model's suggestions"}
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
            // A labelled or dismissed image leaves the list, so the viewer moves on to the next one;
            // a key pressed before that renders acts on the next one too.
            const item = listNow()[k] as InboxEntry | undefined;
            return item ? keyAction(e.key, item).handled : /^([1-9]|Enter|Delete|Backspace)$/.test(e.key);
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
            <button disabled={sure.length === 0}
              onClick={() => { label(Object.fromEntries(sure.map((i) => [i.id, suggested(i) ?? ""]))); }}>
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
              <button className="small" disabled={!selection.some(suggested)}
                onClick={() => { label(Object.fromEntries(selection.filter(suggested).map((i) => [i.id, suggested(i) ?? ""]))); }}>
                Accept suggestions
              </button>
              <select value="" aria-label="Label selected as" onChange={(e) => {
                if (e.target.value) label(Object.fromEntries(selection.map((i) => [i.id, e.target.value])));
              }}>
                <option value="">Label as…</option>
                {task.classes.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
              <button className="small danger" onClick={() => { dismiss(selection.map((i) => i.id)); }}>Dismiss</button>
              <button className="ghost small" onClick={() => { setPicked(new Set()); }}>Clear selection</button>
            </div>
          )}
          <div className={`inbox-grid ${reordering ? "stale" : ""}`}>
            {list.slice(0, shown).map((i, n) => {
              const p = i.prediction;
              const s = suggested(i);
              return (
                <article key={i.id} className={`inbox-card ${cur?.id === i.id ? "current" : ""} ${picked.has(i.id) ? "picked" : ""}`}
                  tabIndex={cur?.id === i.id ? 0 : -1} ref={(el) => { if (el) cards.current.set(i.id, el); else cards.current.delete(i.id); }}
                  onClick={() => { setCurrentNow(i.id); }} onFocus={() => { if (currentNow.current !== i.id) setCurrentNow(i.id); }}>
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
                      {s && <button className="small" onClick={(e) => { e.stopPropagation(); label({ [i.id]: s }); }}>✓ {s}</button>}
                      <ClassSelect className="small" classes={classNames} numbered placeholder="Label as…" aria-label="Label as"
                        onPick={(c) => { label({ [i.id]: c }); }} />
                      <button className="ghost small danger" aria-label="Dismiss" title="Dismiss (Delete)"
                        onClick={(e) => { e.stopPropagation(); dismiss([i.id]); }}>×</button>
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
