import { useCallback, useEffect, useRef, useState } from "react";
import { api, AUTH_EVENT, HttpError, type CompareReport, type EvalReport, type ModelInfo, type TaskDetail, type TaskSummary } from "./api";
import Evaluate from "./Evaluate";
import Dashboard from "./Dashboard";
import Examples from "./Examples";
import Inbox from "./Inbox";
import NewTask from "./NewTask";
import { taskUndoKey, topModal, typingAt } from "./keys";
import { usePredictResults } from "./predictResults";
import { Gate, Latest, stepToast, useByTask } from "./state";
import { useUploads, type Stopped, type UploadStop } from "./useUploads";
import Predict from "./Predict";
import Settings from "./Settings";
import Shortcuts from "./Shortcuts";
import SignIn from "./SignIn";
import { compareAfter, evalAfter, jobLost, type CompareRun, type EvalRun } from "./runs";
import { ErrorBox, JobPoller } from "./ui";
import { BATCH, batchLimits, isAbort, uploadSteps } from "./upload";

const TABS = ["examples", "inbox", "predict", "evaluate", "settings"] as const;
type Tab = (typeof TABS)[number];

/** decodeURIComponent, but malformed input (e.g. a stray "%") is kept as typed instead of throwing. */
function safeDecode(s: string): string {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}

function parseHash(): { task: string | null; tab: Tab } {
  // Split before decoding, so an encoded "/" stays inside its segment.
  const [, name = "", tab = ""] = location.hash.slice(1).split("/").map(safeDecode);
  return { task: name || null, tab: (TABS as readonly string[]).includes(tab) ? (tab as Tab) : "examples" };
}
const go = (task: string | null, tab: Tab = "examples") => {
  location.hash = task ? `/${encodeURIComponent(task)}/${tab}` : "";
};

export default function App() {
  const [route, setRoute] = useState(parseHash);
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [task, setTask] = useState<TaskDetail | null>(null);
  const [models, setModels] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [signIn, setSignIn] = useState(false); // the server wants its token: the sign-in dialog is open
  const [authRequired, setAuthRequired] = useState(false); // the server has a token (so one can log out)
  const signedIn = useRef(false); // a request has worked: a sign-in after that is "again"
  // Uploads that stopped for want of a session, said once signed in again (an error said while
  // the sign-in dialog covers the page would go unseen).
  const [missed, setMissed] = useState<string[]>([]);
  const [urlFetch, setUrlFetch] = useState(false); // the server downloads image URLs
  const [limits, setLimits] = useState(BATCH); // upload batch sizes the server takes
  // A message for a few seconds. ``then`` offers the reverse of an undo or redo, while the task is
  // still ``task`` at revision ``rev`` (after another change it would undo something else).
  const [toast, setToast] = useState<{ text: string; then?: "undo" | "redo" | null; task?: string; rev?: number; key: number } | null>(null);
  const [menuOpen, setMenuOpen] = useState(false); // the task list, on narrow screens
  const [help, setHelp] = useState(false); // the keyboard shortcuts
  const currentTask = useRef(route.task); // read by responses that arrive after the user moved on
  // Prediction results live here so they survive switching tabs (they reset per task).
  const predictState = usePredictResults(route.task);
  // Each task's last evaluation and comparison, and their running jobs, which are polled here
  // until they end, whatever is open, so a run can still be followed or cancelled after leaving it.
  const evals = useByTask<EvalRun>();
  const compares = useByTask<CompareRun>();

  const taskLoads = useRef(new Latest()).current;
  /** Fetch the open task again, e.g. after it was changed elsewhere (the CLI, another tab); a
   * response older than the task shown is ignored. If it is gone, say so. */
  const reloadTask = useCallback(() => {
    const name = currentTask.current;
    if (!name) return;
    const latest = taskLoads.begin();
    api.task(name).then((t) => {
      if (!latest() || currentTask.current !== name) return;
      setTask((cur) => (cur && cur.name === t.name && cur.revision >= t.revision ? cur : t));
    }, (e: unknown) => {
      if (!latest() || currentTask.current !== name || !(e instanceof HttpError && e.status === 404)) return;
      setTask(null);
      setLoadFailed(true);
      setError(e.message);
    });
  }, [taskLoads]);

  const run = useCallback(async <T,>(p: Promise<T>): Promise<T | undefined> => {
    const from = currentTask.current;
    try {
      return await p;
    } catch (e) {
      // An error from a task the user has since left would only confuse the one now shown, a
      // cancelled request was meant to stop, and one refused for want of a session opens the
      // sign-in dialog instead.
      if (currentTask.current !== from || isAbort(e) || (e instanceof HttpError && e.status === 401)) return undefined;
      setError(e instanceof Error ? e.message : String(e));
      // Refused (say, an example or class that is gone): the task may have changed elsewhere.
      if (e instanceof HttpError && [400, 404, 409].includes(e.status)) {
        reloadTask();
        if (e.status === 404) {
          setError(`That is no longer there (${e.message}): it was probably changed in another tab or by the CLI. The task now shows as it is.`);
        }
      }
      return undefined;
    }
  }, [reloadTask]);

  /** What the server says about itself, and its tasks: at startup, and again after signing in. */
  const loadServer = useCallback(() => {
    void refreshTasks();
    void run(api.models()).then((m) => { if (m) setModels(m); });
    void api.health().then((h) => {
      setUrlFetch(h.url_fetch);
      setAuthRequired(h.auth_required);
      setLimits(batchLimits(h.max_request_bytes));
    }, () => { /* an old server: no URL fetching, default batches */ });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps -- refreshTasks and run are stable
  const taskLists = useRef(new Latest()).current;
  const refreshTasks = useCallback(async () => {
    const latest = taskLists.begin();
    const t = await run(api.tasks());
    if (t) signedIn.current = true;
    if (t && latest()) setTasks(t); // not an older list that arrived after a newer one
  }, [run, taskLists]);

  useEffect(() => {
    const auth = () => { setSignIn(true); };
    window.addEventListener(AUTH_EVENT, auth);
    return () => { window.removeEventListener(AUTH_EVENT, auth); };
  }, []);

  useEffect(() => {
    const on = () => { setRoute(parseHash()); };
    window.addEventListener("hashchange", on);
    loadServer();
    // Tasks made or changed elsewhere (the CLI, another tab) show up when the window comes back.
    const back = () => { if (document.visibilityState === "visible") void refreshTasks(); };
    window.addEventListener("focus", back);
    document.addEventListener("visibilitychange", back);
    return () => {
      window.removeEventListener("hashchange", on);
      window.removeEventListener("focus", back);
      document.removeEventListener("visibilitychange", back);
    };
  }, [refreshTasks, loadServer]);
  useEffect(() => { // "?" shows the keyboard shortcuts
    const on = (e: KeyboardEvent) => {
      if (e.key !== "?" || e.ctrlKey || e.metaKey || e.altKey) return;
      if (typingAt(e.target) || topModal()) return; // not over the image viewer or another dialog
      e.preventDefault();
      setHelp(true);
    };
    window.addEventListener("keydown", on);
    return () => { window.removeEventListener("keydown", on); };
  }, []);

  const [attempt, setAttempt] = useState(0); // loading the task again after signing in
  useEffect(() => {
    currentTask.current = route.task;
    setMenuOpen(false);
    setToast(null);
    setError(null);
    setTask(null);
    setLoadFailed(false);
    if (!route.task) return;
    const latest = taskLoads.begin(); // false once the user opens another task: a late response must not show
    api.task(route.task).then(
      (t) => { if (latest()) setTask(t); },
      (e: unknown) => {
        if (!latest()) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoadFailed(true);
      },
    );
  }, [route.task, taskLoads, attempt]);

  // The task list (fetched again whenever the window comes back, and after each change) shows
  // the open task changed elsewhere, or gone: fetch it again.
  const listed = tasks.find((s) => s.name === task?.name);
  const listedRevision = tasks.length && task ? listed?.revision ?? -1 : null; // -1: not listed
  useEffect(() => {
    if (task && listedRevision !== null && (listedRevision > task.revision || listedRevision === -1)) reloadTask();
  }, [listedRevision, task?.revision, reloadTask]); // eslint-disable-line react-hooks/exhaustive-deps -- only on these changes

  /** A background job of task ``name`` failed: say so if that task is open. */
  const jobFailed = (name: string, message: string) => { if (currentTask.current === name) setError(message); };

  /** Show a task returned by a mutation, unless it is for another task or older than what is shown. */
  const updateTask = (t: TaskDetail) => {
    if (t.name === currentTask.current) {
      setTask((cur) => (cur && cur.name === t.name && cur.revision > t.revision ? cur : t));
    }
    void refreshTasks();
  };

  /** An upload stopped part-way: say how much of it got in (of a task the user has left, too). */
  const uploadStopped = ({ task: name, added, total, error }: UploadStop) => {
    const text = `Upload${name === currentTask.current ? "" : ` to “${name}”`} stopped: ${String(added)} of ${String(total)} added`;
    if (error instanceof HttpError && error.status === 401) setMissed((m) => [...m, `${text}; add the other ${String(total - added)} again`]);
    else if (error) setError(`${error.message}. ${text}`);
    else setToast({ text, key: Date.now() });
  };
  // Uploads live here so they go on while tabs are switched; opening another task stops them.
  const exampleUploads = useUploads(route.task, { onTask: updateTask, onStop: uploadStopped, limits });
  const inboxUploads = useUploads(route.task, { onTask: updateTask, onStop: uploadStopped, limits });

  /** Undo or redo the task's last change, and say what it was in a toast that offers the reverse.
   * One step at a time: presses while one is in flight are ignored, so a double press or a held
   * key doesn't walk back several versions. Undo while images are being uploaded to the task
   * stops the uploads first (after their batches in flight, so none lands after the undo) and
   * then undoes all of them: every undo step on top that their batches made (uploads running at
   * once share an undo group, so these are usually one, but an inbox upload alongside example
   * uploads, or a rename meanwhile, makes more). A change made meanwhile (a relabel, say) is not
   * undone with them: undoing stops there and says how much of the upload is under it. If none
   * of the uploads had landed, nothing is undone. */
  const stepGate = useRef(new Gate()).current;
  const step = (which: "undo" | "redo") => stepGate.run(async () => {
    if (!task) return;
    let shown = task;
    if (which === "undo" && (exampleUploads.busy || inboxUploads.busy)) {
      const [a, b] = await Promise.all([exampleUploads.stop(), inboxUploads.stop()]);
      const batches = new Map([...a, ...b]);
      if (currentTask.current !== shown.name) return;
      if (!batches.size) { setToast({ text: "Stopped the upload; none of it had been added", key: Date.now() }); return; }
      const fresh = await run(api.task(shown.name)); // with every batch that landed
      if (!fresh) return;
      updateTask(fresh);
      shown = fresh;
      await undoUploads(shown, batches);
      return;
    }
    if (!(which === "undo" ? shown.can_undo : shown.can_redo)) return;
    const t = await run(which === "undo" ? api.undo(shown.name) : api.redo(shown.name));
    if (!t) return;
    updateTask(t);
    const said = stepToast(which, t, which === "undo" ? shown.undo_label : shown.redo_label);
    setToast({ ...said, task: t.name, rev: t.revision, key: Date.now() });
  });
  /** Undo the steps on top of ``shown`` that the stopped uploads' ``batches`` made (see ``step``). */
  const undoUploads = async (shown: TaskDetail, batches: Stopped) => {
    const steps = uploadSteps(batches);
    let t = shown, undone = 0;
    const firstLabel = shown.undo_label;
    while (t.can_undo && t.undo_group && batches.has(t.undo_group)) {
      const next = await run(api.undo(t.name));
      if (!next) break; // the error shows
      updateTask(next);
      t = next;
      undone++;
      if (currentTask.current !== t.name) return;
    }
    const left = steps - undone;
    const under = left > 0 ? ` ${left === steps ? "It is" : "The rest of it is"} under a change made meanwhile${t.undo_label ? ` (${t.undo_label})` : ""}: undo that to reach it.` : "";
    if (undone === 1) {
      const said = stepToast("undo", t, firstLabel);
      setToast({ ...said, text: `Stopped the upload. ${said.text}${under ? `.${under}` : ""}`, task: t.name, rev: t.revision, key: Date.now() });
    } else {
      // Redo would make only the last of several steps again: not offered.
      setToast({ text: `Stopped the upload.${undone ? ` Undid it (${String(undone)} steps).` : ""}${under}`, key: Date.now() });
    }
  };
  const undo = () => step("undo");
  useEffect(() => { // the toast goes after a few seconds
    if (!toast) return;
    const timer = setTimeout(() => { setToast(null); }, 6000);
    return () => { clearTimeout(timer); };
  }, [toast]);
  // Ctrl/Cmd+Z undoes the task's last change and Ctrl/Cmd+Shift+Z (or Ctrl+Y) redoes it, unless
  // typing in a field (where they undo the typing); see taskUndoKey.
  const stepRef = useRef(step);
  stepRef.current = step;
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const which = taskUndoKey(e);
      if (!which) return;
      e.preventDefault();
      if (!e.repeat) void stepRef.current(which); // a held key steps once
    };
    window.addEventListener("keydown", on);
    return () => { window.removeEventListener("keydown", on); };
  }, []);

  /** Signed in (again): close the dialog and fetch what failed meanwhile; everything shown stays. */
  const afterSignIn = () => {
    setSignIn(false);
    loadServer();
    if (task) reloadTask();
    else setAttempt((n) => n + 1);
    if (missed.length) { setError(`While you were signed out: ${missed.join(". ")}.`); setMissed([]); }
  };
  const logOut = () => {
    if ((exampleUploads.busy || inboxUploads.busy) && !confirm("An upload is still running. Log out and stop it?")) return;
    // A fresh page after it: nothing of the session is left on screen.
    const fresh = () => { location.hash = ""; location.reload(); };
    void api.logout().then(fresh, (e: unknown) => {
      // 401: the session had already ended, so there is nothing left to log out of.
      if (e instanceof HttpError && e.status === 401) fresh();
      else setError(e instanceof Error ? e.message : String(e));
    });
  };

  return (
    <div className="app">
      <aside className={`sidebar ${menuOpen ? "open" : ""}`}>
        <div className="row brand-row">
          <a className="brand" href="#">fewshot</a>
          <button className="ghost small menu-toggle" aria-expanded={menuOpen} aria-controls="task-list"
            onClick={() => { setMenuOpen((o) => !o); }}>
            {route.task ?? "Tasks"} ({tasks.length}) {menuOpen ? "▴" : "▾"}
          </button>
        </div>
        <nav id="task-list">
          {tasks.map((t) => (
            <a key={t.name} href={`#/${encodeURIComponent(t.name)}/examples`} className={route.task === t.name ? "active" : ""}>
              <span>{t.name}</span>
              <small>{t.n_classes} classes · {t.n_examples} images</small>
            </a>
          ))}
        </nav>
        <button className="secondary" onClick={() => { go(null); }}>+ New task</button>
        <button className="ghost small" onClick={() => { setHelp(true); }} title="Keyboard shortcuts (?)">Keyboard shortcuts <kbd>?</kbd></button>
        {authRequired && <button className="ghost small" onClick={logOut} title="End this session (in every tab)">Log out</button>}
      </aside>

      <main>
        <ErrorBox error={error} onClose={() => { setError(null); }} />
        {!route.task && <Dashboard tasks={tasks} />}
        {!route.task && <NewTask models={models} run={run} onCreated={(t) => { void refreshTasks(); go(t.name); }} />}
        {route.task && !task && (loadFailed
          ? <p className="muted">Task “{route.task}” could not be loaded. <a href="#">Back to tasks</a></p>
          : <p className="muted">Loading…</p>)}
        {task && (
          <>
            <header className="task-head">
              <h1>{task.name}</h1>
              <span className="muted">
                {models?.embedders.find((e) => e.spec === task.embedder)?.label ?? task.embedder} · {task.classifier}
              </span>
              <span className="spacer" />
              <button className="ghost small" disabled={!task.can_undo && !exampleUploads.busy && !inboxUploads.busy} onClick={() => { void undo(); }}
                title={exampleUploads.busy || inboxUploads.busy ? "Stop the upload and undo it (Ctrl+Z)" : `Undo ${task.undo_label || "the last change"} (Ctrl+Z)`}
                onMouseEnter={reloadTask} onFocus={reloadTask}>↶ Undo</button>
              <button className="ghost small" disabled={!task.can_redo} onClick={() => { void step("redo"); }}
                title={`Redo ${task.redo_label || "the last undone change"} (Ctrl+Shift+Z)`} onMouseEnter={reloadTask} onFocus={reloadTask}>↷ Redo</button>
            </header>
            <div className="tabs" role="tablist" aria-label="Task views" onKeyDown={(e) => {
              // Arrow keys, Home and End move between tabs (the WAI-ARIA tabs pattern).
              const i = TABS.indexOf(route.tab);
              const moves: Record<string, number> = { ArrowLeft: i - 1, ArrowRight: i + 1, Home: 0, End: TABS.length - 1 };
              if (!(e.key in moves)) return;
              const j = moves[e.key];
              e.preventDefault();
              const next = TABS[(j + TABS.length) % TABS.length];
              go(task.name, next);
              document.getElementById(`tab-${next}`)?.focus();
            }}>
              {TABS.map((t) => (
                <button
                  key={t}
                  id={`tab-${t}`}
                  role="tab"
                  aria-selected={route.tab === t}
                  aria-controls="tab-panel"
                  tabIndex={route.tab === t ? 0 : -1}
                  className={route.tab === t ? "tab active" : "tab"}
                  onClick={() => { go(task.name, t); }}
                >
                  {t[0].toUpperCase() + t.slice(1)}{t === "inbox" && task.inbox_count ? ` (${String(task.inbox_count)})` : ""}
                </button>
              ))}
            </div>
            <div id="tab-panel" role="tabpanel" aria-labelledby={`tab-${route.tab}`}>
              {route.tab === "examples" && <Examples task={task} setTask={updateTask} run={run} uploads={exampleUploads} />}
              {route.tab === "inbox" && <Inbox task={task} setTask={updateTask} run={run} uploads={inboxUploads} />}
              {route.tab === "predict" && <Predict task={task} setTask={updateTask} run={run} state={predictState} urlFetch={urlFetch} limits={limits} />}
              {route.tab === "evaluate" && <Evaluate key={task.name} task={task} setTask={updateTask} run={run}
                state={evals.map[task.name] ?? null} setState={evals.setFor(task.name)}
                models={models} compare={compares.map[task.name] ?? null} setCompare={compares.setFor(task.name)} />}
              {route.tab === "settings" && (
                <Settings task={task} models={models} setTask={updateTask} run={run} onDeleted={() => { evals.setFor(task.name)(null); compares.setFor(task.name)(null); void refreshTasks(); go(null); }}
                  onDuplicated={(t) => { void refreshTasks(); go(t.name); }}
                  onRenamed={(t) => {
                    // Results, and uploads in progress, move with the task to its new name.
                    predictState.rename(t.name);
                    exampleUploads.rename(t.name);
                    inboxUploads.rename(t.name);
                    evals.rename(task.name, t.name);
                    compares.rename(task.name, t.name);
                    void refreshTasks();
                    go(t.name, "settings");
                  }} />
              )}
            </div>
          </>
        )}
      </main>
      {Object.entries(evals.map).map(([name, r]) => r?.job && (
        <JobPoller<EvalReport> key={r.job.id} id={r.job.id}
          onJob={(j) => { evals.setFor(name)((s) => evalAfter(s, j)); if (j.status === "failed") jobFailed(name, `Evaluation failed: ${j.error ?? "unknown error"}`); }}
          onError={(e) => { const id = r.job?.id ?? ""; evals.setFor(name)((s) => jobLost(s, id)); jobFailed(name, e.message); }} />
      ))}
      {Object.entries(compares.map).map(([name, r]) => r?.job && (
        <JobPoller<CompareReport> key={r.job.id} id={r.job.id}
          onJob={(j) => { compares.setFor(name)((s) => compareAfter(s, j)); if (j.status === "failed") jobFailed(name, `Comparison failed: ${j.error ?? "unknown error"}`); }}
          onError={(e) => { const id = r.job?.id ?? ""; compares.setFor(name)((s) => jobLost(s, id)); jobFailed(name, e.message); }} />
      ))}
      {help && <Shortcuts onClose={() => { setHelp(false); }} />}
      {signIn && <SignIn again={signedIn.current} onSignedIn={afterSignIn} />}
      {toast && (
        <div className="toast" role="status" key={toast.key}>
          <span>{toast.text}</span>
          {toast.then && task && task.name === toast.task && task.revision === toast.rev && (
            <button className="small secondary" onClick={() => { const then = toast.then; setToast(null); if (then) void step(then); }}>
              {toast.then === "redo" ? "Redo" : "Undo"}
            </button>
          )}
          <button className="ghost small" aria-label="Dismiss" onClick={() => { setToast(null); }}>×</button>
        </div>
      )}
    </div>
  );
}
