import { useCallback, useEffect, useRef, useState } from "react";
import { api, AUTH_EVENT, HttpError, type CompareReport, type EvalReport, type ModelInfo, type TaskDetail, type TaskSummary } from "./api";
import Evaluate from "./Evaluate";
import Dashboard from "./Dashboard";
import Examples from "./Examples";
import Inbox from "./Inbox";
import { Gate, Latest, stepToast, taskUndoKey, typingAt, useByTask, usePredictResults } from "./hooks";
import Predict from "./Predict";
import Settings from "./Settings";
import Shortcuts from "./Shortcuts";
import { compareAfter, evalAfter, jobLost, type CompareRun, type EvalRun } from "./runs";
import { ErrorBox, JobPoller } from "./ui";
import { isAbort } from "./upload";

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
  const [signIn, setSignIn] = useState(false); // the server wants its token
  const [urlFetch, setUrlFetch] = useState(false); // the server downloads image URLs
  const [toast, setToast] = useState<{ text: string; then: "undo" | "redo" | null; key: number } | null>(null);
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
      // An error from a task the user has since left would only confuse the one now shown, and
      // a cancelled request was meant to stop.
      if (currentTask.current !== from || isAbort(e)) return undefined;
      setError(e instanceof Error ? e.message : String(e));
      // Refused (say, an example or class that is gone): the task may have changed elsewhere.
      if (e instanceof HttpError && [400, 404, 409].includes(e.status)) reloadTask();
      return undefined;
    }
  }, [reloadTask]);

  const taskLists = useRef(new Latest()).current;
  const refreshTasks = useCallback(async () => {
    const latest = taskLists.begin();
    const t = await run(api.tasks());
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
    void refreshTasks();
    void run(api.models()).then((m) => { if (m) setModels(m); });
    void api.health().then((h) => { setUrlFetch(h.url_fetch); }, () => { /* an old server: no URL fetching */ });
    // Tasks made or changed elsewhere (the CLI, another tab) show up when the window comes back.
    const back = () => { if (document.visibilityState === "visible") void refreshTasks(); };
    window.addEventListener("focus", back);
    document.addEventListener("visibilitychange", back);
    return () => {
      window.removeEventListener("hashchange", on);
      window.removeEventListener("focus", back);
      document.removeEventListener("visibilitychange", back);
    };
  }, [refreshTasks, run]);
  useEffect(() => { // "?" shows the keyboard shortcuts
    const on = (e: KeyboardEvent) => {
      if (e.key !== "?" || e.ctrlKey || e.metaKey || e.altKey) return;
      if (typingAt(e.target)) return;
      e.preventDefault();
      setHelp(true);
    };
    window.addEventListener("keydown", on);
    return () => { window.removeEventListener("keydown", on); };
  }, []);

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
  }, [route.task, taskLoads]);

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

  /** Undo or redo the task's last change, and say what it was in a toast that offers the reverse.
   * One step at a time: presses while one is in flight are ignored, so a double press or a held
   * key doesn't walk back several versions. */
  const stepGate = useRef(new Gate()).current;
  const step = (which: "undo" | "redo") => stepGate.run(async () => {
    if (!task || !(which === "undo" ? task.can_undo : task.can_redo)) return;
    const t = await run(which === "undo" ? api.undo(task.name) : api.redo(task.name));
    if (!t) return;
    updateTask(t);
    setToast({ ...stepToast(which, t, which === "undo" ? task.undo_label : task.redo_label), key: Date.now() });
  });
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

  if (signIn) return <SignIn />;

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
              <button className="ghost small" disabled={!task.can_undo} onClick={() => { void undo(); }}
                title={`Undo ${task.undo_label || "the last change"} (Ctrl+Z)`}>↶ Undo</button>
              <button className="ghost small" disabled={!task.can_redo} onClick={() => { void step("redo"); }}
                title={`Redo ${task.redo_label || "the last undone change"} (Ctrl+Shift+Z)`}>↷ Redo</button>
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
              {route.tab === "examples" && <Examples task={task} setTask={updateTask} run={run} />}
              {route.tab === "inbox" && <Inbox task={task} setTask={updateTask} run={run} />}
              {route.tab === "predict" && <Predict task={task} setTask={updateTask} run={run} state={predictState} urlFetch={urlFetch} />}
              {route.tab === "evaluate" && <Evaluate key={task.name} task={task} setTask={updateTask} run={run}
                state={evals.map[task.name] ?? null} setState={evals.setFor(task.name)}
                models={models} compare={compares.map[task.name] ?? null} setCompare={compares.setFor(task.name)} />}
              {route.tab === "settings" && (
                <Settings task={task} models={models} setTask={updateTask} run={run} onDeleted={() => { evals.setFor(task.name)(null); compares.setFor(task.name)(null); void refreshTasks(); go(null); }}
                  onDuplicated={(t) => { void refreshTasks(); go(t.name); }}
                  onRenamed={(t) => {
                    // Results move with the task to its new name.
                    predictState.rename(t.name);
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
      {toast && (
        <div className="toast" role="status" key={toast.key}>
          <span>{toast.text}</span>
          {toast.then && (
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

/** Asks for the server's token (set with `fewshot serve --token`); the server keeps it in a cookie. */
function SignIn() {
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <main className="sign-in">
      <form
        className="new-task"
        onSubmit={(e) => {
          e.preventDefault();
          setBusy(true);
          api.login(token).then(
            () => { location.reload(); },
            (err: unknown) => { setBusy(false); setError(err instanceof Error ? err.message : String(err)); },
          );
        }}
      >
        <h1>Sign in</h1>
        <p className="muted">This fewshot server needs its access token (the one it was started with, <code>fewshot serve --token</code>).</p>
        <ErrorBox error={error} onClose={() => { setError(null); }} />
        <label>
          <span>Token</span>
          <input type="password" autoComplete="current-password" value={token} onChange={(e) => { setToken(e.target.value); }} required autoFocus />
        </label>
        <button type="submit" disabled={busy || !token}>{busy ? "Signing in…" : "Sign in"}</button>
      </form>
    </main>
  );
}

function NewTask({ models, run, onCreated }: {
  models: ModelInfo | null; run: <T>(p: Promise<T>) => Promise<T | undefined>; onCreated: (t: TaskDetail) => void;
}) {
  const [name, setName] = useState("");
  const [embedder, setEmbedder] = useState("");
  const [busy, setBusy] = useState(false);
  const spec = embedder || (models?.embedders[0]?.spec ?? "");
  const importZip = useRef<HTMLInputElement>(null);
  return (
    <>
    <form
      className="new-task"
      onSubmit={(e) => {
        e.preventDefault();
        setBusy(true);
        void run(api.createTask(name.trim(), spec || undefined, "gda")).then((t) => {
          setBusy(false);
          if (t) onCreated(t);
        });
      }}
    >
      <h1>New task</h1>
      <p className="muted">
        A task is a set of classes plus a few labeled images of each. New images are embedded with a pretrained vision model and
        assigned to the closest class. No training run is needed: add, remove, or relabel examples and predictions update immediately.
      </p>
      <label>
        <span>Name</span>
        <input value={name} onChange={(e) => { setName(e.target.value); }} placeholder="e.g. weld-defects" pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}" required autoFocus />
      </label>
      <label>
        <span>Embedder</span>
        <select value={spec} onChange={(e) => { setEmbedder(e.target.value); }}>
          {models?.embedders.map((m) => <option key={m.spec} value={m.spec}>{m.label}</option>)}
        </select>
        <small className="muted">{models?.embedders.find((m) => m.spec === spec)?.notes}</small>
      </label>
      <button type="submit" disabled={busy || !name.trim()}>{busy ? "Creating… (loads the model on first use)" : "Create task"}</button>
    </form>
    <div className="new-task import">
      <h3>Or import a task</h3>
      <p className="muted">From a zip made with “Export” on a task's Settings tab, or <code>fewshot export</code>.</p>
      <input ref={importZip} type="file" accept=".zip,application/zip" hidden onChange={(e) => {
        const f = e.target.files?.[0];
        e.target.value = "";
        if (!f) return;
        setBusy(true);
        void run(api.importTask(f, name.trim() || undefined)).then((t) => {
          setBusy(false);
          if (t) onCreated(t);
        });
      }} />
      <button className="secondary" disabled={busy} onClick={() => { importZip.current?.click(); }}>
        Import zip{name.trim() ? ` as “${name.trim()}”` : ""}…
      </button>
    </div>
    </>
  );
}
