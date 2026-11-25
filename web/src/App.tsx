import { useCallback, useEffect, useRef, useState } from "react";
import { api, AUTH_EVENT, type ModelInfo, type TaskDetail, type TaskSummary } from "./api";
import Evaluate, { type EvalRun } from "./Evaluate";
import Dashboard from "./Dashboard";
import Examples from "./Examples";
import Inbox from "./Inbox";
import { usePredictResults } from "./hooks";
import Predict from "./Predict";
import Settings from "./Settings";
import Shortcuts from "./Shortcuts";
import { ErrorBox } from "./ui";

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
  // Prediction and evaluation results live here so they survive switching tabs; both reset per task.
  const predictState = usePredictResults(route.task);
  const [evalRun, setEvalRun] = useState<EvalRun | null>(null);

  const run = useCallback(async <T,>(p: Promise<T>): Promise<T | undefined> => {
    const from = currentTask.current;
    try {
      return await p;
    } catch (e) {
      // An error from a task the user has since left would only confuse the one now shown.
      if (currentTask.current === from) setError(e instanceof Error ? e.message : String(e));
      return undefined;
    }
  }, []);

  const refreshTasks = useCallback(
    () => run(api.tasks()).then((t) => { if (t) setTasks(t); }),
    [run],
  );

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
      if (e.target instanceof Element && e.target.closest("input, textarea, select, [contenteditable]")) return;
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
    setEvalRun((s) => (s?.task === route.task ? s : null)); // kept when this is the task's new name
    setError(null);
    setTask(null);
    setLoadFailed(false);
    if (!route.task) return;
    let current = true; // false once the user opens another task: a late response must not show
    api.task(route.task).then(
      (t) => { if (current) setTask(t); },
      (e: unknown) => {
        if (!current) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoadFailed(true);
      },
    );
    return () => { current = false; };
  }, [route.task]);

  /** Show a task returned by a mutation, unless it is for another task or older than what is shown. */
  const updateTask = (t: TaskDetail) => {
    if (t.name === currentTask.current) {
      setTask((cur) => (cur && cur.name === t.name && cur.revision > t.revision ? cur : t));
    }
    void refreshTasks();
  };

  /** Undo or redo the task's last change, and say what it was in a toast that offers the reverse. */
  const step = async (which: "undo" | "redo") => {
    if (!task || !(which === "undo" ? task.can_undo : task.can_redo)) return;
    const what = which === "undo" ? task.undo_label : task.redo_label;
    const t = await run(which === "undo" ? api.undo(task.name) : api.redo(task.name));
    if (!t) return;
    updateTask(t);
    setToast({
      text: `${which === "undo" ? "Undid" : "Redid"} ${what || "the last change"}`,
      then: which === "undo" ? (t.can_redo ? "redo" : null) : (t.can_undo ? "undo" : null),
      key: Date.now(),
    });
  };
  const undo = () => step("undo");
  useEffect(() => { // the toast goes after a few seconds
    if (!toast) return;
    const timer = setTimeout(() => { setToast(null); }, 6000);
    return () => { clearTimeout(timer); };
  }, [toast]);
  // Ctrl/Cmd+Z undoes the task's last change and Ctrl/Cmd+Shift+Z (or Ctrl+Y) redoes it, unless
  // typing in a field (where they undo the typing).
  const stepRef = useRef(step);
  stepRef.current = step;
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (!(e.ctrlKey || e.metaKey) || e.altKey || (key !== "z" && key !== "y")) return;
      if (e.target instanceof Element && e.target.closest("input, textarea, select, [contenteditable]")) return;
      e.preventDefault();
      void stepRef.current(key === "y" || e.shiftKey ? "redo" : "undo");
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
              {route.tab === "evaluate" && <Evaluate task={task} setTask={updateTask} run={run} state={evalRun} setState={setEvalRun} />}
              {route.tab === "settings" && (
                <Settings task={task} models={models} setTask={updateTask} run={run} onDeleted={() => { void refreshTasks(); go(null); }}
                  onDuplicated={(t) => { void refreshTasks(); go(t.name); }}
                  onRenamed={(t) => {
                    // Results move with the task to its new name.
                    predictState.rename(t.name);
                    setEvalRun((s) => (s?.task === task.name ? { ...s, task: t.name } : s));
                    void refreshTasks();
                    go(t.name, "settings");
                  }} />
              )}
            </div>
          </>
        )}
      </main>
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
        void run(api.createTask(name.trim(), spec || undefined, "prototype")).then((t) => {
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
