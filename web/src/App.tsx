import { useCallback, useEffect, useRef, useState } from "react";
import { api, type ModelInfo, type TaskDetail, type TaskSummary } from "./api";
import Evaluate, { type EvalRun } from "./Evaluate";
import Examples from "./Examples";
import { usePredictResults } from "./hooks";
import Predict from "./Predict";
import Settings from "./Settings";
import { ErrorBox } from "./ui";

const TABS = ["examples", "predict", "evaluate", "settings"] as const;
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
  const currentTask = useRef(route.task); // read by responses that arrive after the user moved on
  // Prediction and evaluation results live here so they survive switching tabs; both reset per task.
  const predictState = usePredictResults(route.task);
  const [evalRun, setEvalRun] = useState<EvalRun | null>(null);

  const run = useCallback(async <T,>(p: Promise<T>): Promise<T | undefined> => {
    try {
      return await p;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return undefined;
    }
  }, []);

  const refreshTasks = useCallback(
    () => run(api.tasks()).then((t) => { if (t) setTasks(t); }),
    [run],
  );

  useEffect(() => {
    const on = () => { setRoute(parseHash()); };
    window.addEventListener("hashchange", on);
    void refreshTasks();
    void run(api.models()).then((m) => { if (m) setModels(m); });
    return () => { window.removeEventListener("hashchange", on); };
  }, [refreshTasks, run]);

  useEffect(() => {
    currentTask.current = route.task;
    setEvalRun(null);
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

  return (
    <div className="app">
      <aside className="sidebar">
        <a className="brand" href="#">fewshot</a>
        <nav>
          {tasks.map((t) => (
            <a key={t.name} href={`#/${encodeURIComponent(t.name)}/examples`} className={route.task === t.name ? "active" : ""}>
              <span>{t.name}</span>
              <small>{t.n_classes} classes · {t.n_examples} images</small>
            </a>
          ))}
        </nav>
        <button className="secondary" onClick={() => { go(null); }}>+ New task</button>
      </aside>

      <main>
        <ErrorBox error={error} onClose={() => { setError(null); }} />
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
                  {t[0].toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
            <div id="tab-panel" role="tabpanel" aria-labelledby={`tab-${route.tab}`}>
              {route.tab === "examples" && <Examples task={task} setTask={updateTask} run={run} />}
              {route.tab === "predict" && <Predict task={task} setTask={updateTask} run={run} state={predictState} />}
              {route.tab === "evaluate" && <Evaluate task={task} setTask={updateTask} run={run} state={evalRun} setState={setEvalRun} />}
              {route.tab === "settings" && (
                <Settings task={task} models={models} setTask={updateTask} run={run} onDeleted={() => { void refreshTasks(); go(null); }} />
              )}
            </div>
          </>
        )}
      </main>
    </div>
  );
}

function NewTask({ models, run, onCreated }: {
  models: ModelInfo | null; run: <T>(p: Promise<T>) => Promise<T | undefined>; onCreated: (t: TaskDetail) => void;
}) {
  const [name, setName] = useState("");
  const [embedder, setEmbedder] = useState("");
  const [busy, setBusy] = useState(false);
  const spec = embedder || (models?.embedders[0]?.spec ?? "");
  return (
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
  );
}
