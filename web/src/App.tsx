import { useCallback, useEffect, useState } from "react";
import { api, type ModelInfo, type TaskDetail, type TaskSummary } from "./api";
import Evaluate from "./Evaluate";
import Examples from "./Examples";
import Predict from "./Predict";
import Settings from "./Settings";
import { ErrorBox } from "./ui";

const TABS = ["examples", "predict", "evaluate", "settings"] as const;
type Tab = (typeof TABS)[number];

function parseHash(): { task: string | null; tab: Tab } {
  const [, name, tab] = decodeURIComponent(location.hash.slice(1)).split("/");
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

  const run = useCallback(async <T,>(p: Promise<T>): Promise<T | undefined> => {
    try {
      return await p;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return undefined;
    }
  }, []);

  const refreshTasks = useCallback(() => run(api.tasks()).then((t) => t && setTasks(t)), [run]);

  useEffect(() => {
    const on = () => setRoute(parseHash());
    window.addEventListener("hashchange", on);
    refreshTasks();
    run(api.models()).then((m) => m && setModels(m));
    return () => window.removeEventListener("hashchange", on);
  }, [refreshTasks, run]);

  useEffect(() => {
    setError(null);
    setTask(null);
    if (route.task) run(api.task(route.task)).then((t) => t && setTask(t));
  }, [route.task, run]);

  const updateTask = (t: TaskDetail) => {
    setTask(t);
    refreshTasks();
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
        <button className="secondary" onClick={() => go(null)}>+ New task</button>
      </aside>

      <main>
        <ErrorBox error={error} onClose={() => setError(null)} />
        {!route.task && <NewTask models={models} run={run} onCreated={(t) => { refreshTasks(); go(t.name); }} />}
        {route.task && !task && <p className="muted">Loading…</p>}
        {task && (
          <>
            <header className="task-head">
              <h1>{task.name}</h1>
              <span className="muted">
                {models?.embedders.find((e) => e.spec === task.embedder)?.label ?? task.embedder} · {task.classifier}
              </span>
            </header>
            <div className="tabs" role="tablist">
              {TABS.map((t) => (
                <button key={t} role="tab" aria-selected={route.tab === t} className={route.tab === t ? "tab active" : "tab"} onClick={() => go(task.name, t)}>
                  {t[0].toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
            {route.tab === "examples" && <Examples task={task} setTask={updateTask} run={run} />}
            {route.tab === "predict" && <Predict task={task} setTask={updateTask} run={run} />}
            {route.tab === "evaluate" && <Evaluate task={task} setTask={updateTask} run={run} />}
            {route.tab === "settings" && (
              <Settings task={task} models={models} setTask={updateTask} run={run} onDeleted={() => { refreshTasks(); go(null); }} />
            )}
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
  const spec = embedder || models?.embedders[0]?.spec || "";
  return (
    <form
      className="new-task"
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        const t = await run(api.createTask(name.trim(), spec, "prototype"));
        setBusy(false);
        if (t) onCreated(t);
      }}
    >
      <h1>New task</h1>
      <p className="muted">
        A task is a set of classes plus a few labeled images of each. New images are embedded with a pretrained vision model and
        assigned to the closest class. No training run is needed: add, remove, or relabel examples and predictions update immediately.
      </p>
      <label>
        <span>Name</span>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. weld-defects" pattern="[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}" required autoFocus />
      </label>
      <label>
        <span>Embedder</span>
        <select value={spec} onChange={(e) => setEmbedder(e.target.value)}>
          {models?.embedders.map((m) => <option key={m.spec} value={m.spec}>{m.label}</option>)}
        </select>
        <small className="muted">{models?.embedders.find((m) => m.spec === spec)?.notes}</small>
      </label>
      <button type="submit" disabled={busy || !name.trim()}>{busy ? "Creating… (loads the model on first use)" : "Create task"}</button>
    </form>
  );
}
