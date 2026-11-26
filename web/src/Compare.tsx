import { useState, type Dispatch, type SetStateAction } from "react";
import { api, type CompareReport, type Job, type ModelInfo, type TaskDetail } from "./api";
import { pct } from "./format";
import { usePollJob } from "./hooks";

/** The last model comparison of a task, kept by the App so it survives switching tabs. */
export interface CompareRun { task: string; report: CompareReport | null; job: Job<CompareReport> | null }

interface Props {
  task: TaskDetail; models: ModelInfo | null; setTask: (t: TaskDetail) => void;
  run: <T>(p: Promise<T>) => Promise<T | undefined>;
  state: CompareRun | null; setState: Dispatch<SetStateAction<CompareRun | null>>;
}

/** Cross-validate the task with other embedders and switch to the most accurate. */
export default function Compare({ task, models, setTask, run, state, setState }: Props) {
  const mine = state?.task === task.name ? state : null;
  const report = mine?.report ?? null;
  const job = mine?.job ?? null;
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const label = (spec: string) => models?.embedders.find((e) => e.spec === spec)?.label ?? spec;
  const others = (models?.embedders ?? []).filter((e) => e.spec !== task.embedder);

  const jobId = job?.id;
  usePollJob<CompareReport>(jobId, (j) => {
    setState((s) => {
      if (!s || s.job?.id !== jobId) return s;
      if (j.status === "running") return { ...s, job: j };
      return j.status === "done" ? { ...s, job: null, report: j.result } : { ...s, job: null };
    });
    if (j.status === "failed") void run(Promise.reject(new Error(`Comparison failed: ${j.error ?? "unknown error"}`)));
  }, (e) => {
    setState((s) => (s && s.job?.id === jobId ? { ...s, job: null } : s));
    void run(Promise.reject(e));
  });

  const go = async () => {
    const j = await run(api.startComparison(task.name, [task.embedder, ...others.map((e) => e.spec).filter((s) => picked.has(s))]));
    if (j) setState((s) => ({ task: task.name, report: s?.task === task.name ? s.report : null, job: j }));
  };
  const toggle = (spec: string) => {
    setPicked((p) => { const n = new Set(p); if (n.has(spec)) n.delete(spec); else n.add(spec); return n; });
  };
  const use = async (spec: string) => {
    const t = await run(api.updateTask(task.name, { embedder: spec }));
    if (t) setTask(t);
  };
  const stale = report && report.revision !== task.revision;
  const [n, stage] = job?.stage.split(" ") ?? ["", ""]; // "2/3 embedding"

  return (
    <section aria-labelledby="compare-h">
      <h3 id="compare-h">Compare models</h3>
      <p className="muted">
        Cross-validates this task's examples with other embedders (the current settings otherwise), to pick a model on your own
        images. A model not used before is downloaded first, which can take a few minutes.
      </p>
      {others.length > 0 && (
        <fieldset className="compare-pick" disabled={job !== null}>
          <legend className="muted">Compare {label(task.embedder)} (current) with:</legend>
          {others.map((e) => (
            <label key={e.spec} className="check" title={e.notes}>
              <input type="checkbox" checked={picked.has(e.spec)} onChange={() => { toggle(e.spec); }} />
              {e.label}
            </label>
          ))}
        </fieldset>
      )}
      <div className="row">
        <button className="secondary" disabled={job !== null || picked.size === 0 || task.examples.length < 2}
          onClick={() => { void go(); }}>
          {job ? "Comparing…" : "Compare"}
        </button>
      </div>
      {job && (
        <div className="row job">
          <span>
            {n ? `Model ${n}: ` : ""}{stage === "embedding" ? "embedding images" : stage === "evaluating" ? "cross-validating" : "starting"}
            {job.total ? ` ${String(job.done)} / ${String(job.total)}` : "…"}
          </span>
          <progress max={job.total || 1} value={job.total ? job.done : undefined} />
          <button className="ghost small" onClick={() => { void run(api.cancelJob(job.id)); }}>Cancel</button>
        </div>
      )}
      {stale && <p className="hint">The task changed since this comparison ran.</p>}
      {report && (
        <div className="scroll-x">
          <table className="table compare">
            <thead><tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th><th>Time</th><th /></tr></thead>
            <tbody>
              {report.results.map((r) => (
                <tr key={r.embedder} className={r.embedder === report.best ? "best" : undefined}>
                  <td>
                    <span title={r.embedder}>{label(r.embedder)}</span>
                    {r.embedder === task.embedder && <span className="badge">current</span>}
                    {r.embedder === report.best && <span className="badge ok">most accurate</span>}
                  </td>
                  {r.error
                    ? <td colSpan={3} className="danger-text">{r.error}</td>
                    : <><td>{pct(r.accuracy)}</td><td>{pct(r.macro_f1)}</td><td>{r.seconds == null ? "–" : `${r.seconds.toFixed(1)} s`}</td></>}
                  <td>
                    {!r.error && r.accuracy !== null && r.embedder !== task.embedder && (
                      <button className="small" onClick={() => { void use(r.embedder); }}>Use this model</button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
