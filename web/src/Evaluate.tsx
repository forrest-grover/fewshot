import { useEffect, type Dispatch, type SetStateAction } from "react";
import { api, type EvalReport, type Job, type TaskDetail } from "./api";
import { pct } from "./format";

/** The last evaluation of a task, kept by the App so it survives switching tabs. While it runs,
 * ``job`` is the server's background job, polled for progress. */
export interface EvalRun { task: string; report: EvalReport | null; rev: number | null; job: Job<EvalReport> | null }

interface Props {
  task: TaskDetail; setTask: (t: TaskDetail) => void; run: <T>(p: Promise<T>) => Promise<T | undefined>;
  state: EvalRun | null; setState: Dispatch<SetStateAction<EvalRun | null>>;
}

const POLL_MS = 500;
const STAGES: Record<string, string> = { starting: "Starting", embedding: "Embedding images", evaluating: "Cross-validating" };

export default function Evaluate({ task, setTask, run, state, setState }: Props) {
  const mine = state?.task === task.name ? state : null; // never show another task's report
  const report = mine?.report ?? null;
  const job = mine?.job ?? null;
  const busy = job !== null;
  const rev = mine?.rev ?? null;
  const update = (p: Partial<EvalRun>) => {
    // A run that finishes after the user opened another task is dropped.
    setState((s) => (s && s.task !== task.name ? s : { task: task.name, report: null, rev: null, job: null, ...s, ...p }));
  };

  const go = async () => {
    const j = await run(api.startEvaluation(task.name));
    if (j) update({ job: j, rev: task.revision });
  };
  // Poll the running job until it ends. Leaving the tab stops polling; coming back resumes it.
  const jobId = job?.id;
  useEffect(() => {
    if (!jobId) return;
    let live = true;
    const timer = setInterval(() => {
      void api.job<EvalReport>(jobId).then((j) => {
        if (!live) return;
        setState((s) => {
          if (!s || s.job?.id !== jobId) return s;
          if (j.status === "running") return { ...s, job: j };
          return { ...s, job: null, report: j.status === "done" ? j.result : s.report };
        });
        if (j.status === "failed") void run(Promise.reject(new Error(`Evaluation failed: ${j.error ?? "unknown error"}`)));
      }, (e: unknown) => {
        if (!live) return;
        setState((s) => (s && s.job?.id === jobId ? { ...s, job: null } : s)); // e.g. the server restarted
        void run(Promise.reject(e instanceof Error ? e : new Error(String(e))));
      });
    }, POLL_MS);
    return () => { live = false; clearInterval(timer); };
  }, [jobId, run, setState]);

  const fix = async (p: Promise<TaskDetail>) => { const t = await run(p); if (t) setTask(t); };
  const stale = report && rev !== task.revision;
  const maxCell = report ? Math.max(1, ...report.confusion.flat()) : 1;

  return (
    <div>
      <div className="row">
        <button onClick={() => { void go(); }} disabled={busy || task.examples.length < 2}>{busy ? "Evaluating…" : report ? "Re-run evaluation" : "Run evaluation"}</button>
        <span className="muted">
          Cross-validates the current settings on this task's examples: each image is predicted by a classifier fitted without it.
        </span>
      </div>
      {job && (
        <div className="row job">
          <span>{STAGES[job.stage] ?? job.stage}{job.total ? ` ${String(job.done)} / ${String(job.total)}` : "…"}</span>
          <progress max={job.total || 1} value={job.total ? job.done : undefined} />
          <button className="ghost small" onClick={() => { void run(api.cancelJob(job.id)); }}>Cancel</button>
        </div>
      )}
      {stale && <p className="hint">The task changed since this evaluation ran. Re-run it to see the effect.</p>}
      {report && (
        <>
          <div className="stats">
            <div className="stat"><div className="stat-value">{pct(report.accuracy)}</div><div className="muted">accuracy</div></div>
            <div className="stat"><div className="stat-value">{report.n_scored}</div><div className="muted">examples scored</div></div>
            <div className="stat"><div className="stat-value">{report.mistakes.length}</div><div className="muted">mistakes</div></div>
            <div className="stat"><div className="stat-value small">{report.method}</div><div className="muted">method</div></div>
          </div>

          <h3>Per class</h3>
          <table className="table">
            <thead><tr><th>Class</th><th>Examples</th><th>Recall</th><th>Precision</th></tr></thead>
            <tbody>
              {Object.entries(report.per_class).map(([c, m]) => (
                <tr key={c}><td>{c}</td><td>{m.support}</td><td>{pct(m.recall)}</td><td>{pct(m.precision)}</td></tr>
              ))}
            </tbody>
          </table>

          <h3>Confusion matrix <span className="muted">(rows: true class, columns: predicted)</span></h3>
          <div className="scroll-x">
            <table className="confusion">
              <thead><tr><th />{report.classes.map((c) => <th key={c} title={c}><span>{c}</span></th>)}</tr></thead>
              <tbody>
                {report.confusion.map((row, i) => (
                  <tr key={i}>
                    <th title={report.classes[i]}>{report.classes[i]}</th>
                    {row.map((v, j) => (
                      <td key={j} className={i === j ? "diag" : v ? "off" : ""} style={{ ["--a" as string]: v / maxCell }}>{v || ""}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {report.mistakes.length > 0 && (
            <>
              <h3>Mistakes <span className="muted">(most confident first; these are often mislabeled examples)</span></h3>
              <div className="thumbs">
                {report.mistakes.map((m) => {
                  const exists = task.examples.find((e) => e.id === m.id);
                  return (
                    <figure key={m.id} className="thumb wide">
                      {/* 400 px: 2x the 200 px tile */}
                      <img src={api.imageUrl(task.name, m.id, 400)} alt="" loading="lazy" />
                      <figcaption className="col">
                        <span><b>{m.true}</b> → {m.pred} ({pct(m.confidence)})</span>
                        {exists && exists.label === m.true ? (
                          <span className="row">
                            <button className="small secondary" onClick={() => { void fix(api.relabel(task.name, m.id, m.pred)); }}>Relabel as {m.pred}</button>
                            <button className="small ghost danger" onClick={() => { void fix(api.deleteExample(task.name, m.id)); }}>Delete</button>
                          </span>
                        ) : <span className="ok">{exists ? `Now ${exists.label}` : "Deleted"}</span>}
                      </figcaption>
                    </figure>
                  );
                })}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
