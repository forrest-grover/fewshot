import { useState } from "react";
import { api, type EvalReport, type TaskDetail } from "./api";
import { pct } from "./format";

interface Props { task: TaskDetail; setTask: (t: TaskDetail) => void; run: <T>(p: Promise<T>) => Promise<T | undefined> }

export default function Evaluate({ task, setTask, run }: Props) {
  const [report, setReport] = useState<EvalReport | null>(null);
  const [rev, setRev] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);

  const go = async () => {
    setBusy(true);
    const r = await run(api.evaluate(task.name));
    setBusy(false);
    if (r) { setReport(r); setRev(task.revision); }
  };
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
