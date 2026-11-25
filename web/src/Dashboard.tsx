import type { TaskSummary } from "./api";
import { pct } from "./format";

/** Every task at a glance: size, inbox, last evaluation (and whether the task changed since), warnings. */
export default function Dashboard({ tasks }: { tasks: TaskSummary[] }) {
  if (!tasks.length) return null;
  return (
    <section className="dashboard">
      <h1>Tasks</h1>
      <div className="task-cards">
        {tasks.map((t) => {
          const ev = t.last_evaluation;
          const stale = ev !== null && ev.revision !== t.revision;
          return (
            <a key={t.name} className="task-card" href={`#/${encodeURIComponent(t.name)}/examples`}>
              <div className="row"><strong>{t.name}</strong><span className="spacer" />
                {ev && ev.accuracy !== null && (
                  <span className={stale ? "muted" : ""} title={`${ev.method ?? ""} on ${ev.at.slice(0, 16).replace("T", " ")}${stale ? "; the task has changed since" : ""}`}>
                    {pct(ev.accuracy)}{stale ? "*" : ""}
                  </span>
                )}
              </div>
              <div className="muted">
                {t.n_classes} classes · {t.n_examples} examples{t.n_inbox ? ` · ${String(t.n_inbox)} in the inbox` : ""}
              </div>
              <div className="muted small-text">{t.classifier} · {t.embedder}</div>
              {!ev && t.n_examples > 1 && <div className="muted small-text">Not evaluated yet</div>}
              {t.warnings.map((w) => <div key={w} className="warn">⚠ {w}</div>)}
            </a>
          );
        })}
      </div>
    </section>
  );
}
