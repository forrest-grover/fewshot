import type { CompareReport, EvalReport, Job } from "./api";

/** A task's last evaluation, and the background job computing the next one (polled by the App
 * until it ends, whichever task or tab is open). */
export interface EvalRun {
  report: EvalReport | null; rev: number | null; // the report shown, and the task revision it is of
  job: Job<EvalReport> | null; // a run in progress
}
export const NO_EVAL: EvalRun = { report: null, rev: null, job: null };

/** A task's last model comparison, and the job computing the next one. */
export interface CompareRun { report: CompareReport | null; job: Job<CompareReport> | null }

/** ``s`` after its job reported ``j``. Only a finished run replaces the report (and its
 * revision, which the report gives: the task may have changed while it ran); a cancelled or
 * failed one leaves the old report. A report from a job no longer tracked is ignored. */
export function evalAfter(s: EvalRun | null, j: Job<EvalReport>): EvalRun | null {
  if (!s || s.job?.id !== j.id) return s;
  if (j.status === "running") return { ...s, job: j };
  return j.status === "done" ? { ...s, job: null, report: j.result, rev: j.result?.revision ?? null } : { ...s, job: null };
}

/** Like ``evalAfter``, for a comparison. */
export function compareAfter(s: CompareRun | null, j: Job<CompareReport>): CompareRun | null {
  if (!s || s.job?.id !== j.id) return s;
  if (j.status === "running") return { ...s, job: j };
  return j.status === "done" ? { ...s, job: null, report: j.result } : { ...s, job: null };
}

/** ``s`` once its job ``id`` can no longer be polled (e.g. the server restarted). */
export function jobLost<R extends { job: { id: string } | null }>(s: R | null, id: string): R | null {
  return s && s.job?.id === id ? { ...s, job: null } : s;
}
