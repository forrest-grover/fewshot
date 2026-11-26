import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { EvalReport, Job } from "./api";
import { useByTask } from "./hooks";
import { evalAfter, jobLost, NO_EVAL, type EvalRun } from "./runs";

const job = (over: Partial<Job<EvalReport>> = {}): Job<EvalReport> => ({
  id: "j1", task: "pets", kind: "evaluate", status: "running", stage: "embedding", done: 1, total: 4, result: null, error: null, ...over,
});
const report = { revision: 7 } as EvalReport;

describe("evalAfter", () => {
  const running: EvalRun = { ...NO_EVAL, job: job() };

  it("follows progress and takes the finished report and its revision", () => {
    expect(evalAfter(running, job({ done: 3 }))?.job?.done).toBe(3);
    expect(evalAfter(running, job({ status: "done", result: report }))).toEqual({ report, rev: 7, job: null });
  });

  it("keeps the old report when a run is cancelled or fails", () => {
    const old: EvalRun = { report, rev: 7, job: job() };
    expect(evalAfter(old, job({ status: "cancelled" }))).toEqual({ report, rev: 7, job: null });
  });

  it("ignores a job it no longer tracks", () => {
    expect(evalAfter(running, job({ id: "other", status: "done", result: report }))).toBe(running);
    expect(evalAfter(null, job())).toBeNull();
    expect(jobLost(running, "other")).toBe(running);
    expect(jobLost(running, "j1")?.job).toBeNull();
  });
});

describe("useByTask", () => {
  it("keeps one value per task, moves it on rename and forgets it on null", () => {
    const { result } = renderHook(() => useByTask<EvalRun>());
    act(() => { result.current.setFor("a")({ ...NO_EVAL, job: job() }); });
    act(() => { result.current.setFor("b")((s) => ({ ...(s ?? NO_EVAL), rev: 2 })); });
    expect(result.current.map.a?.job?.id).toBe("j1"); // a's run survives working on b
    expect(result.current.map.b?.rev).toBe(2);
    act(() => { result.current.rename("a", "a2"); });
    expect(result.current.map.a).toBeUndefined();
    expect(result.current.map.a2?.job?.id).toBe("j1");
    act(() => { result.current.setFor("b")(null); });
    expect(result.current.map.b).toBeUndefined();
  });
});
