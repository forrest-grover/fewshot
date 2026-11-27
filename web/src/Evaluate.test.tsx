import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { EvalReport } from "./api";
import Evaluate from "./Evaluate";
import { NO_EVAL } from "./runs";
import { taskDetail } from "./test-data";

afterEach(cleanup);

const withCalibration = (ece: number, eceCalibrated: number, helps: boolean) => ({
  accuracy: 0.9, macro_f1: 0.9, classes: ["a", "b"], confusion: [[9, 1], [1, 9]], method: "loo",
  calibration: { temperature: 2, ece, ece_calibrated: eceCalibrated, temperature_helps: helps, reliability: [], reliability_calibrated: [] },
  mistakes: [], predictions: [], n_examples: 20, n_scored: 20, per_class: {}, revision: 3,
  suggested_class_thresholds: {}, suggested_reject_below: null, top_k_accuracy: {},
}) as unknown as EvalReport;

const show = (report: EvalReport) => render(<Evaluate task={taskDetail()} setTask={vi.fn()} run={(p) => p} models={null}
  state={{ ...NO_EVAL, report, rev: 3 }} setState={vi.fn()} compare={null} setCompare={vi.fn()} />);

describe("Evaluate's calibration advice", () => {
  it("offers the temperature when the evaluation says it helps", () => {
    show(withCalibration(0.2, 0.05, true));
    expect(screen.getByText(/with temperature 2 it would be/)).toBeDefined();
    expect(screen.getByRole("button", { name: "Apply temperature" })).toBeDefined();
  });

  it("offers nothing when the improvement is too small to matter, as the CLI does", () => {
    show(withCalibration(0.2, 0.198, false));
    expect(screen.getByText(/no temperature would make it noticeably better/)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Apply temperature" })).toBeNull();
  });
});
