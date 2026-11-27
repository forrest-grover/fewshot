import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EvalReport, Example, HeldOut } from "./api";
import Evaluate from "./Evaluate";
import Examples from "./Examples";
import { NO_EVAL } from "./runs";
import { idleUploads, taskDetail } from "./test-data";

afterEach(cleanup);

/** A stand-in IntersectionObserver whose elements are "near" the viewport when ``show`` says so. */
class FakeObserver {
  static all: FakeObserver[] = [];
  el: Element | null = null;
  constructor(private readonly cb: IntersectionObserverCallback) { FakeObserver.all.push(this); }
  observe(el: Element) { this.el = el; }
  disconnect() { this.el = null; }
  static show(near: (el: Element) => boolean) {
    act(() => {
      for (const o of FakeObserver.all) {
        if (o.el) o.cb([{ isIntersecting: near(o.el) } as IntersectionObserverEntry], o as unknown as IntersectionObserver);
      }
    });
  }
}

const big = (classes: number, per: number) => {
  const names = Array.from({ length: classes }, (_, c) => `c${String(c)}`);
  const examples: Example[] = names.flatMap((label) => Array.from({ length: per }, (_, i) => ({
    id: `${label}-${String(i)}`, label, added: "2026-01-01T00:00:00", file: "x.jpg", note: "", source: `${label}-${String(i)}.jpg`, tags: [],
  })));
  return taskDetail({ classes: names.map((name) => ({ name, description: "", count: per, reject_below: null })), examples });
};
const props = { setTask: vi.fn(), run: <T,>(p: Promise<T>) => p, uploads: idleUploads() };

describe("Examples on a big task", () => {
  beforeEach(() => { FakeObserver.all = []; vi.stubGlobal("IntersectionObserver", FakeObserver); });
  afterEach(() => { vi.unstubAllGlobals(); });

  it("mounts thumbnails only for classes near the viewport", () => {
    const { container } = render(<Examples task={big(20, 5)} {...props} />);
    const thumbs = (c: string) => container.querySelectorAll(`img[alt^="${c}-"]`).length;
    expect(container.querySelectorAll(".thumb")).toHaveLength(15); // the first three classes, at once
    expect(screen.getAllByRole("heading", { level: 3 })).toHaveLength(20); // every class's header
    const card = (el: Element) => el.closest("section")?.querySelector("h3")?.textContent ?? "";
    FakeObserver.show((el) => ["c10", "c11"].includes(card(el)));
    expect([thumbs("c0"), thumbs("c10"), thumbs("c11"), thumbs("c12")]).toEqual([0, 5, 5, 0]);
    expect(container.querySelectorAll(".thumb")).toHaveLength(10);
  });

  it("ticks, searches and moves with the right examples when only some are mounted", () => {
    render(<Examples task={big(20, 5)} {...props} />);
    fireEvent.click(screen.getByLabelText("Select c0-1.jpg"));
    expect(screen.getByText("1 selected")).toBeDefined();
    fireEvent.change(screen.getByLabelText("Search examples"), { target: { value: "c1-" } });
    expect(screen.getByText("5 examples (5 shown)")).toBeDefined(); // c1
    expect(screen.getAllByText("5 examples (0 shown)")).toHaveLength(19);
  });
});

describe("Evaluate on a big task", () => {
  it("lists mistakes a page at a time", () => {
    const mistakes: HeldOut[] = Array.from({ length: 500 }, (_, i) => ({
      id: `e${String(i)}`, source: `e${String(i)}.jpg`, true: "a", pred: "b", correct: false, confidence: 1 - i / 1000, similarity: 0.5,
    }));
    const report = {
      accuracy: 0.5, macro_f1: 0.5, calibration: null, classes: ["a", "b"], confusion: [[0, 500], [0, 0]], method: "loo",
      mistakes, predictions: mistakes, n_examples: 500, n_scored: 500, per_class: {}, revision: 3,
      suggested_class_thresholds: {}, suggested_reject_below: null, top_k_accuracy: {},
    } as unknown as EvalReport;
    const { container } = render(<Evaluate task={taskDetail()} setTask={vi.fn()} run={(p) => p} models={null}
      state={{ ...NO_EVAL, report, rev: 3 }} setState={vi.fn()} compare={null} setCompare={vi.fn()} />);
    const listedImgs = () => container.querySelectorAll(".thumb.wide img").length;
    expect(listedImgs()).toBe(60);
    fireEvent.click(screen.getByRole("button", { name: "Show more (440 hidden)" }));
    expect(listedImgs()).toBe(300);
  });
});
