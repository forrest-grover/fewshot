import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api, type HeldOut, type InboxEntry } from "./api";
import App from "./App";
import Examples from "./Examples";
import Inbox from "./Inbox";
import { idleUploads, taskDetail } from "./test-data";
import Threshold from "./Threshold";

afterEach(() => { cleanup(); location.hash = ""; });

const json = (v: unknown) => new Response(JSON.stringify(v), { status: 200, headers: { "Content-Type": "application/json" } });
const pets = taskDetail({
  classes: [{ name: "cat", description: "", count: 1, reject_below: null }],
  examples: [{ id: "e1", label: "cat", added: "2026-01-01T00:00:00", file: "e1.jpg", note: "", source: "e1.jpg", tags: [] }],
});

describe("Threshold chart", () => {
  const preds: HeldOut[] = Array.from({ length: 40 }, (_, i) => ({
    id: String(i), true: "a", pred: "a", correct: i % 4 !== 0, confidence: 0.9, similarity: 0.5 + i / 100,
  }));

  it("is one tab stop whose arrow keys choose a bar and Enter picks it", () => {
    const { container } = render(<Threshold predictions={preds} current={null} suggested={null} onSet={vi.fn()} />);
    expect(container.querySelectorAll("[tabindex='0']")).toHaveLength(1);
    const chart = screen.getByRole("slider");
    fireEvent.focus(chart);
    expect(chart.getAttribute("aria-valuetext")).toMatch(/^0\.500 to /);
    fireEvent.keyDown(chart, { key: "ArrowRight" });
    fireEvent.keyDown(chart, { key: "ArrowRight" });
    expect(chart.getAttribute("aria-valuenow")).toBe("2");
    fireEvent.keyDown(chart, { key: "Enter" });
    const picked = screen.getByRole<HTMLInputElement>("spinbutton").value;
    expect(Number(picked)).toBeCloseTo(0.5 + 2 * (0.39 / 24), 3);
    fireEvent.keyDown(chart, { key: "End" });
    expect(chart.getAttribute("aria-valuenow")).toBe("23");
  });
});

describe("live regions and labels", () => {
  it("announce upload progress", () => {
    const uploads = idleUploads({ cat: { skipped: [], progress: { done: 100, total: 300, cancel: vi.fn() } } });
    render(<Examples task={pets} setTask={vi.fn()} run={(p) => p} uploads={uploads} />);
    expect(screen.getByRole("status").textContent).toBe("Uploading to cat: 100 / 300");
  });

  it("name each inbox card by its image and suggestion", async () => {
    const entry: InboxEntry = { id: "i1", source: "a.jpg", file: "a.jpg", added: "", margin: null,
      prediction: { label: "cat", labels: ["cat"], top_label: "cat", rejected: false, confidence: 0.8, similarity: 0.5, scores: {}, evidence: [], error: null } };
    vi.spyOn(api, "inbox").mockResolvedValue({ items: [entry], total: 1, offset: 0 });
    render(<Inbox task={{ ...pets, inbox_count: 1 }} setTask={vi.fn()} run={(p) => p} uploads={idleUploads()} />);
    const card = await screen.findByRole("article", { name: "a.jpg: suggested cat, 80.0%" });
    expect(card.getAttribute("aria-current")).toBe("true");
  });
});

describe("api paths", () => {
  it("encode example and job ids", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(() => Promise.resolve(json({})));
    await api.relabel("my task", "a/b?c", "cat");
    await api.job("j/1");
    expect(fetch.mock.calls.map((c) => c[0])).toEqual(["/api/tasks/my%20task/examples/a%2Fb%3Fc", "/api/jobs/j%2F1"]);
    expect(api.imageUrl("t", "#x", 64)).toBe("/api/tasks/t/examples/%23x/image?size=64");
  });
});

describe("the ? key", () => {
  it("doesn't open the shortcuts over the image viewer", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = typeof input === "string" ? input : "";
      if (url === "/api/tasks") return Promise.resolve(json([]));
      if (url === "/api/models") return Promise.resolve(json({ classifiers: [], embedders: [] }));
      if (url === "/api/health") return Promise.resolve(json({ url_fetch: false, auth_required: false, max_request_bytes: 1 << 30 }));
      return Promise.resolve(json(pets));
    });
    location.hash = "#/pets/examples";
    render(<App />);
    fireEvent.click(await screen.findByAltText("e1.jpg"));
    await screen.findByRole("dialog");
    fireEvent.keyDown(document.activeElement ?? document.body, { key: "?" });
    expect(screen.queryByRole("dialog", { name: "Keyboard shortcuts" })).toBeNull();
    fireEvent.keyDown(document.activeElement ?? document.body, { key: "Escape" });
    fireEvent.keyDown(document.body, { key: "?" });
    expect(screen.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeDefined();
  });
});
