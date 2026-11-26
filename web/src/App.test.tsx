import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Prediction, TaskDetail } from "./api";
import App from "./App";
import { taskDetail } from "./test-data";

/** One request the App made. */
interface Call { method: string; url: string; body: unknown; signal: AbortSignal | null }
type Answer = Response | Promise<Response>;
type Route = (c: Call) => Answer | undefined;

const json = (v: unknown, status = 200) => new Response(JSON.stringify(v), { status, headers: { "Content-Type": "application/json" } });
const pets: TaskDetail = taskDetail({
  classes: [{ name: "cat", description: "", count: 1, reject_below: null }, { name: "dog", description: "", count: 0, reject_below: null }],
  examples: [{ id: "e1", label: "cat", added: "2026-01-01T00:00:00", file: "e1.jpg", note: "", source: "e1.jpg", tags: [] }],
});
const birds = taskDetail({ name: "birds", revision: 1 });
const summary = (t: TaskDetail) => ({
  name: t.name, classifier: t.classifier, embedder: t.embedder, counts: {}, created: "", last_evaluation: null,
  n_classes: t.classes.length, n_examples: t.examples.length, n_inbox: 0, revision: t.revision, warnings: [],
});

let calls: Call[] = [];
/** Answer the App's requests: ``route`` first, then a server holding ``pets`` and ``birds``. */
function serve(route: Route = () => undefined) {
  calls = [];
  vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
    const c: Call = { method: init?.method ?? "GET", url: typeof input === "string" ? input : input instanceof URL ? input.href : input.url, body: init?.body, signal: init?.signal ?? null };
    calls.push(c);
    const mine = route(c);
    if (mine) return Promise.resolve(mine);
    const tasks: Record<string, TaskDetail> = { pets, birds };
    if (c.url === "/api/tasks") return Promise.resolve(json([summary(pets), summary(birds)]));
    if (c.url === "/api/models") return Promise.resolve(json({ classifiers: ["gda"], embedders: [] }));
    if (c.url === "/api/health") {
      return Promise.resolve(json({ status: "ok", auth_required: false, models_loaded: [], jobs_running: 0, url_fetch: false,
        max_request_bytes: 1 << 30, max_upload_bytes: 1 << 26 }));
    }
    const m = /^\/api\/tasks\/([^/?]+)$/.exec(c.url);
    if (m && c.method === "GET") return Promise.resolve(json(tasks[m[1]]));
    if (c.url.includes("/inbox?")) return Promise.resolve(json({ items: [], total: 0, offset: 0 }));
    return Promise.resolve(json({ detail: `unexpected ${c.method} ${c.url}` }, 500));
  });
}
/** A response that waits until the request is aborted, like a slow upload. */
const pending = (c: Call): Promise<Response> => new Promise((_res, rej) => {
  c.signal?.addEventListener("abort", () => { rej(new DOMException("aborted", "AbortError")); });
});
const taskLoads = (name: string) => calls.filter((c) => c.method === "GET" && c.url === `/api/tasks/${name}`).length;

async function open(hash: string) {
  await act(async () => {
    location.hash = hash;
    window.dispatchEvent(new HashChangeEvent("hashchange"));
    await Promise.resolve();
  });
}

beforeEach(() => { location.hash = "#/pets/examples"; });
afterEach(() => { cleanup(); location.hash = ""; });

describe("App", () => {
  it.each([400, 404, 409])("fetches the task again when a change is refused with %i", async (status) => {
    serve((c) => (c.method === "DELETE" ? json({ detail: "No such example" }, status) : undefined));
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete example" }));
    expect(await screen.findByText("No such example")).toBeDefined();
    await waitFor(() => { expect(taskLoads("pets")).toBe(2); });
  });

  it("doesn't fetch the task again after other errors", async () => {
    serve((c) => (c.method === "DELETE" ? json({ detail: "Disk full" }, 500) : undefined));
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete example" }));
    await screen.findByText("Disk full");
    expect(taskLoads("pets")).toBe(1);
  });

  it("Enter doesn't add a rejected prediction; a number key does", async () => {
    const rejected: Prediction = { label: null, labels: [], top_label: "cat", rejected: true, confidence: 0.3, similarity: 0.1,
      scores: { cat: 0.3, dog: 0.2 }, evidence: [], error: null };
    serve((c) => {
      if (c.url.includes("/predict")) return json([rejected]);
      if (c.method === "POST" && c.url.endsWith("/examples")) return json({ added: [{}], errors: [], task: pets }, 201);
      return undefined;
    });
    URL.createObjectURL = vi.fn(() => "blob:x");
    URL.revokeObjectURL = vi.fn();
    location.hash = "#/pets/predict";
    const { container } = render(<App />);
    await screen.findByText(/Drop images to classify/);
    const input = container.querySelector<HTMLInputElement>("input[type=file]");
    if (!input) throw new Error("no file input");
    fireEvent.change(input, { target: { files: [new File(["x"], "a.png", { type: "image/png" })] } });
    await screen.findByText(/Unknown \(closest: cat\)/);
    const adds = () => calls.filter((c) => c.method === "POST" && c.url.endsWith("/examples"));
    fireEvent.keyDown(document.body, { key: "Enter" });
    await new Promise((r) => setTimeout(r, 20));
    expect(adds()).toHaveLength(0);
    fireEvent.keyDown(document.body, { key: "2" });
    await waitFor(() => { expect(adds()).toHaveLength(1); });
    expect((adds()[0].body as FormData).get("label")).toBe("dog");
  });

  it("an upload goes on while tabs are switched, and stops (saying so) when another task opens", async () => {
    serve((c) => (c.method === "POST" && c.url.endsWith("/examples") ? pending(c) : undefined));
    const { container } = render(<App />);
    await screen.findByRole("heading", { name: "cat" });
    const card = container.querySelectorAll(".class-card")[0] as HTMLElement;
    const input = card.querySelector<HTMLInputElement>("input[type=file]");
    if (!input) throw new Error("no file input");
    fireEvent.change(input, { target: { files: [new File(["x"], "a.png", { type: "image/png" })] } });
    await within(card).findByText(/Uploading 0 \/ 1/);
    const upload = calls.find((c) => c.method === "POST");
    expect((upload?.body as FormData).get("undo_group")).toBeTruthy();

    await open("#/pets/predict");
    await screen.findByText(/Drop images to classify/);
    await open("#/pets/examples");
    await screen.findByText(/Uploading 0 \/ 1/); // still going, shown again
    expect(upload?.signal?.aborted).toBe(false);

    await open("#/birds/examples");
    expect(upload?.signal?.aborted).toBe(true);
    expect(await screen.findByText("Upload to “pets” stopped: 0 of 1 added")).toBeDefined();
  });
});
