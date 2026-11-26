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
/** Uploads that the server answers only when ``release`` is called, a batch at a time. */
function holdUploads() {
  const waiting: (() => void)[] = [];
  let n = 0;
  const route: Route = (c) => {
    if (!(c.method === "POST" && c.url.endsWith("/examples"))) return undefined;
    const files = (c.body as FormData).getAll("files").length;
    return new Promise<Response>((res) => {
      waiting.push(() => { n += files; res(json({ added: Array.from({ length: files }, () => ({})), errors: [], task: { ...pets, revision: 3 + n } }, 201)); });
    });
  };
  return { route, release: () => { waiting.shift()?.(); } };
}
/** Drop ``n`` images on the first class. */
function drop(container: HTMLElement, n: number) {
  const input = container.querySelector<HTMLInputElement>(".class-card input[type=file]");
  if (!input) throw new Error("no file input");
  const files = Array.from({ length: n }, (_, i) => new File(["x"], `${String(i)}.png`, { type: "image/png" }));
  fireEvent.change(input, { target: { files } });
}
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
    expect(await screen.findByText(/No such example/)).toBeDefined();
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
    const held = holdUploads();
    serve(held.route);
    const { container } = render(<App />);
    await screen.findByRole("heading", { name: "cat" });
    drop(container, 150); // two batches
    await screen.findByText(/Uploading 0 \/ 150/);
    const upload = calls.find((c) => c.method === "POST");
    expect((upload?.body as FormData).get("undo_group")).toBeTruthy();

    await open("#/pets/predict");
    await screen.findByText(/Drop images to classify/);
    await open("#/pets/examples");
    await screen.findByText(/Uploading 0 \/ 150/); // still going, shown again
    expect(upload?.signal?.aborted).toBe(false);

    await open("#/birds/examples");
    expect(upload?.signal?.aborted).toBe(false); // the server may store it: it is waited for and counted
    act(() => { held.release(); });
    expect(await screen.findByText("Upload to “pets” stopped: 100 of 150 added")).toBeDefined();
    expect(calls.filter((c) => c.method === "POST")).toHaveLength(1);
  });

  it("Ctrl+Z during an upload stops it after the batch in flight, then undoes all of it", async () => {
    const held = holdUploads();
    serve((c) => held.route(c) ?? (c.url === "/api/tasks/pets/undo" ? json({ ...pets, revision: 1000, can_redo: true, redo_label: "add 200 examples to cat" }) : undefined));
    const { container } = render(<App />);
    await screen.findByRole("heading", { name: "cat" });
    drop(container, 250);
    await screen.findByText(/Uploading 0 \/ 250/);
    act(() => { held.release(); });
    await screen.findByText(/Uploading 100 \/ 250/);
    expect(screen.getByRole("button", { name: "↶ Undo" }).title).toBe("Stop the upload and undo it (Ctrl+Z)");
    fireEvent.keyDown(document.body, { key: "z", ctrlKey: true });
    await screen.findByText(/Stopping 100 \/ 250/);
    expect(calls.some((c) => c.url.endsWith("/undo"))).toBe(false); // not before the batch in flight lands
    act(() => { held.release(); });
    expect(await screen.findByText("Stopped the upload. Undid add 200 examples to cat")).toBeDefined();
    const posts = calls.filter((c) => c.method === "POST").map((c) => c.url);
    expect(posts).toEqual(["/api/tasks/pets/examples", "/api/tasks/pets/examples", "/api/tasks/pets/undo"]);
    // The toast offers Redo, until the task changes again (then it would redo nothing).
    expect(screen.getByRole("button", { name: "Redo" })).toBeDefined();
  });

  it("signing in again keeps the page, and says what uploads stopped meanwhile", async () => {
    let signedOut = false;
    let posts = 0;
    serve((c) => {
      if (c.url === "/api/login") { signedOut = false; return new Response(null, { status: 204 }); }
      if (signedOut) return json({ detail: "Not signed in" }, 401);
      if (c.method === "POST" && c.url.endsWith("/examples")) {
        posts += 1;
        if (posts === 2) { signedOut = true; return json({ detail: "Not signed in" }, 401); } // the session ended mid-upload
        return json({ added: Array.from({ length: 100 }, () => ({})), errors: [], task: { ...pets, revision: 4 } }, 201);
      }
      return undefined;
    });
    const { container } = render(<App />);
    await screen.findByRole("heading", { name: "cat" });
    drop(container, 300);
    const dialog = await screen.findByRole("dialog", { name: "Sign in" });
    expect(screen.getByText(/Your session has ended/)).toBeDefined();
    expect(screen.getByRole("heading", { name: "cat" })).toBeDefined(); // the page stays behind it
    expect(screen.queryByRole("alert")).toBeNull(); // no "Not signed in" error behind the dialog
    fireEvent.change(within(dialog).getByLabelText("Token"), { target: { value: "secret" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Sign in" }));
    expect(await screen.findByText("While you were signed out: Upload stopped: 100 of 300 added; add the other 200 again.")).toBeDefined();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByRole("heading", { name: "cat" })).toBeDefined();
    const login = calls.find((c) => c.url === "/api/login");
    expect(JSON.parse(login?.body as string)).toEqual({ token: "secret" });
  });

  it("offers Log out when the server has a token", async () => {
    serve((c) => {
      if (c.url === "/api/health") {
        return json({ status: "ok", auth_required: true, models_loaded: [], jobs_running: 0, url_fetch: false, max_request_bytes: 1 << 30, max_upload_bytes: 1 << 26 });
      }
      if (c.url === "/api/logout") return new Response(null, { status: 204 });
      return undefined;
    });
    const reload = vi.fn();
    vi.spyOn(window, "location", "get").mockReturnValue({ ...window.location, reload, hash: "#/pets/examples" } as Location);
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Log out" }));
    await waitFor(() => { expect(reload).toHaveBeenCalled(); });
    expect(calls.filter((c) => c.url === "/api/logout").map((c) => c.method)).toEqual(["POST"]);
  });

  it("drops the toast's Redo once the task changed", async () => {
    serve((c) => {
      if (c.url === "/api/tasks/pets/undo") return json({ ...pets, revision: 4, can_redo: true, redo_label: "x" });
      if (c.method === "DELETE") return json({ ...pets, revision: 5, examples: [], can_redo: false });
      return undefined;
    });
    render(<App />);
    await screen.findByRole("heading", { name: "cat" });
    fireEvent.keyDown(document.body, { key: "z", ctrlKey: true });
    await screen.findByRole("button", { name: "Redo" });
    fireEvent.click(screen.getByRole("button", { name: "Delete example" }));
    await waitFor(() => { expect(screen.queryByRole("button", { name: "Redo" })).toBeNull(); });
    expect(screen.getByText(/Undid/)).toBeDefined();
  });
});
