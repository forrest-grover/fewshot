import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api, type Prediction, type TaskDetail } from "./api";
import Predict from "./Predict";
import { usePredictResults } from "./predictResults";
import { taskDetail } from "./test-data";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

const pets = taskDetail({ classes: [{ name: "cat", description: "", count: 1, reject_below: null }, { name: "dog", description: "", count: 1, reject_below: null }] });
const pred = (label: string): Prediction => ({
  label, labels: [label], top_label: label, confidence: 0.9, similarity: 0.8, rejected: false, error: null,
  scores: { [label]: 0.9 }, evidence: [],
});

/** The Predict tab with its results state held above it, as the App does; ``errors`` collects
 * what the error box would show. */
function Host({ task, errors, onState }: { task: TaskDetail; errors: string[]; onState?: (s: ReturnType<typeof usePredictResults>) => void }) {
  const state = usePredictResults(task.name);
  onState?.(state);
  const run = <T,>(p: Promise<T>) => p.catch((e: unknown) => { errors.push(e instanceof Error ? e.message : String(e)); return undefined; });
  return <Predict task={task} setTask={vi.fn()} run={run} state={state} urlFetch limits={{ files: 1, bytes: 1 << 20 }} />;
}

describe("Predict", () => {
  it("shows a URL's host instead of loading the remote image, and says why Add as added nothing", async () => {
    vi.spyOn(api, "predictUrls").mockResolvedValue([pred("cat")]);
    const add = vi.spyOn(api, "addExampleUrls").mockResolvedValue({ added: [], unchanged: [], created_classes: [], errors: [{ file: "https://example.com/c.jpg", error: "timed out" }], task: pets, moved: [] });
    const errors: string[] = [];
    const { container } = render(<Host task={pets} errors={errors} />);
    fireEvent.change(screen.getByRole("textbox", { name: "Image URLs to classify" }), { target: { value: "https://example.com/c.jpg" } });
    fireEvent.click(screen.getByRole("button", { name: "Classify URLs" }));
    // The page's content policy blocks images from other sites: none is asked for.
    expect(await screen.findByRole("img", { name: "https://example.com/c.jpg" })).toBeDefined();
    expect(screen.getByText("example.com")).toBeDefined();
    expect([...container.querySelectorAll("img")].map((i) => i.getAttribute("src"))).toEqual([]);
    fireEvent.click(screen.getByRole("button", { name: "Add as cat" }));
    await waitFor(() => { expect(errors).toEqual(["Not added: timed out"]); });
    expect(add).toHaveBeenCalledWith("pets", "cat", ["https://example.com/c.jpg"]);
  });

  it("sends a classification's later batches to the task's new name after a rename", async () => {
    const releases: (() => void)[] = [];
    const predict = vi.spyOn(api, "predict").mockImplementation((_n, fs) => new Promise((res) => {
      releases.push(() => { res(fs.map(() => pred("cat"))); });
    }));
    let blobs = 0; // jsdom has no object URLs
    URL.createObjectURL = () => `blob:${String(++blobs)}`;
    URL.revokeObjectURL = () => { /* nothing to release */ };
    const errors: string[] = [];
    let state: ReturnType<typeof usePredictResults> | null = null;
    const { container, rerender } = render(<Host task={pets} errors={errors} onState={(s) => { state = s; }} />);
    const input = container.querySelector<HTMLInputElement>("input[type=file]");
    if (!input) throw new Error("no file input");
    const files = ["a", "b", "c"].map((n) => new File(["x"], `${n}.png`, { type: "image/png" }));
    fireEvent.change(input, { target: { files } });
    await waitFor(() => { expect(predict).toHaveBeenCalledTimes(1); });
    act(() => { state?.rename("cats"); }); // Settings renamed the task; the route follows later
    act(() => { releases[0](); });
    await waitFor(() => { expect(predict).toHaveBeenCalledTimes(2); });
    rerender(<Host task={{ ...pets, name: "cats" }} errors={errors} onState={(s) => { state = s; }} />);
    act(() => { releases[1](); });
    await waitFor(() => { expect(predict).toHaveBeenCalledTimes(3); });
    act(() => { releases[2](); });
    expect(predict.mock.calls.map((c) => c[0])).toEqual(["pets", "cats", "cats"]);
    await waitFor(() => { expect(screen.getAllByText(/^[abc]\.png$/)).toHaveLength(3); }); // every result landed
    expect(errors).toEqual([]);
  });
});
