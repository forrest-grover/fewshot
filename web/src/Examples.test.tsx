import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api, type Example, type TaskDetail } from "./api";
import Examples from "./Examples";
import { idleUploads, taskDetail } from "./test-data";

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

const example = (id: string, label: string, over: Partial<Example> = {}): Example => ({
  id, label, added: "2026-01-01T00:00:00", file: `${id}.jpg`, note: "", source: `${id}.jpg`, tags: [], ...over,
});
/** The task as a fresh response would bring it: new objects throughout. */
const pets = (rev: number, tags: string[] = []): TaskDetail => taskDetail({
  revision: rev,
  classes: [{ name: "cat", description: "", count: 2, reject_below: null }, { name: "dog", description: "", count: 1, reject_below: null }],
  examples: [example("a", "cat", { tags }), example("b", "cat"), example("c", "dog")],
});

describe("Examples while changing", () => {
  it("can't rename or delete a class whose upload is running, which would bring the old name back", () => {
    const uploads = idleUploads({ cat: { skipped: [], progress: { done: 0, total: 300, cancel: vi.fn() } } });
    render(<Examples task={pets(1)} setTask={vi.fn()} run={(p) => p} uploads={uploads} />);
    const [catRename, dogRename] = screen.getAllByRole("button", { name: "Rename" });
    const [catDelete, dogDelete] = screen.getAllByRole("button", { name: "Delete" });
    expect([catRename, catDelete].map((b) => (b as HTMLButtonElement).disabled)).toEqual([true, true]);
    expect([dogRename, dogDelete].map((b) => (b as HTMLButtonElement).disabled)).toEqual([false, false]);
  });

  it("in the viewer, deletes once for a double or held Delete, while the first is unanswered", async () => {
    let answer: (t: TaskDetail) => void = () => { /* set below */ };
    const del = vi.spyOn(api, "deleteExample").mockImplementation(() => new Promise((res) => { answer = res; }));
    const relabel = vi.spyOn(api, "relabel").mockResolvedValue(pets(2));
    render(<Examples task={pets(1)} setTask={vi.fn()} run={(p) => p} uploads={idleUploads()} />);
    fireEvent.click(screen.getByAltText("a.jpg"));
    await screen.findByRole("dialog");
    fireEvent.keyDown(window, { key: "Delete" });
    fireEvent.keyDown(window, { key: "Delete" }); // pressed again before the answer
    fireEvent.keyDown(window, { key: "Delete", repeat: true }); // held
    fireEvent.keyDown(window, { key: "2", repeat: true });
    expect(del.mock.calls).toEqual([["pets", "a"]]);
    expect(relabel).not.toHaveBeenCalled();
    await act(async () => { answer(pets(2)); await Promise.resolve(); });
    fireEvent.keyDown(window, { key: "Delete" }); // answered: a new press acts again
    expect(del).toHaveBeenCalledTimes(2);
    fireEvent.keyDown(window, { key: "2" }); // nor another change to it while that is unanswered
    expect(relabel).not.toHaveBeenCalled();
    await act(async () => { answer(pets(3)); await Promise.resolve(); });
    fireEvent.keyDown(window, { key: "2" });
    expect(relabel).toHaveBeenCalledTimes(1);
  });
});

describe("Examples thumbnails", () => {
  it("re-render only when what they show changes, not on every response", () => {
    const urls = vi.spyOn(api, "imageUrl");
    const uploads = idleUploads();
    const props = { setTask: vi.fn(), run: <T,>(p: Promise<T>) => p, uploads };
    const { rerender } = render(<Examples task={pets(1)} {...props} />);
    expect(urls).toHaveBeenCalledTimes(3);
    rerender(<Examples task={pets(2)} {...props} />); // say, after an unrelated change
    expect(urls).toHaveBeenCalledTimes(3);
    rerender(<Examples task={pets(3, ["blurry"])} {...props} />);
    expect(urls).toHaveBeenCalledTimes(4); // only the tagged one
    expect(urls).toHaveBeenLastCalledWith("pets", "a", expect.any(Number));
  });

  it("open in the viewer after a response brought new objects for the same examples", async () => {
    const props = { setTask: vi.fn(), run: <T,>(p: Promise<T>) => p, uploads: idleUploads() };
    const { rerender } = render(<Examples task={pets(1)} {...props} />);
    rerender(<Examples task={pets(2)} {...props} />); // say, after an undo: the thumbnails keep the old objects
    fireEvent.click(screen.getByAltText("b.jpg"));
    expect((await screen.findByRole("dialog")).textContent).toContain("2 / 2");
  });
});
