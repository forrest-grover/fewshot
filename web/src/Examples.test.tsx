import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api, type Example, type TaskDetail } from "./api";
import Examples from "./Examples";
import type { Uploads } from "./useUploads";
import { taskDetail } from "./test-data";

afterEach(cleanup);

const example = (id: string, label: string, over: Partial<Example> = {}): Example => ({
  id, label, added: "2026-01-01T00:00:00", file: `${id}.jpg`, note: "", source: `${id}.jpg`, tags: [], ...over,
});
/** The task as a fresh response would bring it: new objects throughout. */
const pets = (rev: number, tags: string[] = []): TaskDetail => taskDetail({
  revision: rev,
  classes: [{ name: "cat", description: "", count: 2, reject_below: null }, { name: "dog", description: "", count: 1, reject_below: null }],
  examples: [example("a", "cat", { tags }), example("b", "cat"), example("c", "dog")],
});

describe("Examples thumbnails", () => {
  it("re-render only when what they show changes, not on every response", () => {
    const urls = vi.spyOn(api, "imageUrl");
    const uploads: Uploads = { slots: {}, start: () => Promise.resolve([]), dismiss: vi.fn() };
    const props = { setTask: vi.fn(), run: <T,>(p: Promise<T>) => p, uploads };
    const { rerender } = render(<Examples task={pets(1)} {...props} />);
    expect(urls).toHaveBeenCalledTimes(3);
    rerender(<Examples task={pets(2)} {...props} />); // say, after an unrelated change
    expect(urls).toHaveBeenCalledTimes(3);
    rerender(<Examples task={pets(3, ["blurry"])} {...props} />);
    expect(urls).toHaveBeenCalledTimes(4); // only the tagged one
    expect(urls).toHaveBeenLastCalledWith("pets", "a", expect.any(Number));
  });
});
