import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Prediction } from "./api";
import { usePredictResults, type Item } from "./predictResults";

const item = (key: number, url: string): Item => ({ key, name: url, url, pred: {} as Prediction });

describe("usePredictResults", () => {
  it("releases preview URLs of results no longer shown, after rendering", () => {
    const revoked = vi.fn();
    Object.defineProperty(URL, "revokeObjectURL", { value: revoked, configurable: true, writable: true }); // jsdom has none
    const { result, rerender } = renderHook(({ task }) => usePredictResults(task), { initialProps: { task: "a" as string | null } });
    act(() => {
      result.current.urls.current.add("blob:1").add("blob:2");
      result.current.setResults((r) => ({ ...r, items: [item(1, "blob:1"), item(2, "blob:2")] }));
    });
    expect(revoked).not.toHaveBeenCalled();
    act(() => { result.current.setResults((r) => ({ ...r, items: r.items.slice(1) })); }); // "blob:1" cleared
    expect(revoked).toHaveBeenCalledWith("blob:1");
    expect(revoked).not.toHaveBeenCalledWith("blob:2");
    rerender({ task: "b" }); // another task: its results go
    expect(result.current.results).toMatchObject({ task: "b", items: [] });
    expect(revoked).toHaveBeenCalledWith("blob:2");
  });

  it("aborts classifications in flight when another task opens, not on a rename", () => {
    const { result, rerender } = renderHook(({ task }) => usePredictResults(task), { initialProps: { task: "a" as string | null } });
    const first = result.current.abortable();
    act(() => { result.current.rename("a2"); });
    rerender({ task: "a2" });
    expect(first.signal.aborted).toBe(false);
    expect(result.current.taskName.current).toBe("a2");
    rerender({ task: "b" });
    expect(first.signal.aborted).toBe(true);
    expect(result.current.taskName.current).toBe("b");
    const second = result.current.abortable();
    act(() => { result.current.cancel(); });
    expect(second.signal.aborted).toBe(true);
  });
});
