import { afterEach, describe, expect, it, vi } from "vitest";
import { api, HttpError } from "./api";

afterEach(() => { vi.restoreAllMocks(); });

describe("api errors", () => {
  it("reads a problem's code and one-line detail", async () => {
    const problem = { type: "urn:fewshot:problem:task-not-found", title: "No such task", status: 404,
      code: "TASK_NOT_FOUND", detail: "No task 'pest'; did you mean 'pets'?", instance: "/api/tasks/pest" };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(problem), {
      status: 404, headers: { "Content-Type": "application/problem+json" } }));
    const e = await api.task("pest").catch((x: unknown) => x);
    expect(e).toBeInstanceOf(HttpError);
    expect([(e as HttpError).status, (e as HttpError).code, (e as HttpError).message]).toEqual(
      [404, "TASK_NOT_FOUND", "No task 'pest'; did you mean 'pets'?"]);
  });

  it("falls back to the status line for a body that isn't a problem", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("<html>bad gateway</html>", { status: 502, statusText: "Bad Gateway" }));
    const e = (await api.tasks().catch((x: unknown) => x)) as HttpError;
    expect([e.status, e.code, e.message]).toEqual([502, undefined, "502 Bad Gateway"]);
  });
});
