import type { components } from "./api-schema";

// Response and body types come from the server's OpenAPI schema (src/api-schema.ts, regenerated
// with `npm run gen:api`), so they can't drift from it unnoticed.
type S = components["schemas"];
export type TaskSummary = S["TaskSummary"];
export type LastEvaluation = S["LastEvaluation"];
/** A class; ``reject_below`` is its own threshold, overriding the task's. */
export type ClassInfo = S["ClassOut"];
export type Example = S["ExampleOut"];
/** A task; ``undo_label`` and ``redo_label`` say what undo and redo would do, e.g. "delete 3 examples". */
export type TaskDetail = S["TaskDetail"];
/** ``scores`` are highest first; ``labels`` is every class the image matches (label first unless
 * rejected); ``evidence`` the most similar labelled examples, if asked for; ``error`` is set when
 * the image could not be read (``top_label``, ``confidence`` and ``similarity`` are then null, the
 * lists empty); ``revision`` the task version used. */
export type Prediction = S["PredictionOut"];
/** An example's prediction by a classifier fitted without it. */
export type HeldOut = S["HeldOutPrediction"];
export type EvalReport = S["EvalReport"];
/** One embedder's cross-validation on a task, or why it couldn't be used. */
export type CompareResult = S["CompareResult"];
export type CompareReport = S["CompareReport"];
/** An inbox image; ``prediction`` and ``margin`` are null while the task has no classes. */
export type InboxEntry = S["InboxEntry"];
export type InboxPage = S["InboxPage"];
export type InboxAdded = S["InboxAdded"];
export type InboxSort = "uncertain" | "confident" | "novel" | "added";
export type DuplicatePair = S["DuplicatePair"];
export type Health = S["Health"];
/** An upload that couldn't be used: its file name (or URL, when known) and why. */
export type FileError = S["FileError"];
/** Uploads are deduplicated by content: ``added`` lists the examples new to the task,
 * ``unchanged`` those already in this class, and ``moved`` those that were examples of another
 * class and were moved to this one. */
export type AddResult = S["AddResult"];
export type ModelInfo = S["ModelsOut"];
/** PATCH /api/tasks/{name}, a JSON Merge Patch: every field is optional, and null clears
 * reject_below, reject_below_zero_shot or temperature (the clear_* flags are deprecated). */
export type TaskSettings = Partial<Omit<S["TaskSettings"], "clear_reject" | "clear_reject_zero_shot" | "clear_temperature">>;
export type JobStatus = S["JobOut"]["status"];

/** Whether a job is still working (a cancelled one runs until its next progress step). */
export const jobActive = (j: { status: JobStatus }): boolean => j.status === "running" || j.status === "cancelling";
/** A background job whose ``result`` (once done) is a ``T``. */
export type Job<T> = Omit<S["JobOut"], "status" | "result"> & { status: JobStatus; result: T | null };

/** Fired on window when a request is refused for want of the server's token. */
export const AUTH_EVENT = "fewshot:auth-required";

/** An error the server answers (application/problem+json; see the README's error codes). */
export type Problem = S["Problem"];

/** A request the server answered with an error status; ``code`` is the problem's stable code
 * (such as TASK_NOT_FOUND or CLASS_NAME_NEAR_MISS), when the server gave one. */
export class HttpError extends Error {
  constructor(readonly status: number, message: string, readonly code?: string) { super(message); }
}

async function req<T>(method: string, url: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const init: RequestInit = { method, signal };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "Content-Type": "application/json" };
  }
  const r = await fetch(url, init);
  if (r.status === 401 && url !== "/api/login") {
    // The server wants its token: the App shows the sign-in form.
    window.dispatchEvent(new Event(AUTH_EVENT));
    throw new HttpError(401, "Not signed in");
  }
  if (!r.ok) {
    let msg = `${String(r.status)} ${r.statusText}`;
    let code: string | undefined;
    try {
      // Error bodies are problems ({code, detail, ...}, detail one line); anything else (a
      // proxy's page) falls back to the status line.
      const body = (await r.json()) as Partial<Problem> | null;
      if (typeof body?.detail === "string") msg = body.detail;
      code = body?.code;
    } catch { /* not JSON */ }
    throw new HttpError(r.status, msg, code);
  }
  // The server's response schemas are mirrored by the interfaces above; this is the one untyped boundary.
  return r.status === 204 ? (undefined as T) : ((await r.json()) as T);
}

const t = (name: string) => `/api/tasks/${encodeURIComponent(name)}`;
/** An example of task ``n``, by id (encoded, like every name or id put in a path). */
const ex = (n: string, id: string) => `${t(n)}/examples/${encodeURIComponent(id)}`;
/** The multipart fields of upload ``B`` besides its files (``undo_group``: uploads with the same
 * one become one undo step). */
type UploadFields<B> = Omit<B, "files">;
const files = (fs: File[], extra: Partial<Record<string, string | null>> = {}) => {
  const fd = new FormData();
  fs.forEach((f) => { fd.append("files", f); });
  Object.entries(extra).forEach(([k, v]) => { if (v != null) fd.append(k, v); });
  return fd;
};

export const api = {
  models: () => req<ModelInfo>("GET", "/api/models"),
  tasks: () => req<TaskSummary[]>("GET", "/api/tasks"),
  task: (n: string) => req<TaskDetail>("GET", t(n)),
  // An undefined embedder is left out of the body, so the server uses its default.
  createTask: (name: string, embedder: string | undefined, classifier: string) =>
    req<TaskDetail>("POST", "/api/tasks", { name, embedder, classifier }),
  updateTask: (n: string, s: TaskSettings) => req<TaskDetail>("PATCH", t(n), s),
  deleteTask: (n: string) => req<undefined>("DELETE", t(n)),
  addClass: (n: string, name: string, description = "") =>
    req<TaskDetail>("POST", `${t(n)}/classes`, { name, description }),
  patchClass: (n: string, c: string, p: { name?: string; description?: string }) =>
    req<TaskDetail>("PATCH", `${t(n)}/classes/${encodeURIComponent(c)}`, p),
  describe: (n: string, template: string, overwrite: boolean) =>
    req<TaskDetail>("POST", `${t(n)}/describe`, { template, overwrite }),
  deleteClass: (n: string, c: string) => req<TaskDetail>("DELETE", `${t(n)}/classes/${encodeURIComponent(c)}`),
  /** Add ``fs`` to class ``label``; uploads given the same ``undoGroup`` are undone as one step. */
  addExamples: (n: string, label: string, fs: File[], signal?: AbortSignal, undoGroup?: string) =>
    req<AddResult>("POST", `${t(n)}/examples`,
      files(fs, { label, undo_group: undoGroup } satisfies Partial<UploadFields<S["Body_add_examples"]>>), signal),
  addExampleUrls: (n: string, label: string, urls: string[]) => req<AddResult>("POST", `${t(n)}/examples/urls`, { label, urls }),
  relabel: (n: string, id: string, label: string) => req<TaskDetail>("PATCH", ex(n, id), { label }),
  annotate: (n: string, id: string, p: { tags?: string[]; note?: string }) => req<TaskDetail>("PATCH", ex(n, id), p),
  tagMany: (n: string, ids: string[], add_tags: string[], remove_tags: string[] = []) =>
    req<TaskDetail>("PATCH", `${t(n)}/examples`, { ids, add_tags, remove_tags }),
  deleteExample: (n: string, id: string) => req<TaskDetail>("DELETE", ex(n, id)),
  relabelMany: (n: string, ids: string[], label: string) => req<TaskDetail>("PATCH", `${t(n)}/examples`, { ids, label }),
  deleteExamples: (n: string, ids: string[]) => req<TaskDetail>("POST", `${t(n)}/examples/delete`, { ids }),
  predict: (n: string, fs: File[], signal?: AbortSignal) =>
    req<Prediction[]>("POST", `${t(n)}/predict?evidence=3`, files(fs), signal),
  predictUrls: (n: string, urls: string[]) => req<Prediction[]>("POST", `${t(n)}/predict/urls?evidence=3`, { urls }),
  duplicates: (n: string, threshold: number) => req<DuplicatePair[]>("GET", `${t(n)}/duplicates?threshold=${String(threshold)}`),
  undo: (n: string) => req<TaskDetail>("POST", `${t(n)}/undo`),
  redo: (n: string) => req<TaskDetail>("POST", `${t(n)}/redo`),
  rename: (n: string, name: string) => req<TaskDetail>("POST", `${t(n)}/rename`, { name }),
  duplicate: (n: string, name: string, examples: boolean) => req<TaskDetail>("POST", `${t(n)}/duplicate`, { name, examples }),
  exportUrl: (n: string) => `${t(n)}/export`,
  importTask: (file: File, name?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    if (name) fd.append("name", name);
    return req<TaskDetail>("POST", "/api/tasks/import", fd);
  },
  health: () => req<Health>("GET", "/api/health"),
  inbox: (n: string, sort: InboxSort, offset = 0, limit = 1000) =>
    req<InboxPage>("GET", `${t(n)}/inbox?sort=${encodeURIComponent(sort)}&offset=${String(offset)}&limit=${String(limit)}`),
  addToInbox: (n: string, fs: File[], signal?: AbortSignal, undoGroup?: string) =>
    req<InboxAdded>("POST", `${t(n)}/inbox`, files(fs, { undo_group: undoGroup } satisfies UploadFields<S["Body_add_to_inbox"]>), signal),
  acceptInbox: (n: string, labels: Record<string, string>) => req<TaskDetail>("POST", `${t(n)}/inbox/accept`, { labels }),
  dismissInbox: (n: string, ids: string[]) => req<TaskDetail>("POST", `${t(n)}/inbox/dismiss`, { ids }),
  evaluate: (n: string) => req<EvalReport>("POST", `${t(n)}/evaluate`),
  startEvaluation: (n: string) => req<Job<EvalReport>>("POST", `${t(n)}/evaluate/jobs`),
  startComparison: (n: string, embedders: string[]) => req<Job<CompareReport>>("POST", `${t(n)}/compare/jobs`, { embedders }),
  setClassThresholds: (n: string, thresholds: Record<string, number | null>) =>
    req<TaskDetail>("PUT", `${t(n)}/class-thresholds`, { thresholds }),
  job: <T>(id: string) => req<Job<T>>("GET", `/api/jobs/${encodeURIComponent(id)}`),
  cancelJob: (id: string) => req<Job<unknown>>("DELETE", `/api/jobs/${encodeURIComponent(id)}`),
  login: (token: string) => req<undefined>("POST", "/api/login", { token }),
  /** End the session (its cookie stops working, here and in every other tab). */
  logout: () => req<undefined>("POST", "/api/logout"),
  /** An example's image; with ``size``, a JPEG scaled down to fit in that many pixels. */
  imageUrl: (n: string, id: string, size?: number) =>
    `${ex(n, id)}/image` + (size ? `?size=${String(size)}` : ""),
};
