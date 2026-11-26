export interface TaskSummary {
  name: string; embedder: string; classifier: string; revision: number;
  n_classes: number; n_examples: number; n_inbox: number; counts: Record<string, number>; created: string;
  last_evaluation: { at: string; revision: number | null; method: string | null; accuracy: number | null } | null;
  warnings: string[];
}
export interface ClassInfo { name: string; description: string; count: number; reject_below: number | null } // reject_below: the class's own threshold
export interface Example {
  id: string; label: string; file: string; source: string; added: string; tags: string[]; note: string;
}
export interface TaskDetail {
  name: string; embedder: string; supports_text: boolean; classifier: string;
  classifier_params: Record<string, unknown>; text_weight: number; reject_below: number | null;
  reject_below_zero_shot: number | null; flip: boolean; temperature: number | null; inbox_count: number;
  can_undo: boolean; can_redo: boolean; undo_label: string; redo_label: string; // labels: e.g. "delete 3 examples"
  revision: number; classes: ClassInfo[]; examples: Example[];
}
export interface Prediction {
  file: string; label: string | null; top_label: string; confidence: number;
  similarity: number; rejected: boolean; scores: Record<string, number>; // scores: highest first
  labels: string[]; // every class the image matches (with a reject threshold), label first
  evidence: { id: string; label: string; similarity: number }[]; // most similar labelled examples, if asked for
  error: string | null; // set when the image could not be read; the other fields are then empty
}
/** An example's prediction by a classifier fitted without it. */
export interface HeldOut { id: string; true: string; pred: string; confidence: number; similarity: number; correct: boolean }
export interface EvalReport {
  method: string; n_scored: number; n_examples: number; accuracy: number | null; macro_f1: number | null;
  top_k_accuracy: Record<string, number>; classes: string[]; confusion: number[][];
  per_class: Record<string, { support: number; recall: number | null; precision: number | null }>;
  mistakes: HeldOut[]; predictions: HeldOut[]; suggested_reject_below: number | null;
  suggested_class_thresholds: Record<string, number>; // per class, for classes with 5+ correct predictions
  calibration: { temperature: number; ece: number; ece_calibrated: number } | null;
}
/** One embedder's cross-validation on a task, or why it couldn't be used. */
export interface CompareResult {
  embedder: string; accuracy: number | null; macro_f1: number | null; top_k_accuracy: Record<string, number>;
  n_scored: number; seconds: number | null; method: string | null; error: string | null;
}
export interface CompareReport { revision: number; current: string; n_examples: number; results: CompareResult[]; best: string | null }
export interface InboxEntry {
  id: string; file: string; source: string; added: string;
  prediction?: Prediction | null; margin?: number | null; // absent while the task has no classes
}
export type InboxSort = "uncertain" | "confident" | "novel" | "added";
export interface DuplicatePair { a: string; b: string; similarity: number; labels: string[]; same_label: boolean }
export interface Health { status: string; auth_required: boolean; models_loaded: string[]; jobs_running: number; url_fetch: boolean }
export interface AddResult { added: Example[]; errors: { file: string; error: string }[]; task: TaskDetail }
export interface Job<T> {
  id: string; task: string; kind: string; status: "running" | "done" | "failed" | "cancelled";
  stage: string; done: number; total: number; result: T | null; error: string | null;
}
export interface ModelInfo {
  embedders: { spec: string; label: string; supports_text: boolean; notes: string }[];
  classifiers: string[];
}

/** Fired on window when a request is refused for want of the server's token. */
export const AUTH_EVENT = "fewshot:auth-required";

/** A request the server answered with an error status. */
export class HttpError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

/** A FastAPI error ``detail``: a message, or (422) a list of {loc, msg} problems. */
function detailText(d: unknown): string {
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    return d.map((p: { loc?: unknown[]; msg?: string }) => [p.loc?.filter((x) => x !== "body").join("."), p.msg].filter(Boolean).join(": ")).join("; ");
  }
  return JSON.stringify(d);
}

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "Content-Type": "application/json" };
  }
  const r = await fetch(url, init);
  if (r.status === 401 && url !== "/api/login") {
    // The server wants its token: the App shows the sign-in form.
    window.dispatchEvent(new Event(AUTH_EVENT));
    throw new Error("Not signed in");
  }
  if (!r.ok) {
    let msg: unknown = `${String(r.status)} ${r.statusText}`;
    try {
      // Error bodies are FastAPI's {"detail": ...}; anything else falls back to the status line.
      const body = (await r.json()) as { detail?: unknown } | null;
      msg = body?.detail ?? msg;
    } catch { /* not JSON */ }
    throw new HttpError(r.status, detailText(msg));
  }
  // The server's response schemas are mirrored by the interfaces above; this is the one untyped boundary.
  return r.status === 204 ? (undefined as T) : ((await r.json()) as T);
}

const t = (name: string) => `/api/tasks/${encodeURIComponent(name)}`;
const files = (fs: File[], extra: Record<string, string> = {}) => {
  const fd = new FormData();
  fs.forEach((f) => { fd.append("files", f); });
  Object.entries(extra).forEach(([k, v]) => { fd.append(k, v); });
  return fd;
};

export const api = {
  models: () => req<ModelInfo>("GET", "/api/models"),
  tasks: () => req<TaskSummary[]>("GET", "/api/tasks"),
  task: (n: string) => req<TaskDetail>("GET", t(n)),
  // An undefined embedder is left out of the body, so the server uses its default.
  createTask: (name: string, embedder: string | undefined, classifier: string) =>
    req<TaskDetail>("POST", "/api/tasks", { name, embedder, classifier }),
  updateTask: (n: string, s: Record<string, unknown>) => req<TaskDetail>("PATCH", t(n), s),
  deleteTask: (n: string) => req<undefined>("DELETE", t(n)),
  addClass: (n: string, name: string, description = "") =>
    req<TaskDetail>("POST", `${t(n)}/classes`, { name, description }),
  patchClass: (n: string, c: string, p: { name?: string; description?: string }) =>
    req<TaskDetail>("PATCH", `${t(n)}/classes/${encodeURIComponent(c)}`, p),
  describe: (n: string, template: string, overwrite: boolean) =>
    req<TaskDetail>("POST", `${t(n)}/describe`, { template, overwrite }),
  deleteClass: (n: string, c: string) => req<TaskDetail>("DELETE", `${t(n)}/classes/${encodeURIComponent(c)}`),
  addExamples: (n: string, label: string, fs: File[]) => req<AddResult>("POST", `${t(n)}/examples`, files(fs, { label })),
  addExampleUrls: (n: string, label: string, urls: string[]) => req<AddResult>("POST", `${t(n)}/examples/urls`, { label, urls }),
  relabel: (n: string, id: string, label: string) => req<TaskDetail>("PATCH", `${t(n)}/examples/${id}`, { label }),
  annotate: (n: string, id: string, p: { tags?: string[]; note?: string }) => req<TaskDetail>("PATCH", `${t(n)}/examples/${id}`, p),
  tagMany: (n: string, ids: string[], add_tags: string[], remove_tags: string[] = []) =>
    req<TaskDetail>("PATCH", `${t(n)}/examples`, { ids, add_tags, remove_tags }),
  deleteExample: (n: string, id: string) => req<TaskDetail>("DELETE", `${t(n)}/examples/${id}`),
  relabelMany: (n: string, ids: string[], label: string) => req<TaskDetail>("PATCH", `${t(n)}/examples`, { ids, label }),
  deleteExamples: (n: string, ids: string[]) => req<TaskDetail>("POST", `${t(n)}/examples/delete`, { ids }),
  predict: (n: string, fs: File[]) => req<Prediction[]>("POST", `${t(n)}/predict?evidence=3`, files(fs)),
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
  inbox: (n: string, sort: InboxSort) => req<{ total: number; items: InboxEntry[] }>("GET", `${t(n)}/inbox?sort=${sort}&limit=1000`),
  addToInbox: (n: string, fs: File[]) =>
    req<{ added: InboxEntry[]; errors: { file: string; error: string }[]; task: TaskDetail }>("POST", `${t(n)}/inbox`, files(fs)),
  acceptInbox: (n: string, labels: Record<string, string>) => req<TaskDetail>("POST", `${t(n)}/inbox/accept`, { labels }),
  dismissInbox: (n: string, ids: string[]) => req<TaskDetail>("POST", `${t(n)}/inbox/dismiss`, { ids }),
  evaluate: (n: string) => req<EvalReport>("POST", `${t(n)}/evaluate`),
  startEvaluation: (n: string) => req<Job<EvalReport>>("POST", `${t(n)}/evaluate/jobs`),
  startComparison: (n: string, embedders: string[]) => req<Job<CompareReport>>("POST", `${t(n)}/compare/jobs`, { embedders }),
  setClassThresholds: (n: string, thresholds: Record<string, number | null>) =>
    req<TaskDetail>("PUT", `${t(n)}/class-thresholds`, { thresholds }),
  job: <T>(id: string) => req<Job<T>>("GET", `/api/jobs/${id}`),
  cancelJob: (id: string) => req<Job<unknown>>("DELETE", `/api/jobs/${id}`),
  login: (token: string) => req<undefined>("POST", "/api/login", { token }),
  /** An example's image; with ``size``, a JPEG scaled down to fit in that many pixels. */
  imageUrl: (n: string, id: string, size?: number) =>
    `${t(n)}/examples/${id}/image` + (size ? `?size=${String(size)}` : ""),
};
