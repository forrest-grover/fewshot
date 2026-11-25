export interface TaskSummary {
  name: string; embedder: string; classifier: string;
  n_classes: number; n_examples: number; counts: Record<string, number>; created: string;
}
export interface ClassInfo { name: string; description: string; count: number }
export interface Example { id: string; label: string; file: string; source: string; added: string }
export interface TaskDetail {
  name: string; embedder: string; supports_text: boolean; classifier: string;
  classifier_params: Record<string, unknown>; text_weight: number; reject_below: number | null;
  revision: number; classes: ClassInfo[]; examples: Example[];
}
export interface Prediction {
  file: string; label: string | null; top_label: string; confidence: number;
  similarity: number; rejected: boolean; scores: Record<string, number>;
  error: string | null; // set when the image could not be read; the other fields are then empty
}
export interface EvalReport {
  method: string; n_scored: number; n_examples: number; accuracy: number | null; classes: string[];
  confusion: number[][];
  per_class: Record<string, { support: number; recall: number | null; precision: number | null }>;
  mistakes: { id: string; true: string; pred: string; confidence: number }[];
}
export interface ModelInfo {
  embedders: { spec: string; label: string; supports_text: boolean; notes: string }[];
  classifiers: string[];
}

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "Content-Type": "application/json" };
  }
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg: unknown = `${String(r.status)} ${r.statusText}`;
    try {
      // Error bodies are FastAPI's {"detail": ...}; anything else falls back to the status line.
      const body = (await r.json()) as { detail?: unknown } | null;
      msg = body?.detail ?? msg;
    } catch { /* not JSON */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
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
  addExamples: (n: string, label: string, fs: File[]) =>
    req<{ added: Example[]; errors: { file: string; error: string }[]; task: TaskDetail }>(
      "POST", `${t(n)}/examples`, files(fs, { label })),
  relabel: (n: string, id: string, label: string) => req<TaskDetail>("PATCH", `${t(n)}/examples/${id}`, { label }),
  deleteExample: (n: string, id: string) => req<TaskDetail>("DELETE", `${t(n)}/examples/${id}`),
  predict: (n: string, fs: File[]) => req<Prediction[]>("POST", `${t(n)}/predict`, files(fs)),
  evaluate: (n: string) => req<EvalReport>("POST", `${t(n)}/evaluate`),
  imageUrl: (n: string, id: string) => `${t(n)}/examples/${id}/image`,
};
