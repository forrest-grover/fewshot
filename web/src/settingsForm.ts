import type { TaskDetail, TaskSettings } from "./api";

/** The Settings tab's fields. Text inputs keep exactly what was typed; it is parsed on save. */
export interface SettingsForm {
  embedder: string; classifier: string; params: Record<string, string>; textWeight: number;
  reject: string; rejectZero: string; temperature: string; flip: boolean; names: boolean;
}
/** Fields the user changed and hasn't saved. Fields not in here show the saved task, so a change
 * made elsewhere (the CLI, another tab, undo) shows up without losing what is being typed. */
export type Edits = Partial<SettingsForm>;

const optional = (v: number | null) => (v == null ? "" : String(v));

/** Classifier params from the server as the text shown in their inputs. */
function paramTexts(p: Record<string, unknown>): Record<string, string> {
  return Object.fromEntries(Object.entries(p).map(([k, v]) => [k, typeof v === "number" || typeof v === "string" ? String(v) : ""]));
}

/** The form showing the saved ``task``. */
export function formOf(task: TaskDetail): SettingsForm {
  return {
    embedder: task.embedder, classifier: task.classifier, params: paramTexts(task.classifier_params),
    textWeight: task.text_weight, reject: optional(task.reject_below), rejectZero: optional(task.reject_below_zero_shot),
    temperature: optional(task.temperature), flip: task.flip, names: task.names_as_text,
  };
}

/** A typed number, or null for blank (use the default / off) or NaN for anything unparseable. */
export function parseNumber(s: string): number | null {
  const t = s.trim();
  return t === "" ? null : Number(t);
}

const sameParams = (a: Record<string, unknown>, b: Record<string, unknown>) => {
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => k in b && a[k] === b[k]);
};

/** The PATCH body that saves ``form`` over ``task``: only the fields that differ from it. That
 * matters: the server keeps a fitted value (thresholds, temperature) given alongside a new
 * embedder or classifier, so resending unchanged ones would keep values fitted to the old model.
 * Also returns what can't be saved as typed. */
export function settingsPatch(task: TaskDetail, form: SettingsForm): { patch: TaskSettings; problems: string[] } {
  const patch: TaskSettings = {};
  const problems: string[] = [];
  if (form.embedder !== task.embedder) patch.embedder = form.embedder;
  if (form.classifier !== task.classifier) patch.classifier = form.classifier;

  const params: Record<string, number> = {};
  for (const [k, text] of Object.entries(form.params)) {
    const n = parseNumber(text);
    if (n === null) continue; // blank: the server uses the default
    if (Number.isFinite(n)) params[k] = n;
    else problems.push(`${k} must be a number`);
  }
  // A new classifier starts from its defaults unless params are given (the server resets them).
  if (patch.classifier !== undefined ? Object.keys(params).length > 0 : !sameParams(params, task.classifier_params)) {
    patch.classifier_params = params;
  }
  if (form.textWeight !== task.text_weight) patch.text_weight = form.textWeight;

  const reject = parseNumber(form.reject);
  if (reject !== null && !Number.isFinite(reject)) problems.push("Reject below must be a number, or blank for off");
  else if (reject === null && task.reject_below !== null) patch.clear_reject = true;
  else if (reject !== null && reject !== task.reject_below) patch.reject_below = reject;

  const rejectZero = parseNumber(form.rejectZero);
  if (rejectZero !== null && !Number.isFinite(rejectZero)) problems.push("Zero-shot reject below must be a number, or blank for off");
  else if (rejectZero === null && task.reject_below_zero_shot !== null) patch.clear_reject_zero_shot = true;
  else if (rejectZero !== null && rejectZero !== task.reject_below_zero_shot) patch.reject_below_zero_shot = rejectZero;

  const temperature = parseNumber(form.temperature);
  if (temperature !== null && !(temperature >= 0.01 && temperature <= 100)) {
    problems.push("Temperature must be between 0.01 and 100, or blank for none");
  } else if (temperature === null && task.temperature !== null) patch.clear_temperature = true;
  else if (temperature !== null && temperature !== task.temperature) patch.temperature = temperature;

  if (form.flip !== task.flip) patch.flip = form.flip;
  if (form.names !== task.names_as_text) patch.names_as_text = form.names;
  return { patch, problems };
}

/** ``edits`` without those that were saved: the ones still exactly as they were when ``sent``. */
export function afterSave(edits: Edits, sent: Edits): Edits {
  return Object.fromEntries(Object.entries(edits).filter(([k, v]) => sent[k as keyof Edits] !== v)) as Edits;
}
