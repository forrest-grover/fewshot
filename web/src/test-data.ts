import type { TaskDetail } from "./api";

/** A saved task for tests, with ``over`` applied. */
export function taskDetail(over: Partial<TaskDetail> = {}): TaskDetail {
  return {
    name: "pets", embedder: "openclip:ViT-B-32@openai", supports_text: true, classifier: "gda",
    classifier_params: {}, text_weight: 1, reject_below: null, reject_below_zero_shot: null, flip: false,
    names_as_text: true, temperature: null, revision: 3, inbox_count: 0, can_undo: true, can_redo: false,
    undo_label: "add 2 examples", redo_label: "", classes: [], examples: [], ...over,
  };
}
