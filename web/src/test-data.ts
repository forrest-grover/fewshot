import type { TaskDetail } from "./api";
import type { Uploads } from "./useUploads";

/** A saved task for tests, with ``over`` applied. */
export function taskDetail(over: Partial<TaskDetail> = {}): TaskDetail {
  return {
    name: "pets", embedder: "openclip:ViT-B-32@openai", supports_text: true, classifier: "gda",
    classifier_params: {}, text_weight: 1, reject_below: null, reject_below_zero_shot: null, flip: false,
    names_as_text: true, temperature: null, revision: 3, inbox_count: 0, can_undo: true, can_redo: false,
    undo_label: "add 2 examples", redo_label: "", classes: [], examples: [], ...over,
  };
}

/** Uploads (see useUploads) with nothing being sent, and ``slots`` as given. */
export function idleUploads(slots: Uploads["slots"] = {}): Uploads {
  const nothing = () => { /* nothing to do */ };
  return { slots, start: () => Promise.resolve([]), dismiss: nothing, stop: () => Promise.resolve(0), rename: nothing, busy: false };
}
