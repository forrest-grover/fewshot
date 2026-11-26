import { useCallback, useState } from "react";
import type { TaskDetail } from "./api";

// Small pieces of App state: values kept per task, ordering overlapping requests, one action at a
// time, and the text after an undo or redo.

/** The toast after an undo or redo that returned ``t``: what was undone is now what redo would do
 * (and the reverse), so the label comes from the response, not from the task as shown before it
 * (``fallback``), which may be out of date. ``then`` offers the reverse if it is possible. */
export function stepToast(which: "undo" | "redo", t: TaskDetail, fallback: string) {
  const what = (which === "undo" ? t.redo_label : t.undo_label) || fallback || "the last change";
  return {
    text: `${which === "undo" ? "Undid" : "Redid"} ${what}`,
    then: which === "undo" ? (t.can_redo ? "redo" as const : null) : (t.can_undo ? "undo" as const : null),
  };
}

/** A state setter for one task's value in ``useByTask``. */
export type SetFor<V> = (f: V | null | ((v: V | null) => V | null)) => void;

/** A value per task (e.g. its last evaluation and the job computing the next), kept by the App so
 * it survives switching tabs and tasks. ``setFor(name)`` is a state setter for one task's value;
 * setting null forgets it. */
export function useByTask<V>() {
  const [map, setMap] = useState<Partial<Record<string, V>>>({});
  const setFor = useCallback((task: string): SetFor<V> => (f) => {
    setMap((m) => {
      const cur = m[task] ?? null;
      const next = typeof f === "function" ? (f as (v: V | null) => V | null)(cur) : f;
      return next === cur ? m : { ...m, [task]: next ?? undefined };
    });
  }, []);
  /** Move a task's value to its new name. */
  const rename = useCallback((from: string, to: string) => {
    setMap((m) => (m[from] === undefined ? m : { ...m, [from]: undefined, [to]: m[from] }));
  }, []);
  return { map, setFor, rename };
}

/** Orders overlapping requests for the same thing: ``begin`` returns a check that is true only
 * while no later request has begun, so an older response arriving last can't overwrite a newer one. */
export class Latest {
  private seq = 0;
  begin(): () => boolean {
    const mine = ++this.seq;
    return () => mine === this.seq;
  }
}

/** Lets one async action run at a time: ``run`` ignores calls (resolving to undefined) while an
 * earlier one is still in flight, e.g. a held-down Ctrl+Z. */
export class Gate {
  private busy = false;
  get running(): boolean { return this.busy; }
  async run<T>(f: () => Promise<T>): Promise<T | undefined> {
    if (this.busy) return undefined;
    this.busy = true;
    try {
      return await f();
    } finally {
      this.busy = false;
    }
  }
}
