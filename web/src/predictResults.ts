import { useCallback, useEffect, useRef, useState } from "react";
import type { Prediction } from "./api";

/** One result. ``file`` is what was classified (scaled down), and is also what "Add as" uploads,
 * so adding it reuses the embedding computed for the prediction. A result for an image URL has no
 * ``file``: its ``url`` is the remote image, which "Add as" has the server fetch again. */
export interface Item {
  key: number; name: string; file?: File; url: string; pred: Prediction; adding?: boolean;
  added?: { label: string; id: string }; // set once added as an example (shown while that example exists)
}

/** Prediction results for one task, kept by the App so they survive switching tabs. */
export interface Results {
  task: string;
  items: Item[];
  batch: number | null; // the classification in progress, if any (it lands here even if the task is renamed meanwhile)
  progress?: { done: number; total: number }; // of that classification: images classified so far
  current: number | null; // key of the result that keyboard shortcuts act on
}

/** State for the Predict tab, held above it. Results (and their preview URLs) are dropped when
 * another task is opened (not when ``rename`` moved them to the task's new name) and when the App
 * unmounts. */
export function usePredictResults(task: string | null) {
  const [results, setResults] = useState<Results>({ task: task ?? "", items: [], batch: null, current: null });
  const urls = useRef(new Set<string>()); // preview object URLs still in use
  const name = task ?? "";
  const renamed = useRef<{ from: string; to: string } | null>(null);
  // Results already moved to the task's new name, while the route still shows the old one, stay.
  // Once the route reaches the new name the rename is over, so a later task reusing the old name
  // (say, a duplicate) starts with fresh results.
  if (renamed.current?.to === name) renamed.current = null;
  const moving = renamed.current?.from === name && renamed.current.to === results.task;
  const [epoch, setEpoch] = useState(0); // counts the times results were dropped for another task
  if (results.task !== name && !moving) { // another task: adjust during render, as React recommends
    setResults({ task: name, items: [], batch: null, current: null });
    setEpoch((e) => e + 1);
  }
  // Preview URLs of results no longer shown are released. In an effect, not during render, which
  // React may repeat or throw away.
  useEffect(() => {
    const shown = new Set(results.items.map((it) => it.url));
    urls.current.forEach((u) => { if (!shown.has(u)) { URL.revokeObjectURL(u); urls.current.delete(u); } });
  }, [results.items]);
  useEffect(() => {
    const live = urls.current;
    return () => { revoke(live); };
  }, []);
  // Classifications in flight stop when another task is opened (not on a rename, nor on leaving
  // the tab: their results are kept) and when the App unmounts.
  const inFlight = useRef(new Set<AbortController>());
  useEffect(() => {
    const live = inFlight.current;
    return () => { live.forEach((c) => { c.abort(); }); live.clear(); };
  }, [epoch]);
  /** Start an abortable classification; call ``done`` when it ends. */
  const abortable = useCallback(() => {
    const c = new AbortController();
    inFlight.current.add(c);
    return { signal: c.signal, done: () => { inFlight.current.delete(c); } };
  }, []);
  /** Stop every classification in flight (the Cancel button). */
  const cancel = useCallback(() => { inFlight.current.forEach((c) => { c.abort(); }); inFlight.current.clear(); }, []);
  /** Keep the results when their task is renamed. */
  const rename = (to: string) => {
    renamed.current = { from: name, to };
    setResults((r) => (r.task === name ? { ...r, task: to } : r));
  };
  return { results, setResults, urls, rename, abortable, cancel };
}

function revoke(urls: Set<string>) {
  urls.forEach((u) => { URL.revokeObjectURL(u); });
  urls.clear();
}
