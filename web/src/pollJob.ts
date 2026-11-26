import { useEffect, useRef } from "react";
import { api, HttpError, type Job, jobActive } from "./api";

const POLL_MS = 500;
const POLL_TRIES = 5; // failed requests in a row before giving up on a job

/** Poll job ``jobId`` (if any) every half second, passing each state to ``onJob`` (the last one is
 * finished). A failed request is retried (with growing pauses) up to POLL_TRIES times in a row,
 * since the job keeps running on the server; then, or at once if the server no longer knows the
 * job, it goes to ``onError`` and polling stops. Each request waits for the one before. Unmounting
 * stops polling too. */
export function usePollJob<T>(jobId: string | undefined, onJob: (j: Job<T>) => void, onError: (e: Error) => void) {
  const latest = useRef({ onJob, onError });
  latest.current = { onJob, onError };
  useEffect(() => {
    if (!jobId) return;
    let live = true;
    let failures = 0;
    let timer: ReturnType<typeof setTimeout>;
    const poll = () => {
      api.job<T>(jobId).then((j) => {
        if (!live) return;
        failures = 0;
        latest.current.onJob(j);
        if (jobActive(j)) timer = setTimeout(poll, POLL_MS);
      }, (e: unknown) => {
        if (!live) return;
        failures += 1;
        if (failures < POLL_TRIES && !(e instanceof HttpError && e.status === 404)) {
          timer = setTimeout(poll, POLL_MS * 2 ** failures);
          return;
        }
        latest.current.onError(e instanceof Error ? e : new Error(String(e)));
      });
    };
    timer = setTimeout(poll, POLL_MS);
    return () => { live = false; clearTimeout(timer); };
  }, [jobId]);
}
