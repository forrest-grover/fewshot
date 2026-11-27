"""Background jobs: long operations run in a thread and polled for progress, with cancel."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .exceptions import error_code, http_status, message

log = logging.getLogger(__name__)


class Cancelled(Exception):
    """Raised by ``Job.progress`` once the job was asked to stop."""


class TooManyJobs(Exception):
    """Raised by ``Jobs.start`` when ``limit`` jobs are already running or stopping."""


@dataclass
class Job:  # pylint: disable=too-many-instance-attributes  # a record of a job's state, one field each
    """A long-running operation in a background thread, polled through /api/jobs/{id}."""

    task: str
    kind: str
    revision: int = 0  # of the task when the job started
    uid: str = ""  # the task's uid: a task deleted or renamed and re-created under its name has another
    key: str = ""  # with uid, kind and revision, what makes two jobs the same work (see ``Jobs.start``)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "running"  # running | cancelling | done | failed | cancelled
    stage: str = "starting"
    done: int = 0
    total: int = 0
    result: Any = None
    error: str | None = None
    code: str | None = None  # the error's stable code (see ``exceptions``), when failed
    error_status: int | None = None  # the HTTP status that would answer the error
    expired: bool = False  # finished long enough ago that its result is no longer kept
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    _state: threading.Lock = field(default_factory=threading.Lock, repr=False)  # status changes

    def to_dict(self) -> dict[str, Any]:
        """JSON view of the job."""
        return {k: getattr(self, k) for k in ("id", "task", "kind", "status", "stage", "done", "total", "result",
                                              "error", "expired")}

    def progress(self, stage: str, done: int, total: int) -> None:
        """Progress callback for the work; raises ``Cancelled`` once cancellation was asked for."""
        if self.cancel.is_set():
            raise Cancelled
        self.stage, self.done, self.total = stage, done, total

    def request_cancel(self) -> None:
        """Ask the job to stop at its next progress step; until then its status is "cancelling"."""
        with self._state:
            self.cancel.set()
            if self.status == "running":
                self.status = "cancelling"

    def end(self, status: str, error: BaseException | None = None) -> None:
        """Record how the job ended, and the error that failed it."""
        with self._state:
            self.status, self.finished = status, time.time()
            if error is not None:
                self.code, self.error_status = error_code(error), http_status(error)
                self.error = message(error) if self.error_status < 500 else "Internal error; see the server log"


class Jobs:
    """The jobs of this process, keeping the ``kept`` most recently finished ones for polling, and
    the results of only the ``kept_results`` most recent (a report can be large)."""

    def __init__(self, kept: int = 100, kept_results: int = 10) -> None:
        self.kept = kept
        self.kept_results = kept_results
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, job: Job, work: Callable[[Callable[[str, int, int], None]], Any], limit: int | None = None) -> Job:
        """Run ``work(progress)`` in a thread; its return value becomes ``job.result``.

        If the same work (the same task, by uid, and kind, revision and key) is already running and not being
        cancelled, that job is returned instead and nothing starts. Otherwise, if ``limit`` jobs
        are running already, raises ``TooManyJobs``. Jobs being cancelled count too: their threads
        run until the work's next progress step, which may be a while (loading a model has none),
        so not counting them would let start-and-cancel pile up threads without bound. Both are
        decided under one lock, so concurrent starts can't pass the limit or duplicate a job.
        A ``ValueError`` or ``KeyError`` fails the job with its message; anything else with
        "Internal error" (details go to the log only: they can hold server paths)."""
        def run() -> None:
            try:
                job.result = work(job.progress)
                job.end("done")
            except Cancelled:
                job.end("cancelled")
            except (ValueError, KeyError) as e:
                job.end("failed", e)
            except Exception as e:  # pylint: disable=broad-exception-caught  # reported to the poller
                log.exception("Job %s failed", job.id)
                job.end("failed", e)
            with self._lock:
                self._forget_old()

        same = (job.uid or job.task, job.kind, job.revision, job.key)
        with self._lock:
            for j in self._jobs.values():
                if j.status == "running" and (j.uid or j.task, j.kind, j.revision, j.key) == same:
                    return j
            if limit is not None and sum(j.finished is None for j in self._jobs.values()) >= limit:
                raise TooManyJobs
            self._jobs[job.id] = job
            self._forget_old()
        threading.Thread(target=run, name=f"fewshot-job-{job.id[:8]}", daemon=True).start()
        return job

    def _forget_old(self) -> None:
        """Forget finished jobs beyond ``kept`` and drop results beyond ``kept_results``. Call with
        the lock held."""
        finished = sorted((j for j in self._jobs.values() if j.finished), key=lambda j: j.finished or 0)
        for old in finished[: max(0, len(finished) - self.kept)]:
            del self._jobs[old.id]
        for old in finished[: max(0, len(finished) - self.kept_results)]:
            if old.result is not None:
                old.result, old.expired = None, True

    def get(self, jid: str) -> Job | None:
        """The job with id ``jid``, if it is still remembered."""
        with self._lock:
            return self._jobs.get(jid)

    def running(self) -> list[Job]:
        """Jobs running and not asked to stop."""
        with self._lock:
            return [j for j in self._jobs.values() if j.status == "running"]

    def stopping(self) -> list[Job]:
        """Jobs asked to stop whose threads haven't yet."""
        with self._lock:
            return [j for j in self._jobs.values() if j.status == "cancelling"]
