"""Background jobs: long operations run in a thread and polled for progress, with cancel."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


class Cancelled(Exception):
    """Raised by ``Job.progress`` once the job was asked to stop."""


@dataclass
class Job:  # pylint: disable=too-many-instance-attributes  # a record of a job's state, one field each
    """A long-running operation in a background thread, polled through /api/jobs/{id}."""

    task: str
    kind: str
    revision: int = 0  # of the task when the job started
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "running"  # running | done | failed | cancelled
    stage: str = "starting"
    done: int = 0
    total: int = 0
    result: Any = None
    error: str | None = None
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict[str, Any]:
        """JSON view of the job."""
        return {k: getattr(self, k) for k in ("id", "task", "kind", "status", "stage", "done", "total", "result",
                                              "error")}

    def progress(self, stage: str, done: int, total: int) -> None:
        """Progress callback for the work; raises ``Cancelled`` once cancellation was asked for."""
        if self.cancel.is_set():
            raise Cancelled
        self.stage, self.done, self.total = stage, done, total


class Jobs:
    """The jobs of this process, keeping the ``kept`` most recently finished ones for polling."""

    def __init__(self, kept: int = 100) -> None:
        self.kept = kept
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, job: Job, work: Callable[[Callable[[str, int, int], None]], Any]) -> Job:
        """Run ``work(progress)`` in a thread; its return value becomes ``job.result``. A
        ``ValueError`` or ``KeyError`` fails the job with its message; anything else with
        "Internal error: ..."."""
        def run() -> None:
            try:
                job.result = work(job.progress)
                job.status = "done"
            except Cancelled:
                job.status = "cancelled"
            except (ValueError, KeyError) as e:
                job.status, job.error = "failed", str(e.args[0]) if e.args else str(e)
            except Exception as e:  # pylint: disable=broad-exception-caught  # reported to the poller
                log.exception("Job %s failed", job.id)
                job.status, job.error = "failed", f"Internal error: {e}"
            job.finished = time.time()

        with self._lock:
            finished = sorted((j for j in self._jobs.values() if j.finished), key=lambda j: j.finished or 0)
            for old in finished[: max(0, len(finished) - self.kept + 1)]:
                del self._jobs[old.id]
            self._jobs[job.id] = job
        threading.Thread(target=run, name=f"fewshot-job-{job.id[:8]}", daemon=True).start()
        return job

    def get(self, jid: str) -> Job | None:
        """The job with id ``jid``, if it is still remembered."""
        with self._lock:
            return self._jobs.get(jid)

    def running(self) -> list[Job]:
        """Jobs not finished yet."""
        with self._lock:
            return [j for j in self._jobs.values() if j.status == "running"]
