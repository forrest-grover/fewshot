"""A task's kept evaluations and comparisons: what the dashboard, ``GET .../evaluation`` and
``fewshot list`` show of them after the evaluation itself has finished.

They are kept beside ``task.json``, outside undo history (undoing a change doesn't undo having
measured it): the last evaluation's headline (``eval.json``) and whole report
(``eval-report.json``), the headlines of the last ``EVAL_HISTORY`` evaluations with how each was
made (``eval-history.json``) and the last comparison of embedders (``compare.json``). Each records
the revision it was made at, so a reader can tell whether the task has changed since.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .fsutil import os_reason
from .fsutil import write_atomic as _write_atomic
from .models import _now

if TYPE_CHECKING:
    from .models import TaskConfig

log = logging.getLogger(__name__)

HEADLINE = ("revision", "method", "accuracy", "macro_f1", "n_scored", "n_examples")  # an evaluation's, in eval.json
PROVENANCE = ("embedder", "embedder_label", "classifier", "classifier_params", "folds")  # how it was made
EVAL_HISTORY = 20  # evaluation headlines kept per task


def _read_json(path: Path) -> Any:
    """The JSON in ``path``, or None if it is missing or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class TaskEvaluations:
    """The kept-evaluations part of ``store.Task``, which provides the attributes and methods declared here."""

    path: Path
    config: TaskConfig

    def _writing(self) -> contextlib.AbstractContextManager[None]:  # pragma: no cover  # Task's
        raise NotImplementedError

    def record_evaluation(self, report: dict[str, Any]) -> None:
        """Remember an evaluation: its headline (``eval.json``, for the dashboard: when, at which
        revision, and how well), the whole report (``eval-report.json``) and its headline with how
        it was made at the end of a short history (``eval-history.json``, the last ``EVAL_HISTORY``).

        Does nothing if the task was deleted, or deleted and made again, since this object read it:
        an evaluation that finishes late must not land on a different task of the same name.
        If the task can't be written to (a workspace this process may only read), it is logged
        and nothing is kept: the evaluation itself still stands."""
        uid = self.config.uid
        at = report.get("finished_at") or _now()
        try:
            with self._writing():
                if self.config.uid == uid:
                    history = [*self.evaluation_history(),  # before eval.json, which it falls back on
                               {"at": at, **{k: report.get(k) for k in (*HEADLINE, *PROVENANCE)}}]
                    _write_atomic(self.path / "eval.json",
                                  json.dumps({"at": at, **{k: report.get(k) for k in HEADLINE}}).encode())
                    _write_atomic(self.path / "eval-report.json", json.dumps(report).encode())
                    _write_atomic(self.path / "eval-history.json", json.dumps(history[-EVAL_HISTORY:]).encode())
        except KeyError:  # deleted
            pass
        except OSError as e:  # a task this process can't write to: the report is still returned
            log.warning("%s: the evaluation couldn't be kept: %s", self.path, os_reason(e))

    def record_comparison(self, report: dict[str, Any]) -> None:
        """Remember a comparison of embedders (``compare.json``), as ``record_evaluation`` does."""
        uid = self.config.uid
        try:
            with self._writing():
                if self.config.uid == uid:
                    _write_atomic(self.path / "compare.json", json.dumps(report).encode())
        except KeyError:  # deleted
            pass
        except OSError as e:  # a task this process can't write to: the report is still returned
            log.warning("%s: the comparison couldn't be kept: %s", self.path, os_reason(e))

    def _carry_evaluation(self, before: int) -> None:
        """After a change that evaluation doesn't measure (reject thresholds), an evaluation of
        revision ``before`` still describes the task: move it (headline and report) to the new
        revision, so it doesn't show as out of date. Call inside ``_writing``, after ``_save``."""
        for name in ("eval.json", "eval-report.json"):
            ev = _read_json(self.path / name)
            if isinstance(ev, dict) and ev.get("revision") == before:
                _write_atomic(self.path / name, json.dumps({**ev, "revision": self.config.revision}).encode())

    def last_evaluation(self) -> dict[str, Any] | None:
        """The headline ``record_evaluation`` last stored, or None."""
        data = _read_json(self.path / "eval.json")
        return data if isinstance(data, dict) else None

    def evaluation_report(self) -> dict[str, Any] | None:
        """The whole report ``record_evaluation`` last stored, or None (also for a task last
        evaluated by a version that kept only the headline)."""
        data = _read_json(self.path / "eval-report.json")
        return data if isinstance(data, dict) else None

    def evaluation_history(self) -> list[dict[str, Any]]:
        """The headlines of the last ``EVAL_HISTORY`` evaluations, oldest first, with how each was
        made (embedder, classifier, folds...). A task evaluated only by a version that kept no
        history has its last headline."""
        data = _read_json(self.path / "eval-history.json")
        if isinstance(data, list):
            return [h for h in data if isinstance(h, dict)]
        last = self.last_evaluation()
        return [last] if last else []

    def last_comparison(self) -> dict[str, Any] | None:
        """The report ``record_comparison`` last stored, or None."""
        data = _read_json(self.path / "compare.json")
        return data if isinstance(data, dict) else None
