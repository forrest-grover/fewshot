"""Operations the command line and the REST API share, so both follow the same rules and report
problems the same way: creating tasks, changing settings, fetching and adding images, predicting,
evaluating and labelling from the inbox.

Errors are raised as ``ValueError`` (bad input) and ``KeyError`` (unknown task, class or id), for
each interface to report in its own way. Nothing here imports torch until it is used, so the
command line stays quick to start.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from .embedders import check_embedder
from .fetch import fetch_url
from .folders import read_or_empty
from .text import check_class_name

if TYPE_CHECKING:
    from .pipeline import Pipeline, Prediction
    from .store import Task, Workspace

SpecCheck = Callable[[str], None]  # raises to refuse an embedder spec
Progress = Callable[[str, int, int], None]  # (stage, done, total)


class Blob(NamedTuple):
    """An image to use: its name (file name or URL), its bytes or file, or why it couldn't be read."""

    name: str
    data: bytes | Path | None  # None if it couldn't be read: then ``error`` says why
    error: str | None = None
    source: str | None = None  # what to store as an added image's source, if not ``name``


# ---- tasks and settings ---------------------------------------------------
def create_task(ws: Workspace, name: str, embedder: str | None = None, classifier: str = "gda",
                embedder_ok: SpecCheck | None = None) -> Task:
    """Create task ``name``, checking ``embedder`` (if given) with ``embedder_ok`` (default
    ``check_embedder``, which loads it) first.

    Raises ``ValueError`` for a bad or taken name, an unknown classifier or an embedder refused."""
    kw = {"classifier": classifier}
    if embedder is not None:
        (embedder_ok or check_embedder)(embedder)
        kw["embedder"] = embedder
    return ws.create(name, **kw)


def change_settings(task: Task, values: dict[str, Any], clear: Iterable[str] = (),  # pylint: disable=too-many-arguments
                    merge_params: dict[str, Any] | None = None, *, embedder_ok: SpecCheck | None = None
                    ) -> list[str]:
    """Change ``task``'s settings: ``values`` are set, the settings named in ``clear`` (say,
    ``reject_below``) unset, and ``merge_params`` set on top of the classifier's params (see
    ``Task.update_settings`` for the rules). A new embedder is checked with ``embedder_ok`` (default
    ``check_embedder``).
    Returns the fitted settings the change cleared.

    Raises ``ValueError`` for a setting both given and cleared, or a bad value."""
    kw = dict(values)
    for key in clear:
        if kw.get(key) is not None:
            raise ValueError(f"Give either {key} or clear it, not both")
        kw[key] = None
    if "embedder" in kw:
        if kw["embedder"] == task.config.embedder:
            del kw["embedder"]
        else:
            (embedder_ok or check_embedder)(kw["embedder"])
    return task.update_settings(merge_params=merge_params, **kw)


def check_specs(specs: Iterable[str], limit: int | None = None) -> list[str]:
    """The distinct non-blank embedder specs among ``specs``, in order; ``ValueError`` unless there
    is at least one (and at most ``limit``)."""
    out = list(dict.fromkeys(s.strip() for s in specs if s.strip()))
    if not out or (limit is not None and len(out) > limit):
        raise ValueError(f"Give between 1 and {limit} embedders" if limit else "Give at least one embedder")
    return out


# ---- images in --------------------------------------------------------------
def fetch_urls(urls: Sequence[str], max_bytes: int, *, public_only: bool = False,  # pylint: disable=too-many-arguments
               seconds: float | None = None, max_total: int | None = None, workers: int = 4) -> list[Blob]:
    """Download ``urls``, ``workers`` at a time; each is at most ``max_bytes``, and with
    ``public_only`` only from public addresses (as the server does). ``seconds`` and ``max_total``
    bound the whole batch's time and size: a URL left when either runs out isn't used, and says so.
    A URL that can't be fetched gets its error instead of data."""
    deadline = None if seconds is None else time.monotonic() + seconds
    total = 0
    lock = threading.Lock()

    def one(url: str) -> Blob:
        nonlocal total
        left = 20.0 if deadline is None else min(20.0, deadline - time.monotonic())
        if left <= 0:
            return Blob(url, None, f"Not fetched: the request's {seconds:g} s budget for URLs ran out")
        try:
            data = fetch_url(url, max_bytes, timeout=left, public_only=public_only)
        except ValueError as e:
            return Blob(url, None, str(e))
        with lock:
            total += len(data)
            over = max_total is not None and total > max_total
        if over:
            return Blob(url, None, f"Not used: the images together exceed the {(max_total or 0) / 2**20:.3g} MB "
                                   "request limit")
        return Blob(url, data)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, urls))


@dataclass
class Added:
    """What adding images did: the examples (or inbox images) added, and each image left out."""

    added: list[Any] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (name, reason)


def _added(task_call: Callable[[Iterable[Any]], Sequence[Any]], blobs: Iterable[Blob],
           item: Callable[[Blob], Any]) -> Added:
    """Run ``task_call`` on the readable ``blobs`` (as ``item(blob)``), lazily, so only the batch
    being stored is held in memory; unreadable blobs and refused images become ``errors``."""
    out = Added()
    names: list[str] = []

    def readable() -> Iterator[Any]:
        for b in blobs:
            if b.data is None:
                out.errors.append((b.name, b.error or "unreadable"))
            else:
                names.append(b.name)
                yield item(b)

    for name, r in zip(names, task_call(readable()), strict=True):
        if isinstance(r, ValueError):
            out.errors.append((name, str(r)))
        else:
            out.added.append(r)
    return out


def _bytes(data: bytes | Path | None) -> bytes:
    """``data`` as bytes (a file is read; one that can't be read gives b"", an unreadable image)."""
    return read_or_empty(data) if isinstance(data, Path) else data or b""


def add_examples(task: Task, label: str, blobs: Iterable[Blob]) -> Added:
    """Add the readable ``blobs`` as examples of ``label``, one undo step for all of them;
    ``ValueError`` for a bad label. Images that can't be added are reported, not raised."""
    label = check_class_name(label)
    return _added(task.add_examples, blobs, lambda b: (_bytes(b.data), label, b.source or b.name))


def add_to_inbox(task: Task, blobs: Iterable[Blob]) -> Added:
    """Put the readable ``blobs`` in the task's inbox, one undo step for all of them. Images that
    can't be added (unreadable, or labelled already) are reported, not raised."""
    return _added(task.add_to_inbox, blobs, lambda b: (_bytes(b.data), b.source or b.name))


# ---- predictions and evaluation ---------------------------------------------
def predict(pipe: Pipeline, task: Task, blobs: Sequence[Blob], **kw: Any) -> list[Prediction]:
    """Classify ``blobs`` (see ``Pipeline.predict`` for ``kw``); one that couldn't be read gets a
    failed prediction carrying its error. Raises ``ValueError`` if the task can't predict."""
    from .pipeline import Prediction  # pylint: disable=import-outside-toplevel,redefined-outer-name  # torch-backed

    preds = pipe.predict(task, [b.data if b.data is not None else b"" for b in blobs], **kw)
    return [Prediction.failed(b.error or "unreadable") if b.data is None else p
            for b, p in zip(blobs, preds, strict=True)]


def evaluate(pipe: Pipeline, task: Task, folds: int | None = None, progress: Progress | None = None,
             apply_class_thresholds: bool = False) -> dict[str, Any]:
    """Cross-validate ``task`` and record the result for the dashboard. With
    ``apply_class_thresholds``, each class then gets its suggested reject threshold (thresholds
    don't change what evaluation measures, so the report stays current). Raises ``ValueError`` if
    the task can't be evaluated, and ``KeyError`` if a class went away before its threshold was set."""
    from .evaluate import evaluate_task  # pylint: disable=import-outside-toplevel  # torch-backed

    report = evaluate_task(pipe, task, folds, progress=progress)
    suggested = report["suggested_class_thresholds"]
    if apply_class_thresholds and suggested:
        current = report["revision"] == task.config.revision
        task.set_class_thresholds(suggested)
        if current:
            report = {**report, "revision": task.config.revision}
    task.record_evaluation(report)
    return report


# ---- inbox ------------------------------------------------------------------
def confident_suggestions(pipe: Pipeline, task: Task, min_confidence: float,
                          ids: Iterable[str] | None = None) -> dict[str, str]:
    """``{inbox id: suggested class}`` for every inbox image (of ``ids``, if given) the model
    suggests a class for at least ``min_confidence`` sure, and doesn't reject."""
    wanted = None if ids is None else set(ids)
    return {i["id"]: i["prediction"]["label"] for i in pipe.inbox_suggestions(task, "confident")
            if i.get("prediction") and (wanted is None or i["id"] in wanted)
            and not i["prediction"]["rejected"] and i["prediction"]["confidence"] >= min_confidence}


def accept_confident(pipe: Pipeline, task: Task, min_confidence: float, ids: Iterable[str] | None = None) -> int:
    """Label as the model suggests every inbox image (of ``ids``, if given) it is at least
    ``min_confidence`` sure of, in one undo step; returns how many were labelled."""
    labels = confident_suggestions(pipe, task, min_confidence, ids)
    return task.accept(labels) if labels else 0
