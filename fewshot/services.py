"""Operations the command line and the REST API share, so both follow the same rules and report
problems the same way: creating tasks, changing settings, fetching and adding images, importing folders, predicting,
evaluating and labelling from the inbox.

Errors are raised as ``ValueError`` (bad input) and ``KeyError`` (unknown task, class or id), for
each interface to report in its own way. Nothing here imports torch until it is used, so the
command line stays quick to start.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from .classifiers import PARAM_MAX
from .embedders import check_embedder
from .fetch import fetch_url
from .folders import class_folders, read_or_empty
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


@dataclass(frozen=True)
class SettingsChange:
    """A change to a task's settings, named by setting (see ``settings.SETTINGS``), whatever
    interface asked for it: ``values`` to set, settings to ``clear`` (unset), and ``merge_params``
    to set on top of the classifier's params (see ``Task.update_settings`` for the rules)."""

    values: Mapping[str, Any] = field(default_factory=dict)
    clear: frozenset[str] = frozenset()
    merge_params: Mapping[str, Any] = field(default_factory=dict)


class SettingConflict(ValueError):
    """A setting a change both sets and clears; ``setting`` names it, for an interface to say
    which of its own options or fields clash."""

    def __init__(self, setting: str) -> None:
        super().__init__(f"{setting} can't be both set and cleared")
        self.setting = setting


def apply_settings(task: Task, change: SettingsChange, *, embedder_ok: SpecCheck | None = None) -> list[str]:
    """Make ``change`` to ``task``'s settings; a new embedder is checked with ``embedder_ok``
    (default ``check_embedder``) first. Returns the fitted settings the change cleared.

    Raises ``SettingConflict`` for a setting both set and cleared, and ``ValueError`` naming the
    setting for a bad value."""
    kw = dict(change.values)
    for key in sorted(change.clear):
        if kw.get(key) is not None:
            raise SettingConflict(key)
        kw[key] = None
    if (w := kw.get("text_weight")) is not None and not 0 <= w <= PARAM_MAX["text_weight"]:
        # checked here too so the message names the setting, not a classifier parameter
        raise ValueError(f"text_weight must be between 0 and {PARAM_MAX['text_weight']:g}, got {w}")
    if "embedder" in kw:
        if kw["embedder"] == task.config.embedder:
            del kw["embedder"]
        else:
            (embedder_ok or check_embedder)(kw["embedder"])
    return task.update_settings(merge_params=dict(change.merge_params) or None, **kw)


def change_settings(task: Task, values: dict[str, Any], clear: dict[str, str] | None = None,
                    *, embedder_ok: SpecCheck | None = None) -> list[str]:
    """``apply_settings`` for the REST API until it builds a ``SettingsChange`` itself: ``clear``
    maps each setting to clear to the request field that clears it, for the error message."""
    try:
        return apply_settings(task, SettingsChange(values, frozenset(clear or {})), embedder_ok=embedder_ok)
    except SettingConflict as e:
        raise ValueError(f"Give either {e.setting} or {(clear or {})[e.setting]}, not both") from None


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
    """What adding images did: the examples (or inbox images) stored, each image left out, and
    which of the stored examples were already examples of another class, so adding them moved
    them (``added`` includes those too)."""

    added: list[Any] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (name, reason)
    moved: list[tuple[Any, str]] = field(default_factory=list)  # (example, the class it was in)


def _added(task_call: Callable[[Iterable[Any]], Sequence[Any]], blobs: Iterable[tuple[Blob, str | None]],
           item: Callable[[Blob, str | None], Any]) -> Added:
    """Run ``task_call`` on the readable ``blobs`` (each with its label, if any, as
    ``item(blob, label)``), lazily, so only the batch being stored is held in memory; unreadable
    blobs and refused images become ``errors``."""
    out = Added()
    names: list[str] = []

    def readable() -> Iterator[Any]:
        for b, label in blobs:
            if b.data is None:
                out.errors.append((b.name, b.error or "unreadable"))
            else:
                names.append(b.name)
                yield item(b, label)

    for name, r in zip(names, task_call(readable()), strict=True):
        if isinstance(r, ValueError):
            out.errors.append((name, str(r)))
        else:
            out.added.append(r)
    return out


def _bytes(data: bytes | Path | None) -> bytes:
    """``data`` as bytes (a file is read; one that can't be read gives b"", an unreadable image)."""
    return read_or_empty(data) if isinstance(data, Path) else data or b""


def _add_labelled(task: Task, blobs: Iterable[tuple[Blob, str | None]], undo_group: str | None) -> Added:
    """Add each readable blob as an example of its label, in one undo step, noting the examples
    that were in another class before (re-adding an image moves it)."""
    with contextlib.suppress(FileNotFoundError):  # deleted: adding raises KeyError
        task.refresh()
    was = {e.id: e.label for e in task.config.examples}
    out = _added(lambda items: task.add_examples(items, undo_group=undo_group), blobs,
                 lambda b, label: (_bytes(b.data), label, b.source or b.name))
    for ex in out.added:  # in order, so an image given twice under two labels counts as moved once
        if was.get(ex.id, ex.label) != ex.label:
            out.moved.append((ex, was[ex.id]))
        was[ex.id] = ex.label
    return out


def add_examples(task: Task, label: str, blobs: Iterable[Blob], undo_group: str | None = None) -> Added:
    """Add the readable ``blobs`` as examples of ``label``, one undo step for all of them (and for
    the calls just before with the same ``undo_group``: see ``Task.add_examples``); ``ValueError``
    for a bad label. Images that can't be added are reported, not raised; images that were
    examples of another class are moved to ``label`` and listed in ``moved``."""
    label = check_class_name(label)
    return _add_labelled(task, ((b, label) for b in blobs), undo_group)


def import_folder(task: Task, folder: Path, shots: int | None = None,
                  update: Callable[[int, int], None] | None = None, undo_group: str | None = None) -> Added:
    """Add the images of a folder laid out as ``folder/<class>/<image>`` (at most ``shots`` per
    class; hidden and system folders and files are skipped, see ``folders``) as examples of their
    folder's class, all in one undo step, however many classes there are. Images that can't be
    added (unreadable, or in a folder whose name can't be a class name) are reported by path, not
    raised. ``update(done, total)`` is called as images are read."""
    files = [(f, label) for label, fs in class_folders(folder, shots).items() for f in fs]

    def blobs() -> Iterator[tuple[Blob, str | None]]:
        for i, (f, label) in enumerate(files):
            if update:
                update(i, len(files))
            yield Blob(str(f), f, source=f.name), label

    return _add_labelled(task, blobs(), undo_group)


def add_to_inbox(task: Task, blobs: Iterable[Blob], undo_group: str | None = None) -> Added:
    """Put the readable ``blobs`` in the task's inbox, one undo step for all of them (and for the
    calls just before with the same ``undo_group``). Images that can't be added (unreadable, or
    labelled already) are reported, not raised."""
    return _added(lambda items: task.add_to_inbox(items, undo_group=undo_group), ((b, None) for b in blobs),
                  lambda b, _: (_bytes(b.data), b.source or b.name))


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
