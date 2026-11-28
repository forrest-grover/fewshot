"""On-disk workspace of few-shot tasks.

Layout::

    <root>/cache.sqlite                 embedding cache shared by all tasks
    <root>/blobs/<id[:2]>/<id>.<ext>    each image stored once, hard-linked into the tasks using it
    <root>/tasks/<name>/task.json       config, classes, example labels
    <root>/tasks/<name>/images/<id>.<ext>
    <root>/tasks/<name>/history/<rev>.json   earlier versions of task.json, for undo

Images are stored once per workspace: a task's image file is a hard link to the blob of the same
content, so adding an image to several tasks costs its disk space once. Where hard links aren't
possible the task gets its own copy. A blob is removed when the last task file linking to it is.

The last ``HISTORY_KEPT`` versions of ``task.json`` are kept for ``Task.undo``. Image files of
deleted examples stay until no kept version names them, so undoing a delete brings them back.

Example ids are the first 32 hex chars (128 bits) of the SHA-256 of the image bytes (16 in tasks
created before that; both are accepted), so adding the same image twice is a no-op (or a
relabel), and cached embeddings are reused across tasks.

Several processes (say ``fewshot serve`` and the CLI) can share a workspace. Each change to a task
holds an exclusive lock on ``<task>/.lock``, re-reads ``task.json`` if another process changed it,
and replaces the file atomically, so no process overwrites another's changes. Reads pick up other
processes' changes on the next ``Workspace.get``. The file lock needs ``fcntl`` (Linux, macOS);
elsewhere only threads within one process are serialised.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import re
import shutil
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import ValidationError

from . import blobs as _blobs
from . import exceptions as exc
from . import preconditions
from .classifiers import CLASSIFIERS
from .embedders import check_embedder, loaded_embedders
from .evaluations import TaskEvaluations
from .folders import class_folders, read_or_empty, readable_name
from .fsutil import FileStamp, create_lock_file, move_back, os_reason, reclaim, replace_reserved, spelt_as_listed
from .fsutil import abandoned as _abandoned
from .fsutil import dir_lock as _dir_lock
from .fsutil import file_lock as _file_lock
from .fsutil import fsync_dir as _fsync_dir
from .fsutil import mkstemp as _mkstemp
from .fsutil import stamp as _stamp
from .history import TaskHistory
from .images import image_ext, image_id
from .inbox import TaskInbox, chunked, listed, plural
from .models import MAX_NOTE, ClassDef, Example, TaskConfig, _copy_config, _now, _read_config
from .settings import THRESHOLDS, check_settings, for_new_embedder, for_new_scoring
from .text import check_class_name, check_description, check_tag, check_template, closest, fill_template, prompts

log = logging.getLogger(__name__)

SpecCheck = Callable[[str], None]  # raises to refuse an embedder spec
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# Images saved per task.json write by Task.add_examples.
ADD_CHUNK = 256
# A task folder without task.json (or a .tmp-/.trash- folder) this old is left over from a crash.
STALE_SECONDS = 600
SETTINGS = {"embedder", "classifier", "classifier_params", "text_weight", "reject_below",
            "reject_below_zero_shot", "flip", "temperature", "names_as_text"}


def default_root() -> Path:
    """Workspace root from ``$FEWSHOT_HOME``, defaulting (also when it is empty) to ``./fewshot_data``."""
    return Path(os.environ.get("FEWSHOT_HOME", "").strip() or "./fewshot_data").resolve()


def embedder_loads(spec: str) -> None:
    """``embedders.check_embedder`` (load the model, ``ValueError`` if it can't be), skipped for
    an embedder loaded already, such as one an interface has just checked."""
    if spec not in loaded_embedders():
        check_embedder(spec)


class _NewImage(NamedTuple):
    """An image checked and ready to store."""

    data: bytes
    id: str
    ext: str  # file extension from the image's actual format
    label: str
    source: str


def _check_item(data: bytes, label: str, source: str) -> _NewImage | ValueError:
    """Validate one image to add (decoding it) and its label; the error instead of raising it."""
    try:
        label = check_class_name(label)
        ext = image_ext(data)
    except ValueError as e:
        return e
    return _NewImage(data, image_id(data), ext, label, source)


class Task(TaskEvaluations, TaskHistory, TaskInbox):  # pylint: disable=too-many-public-methods,too-many-instance-attributes  # the whole editing API of a task
    """A task folder: its config plus example images. Every mutation is saved immediately.

    Mutations are serialised across threads and processes, and start from the latest
    ``task.json`` on disk. A mutation on a task deleted by another process raises ``KeyError``.

    ``config`` is a snapshot of the saved task: other threads only ever see versions that were
    written to ``task.json``, and a snapshot is never changed once published, so a reader can hold
    one for a whole prediction or evaluation. Inside a mutation (``_writing``), and only in the
    thread making it, ``config`` is the draft being edited; it is published when ``_save``
    commits it, and dropped if the mutation fails first.
    """

    def __init__(self, path: Path, config: TaskConfig, stamp: FileStamp | None = None,
                 blobs: Path | None = None, embedder_ok: SpecCheck | None = None) -> None:
        self.path = path
        if stamp is not None:  # read from task.json
            config = self._ahead(config, stamp[0])
        self.blobs = blobs  # the workspace's shared image store, if any
        self.embedder_ok = embedder_ok or embedder_loads  # see Workspace
        self._writer: int | None = None  # thread id of the mutation in progress, if any
        self._draft: TaskConfig | None = None  # the config that mutation is editing
        self._published = config
        self._lock = threading.RLock()  # held for a whole mutation: serialises writers
        self._state = threading.Lock()  # held briefly to swap ``_published`` and ``_stamp`` together
        self._stamp = stamp  # of the task.json that ``_published`` matches
        self._depth = 0  # nesting of ``_writing`` in the thread holding ``_lock``
        self._reader: int | None = None  # thread id of the ``locked`` read in progress, if any
        self._commits = 0  # saves that reached task.json, to tell a failed save from a later failure
        self._pending_ino: int | None = None  # inode of the task.json the save in progress wrote
        self._unsynced: set[Path] = set()  # folders with new image files not yet made durable

    # ---- persistence -------------------------------------------------
    @property
    def config(self) -> TaskConfig:
        """The task's settings, classes and examples: the latest saved version, or within a
        mutation, in the thread making it, the draft it edits."""
        if self._writer == threading.get_ident() and self._draft is not None:
            return self._draft
        return self._published

    @config.setter
    def config(self, cfg: TaskConfig) -> None:
        if self._writer == threading.get_ident():
            self._draft = cfg
        else:  # a task being built by its only user (Workspace.build_task)
            self._published = cfg

    @property
    def name(self) -> str:
        """The task name."""
        return self.config.name

    def refresh(self) -> bool:
        """Reload ``task.json`` if it changed on disk; returns whether it did.

        Never waits for a mutation in progress (whose result is published when it commits).
        Raises ``FileNotFoundError`` if the task was deleted.
        """
        with self._state:
            if self._stamp == _stamp(os.stat(self.path / "task.json")):
                return False
            cfg, self._stamp = _read_config(self.path)
            self._published = self._ahead(cfg, self._stamp[0])
            return True

    @contextlib.contextmanager
    def _writing(self) -> Iterator[None]:
        """Context for a mutation: exclusive across threads and processes, starting from the
        latest saved config.

        The mutation edits a draft that only its own thread sees as ``config``; other threads see
        each version once ``_save`` has committed it. If the mutation fails, the draft is dropped
        and ``config`` stays at the last version saved, which may be one this mutation saved
        before failing.
        """
        with self._lock:
            if self._depth:  # nested in a mutation that already holds the locks
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            if self._reader == threading.get_ident():  # its shared lock would block our exclusive one
                raise RuntimeError(f"Can't change task {self.name!r} while reading it in Task.locked")
            with contextlib.ExitStack() as stack:
                try:
                    stack.enter_context(_file_lock(self.path / ".lock", guard=self.path / "task.json"))
                    self.refresh()
                except FileNotFoundError as e:  # deleted by another process
                    raise exc.TaskNotFound(f"No task {self.name!r}", task=self.name) from e
                name, commits = self._published.name, self._commits
                preconditions.check(name, self._published.revision)  # If-Match, atomically
                self._draft = _copy_config(self._published)
                self._writer = threading.get_ident()
                self._depth = 1
                try:
                    yield
                finally:
                    self._writer = None
                    self._draft = None
                    self._depth = 0
                    if self._committed_since(commits):  # (even if it failed after) it moved the revision on
                        preconditions.settle(name)

    @contextlib.contextmanager
    def locked(self) -> Iterator[TaskConfig]:
        """Hold the task still while reading it: yields its latest saved version, and changes from
        other threads and processes wait until the block ends (other readers don't). Needs only
        read access to the task. Don't change the task inside the block (``RuntimeError``).
        Raises ``KeyError`` if the task was deleted."""
        with self._lock:
            if self._depth:  # inside this thread's own change: already held still
                yield self.config
                return
            with contextlib.ExitStack() as stack:
                try:
                    stack.enter_context(_file_lock(self.path / ".lock", shared=True, guard=self.path / "task.json"))
                    self.refresh()
                except FileNotFoundError as e:  # deleted by another process
                    raise exc.TaskNotFound(f"No task {self.name!r}", task=self.name) from e
                self._reader = threading.get_ident()
                try:
                    yield self._published
                finally:
                    self._reader = None

    def _save(self, gone: Iterable[str] = (), record: bool = True, label: str = "", keep_redo: bool = False) -> None:
        """Bump the revision and write ``task.json`` atomically and durably. Call inside ``_writing``.

        With ``record``, the version being replaced is kept in ``history/`` for ``undo``, with
        ``label`` describing the change, and (unless ``keep_redo``) undone changes can no longer
        be redone. ``gone`` names the image files this change stopped using, which are deleted
        once no kept version uses them.

        The change commits when the new task.json replaces the old one. If anything fails before
        that, nothing changed and the error is raised; the clean-up after it (forgetting undone
        changes, pruning history, deleting unused images) is logged if it fails, never raised,
        because the change has been made.

        A recorded change, inside ``_writing``, that leaves the task as it was saved (moving an
        example to its own class, say) isn't saved: an undo step that changes nothing looks like
        one a crash left, which ``undo`` skips, so it would undo the change before it instead.
        """
        cfg = self.config
        if (record and self._writer == threading.get_ident()
                and cfg.model_copy(update={"revision": self._published.revision}) == self._published):
            return
        old_rev = cfg.revision
        cfg.revision += 1
        made: list[Path] = []
        stale_redo: Path | None = None
        self._pending_ino = None
        fd, tmp = _mkstemp(self.path / "task.json")
        try:
            with os.fdopen(fd, "wb") as f:
                self._pending_ino = os.fstat(f.fileno()).st_ino
                f.write(cfg.model_dump_json(indent=2).encode())
                f.flush()
                os.fsync(f.fileno())
            while self._unsynced:  # the new image files' names, before a task.json naming them
                _fsync_dir(self._unsynced.pop())
            if record and (self.path / "task.json").exists():
                made = self._snapshot(old_rev, sorted(set(gone)), label)
            if record and not keep_redo:
                # Out of reach before the commit: a crash between the two then costs the undone
                # changes, never lets a redo overwrite this one.
                stale_redo = self._set_redo_aside()
            os.replace(tmp, self.path / "task.json")
        except BaseException:
            if self._landed():  # interrupted just after the commit: the change stands
                self._publish(cfg)
                raise
            cfg.revision = old_rev
            for leftover in [Path(tmp), *made]:  # a version that never got replaced is no step to undo
                with contextlib.suppress(OSError):
                    os.unlink(leftover)
            if stale_redo is not None:
                with contextlib.suppress(OSError):
                    os.rename(stale_redo, self.path / "history" / "redo")
            raise
        self._publish(cfg)
        self._after_commit(stale_redo, record)

    def _landed(self) -> bool:
        """Whether task.json is the file the last ``_save`` wrote: its commit happened, even if an
        interrupt arrived before the save could record it. Decided from the disk, since a
        signal can land between any two bytecodes."""
        try:
            return self._pending_ino is not None and os.stat(self.path / "task.json").st_ino == self._pending_ino
        except OSError:
            return False

    def _committed_since(self, commits: int) -> bool:
        """Whether a save committed since ``_commits`` was ``commits``."""
        return self._commits != commits or self._landed()

    def _publish(self, cfg: TaskConfig) -> None:
        """Make ``cfg``, just written to task.json, what other threads see."""
        try:
            st: FileStamp | None = _stamp(os.stat(self.path / "task.json"))
        except OSError:  # pragma: no cover  # re-read on the next refresh instead
            st = None
        with self._state:
            # Inside a mutation the draft may still be edited: publish a copy no edit reaches.
            self._published = _copy_config(cfg) if self._writer == threading.get_ident() else cfg
            self._stamp = st
            self._commits += 1

    def _after_commit(self, stale_redo: Path | None, record: bool) -> None:
        """Make the commit durable and clean up after it; failures are logged, since raising
        would report a change that was made as one that failed."""
        steps: list[tuple[str, Any]] = [("sync the task folder", lambda: _fsync_dir(self.path))]
        if stale_redo is not None:
            steps.append(("forget undone changes", lambda: self._clear_redo(stale_redo)))
        if record:
            steps.append(("prune history", self._prune_history))
        for what, step in steps:
            try:
                step()
            except Exception:  # pylint: disable=broad-exception-caught  # the change is saved regardless
                log.exception("%s: the change was saved, but failed to %s", self.path, what)

    # ---- image files (shared through the workspace's store: see blobs.py) ----
    def _blob(self, file: str) -> Path | None:
        return _blobs.blob_path(self.blobs, file)

    def _store_image(self, file: str, data: bytes) -> bool:
        return _blobs.store_image(self.path, file, data, self.blobs, self._unsynced)

    def _add_file(self, file: str, source: bytes | Path) -> None:
        """Store ``file`` from its bytes, or from (a hard link to) another task's image file."""
        if isinstance(source, bytes):
            self._store_image(file, source)
        else:
            _blobs.link_image(self.path, file, source, self.blobs, self._unsynced)

    def _drop_file(self, file: str) -> None:
        _blobs.drop_file(self.path, file, self.blobs)

    def destroy(self) -> None:
        """Delete the task folder, waiting for any mutation in progress (in any process).

        The folder is first moved aside in one step, so a crash part-way leaves either the whole
        task or a ``.trash-`` folder that ``Workspace.gc`` removes, never a half-deleted task.
        """
        with self._writing():
            files = self.config.files() | self._redo_files()
            files |= set().union(*(self._gone(p) for _, p in self._history()))
            trash = self.path.parent / f".trash-{uuid.uuid4().hex}"
            os.rename(self.path, trash)
            _fsync_dir(self.path.parent)
            shutil.rmtree(trash)
            for f in files:  # the shared copies no other task links to
                _blobs.release(self._blob(f))

    # ---- settings ----------------------------------------------------
    def update_settings(self, merge_params: dict[str, Any] | None = None, **kw: Any) -> list[str]:
        """Change embedder/classifier settings (see ``SETTINGS``); unknown keys raise ``ValueError``.

        The rules every interface shares: switching to another classifier starts its params afresh
        unless ``classifier_params`` is given; ``merge_params`` are then set on top of the params
        (the given, fresh or current ones) instead of replacing them; a ``temperature`` of 1 means
        uncalibrated and is stored as None. Changing the embedder also clears the temperature and
        reject thresholds (task and per class) fitted to the old one, and changing a
        ``settings.SCORING`` setting the temperature, unless given in ``kw``; returns the names of
        those cleared. Changing nothing saves nothing. A new embedder must pass ``embedder_ok``
        (see ``Workspace``)."""
        bad = set(kw) - SETTINGS
        if bad:
            raise ValueError(f"Unknown settings: {sorted(bad)}")
        if kw.get("temperature") == 1:
            kw["temperature"] = None
        with self._writing():
            if "classifier_params" not in kw and kw.get("classifier", self.config.classifier) != self.config.classifier:
                kw["classifier_params"] = {}
            if merge_params:
                kw["classifier_params"] = {**kw.get("classifier_params", self.config.classifier_params),
                                           **merge_params}
            try:
                new = TaskConfig.model_validate({**self.config.model_dump(), **kw})
            except ValidationError as e:
                err = e.errors()[0]
                raise ValueError(f"{'.'.join(map(str, err['loc']))}: {err['msg']}") from None
            check_settings(new)
            if new.embedder != self.config.embedder:
                self.embedder_ok(new.embedder)
            if all(getattr(new, k) == getattr(self.config, k) for k in kw):
                return []
            cleared = for_new_embedder(self.config, keep=kw) if new.embedder != self.config.embedder else []
            cleared = for_new_scoring(self.config, new, keep=kw) + cleared
            for k in kw:
                setattr(self.config, k, getattr(new, k))
            before = self.config.revision
            self._save(label="change settings")
            if set(kw) <= THRESHOLDS:
                self._carry_evaluation(before)
        return cleared

    # ---- classes -----------------------------------------------------
    def class_names(self) -> list[str]:
        """Class names in definition order."""
        return [c.name for c in self.config.classes]

    def upsert_class(self, name: str, description: str | None = None) -> ClassDef:
        """Create a class, or update its description if it exists (``None`` keeps it).

        Raises ``ValueError`` for a name ``check_class_name`` refuses.
        """
        name = name.strip()
        if description is not None:
            check_description(description)
        with self._writing():
            for i, c in enumerate(self.config.classes):
                if c.name == name:
                    if description is not None:
                        c = self.config.classes[i] = c.model_copy(update={"description": description})
                    self._save(label=f"edit class {name}")
                    return c
            c = ClassDef(name=check_class_name(name), description=description or "")
            self.config.classes.append(c)
            self._save(label=f"add class {c.name}")
            return c

    def edit_class(self, name: str, new_name: str | None = None, description: str | None = None) -> ClassDef:
        """Rename class ``name`` (relabelling its examples) and/or set its description, in one
        change and one undo step. Raises ``KeyError`` for an unknown class and ``ValueError`` for
        a bad or taken new name or a description ``check_description`` refuses; then nothing
        changes."""
        new = None if new_name is None or new_name.strip() == name else check_class_name(new_name)
        if description is not None:
            check_description(description)
        with self._writing():
            c = self._class(name)
            if new is not None and new in self.class_names():
                raise exc.ClassExists(f"Class {new!r} already exists", label=new)
            cfg = self.config
            update: dict[str, Any] = {} if description is None else {"description": description}
            if new is not None:
                update["name"] = new
                cfg.examples = [e.model_copy(update={"label": new}) if e.label == name else e for e in cfg.examples]
            edited = c.model_copy(update=update)
            cfg.classes = [edited if k.name == name else k for k in cfg.classes]
            self._save(label=f"rename class {name} to {new}" if new is not None and description is None
                       else f"edit class {name}")
            return edited

    def describe_classes(self, template: str, overwrite: bool = False) -> int:
        """Set descriptions from ``template`` ('{}' -> class name); returns how many changed.

        A template of several lines gives each class several prompts, which are averaged.
        """
        check_template(template)
        template = "\n".join(prompts(template))
        n = 0
        with self._writing():
            classes = self.config.classes
            for i, c in enumerate(classes):
                if overwrite or not c.description.strip():
                    desc = check_description(fill_template(template, c.name))
                    classes[i] = c.model_copy(update={"description": desc})
                    n += 1
            self._save(label="describe classes")
        return n

    def rename_class(self, old: str, new: str) -> None:
        """Rename a class and relabel its examples."""
        if check_class_name(new) == old:
            raise exc.ClassExists(f"Class {old!r} already exists", label=old)
        self.edit_class(old, new)

    def delete_class(self, name: str) -> int:
        """Delete a class and its examples; returns the number of examples removed."""
        with self._writing():
            self._class(name)
            gone = [e for e in self.config.examples if e.label == name]
            self.config.examples = [e for e in self.config.examples if e.label != name]
            self.config.classes = [c for c in self.config.classes if c.name != name]
            self._save(gone=[e.file for e in gone], label=f"delete class {name}")  # files stay while undo needs them
            return len(gone)

    def set_class_thresholds(self, thresholds: dict[str, float | None]) -> None:
        """Set (or, with None, clear) per-class reject thresholds in one change. Raises ``KeyError``
        for an unknown class and ``ValueError`` for a value outside -1..1; then nothing changes.
        An empty dict changes nothing (and adds no undo step)."""
        if not thresholds:
            return
        for k, v in thresholds.items():
            if v is not None and not -1.0 <= v <= 1.0:
                raise ValueError(f"The threshold of {k!r} must be between -1 and 1, got {v}")
        with self._writing():
            for k in thresholds:
                self._class(k)
            cfg = self.config
            cfg.classes = [c.model_copy(update={"reject_below": thresholds[c.name]}) if c.name in thresholds else c
                           for c in cfg.classes]
            n = len(thresholds)
            before = cfg.revision
            self._save(label=f"set the threshold of {next(iter(thresholds))}" if n == 1
                       else f"set thresholds of {n} classes")
            self._carry_evaluation(before)

    def _class(self, name: str) -> ClassDef:
        for c in self.config.classes:
            if c.name == name:
                return c
        raise exc.ClassNotFound(f"No class {name!r}", label=name, similar_to=closest(name, self.class_names()))

    # ---- examples ----------------------------------------------------
    def add_example(self, data: bytes, label: str, source: str = "") -> Example:
        """Store an image under ``label`` (creating the class); re-adding an image relabels it.

        Raises ``ValueError`` for an unreadable image or a bad class name.
        """
        result = self.add_examples([(data, label, source)])[0]
        if isinstance(result, ValueError):
            raise result
        return result

    def add_examples(self, items: Iterable[tuple[bytes, str, str]],
                     chunk: int = ADD_CHUNK, undo_group: str | None = None) -> list[Example | ValueError]:
        """Store ``(image bytes, label, source)`` items, like ``add_example`` but with one save per
        ``chunk`` items instead of one per image, and one undo step for the whole call. Calls with
        the same ``undo_group`` (any string the caller chooses, e.g. for one upload sent in several
        requests) that follow each other, with no other change in between, share one undo step too.

        Returns one entry per item: its ``Example``, or the ``ValueError`` that kept it out (an
        unreadable image or a bad class name). Images are decoded to check them before the task is
        locked, so a long import doesn't block other writers for its whole length; ``items`` may be
        a generator, and only one chunk of image bytes is held at a time.
        """
        group = ("examples", undo_group or f"call:{uuid.uuid4().hex}")
        results: list[Example | ValueError] = []
        for batch in chunked(items, chunk):
            results += self._add_batch(batch, group)
        return results

    def _add_batch(self, batch: list[tuple[bytes, str, str]], group: tuple[str, str]) -> list[Example | ValueError]:
        """Add one chunk (saving unless every item failed), in undo group ``group`` (see
        ``_save_in_group``), so one undo takes back the whole call or group."""
        checked = [_check_item(*item) for item in batch]  # decoding happens here, before locking
        out: list[Example | ValueError] = []
        created: list[str] = []
        changed: list[str] = []  # the label of each image added or moved (not one there already)
        with self._writing(), self._new_files(created):  # storing some may fail (a full disk)
            examples = self.config.examples
            at = {e.id: i for i, e in enumerate(examples)}  # id -> position
            legacy = {e.id: i for i, e in enumerate(examples) if len(e.id) == 16}  # 16 chars: older tasks
            waiting = {x.id for x in self.config.inbox}
            for c in checked:
                if isinstance(c, ValueError):
                    out.append(c)
                    continue
                self._ensure_class(c.label)
                if c.id in waiting:  # labelled now: no longer waiting in the inbox
                    self.config.inbox = [x for x in self.config.inbox if x.id != c.id]
                    waiting.discard(c.id)
                i = at.get(c.id, legacy.get(c.id[:16]))
                if i is not None:  # already stored: just relabel (if it was in another class)
                    changed += [c.label] if examples[i].label != c.label else []
                    ex = examples[i] = examples[i].model_copy(update={"label": c.label})
                else:
                    changed.append(c.label)
                    ex = Example(id=c.id, label=c.label, file=f"images/{c.id}.{c.ext}", source=c.source)
                    if self._store_image(ex.file, c.data):
                        created.append(ex.file)
                    at[c.id] = len(examples)
                    examples.append(ex)
                out.append(ex)
            if changed:  # a batch of images all there already changes nothing: no save, no undo step
                self._save_in_group(created, group, len(changed), list(dict.fromkeys(changed)),
                                    lambda n, names: f"add {_n(n)} to {listed(names)}")
        return out

    def _ensure_class(self, label: str) -> None:
        """Add class ``label`` (already checked) if missing, without saving."""
        if label not in self.class_names():
            self.config.classes.append(ClassDef(name=label))

    def add_examples_from_dir(self, folder: Path | str, shots: int | None = None) -> int:
        """Import a folder laid out as ``folder/<class>/<image>`` (at most ``shots`` images per
        class; hidden and system folders and files are skipped, see ``folders``); returns how many
        images were added.

        Raises ``ValueError`` for the first image that can't be read (after adding the others).
        """
        folder = Path(folder)
        files = [(f, label) for label, fs in class_folders(folder, shots).items() for f in fs]
        results = self.add_examples((read_or_empty(f), label, readable_name(f.name)) for f, label in files)
        for (f, _), r in zip(files, results, strict=True):
            if isinstance(r, ValueError):
                raise ValueError(f"{f}: {r}")
        return len(results)

    def example(self, eid: str) -> Example:
        """Look up an example by id; raises ``KeyError`` if missing."""
        for e in self.config.examples:
            if e.id == eid:
                return e
        raise exc.ExampleNotFound(f"No example {eid!r}", example_id=eid)

    def relabel(self, eid: str, label: str) -> Example:
        """Move an example to ``label``, creating the class if needed."""
        with self._writing():
            self.relabel_many([eid], label)
            return self.example(eid)

    def relabel_many(self, eids: Iterable[str], label: str) -> int:
        """Move several examples to ``label`` in one save, creating the class if needed.

        Raises ``KeyError`` (changing nothing) if any id is unknown; returns how many moved.
        """
        return self.edit_examples(eids, label=label)

    def delete_example(self, eid: str) -> None:
        """Remove an example (its image file goes once undo no longer needs it)."""
        self.delete_examples([eid])

    def delete_examples(self, eids: Iterable[str]) -> int:
        """Remove several examples in one save; their image files go once undo no longer needs them.

        Raises ``KeyError`` (changing nothing) if any id is unknown; returns how many were removed.
        """
        with self._writing():
            gone = self._examples(eids)
            ids = {e.id for e in gone}
            self.config.examples = [x for x in self.config.examples if x.id not in ids]
            self._save(gone=[e.file for e in gone], label=f"delete {_n(len(gone))}")
            return len(gone)

    def annotate(self, eids: Iterable[str], *, tags: Iterable[str] | None = None, add_tags: Iterable[str] = (),
                 remove_tags: Iterable[str] = (), note: str | None = None) -> int:
        """Set the tags (replacing them, or adding and removing some) and/or the note of examples.

        Raises ``KeyError`` (changing nothing) if any id is unknown and ``ValueError`` for a bad tag
        or a note over ``MAX_NOTE`` characters; returns how many examples were changed.
        """
        return self.edit_examples(eids, tags=tags, add_tags=add_tags, remove_tags=remove_tags, note=note)

    def edit_examples(self, eids: Iterable[str], *, label: str | None = None,  # pylint: disable=too-many-arguments  # one per field
                      tags: Iterable[str] | None = None, add_tags: Iterable[str] = (),
                      remove_tags: Iterable[str] = (), note: str | None = None) -> int:
        """Move examples to ``label`` (creating the class) and/or set their tags (replacing them,
        or adding and removing some) and/or note, all in one change and one undo step.

        Raises ``KeyError`` (changing nothing) if any id is unknown and ``ValueError`` for a bad
        class name or tag or a note over ``MAX_NOTE`` characters; returns how many examples were
        changed.
        """
        add = [check_tag(t) for t in add_tags]
        remove = {check_tag(t) for t in remove_tags}
        base = None if tags is None else [check_tag(t) for t in tags]
        if note is not None and len(note) > MAX_NOTE:
            raise ValueError(f"Note is longer than {MAX_NOTE} characters")
        label = None if label is None else check_class_name(label)
        tagging = base is not None or add or remove or note is not None

        def edited(e: Example) -> Example:
            update: dict[str, Any] = {} if label is None else {"label": label}
            if tagging:
                new = [*(e.tags if base is None else base), *add]
                update["tags"] = tuple(t for t in dict.fromkeys(new) if t not in remove)
            if note is not None:
                update["note"] = note.strip()
            return e.model_copy(update=update)

        with self._writing():
            ids = {e.id for e in self._examples(eids)}
            if label is not None:
                self._ensure_class(label)
            self.config.examples = [edited(e) if e.id in ids else e for e in self.config.examples]
            verb = "tag" if note is None and label is None else "edit"
            self._save(label=f"{verb} {_n(len(ids))}" if tagging else f"move {_n(len(ids))} to {label}")
            return len(ids)

    def find_examples(self, label: str | None = None, tag: str | None = None) -> list[Example]:
        """Examples with ``label`` and/or ``tag`` (all of them if neither is given)."""
        return [e for e in self.config.examples
                if (label is None or e.label == label) and (tag is None or tag in e.tags)]

    def resolve_ids(self, prefixes: Iterable[str], inbox: bool = False) -> list[str]:
        """Full example ids (with ``inbox``, inbox image ids) for ids or unique id prefixes (at
        least 4 characters).

        Raises ``KeyError`` for a prefix that is too short or matches no image or several.
        """
        what = "inbox image" if inbox else "example"
        ids = [x.id for x in (self.config.inbox if inbox else self.config.examples)]
        out = []
        for p in prefixes:
            if len(p) < 4:
                raise exc.ExampleNotFound(f"The {what} id {p!r} is too short: give at least 4 characters", example_id=p)
            hits = [i for i in ids if i.startswith(p)]
            if len(hits) != 1:
                raise exc.ExampleNotFound(f"No {what} {p!r}" if not hits else f"The {what} id {p!r} is ambiguous")
            out.append(hits[0])
        return out

    def _examples(self, eids: Iterable[str]) -> list[Example]:
        """The examples with these ids (each once, in task order); ``KeyError`` names the first unknown."""
        by_id = {e.id: e for e in self.config.examples}
        wanted = list(dict.fromkeys(eids))
        for eid in wanted:
            if eid not in by_id:
                raise exc.ExampleNotFound(f"No example {eid!r}", example_id=eid)
        keep = set(wanted)
        return [e for e in self.config.examples if e.id in keep]

    def image_path(self, eid: str) -> Path:
        """Path of an example's (or inbox image's) file; ``KeyError`` if neither has the id."""
        for x in self.config.inbox:
            if x.id == eid:
                return self.path / x.file
        return self.path / self.example(eid).file

    def warnings(self, counts: dict[str, int]) -> list[str]:
        """Problems worth fixing, given the examples per class: classes that can't be scored or
        predicted, strong imbalance."""
        out = []
        described = {c.name for c in self.config.classes if prompts(c.description)}
        thin = [c for c, n in counts.items() if n < 2 and c not in described]
        if thin:
            out.append(f"Fewer than 2 examples in {', '.join(thin[:3])}{', ...' if len(thin) > 3 else ''}: "
                       "evaluation can't score them")
        sizes = [n for n in counts.values() if n]
        if len(sizes) > 1 and max(sizes) > 5 * min(sizes):
            out.append(f"Classes are unbalanced ({min(sizes)} to {max(sizes)} examples)")
        if len(counts) < 2 and not self.config.examples:
            out.append("Add classes and examples to start")
        elif len(with_examples := [c for c, n in counts.items() if n]) == 1:
            out.append(f"Only one class has examples ({with_examples[0]}): evaluation's accuracy is always 1.0")
        return out

    def summary(self) -> dict[str, Any]:
        """Overview of the task: settings, per-class example counts, the last evaluation and warnings."""
        counts = {c.name: 0 for c in self.config.classes}
        for e in self.config.examples:
            counts[e.label] = counts.get(e.label, 0) + 1
        return {
            "last_evaluation": self.last_evaluation(),
            "warnings": self.warnings(counts),
            "name": self.name,
            "embedder": self.config.embedder,
            "classifier": self.config.classifier,
            "n_classes": len(self.config.classes),
            "n_examples": len(self.config.examples),
            "n_inbox": len(self.config.inbox),
            "counts": counts,
            "created": self.config.created,
            "revision": self.config.revision,
        }


class Workspace:
    """A folder of tasks plus the shared embedding cache."""

    def __init__(self, root: Path | str | None = None, create: bool = True,
                 embedder_ok: SpecCheck | None = None) -> None:
        """The workspace in ``root`` (default ``default_root()``), made if missing and ``create``.

        ``embedder_ok`` checks an embedder spec chosen for a task (by ``create`` or
        ``Task.update_settings``) and raises ``ValueError`` to refuse it: by default
        ``embedder_loads``, so a bad spec fails there rather than at the first prediction."""
        self.root = Path(root).resolve() if root else default_root()
        if create:
            (self.root / "tasks").mkdir(parents=True, exist_ok=True)
        self.blobs = self.root / "blobs"
        self.embedder_ok = embedder_ok or embedder_loads
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def list_tasks(self) -> list[Task]:
        """All readable tasks, sorted by name; a task whose task.json is invalid or can't be read
        (say, for lack of permission) is logged and skipped."""
        tasks = []
        for p in self.task_folders():
            if (p / "task.json").exists():
                try:
                    tasks.append(self.get(p.name))
                except KeyError as e:
                    log.warning("Skipping task folder %s: %s", p, e.args[0])
        return tasks

    def task_folders(self) -> list[Path]:
        """Task folders, sorted by name (leaving out the hidden ones used while creating and deleting)."""
        tasks = self.root / "tasks"
        if not tasks.is_dir():  # a workspace not made yet (see ``create``)
            return []
        return sorted(p for p in tasks.iterdir() if p.is_dir() and not p.name.startswith("."))

    def get(self, name: str) -> Task:
        """Load a task, or refresh the cached one if another process changed it.

        Raises ``KeyError`` if the task does not exist (or was deleted by another process), or if
        its task.json is invalid or can't be read; the message says which.
        """
        with self._lock:
            t = self._tasks.get(name)
            if t is not None and t.path.name != name:  # renamed since (a rename racing this one)
                del self._tasks[name]
                t = None
            if t is None:
                if not NAME_RE.fullmatch(name) or not spelt_as_listed(self.root / "tasks", name):
                    raise exc.TaskNotFound(f"No task {name!r}", task=name)
                p = self.root / "tasks" / name
                try:
                    t = Task(p, *_read_config(p), blobs=self.blobs, embedder_ok=self.embedder_ok)
                except (FileNotFoundError, NotADirectoryError):
                    raise exc.TaskNotFound(f"No task {name!r}", task=name) from None
                except (ValueError, OSError) as e:
                    raise exc.TaskUnreadable(f"Task {name!r} can't be loaded: {os_reason(e)}", task=name) from None
                self._tasks[name] = t
                return t
        try:
            t.refresh()
        except (OSError, ValueError) as e:
            with self._lock:
                self._tasks.pop(name, None)
            if not isinstance(e, FileNotFoundError):
                raise exc.TaskUnreadable(f"Task {name!r} can't be loaded: {os_reason(e)}", task=name) from None
            raise exc.TaskNotFound(f"No task {name!r}", task=name) from None
        return t

    def create(self, name: str, **settings: Any) -> Task:
        """Create a new task folder; an ``embedder`` given in ``settings`` must pass
        ``embedder_ok``. Raises ``ValueError`` for a bad or taken name, an unknown classifier or
        a refused embedder."""
        cfg = TaskConfig(name=name, **settings)
        if cfg.classifier not in CLASSIFIERS:
            raise exc.UnknownClassifier(f"Unknown classifier {cfg.classifier!r}; choose from {sorted(CLASSIFIERS)}")
        if "embedder" in settings:
            self.embedder_ok(cfg.embedder)
        return self.build_task(cfg)

    def build_task(self, cfg: TaskConfig, images: Iterable[tuple[str, bytes | Path]] = ()) -> Task:
        """Make task ``cfg.name`` from ``cfg`` (saved as it is: check it first) and its image
        files, each ``(file, source)`` stored as the task's ``file`` from the image's bytes or
        from an image file of another task (hard-linked where possible, so it is shared).
        ``images`` may be a generator, and raise to stop the build.

        The task is built in a hidden folder and moved into place at the end, so a crash part-way
        leaves no half-made task (``gc`` removes the leftover). The name's folder and the hidden
        one stay locked meanwhile, so gc, or a create of the same name, can't take them for
        leftovers however long it takes. Raises ``ValueError`` for a bad or taken name, and
        whatever stopped the build (after removing what it made)."""
        p = self._reserve(cfg.name)
        build = self.root / "tasks" / f".tmp-{uuid.uuid4().hex}"
        try:
            build.mkdir()
            with _dir_lock(p), _dir_lock(build):
                (build / "images").mkdir()
                create_lock_file(build)
                t = Task(build, cfg, blobs=self.blobs, embedder_ok=self.embedder_ok)
                for file, source in images:
                    t._add_file(file, source)  # pylint: disable=protected-access  # a task only we can see yet
                t._save(record=False)  # pylint: disable=protected-access
                replace_reserved(build, p)
        except BaseException:
            shutil.rmtree(build, ignore_errors=True)
            with contextlib.suppress(OSError):  # only our empty reservation: never a task moved in
                p.rmdir()
            raise
        t.path = p
        _fsync_dir(p.parent)
        with self._lock:
            self._tasks[cfg.name] = t
        return t

    def _reserve(self, name: str) -> Path:
        """Create the empty folder for new task ``name``; ``ValueError`` if the name is bad or taken."""
        if not NAME_RE.fullmatch(name):
            raise ValueError("Task names use letters, digits, '_', '-', '.' (max 64 chars)")
        p = self.root / "tasks" / name
        try:
            p.mkdir()  # atomic: of two concurrent creates, exactly one succeeds
        except FileExistsError:
            if not reclaim(p, STALE_SECONDS):  # unless left by a crash: then take the name back
                raise exc.TaskExists(f"Task {name!r} already exists", task=name) from None
            try:
                p.mkdir()
            except FileExistsError:  # pragma: no cover  # another process took it in between
                raise exc.TaskExists(f"Task {name!r} already exists", task=name) from None
        return p

    def gc(self, dry_run: bool = False) -> dict[str, int]:
        """Clean up what crashes and undo history leave behind: hidden ``.tmp-``/``.trash-``
        folders and task folders without task.json older than ``STALE_SECONDS``, image files no
        version of their task names, and shared copies no task links to. Returns counts (with
        ``dry_run``, of what would go, removing nothing)."""
        out = {"folders": 0, "images": 0, "blobs": 0}
        for p in (self.root / "tasks").iterdir() if (self.root / "tasks").is_dir() else []:
            hidden = p.name.startswith((".tmp-", ".trash-"))
            if p.is_dir() and (hidden or not (p / "task.json").exists()) and (
                    _abandoned(p, STALE_SECONDS) if dry_run else reclaim(p, STALE_SECONDS)):
                out["folders"] += 1
        for t in self.list_tasks():
            with contextlib.suppress(KeyError):  # deleted by another process meanwhile
                out["images"] += t._drop_unused_images(dry_run)  # pylint: disable=protected-access
        out["blobs"] = _blobs.gc_blobs(self.blobs, dry_run=dry_run)
        return out

    def duplicate(self, name: str, new: str, examples: bool = True, inbox: bool = False) -> Task:
        """A copy of task ``name`` as ``new``, with its settings and classes, and its examples and
        inbox if asked; images are shared through the image store, the history starts empty. An
        image file missing from the task (only if it was damaged) is left out with its example,
        as ``export_task`` does."""
        src = self.get(name)
        with src.locked() as cfg:  # a consistent copy, not one mid-change
            copy_cfg = cfg.model_copy(update={
                "name": new, "uid": uuid.uuid4().hex, "revision": 0, "created": _now(),
                "examples": [e for e in cfg.examples if examples and (src.path / e.file).is_file()],
                "inbox": [x for x in cfg.inbox if inbox and (src.path / x.file).is_file()]})
            files = [(e.file, src.path / e.file) for e in [*copy_cfg.examples, *copy_cfg.inbox]]
            return self.build_task(copy_cfg, files)  # linked: the source's bytes aren't read

    def rename(self, old: str, new: str) -> Task:
        """Rename task ``old`` to ``new``; raises ``KeyError`` if ``old`` is missing and
        ``ValueError`` if ``new`` is a bad or taken name. Not recorded for undo."""
        t = self.get(old)
        if new == old:
            return t
        with t._writing():  # pylint: disable=protected-access  # the workspace owns its tasks' folders
            if t.path.name != old:  # a concurrent rename got there first
                raise exc.TaskNotFound(f"No task {old!r}", task=old)
            dest = self._reserve(new)
            src = t.path
            try:
                os.replace(src, dest)  # replaces the empty reserved folder (POSIX)
            except OSError:  # pragma: no cover  # Windows won't replace a folder: free it first
                dest.rmdir()
                os.rename(src, dest)
            t.path = dest
            t.config.name = new
            try:
                t._save(record=False)  # pylint: disable=protected-access
            except BaseException:
                if move_back(dest, src):
                    t.path = src
                else:  # another process took the old name meanwhile: the task opens under the new one
                    log.warning("%s: the rename failed to save, and its name was taken since: now %s", src, dest)
                raise
            _fsync_dir(dest.parent)
        with self._lock:
            self._tasks.pop(old, None)
            self._tasks[new] = t
        return t

    def gc_blobs(self) -> int:
        """Delete shared image copies that no task links to any more; returns how many."""
        return _blobs.gc_blobs(self.blobs)

    def delete(self, name: str) -> None:
        """Delete a task folder and everything in it; raises ``KeyError`` if it is already gone."""
        t = self.get(name)
        try:
            t.destroy()
        finally:
            with self._lock:
                self._tasks.pop(name, None)

    @property
    def cache_path(self) -> Path:
        """Location of the shared embedding cache."""
        return self.root / "cache.sqlite"


_n = functools.partial(plural, what="example")  # "1 example", "3 examples"
