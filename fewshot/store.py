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

Example ids are the first 32 hex chars (128 bits) of the SHA-256 of the image
bytes (16 in tasks created before that; both are accepted), so
adding the same image twice is a no-op (or a relabel), and cached embeddings
are reused across tasks.

Several processes (say ``fewshot serve`` and the CLI) can share a workspace.
Each change to a task holds an exclusive lock on ``<task>/.lock``, re-reads
``task.json`` if another process changed it, and replaces the file atomically,
so no process overwrites another's changes. Reads pick up other processes'
changes on the next ``Workspace.get``. The file lock needs ``fcntl`` (Linux,
macOS); elsewhere only threads within one process are serialised.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .classifiers import CLASSIFIERS, check_params
from .embedders import DEFAULT_SPEC
from .fsutil import FileStamp
from .fsutil import create_exclusive as _create_exclusive
from .fsutil import file_lock as _file_lock
from .fsutil import stamp as _stamp
from .fsutil import write_atomic as _write_atomic
from .history import TaskHistory
from .images import IMAGE_EXTS, image_ext, image_id
from .inbox import TaskInbox
from .text import check_class_name, check_tag, check_template, fill_template, prompts

log = logging.getLogger(__name__)

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# Example files are always written as images/<id>.<ext>; anything else in task.json was not
# written by us, and following it could read or delete files outside the task.
EXAMPLE_FILE_RE = re.compile(r"^images/[0-9a-f]{16}(?:[0-9a-f]{16})?\.[a-z0-9]{1,10}$")
# Images saved per task.json write by Task.add_examples.
ADD_CHUNK = 256
MAX_NOTE = 2000
# A task folder without task.json (or a .tmp-/.trash- folder) this old is left over from a crash.
STALE_SECONDS = 600
SETTINGS = {"embedder", "classifier", "classifier_params", "text_weight", "reject_below",
            "reject_below_zero_shot", "flip", "temperature"}


def default_root() -> Path:
    """Workspace root from ``$FEWSHOT_HOME``, defaulting to ``./fewshot_data``."""
    return Path(os.environ.get("FEWSHOT_HOME", "./fewshot_data")).resolve()


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


def _copy_config(cfg: TaskConfig) -> TaskConfig:
    """A copy of ``cfg`` that edits can't reach back through. Classes and examples are immutable,
    so new lists of the same objects are enough; only the params dict is copied. This is far
    cheaper than a deep copy, which matters because every write to a task makes one."""
    return cfg.model_copy(update={
        "classifier_params": copy.deepcopy(cfg.classifier_params),
        "classes": list(cfg.classes),
        "examples": list(cfg.examples),
        "inbox": list(cfg.inbox),
    })


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_config(path: Path) -> tuple[TaskConfig, FileStamp]:
    """Parse ``path/task.json`` and stamp the exact file that was read.

    The folder's name is the task's name: a task.json naming another (a folder renamed by hand, or
    a rename interrupted after moving the folder) is taken under the folder's name, and the next
    save writes that. Raises ``ValueError`` if the file is not a valid config.
    """
    with open(path / "task.json", "rb") as f:
        stamp = _stamp(os.fstat(f.fileno()))
        data = f.read()
    try:
        cfg = TaskConfig.model_validate_json(data)
    except ValidationError as e:
        err = e.errors()[0]
        where = ".".join(map(str, err["loc"]))
        raise ValueError(f"{path / 'task.json'} is invalid: {where + ': ' if where else ''}{err['msg']}") from None
    if cfg.name != path.name:
        log.warning("%s names task %r; using its folder name %r", path / "task.json", cfg.name, path.name)
        cfg = cfg.model_copy(update={"name": path.name})
    return cfg, stamp


class ClassDef(BaseModel):
    """A class label with an optional text description. Immutable: edits replace the object."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""  # used as a CLIP text prompt when non-empty
    # Similarity under which a prediction of this class becomes "unknown"; overrides the task's
    # reject_below (classes differ in how tightly their images cluster). Not used zero-shot.
    reject_below: float | None = None


class ImageRef(BaseModel):
    """An image stored in the task folder, named by its content id. Immutable: edits replace it."""

    model_config = ConfigDict(frozen=True)

    id: str
    file: str  # images/<id>.<ext>, relative to the task folder
    source: str = ""
    added: str = Field(default_factory=_now)

    @field_validator("file")
    @classmethod
    def _inside_images(cls, v: str) -> str:
        if not EXAMPLE_FILE_RE.fullmatch(v):
            raise ValueError(f"must be images/<id>.<ext>, got {v!r}")
        return v

    @model_validator(mode="after")
    def _named_by_id(self) -> ImageRef:
        # The file name is how the shared image store finds the content, so it must be the id.
        if self.file.split("/", 1)[1].rsplit(".", 1)[0] != self.id:
            raise ValueError(f"file {self.file!r} is not named by the example id {self.id!r}")
        return self


class InboxItem(ImageRef):
    """An unlabelled image waiting in the task's inbox to be labelled (or dismissed)."""


class Example(ImageRef):
    """A labeled example image stored in the task folder. Immutable: edits replace the object."""

    label: str
    tags: tuple[str, ...] = ()
    note: str = ""

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(check_tag(t) for t in v))

    @field_validator("note")
    @classmethod
    def _short_note(cls, v: str) -> str:
        if len(v) > MAX_NOTE:
            raise ValueError(f"note is longer than {MAX_NOTE} characters")
        return v


class TaskConfig(BaseModel):
    """Everything persisted in ``task.json``."""

    name: str
    embedder: str = DEFAULT_SPEC
    classifier: str = "prototype"
    classifier_params: dict[str, Any] = Field(default_factory=dict)
    text_weight: float = 1.0
    reject_below: float | None = None  # similarity under which predictions become "unknown"
    # The same for zero-shot predictions (no example images), whose image-to-text similarities are
    # far lower (about 0.2-0.35 for CLIP) than image-to-image ones; None never rejects them.
    reject_below_zero_shot: float | None = None
    flip: bool = False  # average each image's embedding with its mirror image's (test-time augmentation)
    temperature: float | None = None  # confidence calibration fitted by evaluation (None: 1, uncalibrated)
    classes: list[ClassDef] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    inbox: list[InboxItem] = Field(default_factory=list)  # unlabelled images waiting to be labelled
    created: str = Field(default_factory=_now)
    revision: int = 0
    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)  # tells a re-created task from the old one

    def files(self) -> set[str]:
        """Every image file this version uses: examples' and the inbox's."""
        return {e.file for e in self.examples} | {i.file for i in self.inbox}


def check_settings(cfg: TaskConfig) -> None:
    """Raise ``ValueError`` unless ``cfg``'s classifier settings and thresholds are usable."""
    for key in ("reject_below", "reject_below_zero_shot"):
        v = getattr(cfg, key)
        if v is not None and not -1.0 <= v <= 1.0:
            raise ValueError(f"{key} must be between -1 and 1, got {v}")
    if cfg.temperature is not None and not 0.01 <= cfg.temperature <= 100:
        raise ValueError(f"temperature must be between 0.01 and 100, got {cfg.temperature}")
    if "text_weight" in cfg.classifier_params:
        raise ValueError("text_weight is a task setting, not a classifier parameter")
    check_params(cfg.classifier, {"text_weight": cfg.text_weight, **cfg.classifier_params})


class Task(TaskHistory, TaskInbox):  # pylint: disable=too-many-public-methods  # the whole editing API of a task
    """A task folder: its config plus example images. Every mutation is saved immediately.

    Mutations are serialised across threads and processes, and start from the latest
    ``task.json`` on disk. A mutation on a task deleted by another process raises ``KeyError``.
    """

    def __init__(self, path: Path, config: TaskConfig, stamp: FileStamp | None = None,
                 blobs: Path | None = None) -> None:
        self.path = path
        self.config = config
        self.blobs = blobs  # the workspace's shared image store, if any
        self._lock = threading.RLock()
        self._stamp = stamp  # of the task.json that ``config`` matches
        self._depth = 0  # nesting of ``_writing`` in the thread holding ``_lock``

    # ---- persistence -------------------------------------------------
    @property
    def name(self) -> str:
        """The task name."""
        return self.config.name

    def refresh(self) -> bool:
        """Reload ``task.json`` if it changed on disk; returns whether it did.

        Raises ``FileNotFoundError`` if the task was deleted.
        """
        with self._lock:
            if self._stamp == _stamp(os.stat(self.path / "task.json")):
                return False
            self.config, self._stamp = _read_config(self.path)
            return True

    @contextlib.contextmanager
    def _writing(self) -> Iterator[None]:
        """Context for a mutation: exclusive across threads and processes, starting from the
        latest saved config.

        The mutation edits a fresh copy of ``config``, so a reader holding the previous config
        object never sees it change mid-read. If the mutation fails, the copy is dropped.
        """
        with self._lock:
            if self._depth:  # nested in a mutation that already holds the locks
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            with contextlib.ExitStack() as stack:
                try:
                    stack.enter_context(_file_lock(self.path / ".lock"))
                    self.refresh()
                except FileNotFoundError as e:  # deleted by another process
                    raise KeyError(f"No task {self.name!r}") from e
                before = self.config
                self.config = _copy_config(before)
                self._depth = 1
                try:
                    yield
                except BaseException:
                    self.config = before
                    self._stamp = None  # re-read on next refresh, in case the save got partway
                    raise
                finally:
                    self._depth = 0

    def _save(self, gone: Iterable[str] = (), record: bool = True, label: str = "", keep_redo: bool = False) -> None:
        """Bump the revision and write ``task.json`` atomically and durably. Call inside ``_writing``.

        With ``record``, the version being replaced is kept in ``history/`` for ``undo``, with
        ``label`` describing the change, and (unless ``keep_redo``) undone changes can no longer
        be redone. ``gone`` names the image files this change stopped using, which are deleted
        once no kept version uses them.
        """
        old_rev = self.config.revision
        self.config.revision += 1
        made: list[Path] = []
        fd, tmp = tempfile.mkstemp(dir=self.path, prefix=".task.json.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(self.config.model_dump_json(indent=2).encode())
                f.flush()
                os.fsync(f.fileno())
            if record and (self.path / "task.json").exists():
                made = self._snapshot(old_rev, sorted(set(gone)), label)
            os.replace(tmp, self.path / "task.json")
        except BaseException:
            for leftover in [Path(tmp), *made]:  # a version that never got replaced is no step to undo
                with contextlib.suppress(OSError):
                    os.unlink(leftover)
            raise
        self._stamp = _stamp(os.stat(self.path / "task.json"))
        if record:
            if not keep_redo:
                self._clear_redo()
            self._prune_history()
        if hasattr(os, "O_DIRECTORY"):  # pragma: no branch  # make the rename itself durable (POSIX)
            dfd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)

    # ---- image files -------------------------------------------------
    def _blob(self, file: str) -> Path | None:
        """The shared-store file for ``images/<name>``, or None without a store."""
        name = file.rsplit("/", 1)[-1]
        # Ids from before 128-bit ids are too short to rule out a crafted collision, which in a
        # shared store would let one task's image stand in for another's: those stay task-local.
        if self.blobs is None or len(name.split(".", 1)[0]) != 32:
            return None
        return self.blobs / name[:2] / name

    def _store_image(self, file: str, data: bytes) -> bool:
        """Write ``data`` (whose id the name of ``file`` carries) as the task's ``file``: a hard link
        to the workspace's copy, made if needed, or a plain copy where links aren't possible.
        Returns whether this call created the file (False if it was there already)."""
        dest = self.path / file
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():  # kept for undo, or re-added: same content, since the name is its hash
            return False
        blob = self._blob(file)
        if blob is not None:
            made = False
            try:
                blob.parent.mkdir(parents=True, exist_ok=True)
                made = _create_exclusive(blob, data)
                os.link(blob, dest)
                return True
            except FileExistsError:  # another writer stored this file in this task first
                return False
            except OSError:  # no hard links here, or the blob vanished under us: copy instead
                if made:  # a blob nothing can link to would only waste space
                    blob.unlink(missing_ok=True)
        _write_atomic(dest, data)
        return True

    def _drop_file(self, file: str) -> None:
        """Delete a task image file, and the shared copy once no task links to it."""
        (self.path / file).unlink(missing_ok=True)
        blob = self._blob(file)
        if blob is not None:
            with contextlib.suppress(FileNotFoundError):
                if blob.stat().st_nlink == 1:
                    blob.unlink()

    @classmethod
    def new(cls, path: Path, config: TaskConfig, blobs: Path | None = None) -> Task:
        """Write a new task's first ``task.json`` into ``path``, a folder only this caller uses yet."""
        t = cls(path, config, blobs=blobs)
        t._save()
        return t

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
            shutil.rmtree(trash)
            for f in files:  # the shared copies no other task links to
                blob = self._blob(f)
                with contextlib.suppress(FileNotFoundError):
                    if blob is not None and blob.stat().st_nlink == 1:
                        blob.unlink()

    # ---- settings ----------------------------------------------------
    def update_settings(self, **kw: Any) -> None:
        """Change embedder/classifier settings (see ``SETTINGS``); unknown keys raise ``ValueError``."""
        bad = set(kw) - SETTINGS
        if bad:
            raise ValueError(f"Unknown settings: {sorted(bad)}")
        with self._writing():
            try:
                new = TaskConfig.model_validate({**self.config.model_dump(), **kw})
            except ValidationError as e:
                err = e.errors()[0]
                raise ValueError(f"{'.'.join(map(str, err['loc']))}: {err['msg']}") from None
            check_settings(new)
            for k in kw:
                setattr(self.config, k, getattr(new, k))
            self._save(label="change settings")

    # ---- classes -----------------------------------------------------
    def class_names(self) -> list[str]:
        """Class names in definition order."""
        return [c.name for c in self.config.classes]

    def upsert_class(self, name: str, description: str | None = None) -> ClassDef:
        """Create a class, or update its description if it exists (``None`` keeps it).

        Raises ``ValueError`` for a name ``check_class_name`` refuses.
        """
        name = name.strip()
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
                    classes[i] = c.model_copy(update={"description": fill_template(template, c.name)})
                    n += 1
            self._save(label="describe classes")
        return n

    def rename_class(self, old: str, new: str) -> None:
        """Rename a class and relabel its examples."""
        new = check_class_name(new)
        with self._writing():
            if new in self.class_names():
                raise ValueError(f"Class {new!r} already exists")
            self._class(old)
            cfg = self.config
            cfg.classes = [c.model_copy(update={"name": new}) if c.name == old else c for c in cfg.classes]
            cfg.examples = [e.model_copy(update={"label": new}) if e.label == old else e for e in cfg.examples]
            self._save(label=f"rename class {old} to {new}")

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
        for an unknown class and ``ValueError`` for a value outside -1..1; then nothing changes."""
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
            self._save(label=f"set the threshold of {next(iter(thresholds))}" if n == 1
                       else f"set thresholds of {n} classes")

    def _class(self, name: str) -> ClassDef:
        for c in self.config.classes:
            if c.name == name:
                return c
        raise KeyError(f"No class {name!r}")

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
                     chunk: int = ADD_CHUNK) -> list[Example | ValueError]:
        """Store ``(image bytes, label, source)`` items, like ``add_example`` but with one save per
        ``chunk`` items instead of one per image.

        Returns one entry per item: its ``Example``, or the ``ValueError`` that kept it out (an
        unreadable image or a bad class name). Images are decoded to check them before the task is
        locked, so a long import doesn't block other writers for its whole length; ``items`` may be
        a generator, and only one chunk of image bytes is held at a time.
        """
        results: list[Example | ValueError] = []
        batch: list[tuple[bytes, str, str]] = []
        for item in items:
            batch.append(item)
            if len(batch) >= chunk:
                results += self._add_batch(batch, record=not results)
                batch = []
        if batch:
            results += self._add_batch(batch, record=not results)
        return results

    def _add_batch(self, batch: list[tuple[bytes, str, str]], record: bool = True) -> list[Example | ValueError]:
        """Add one chunk. Only the first chunk of an ``add_examples`` call keeps the version before
        it for undo (``record``), so one undo takes back the whole call."""
        checked = [_check_item(*item) for item in batch]  # decoding happens here, before locking
        out: list[Example | ValueError] = []
        created: list[str] = []
        with self._writing():
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
                if i is not None:  # already stored: just relabel
                    ex = examples[i] = examples[i].model_copy(update={"label": c.label})
                else:
                    ex = Example(id=c.id, label=c.label, file=f"images/{c.id}.{c.ext}", source=c.source)
                    if self._store_image(ex.file, c.data):
                        created.append(ex.file)
                    at[c.id] = len(examples)
                    examples.append(ex)
                out.append(ex)
            if any(not isinstance(r, ValueError) for r in out):
                self._save_new(created, record, _add_label(out))
        return out

    def _save_new(self, created: list[str], record: bool, label: str) -> None:
        """``_save`` after storing new image files; if the save fails, the files ``created`` (named
        by no saved version) are removed again."""
        try:
            self._save(record=record, label=label)
        except BaseException:
            for f in created:
                self._drop_file(f)
            raise

    def _ensure_class(self, label: str) -> None:
        """Add class ``label`` (already checked) if missing, without saving."""
        if label not in self.class_names():
            self.config.classes.append(ClassDef(name=label))

    def add_examples_from_dir(self, folder: Path) -> int:
        """Import a folder laid out as ``folder/<class>/<image>``; returns how many images were added.

        Raises ``ValueError`` for the first image that can't be read (after adding the others).
        """
        items = ((f.read_bytes(), d.name, f.name)
                 for d in sorted(p for p in folder.iterdir() if p.is_dir())
                 for f in sorted(d.iterdir()) if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        results = self.add_examples(items)
        for r in results:
            if isinstance(r, ValueError):
                raise r
        return len(results)

    def example(self, eid: str) -> Example:
        """Look up an example by id; raises ``KeyError`` if missing."""
        for e in self.config.examples:
            if e.id == eid:
                return e
        raise KeyError(f"No example {eid!r}")

    def relabel(self, eid: str, label: str) -> Example:
        """Move an example to ``label``, creating the class if needed."""
        with self._writing():
            self.relabel_many([eid], label)
            return self.example(eid)

    def relabel_many(self, eids: Iterable[str], label: str) -> int:
        """Move several examples to ``label`` in one save, creating the class if needed.

        Raises ``KeyError`` (changing nothing) if any id is unknown; returns how many moved.
        """
        with self._writing():
            ids = {e.id for e in self._examples(eids)}
            label = check_class_name(label)
            self._ensure_class(label)
            self.config.examples = [e.model_copy(update={"label": label}) if e.id in ids else e
                                    for e in self.config.examples]
            self._save(label=f"move {_n(len(ids))} to {label}")
            return len(ids)

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
        add = [check_tag(t) for t in add_tags]
        remove = {check_tag(t) for t in remove_tags}
        base = None if tags is None else [check_tag(t) for t in tags]
        if note is not None and len(note) > MAX_NOTE:
            raise ValueError(f"Note is longer than {MAX_NOTE} characters")
        with self._writing():
            ids = {e.id for e in self._examples(eids)}
            examples = self.config.examples
            for i, e in enumerate(examples):
                if e.id in ids:
                    new = [*(e.tags if base is None else base), *add]
                    update: dict[str, Any] = {"tags": tuple(t for t in dict.fromkeys(new) if t not in remove)}
                    if note is not None:
                        update["note"] = note.strip()
                    examples[i] = e.model_copy(update=update)
            self._save(label=f"tag {_n(len(ids))}" if note is None else f"edit {_n(len(ids))}")
            return len(ids)

    def find_examples(self, label: str | None = None, tag: str | None = None) -> list[Example]:
        """Examples with ``label`` and/or ``tag`` (all of them if neither is given)."""
        return [e for e in self.config.examples
                if (label is None or e.label == label) and (tag is None or tag in e.tags)]

    def resolve_ids(self, prefixes: Iterable[str]) -> list[str]:
        """Full example ids for ids or unique id prefixes (at least 4 characters).

        Raises ``KeyError`` for a prefix that matches no example or several.
        """
        ids = [e.id for e in self.config.examples]
        out = []
        for p in prefixes:
            if len(p) < 4:
                raise KeyError(f"Example id {p!r} is too short: give at least 4 characters")
            hits = [i for i in ids if i.startswith(p)]
            if len(hits) != 1:
                raise KeyError(f"No example {p!r}" if not hits else f"Example id {p!r} is ambiguous")
            out.append(hits[0])
        return out

    def _examples(self, eids: Iterable[str]) -> list[Example]:
        """The examples with these ids (each once, in task order); ``KeyError`` names the first unknown."""
        by_id = {e.id: e for e in self.config.examples}
        wanted = list(dict.fromkeys(eids))
        for eid in wanted:
            if eid not in by_id:
                raise KeyError(f"No example {eid!r}")
        keep = set(wanted)
        return [e for e in self.config.examples if e.id in keep]

    def image_path(self, eid: str) -> Path:
        """Path of an example's (or inbox image's) file; ``KeyError`` if neither has the id."""
        for x in self.config.inbox:
            if x.id == eid:
                return self.path / x.file
        return self.path / self.example(eid).file

    def record_evaluation(self, report: dict[str, Any]) -> None:
        """Remember the headline of an evaluation (in ``eval.json``, outside undo history) for the
        dashboard: when, at which revision, and how well."""
        keep = ("revision", "method", "accuracy", "macro_f1", "n_scored", "n_examples")
        _write_atomic(self.path / "eval.json", json.dumps({"at": _now(), **{k: report.get(k) for k in keep}}).encode())

    def last_evaluation(self) -> dict[str, Any] | None:
        """What ``record_evaluation`` last stored, or None."""
        try:
            data = json.loads((self.path / "eval.json").read_text())
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

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

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root).resolve() if root else default_root()
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)
        self.blobs = self.root / "blobs"
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def list_tasks(self) -> list[Task]:
        """All readable tasks, sorted by name; a task whose task.json is invalid is logged and skipped."""
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
        return sorted(p for p in (self.root / "tasks").iterdir() if p.is_dir() and not p.name.startswith("."))

    def get(self, name: str) -> Task:
        """Load a task, or refresh the cached one if another process changed it.

        Raises ``KeyError`` if the task does not exist (or was deleted by another process), or if
        its task.json is invalid; the message says which.
        """
        with self._lock:
            t = self._tasks.get(name)
            if t is None:
                if not NAME_RE.fullmatch(name):
                    raise KeyError(f"No task {name!r}")
                p = self.root / "tasks" / name
                try:
                    t = Task(p, *_read_config(p), blobs=self.blobs)
                except (FileNotFoundError, NotADirectoryError):
                    raise KeyError(f"No task {name!r}") from None
                except ValueError as e:
                    raise KeyError(f"Task {name!r} can't be loaded: {e}") from None
                self._tasks[name] = t
                return t
        try:
            t.refresh()
        except (FileNotFoundError, ValueError) as e:
            with self._lock:
                self._tasks.pop(name, None)
            if isinstance(e, ValueError):
                raise KeyError(f"Task {name!r} can't be loaded: {e}") from None
            raise KeyError(f"No task {name!r}") from None
        return t

    def create(self, name: str, **settings: Any) -> Task:
        """Create a new task folder; raises ``ValueError`` for bad or taken names."""
        cfg = TaskConfig(name=name, **settings)
        if cfg.classifier not in CLASSIFIERS:
            raise ValueError(f"Unknown classifier {cfg.classifier!r}")
        p = self._reserve(name)
        (p / "images").mkdir()
        t = Task.new(p, cfg, self.blobs)
        with self._lock:
            self._tasks[name] = t
        return t

    def _reserve(self, name: str) -> Path:
        """Create the empty folder for new task ``name``; ``ValueError`` if the name is bad or taken."""
        if not NAME_RE.fullmatch(name):
            raise ValueError("Task names use letters, digits, '_', '-', '.' (max 64 chars)")
        p = self.root / "tasks" / name
        try:
            p.mkdir()  # atomic: of two concurrent creates, exactly one succeeds
        except FileExistsError:
            if not _abandoned(p):
                raise ValueError(f"Task {name!r} already exists") from None
            shutil.rmtree(p, ignore_errors=True)  # left by a crash: take the name back
            return self._reserve_again(name)
        return p

    def _reserve_again(self, name: str) -> Path:
        p = self.root / "tasks" / name
        try:
            p.mkdir()
        except FileExistsError:  # pragma: no cover  # another process took it in between
            raise ValueError(f"Task {name!r} already exists") from None
        return p

    def gc(self) -> dict[str, int]:
        """Clean up what crashes and undo history leave behind: hidden ``.tmp-``/``.trash-``
        folders and task folders without task.json older than ``STALE_SECONDS``, image files no
        version of their task names, and shared copies no task links to. Returns counts."""
        out = {"folders": 0, "images": 0, "blobs": 0}
        for p in (self.root / "tasks").iterdir():
            hidden = p.name.startswith((".tmp-", ".trash-"))
            if p.is_dir() and (hidden or not (p / "task.json").exists()) and _abandoned(p):
                shutil.rmtree(p, ignore_errors=True)
                out["folders"] += 1
        for t in self.list_tasks():
            out["images"] += t._drop_unused_images()  # pylint: disable=protected-access
        out["blobs"] = self.gc_blobs()
        return out

    def duplicate(self, name: str, new: str, examples: bool = True, inbox: bool = False) -> Task:
        """A copy of task ``name`` as ``new``, with its settings and classes, and its examples and
        inbox if asked; images are shared through the image store, the history starts empty."""
        src = self.get(name)
        with src._writing():  # pylint: disable=protected-access  # a consistent copy, not one mid-change
            cfg = src.config
            copy_cfg = cfg.model_copy(update={
                "name": new, "uid": uuid.uuid4().hex, "revision": 0, "created": _now(),
                "examples": list(cfg.examples) if examples else [], "inbox": list(cfg.inbox) if inbox else []})
            p = self._reserve(new)
            try:
                (p / "images").mkdir()
                t = Task(p, copy_cfg, blobs=self.blobs)
                for e in [*copy_cfg.examples, *copy_cfg.inbox]:
                    t._store_image(e.file, (src.path / e.file).read_bytes())  # pylint: disable=protected-access
                t._save(record=False)  # pylint: disable=protected-access
            except BaseException:
                shutil.rmtree(p, ignore_errors=True)
                raise
        with self._lock:
            self._tasks[new] = t
        return t

    def rename(self, old: str, new: str) -> Task:
        """Rename task ``old`` to ``new``; raises ``KeyError`` if ``old`` is missing and
        ``ValueError`` if ``new`` is a bad or taken name. Not recorded for undo."""
        t = self.get(old)
        if new == old:
            return t
        with t._writing():  # pylint: disable=protected-access  # the workspace owns its tasks' folders
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
                os.rename(dest, src)
                t.path = src
                raise
        with self._lock:
            self._tasks.pop(old, None)
            self._tasks[new] = t
        return t

    def gc_blobs(self) -> int:
        """Delete shared image copies that no task links to any more; returns how many."""
        n = 0
        for f in self.blobs.glob("*/*") if self.blobs.is_dir() else []:
            with contextlib.suppress(FileNotFoundError):
                if f.is_file() and f.stat().st_nlink == 1:
                    f.unlink()
                    n += 1
        return n

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


def _abandoned(folder: Path) -> bool:
    """Whether ``folder`` is a leftover of a crash: no task.json (or hidden) and untouched for
    ``STALE_SECONDS``, so no create or import in progress can own it."""
    try:
        age = datetime.now(timezone.utc).timestamp() - folder.stat().st_mtime
    except FileNotFoundError:
        return False
    return age > STALE_SECONDS and (folder.name.startswith(".") or not (folder / "task.json").exists())


def _n(count: int) -> str:
    """ "1 example", "3 examples"."""
    return f"{count} example{'' if count == 1 else 's'}"


def _add_label(results: list[Example | ValueError]) -> str:
    """Undo description of an add: "add 3 examples to red" (classes listed, at most three)."""
    labels = list(dict.fromkeys(r.label for r in results if isinstance(r, Example)))
    where = ", ".join(labels[:3]) + (", ..." if len(labels) > 3 else "")
    return f"add {_n(sum(isinstance(r, Example) for r in results))} to {where}"
