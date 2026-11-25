"""On-disk workspace of few-shot tasks.

Layout::

    <root>/cache.sqlite                 embedding cache shared by all tasks
    <root>/tasks/<name>/task.json       config, classes, example labels
    <root>/tasks/<name>/images/<id>.<ext>

Example ids are the first 16 hex chars of the SHA-256 of the image bytes, so
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
import hashlib
import io
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError, field_validator

from .classifiers import CLASSIFIERS, check_params
from .embedders import DEFAULT_SPEC

try:
    import fcntl
except ImportError:  # pragma: no cover  # Windows: no cross-process lock
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# Example files are always written as images/<id>.<ext>; anything else in task.json was not
# written by us, and following it could read or delete files outside the task.
EXAMPLE_FILE_RE = re.compile(r"^images/[0-9a-f]{16}\.[a-z0-9]{1,10}$")
MAX_CLASS_NAME = 200


def check_class_name(name: str) -> str:
    """Return ``name`` stripped; raise ``ValueError`` if it can't be a class name.

    Class names appear as URL path segments, so ``/``, ``\\``, ``.`` and ``..`` are refused (a
    client would resolve ``.../classes/..`` to the task itself), as are control characters.
    """
    name = name.strip()
    if not name:
        raise ValueError("Class name is empty")
    if name in {".", ".."} or any(ch in "/\\" or ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise ValueError(f"Class name {name!r} can't be '.' or '..' or contain '/', '\\' or control characters")
    if len(name) > MAX_CLASS_NAME:
        raise ValueError(f"Class name is longer than {MAX_CLASS_NAME} characters")
    return name


def default_root() -> Path:
    """Workspace root from ``$FEWSHOT_HOME``, defaulting to ``./fewshot_data``."""
    return Path(os.environ.get("FEWSHOT_HOME", "./fewshot_data")).resolve()


def image_id(data: bytes) -> str:
    """Content id of an image: the first 16 hex chars of its SHA-256."""
    return hashlib.sha256(data).hexdigest()[:16]


def load_image(data: bytes) -> Image.Image:
    """Decode image bytes to an upright RGB image; raises ``ValueError`` if unreadable."""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except UnidentifiedImageError as e:
        raise ValueError("Not a readable image: unrecognised file format") from e
    except Image.DecompressionBombError as e:  # not an OSError subclass
        raise ValueError(f"Not a readable image: {e}") from e
    except OSError as e:
        raise ValueError(f"Not a readable image: {e}") from e
    return ImageOps.exif_transpose(im).convert("RGB")


def fill_template(template: str, name: str) -> str:
    """Text for class ``name`` from a template such as 'a photo of a {}' (underscores become spaces).

    Only the literal ``{}`` is substituted; any other braces are kept as written.
    """
    return template.replace("{}", name.replace("_", " "))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


FileStamp = tuple[int, int, int]  # (inode, mtime_ns, size): changes whenever task.json is replaced


def _stamp(st: os.stat_result) -> FileStamp:
    return (st.st_ino, st.st_mtime_ns, st.st_size)


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``path`` (created if missing) across processes.

    Raises ``FileNotFoundError`` if the folder holding ``path`` is gone.
    """
    if fcntl is None:  # pragma: no cover
        yield
        return
    while True:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            # If the task was deleted (and maybe recreated) while we waited, we hold a lock on a
            # file that is no longer at ``path``; start again on the current one.
            with contextlib.suppress(FileNotFoundError):
                if os.fstat(fd).st_ino == os.stat(path).st_ino:
                    break
        except BaseException:
            os.close(fd)
            raise
        os.close(fd)
    try:
        yield
    finally:
        os.close(fd)  # releases the lock


def _read_config(path: Path) -> tuple[TaskConfig, FileStamp]:
    """Parse ``path/task.json`` and stamp the exact file that was read.

    Raises ``ValueError`` if the file is not a valid config for the task folder ``path``.
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
        raise ValueError(f"{path / 'task.json'} is invalid: name {cfg.name!r} does not match its folder")
    return cfg, stamp


class ClassDef(BaseModel):
    """A class label with an optional text description."""

    name: str
    description: str = ""  # used as a CLIP text prompt when non-empty


class Example(BaseModel):
    """A labeled example image stored in the task folder."""

    id: str
    label: str
    file: str  # images/<id>.<ext>, relative to the task folder
    source: str = ""
    added: str = Field(default_factory=_now)

    @field_validator("file")
    @classmethod
    def _inside_images(cls, v: str) -> str:
        if not EXAMPLE_FILE_RE.fullmatch(v):
            raise ValueError(f"must be images/<id>.<ext>, got {v!r}")
        return v


class TaskConfig(BaseModel):
    """Everything persisted in ``task.json``."""

    name: str
    embedder: str = DEFAULT_SPEC
    classifier: str = "prototype"
    classifier_params: dict[str, Any] = Field(default_factory=dict)
    text_weight: float = 1.0
    reject_below: float | None = None  # similarity under which predictions become "unknown"
    classes: list[ClassDef] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    created: str = Field(default_factory=_now)
    revision: int = 0
    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)  # tells a re-created task from the old one


class Task:
    """A task folder: its config plus example images. Every mutation is saved immediately.

    Mutations are serialised across threads and processes, and start from the latest
    ``task.json`` on disk. A mutation on a task deleted by another process raises ``KeyError``.
    """

    def __init__(self, path: Path, config: TaskConfig, stamp: FileStamp | None = None) -> None:
        self.path = path
        self.config = config
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
                self.config = before.model_copy(deep=True)
                self._depth = 1
                try:
                    yield
                except BaseException:
                    self.config = before
                    self._stamp = None  # re-read on next refresh, in case the save got partway
                    raise
                finally:
                    self._depth = 0

    def _save(self) -> None:
        """Bump the revision and write ``task.json`` atomically and durably. Call inside ``_writing``."""
        self.config.revision += 1
        fd, tmp = tempfile.mkstemp(dir=self.path, prefix=".task.json.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(self.config.model_dump_json(indent=2).encode())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path / "task.json")
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        self._stamp = _stamp(os.stat(self.path / "task.json"))
        if hasattr(os, "O_DIRECTORY"):  # pragma: no branch  # make the rename itself durable (POSIX)
            dfd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)

    @classmethod
    def new(cls, path: Path, config: TaskConfig) -> Task:
        """Write a new task's first ``task.json`` into ``path``, a folder only this caller uses yet."""
        t = cls(path, config)
        t._save()
        return t

    def destroy(self) -> None:
        """Delete the task folder, waiting for any mutation in progress (in any process)."""
        with self._writing():
            shutil.rmtree(self.path)

    # ---- settings ----------------------------------------------------
    def update_settings(self, **kw: Any) -> None:
        """Change embedder/classifier settings; unknown keys raise ``ValueError``."""
        allowed = {"embedder", "classifier", "classifier_params", "text_weight", "reject_below"}
        bad = set(kw) - allowed
        if bad:
            raise ValueError(f"Unknown settings: {sorted(bad)}")
        with self._writing():
            try:
                new = TaskConfig.model_validate({**self.config.model_dump(), **kw})
            except ValidationError as e:
                err = e.errors()[0]
                raise ValueError(f"{'.'.join(map(str, err['loc']))}: {err['msg']}") from None
            if new.reject_below is not None and not -1.0 <= new.reject_below <= 1.0:
                raise ValueError(f"reject_below must be between -1 and 1, got {new.reject_below}")
            if "text_weight" in new.classifier_params:
                raise ValueError("text_weight is a task setting, not a classifier parameter")
            check_params(new.classifier, {"text_weight": new.text_weight, **new.classifier_params})
            for k in kw:
                setattr(self.config, k, getattr(new, k))
            self._save()

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
            for c in self.config.classes:
                if c.name == name:
                    if description is not None:
                        c.description = description
                    self._save()
                    return c
            c = ClassDef(name=check_class_name(name), description=description or "")
            self.config.classes.append(c)
            self._save()
            return c

    def describe_classes(self, template: str, overwrite: bool = False) -> int:
        """Set descriptions from ``template`` ('{}' -> class name); returns how many changed."""
        if "{}" not in template:
            raise ValueError("Template must contain '{}'")
        n = 0
        with self._writing():
            for c in self.config.classes:
                if overwrite or not c.description.strip():
                    c.description = fill_template(template, c.name)
                    n += 1
            self._save()
        return n

    def rename_class(self, old: str, new: str) -> None:
        """Rename a class and relabel its examples."""
        new = check_class_name(new)
        with self._writing():
            if new in self.class_names():
                raise ValueError(f"Class {new!r} already exists")
            c = self._class(old)
            c.name = new
            for e in self.config.examples:
                if e.label == old:
                    e.label = new
            self._save()

    def delete_class(self, name: str) -> int:
        """Delete a class and its examples; returns the number of examples removed."""
        with self._writing():
            self._class(name)
            gone = [e for e in self.config.examples if e.label == name]
            self.config.examples = [e for e in self.config.examples if e.label != name]
            self.config.classes = [c for c in self.config.classes if c.name != name]
            self._save()  # before deleting files, so task.json never names a missing image
            for e in gone:
                (self.path / e.file).unlink(missing_ok=True)
            return len(gone)

    def _class(self, name: str) -> ClassDef:
        for c in self.config.classes:
            if c.name == name:
                return c
        raise KeyError(f"No class {name!r}")

    # ---- examples ----------------------------------------------------
    def add_example(self, data: bytes, label: str, source: str = "") -> Example:
        """Store an image under ``label`` (creating the class); re-adding an image relabels it."""
        load_image(data)  # validates before anything is written
        eid = image_id(data)
        label = label.strip()
        with self._writing():
            if label not in self.class_names():
                self.upsert_class(label)
            for e in self.config.examples:
                if e.id == eid:
                    e.label = label
                    self._save()
                    return e
            fmt = (Image.open(io.BytesIO(data)).format or "png").lower()
            ext = {"jpeg": "jpg"}.get(fmt, fmt)
            rel = f"images/{eid}.{ext}"
            (self.path / "images").mkdir(exist_ok=True)
            (self.path / rel).write_bytes(data)
            ex = Example(id=eid, label=label, file=rel, source=source)
            self.config.examples.append(ex)
            self._save()
            return ex

    def add_examples_from_dir(self, folder: Path) -> int:
        """Import a folder laid out as ``folder/<class>/<image>``."""
        n = 0
        for cls_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            for f in sorted(cls_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    self.add_example(f.read_bytes(), cls_dir.name, f.name)
                    n += 1
        return n

    def example(self, eid: str) -> Example:
        """Look up an example by id; raises ``KeyError`` if missing."""
        for e in self.config.examples:
            if e.id == eid:
                return e
        raise KeyError(f"No example {eid!r}")

    def relabel(self, eid: str, label: str) -> Example:
        """Move an example to ``label``, creating the class if needed."""
        with self._writing():
            e = self.example(eid)
            label = label.strip()
            if label not in self.class_names():
                self.upsert_class(label)
            e.label = label
            self._save()
            return e

    def delete_example(self, eid: str) -> None:
        """Remove an example and its image file."""
        with self._writing():
            e = self.example(eid)
            self.config.examples = [x for x in self.config.examples if x.id != eid]
            self._save()
            (self.path / e.file).unlink(missing_ok=True)

    def image_path(self, eid: str) -> Path:
        """Path of an example's image file."""
        return self.path / self.example(eid).file

    def summary(self) -> dict[str, Any]:
        """Overview of the task: settings and per-class example counts."""
        counts = {c.name: 0 for c in self.config.classes}
        for e in self.config.examples:
            counts[e.label] = counts.get(e.label, 0) + 1
        return {
            "name": self.name,
            "embedder": self.config.embedder,
            "classifier": self.config.classifier,
            "n_classes": len(self.config.classes),
            "n_examples": len(self.config.examples),
            "counts": counts,
            "created": self.config.created,
        }


class Workspace:
    """A folder of tasks plus the shared embedding cache."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root).resolve() if root else default_root()
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def list_tasks(self) -> list[Task]:
        """All readable tasks, sorted by name; a task whose task.json is invalid is logged and skipped."""
        tasks = []
        for p in sorted((self.root / "tasks").iterdir()):
            if (p / "task.json").exists():
                try:
                    tasks.append(self.get(p.name))
                except KeyError as e:
                    log.warning("Skipping task folder %s: %s", p, e.args[0])
        return tasks

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
                    t = Task(p, *_read_config(p))
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
        if not NAME_RE.fullmatch(name):
            raise ValueError("Task names use letters, digits, '_', '-', '.' (max 64 chars)")
        p = self.root / "tasks" / name
        cfg = TaskConfig(name=name, **settings)
        if cfg.classifier not in CLASSIFIERS:
            raise ValueError(f"Unknown classifier {cfg.classifier!r}")
        try:
            p.mkdir()  # atomic: of two concurrent creates, exactly one succeeds
        except FileExistsError:
            raise ValueError(f"Task {name!r} already exists") from None
        (p / "images").mkdir()
        t = Task.new(p, cfg)
        with self._lock:
            self._tasks[name] = t
        return t

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
