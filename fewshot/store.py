"""On-disk workspace of few-shot tasks.

Layout::

    <root>/cache.sqlite                 embedding cache shared by all tasks
    <root>/tasks/<name>/task.json       config, classes, example labels
    <root>/tasks/<name>/images/<id>.<ext>

Example ids are the first 16 hex chars of the SHA-256 of the image bytes, so
adding the same image twice is a no-op (or a relabel), and cached embeddings
are reused across tasks.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from .classifiers import CLASSIFIERS
from .embedders import DEFAULT_SPEC

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def default_root() -> Path:
    return Path(os.environ.get("FEWSHOT_HOME", "./fewshot_data")).resolve()


def image_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def load_image(data: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Not a readable image: {e}") from e
    return ImageOps.exif_transpose(im).convert("RGB")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ClassDef(BaseModel):
    name: str
    description: str = ""  # used as a CLIP text prompt when non-empty


class Example(BaseModel):
    id: str
    label: str
    file: str
    source: str = ""
    added: str = Field(default_factory=_now)


class TaskConfig(BaseModel):
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


class Task:
    def __init__(self, path: Path, config: TaskConfig) -> None:
        self.path = path
        self.config = config
        self._lock = threading.RLock()

    # ---- persistence -------------------------------------------------
    @property
    def name(self) -> str:
        return self.config.name

    def save(self) -> None:
        with self._lock:
            self.config.revision += 1
            tmp = self.path / "task.json.tmp"
            tmp.write_text(self.config.model_dump_json(indent=2))
            tmp.replace(self.path / "task.json")

    # ---- settings ----------------------------------------------------
    def update_settings(self, **kw: Any) -> None:
        allowed = {"embedder", "classifier", "classifier_params", "text_weight", "reject_below"}
        bad = set(kw) - allowed
        if bad:
            raise ValueError(f"Unknown settings: {sorted(bad)}")
        if "classifier" in kw and kw["classifier"] not in CLASSIFIERS:
            raise ValueError(f"Unknown classifier {kw['classifier']!r}")
        with self._lock:
            for k, v in kw.items():
                setattr(self.config, k, v)
            self.save()

    # ---- classes -----------------------------------------------------
    def class_names(self) -> list[str]:
        return [c.name for c in self.config.classes]

    def upsert_class(self, name: str, description: str | None = None) -> ClassDef:
        name = name.strip()
        if not name:
            raise ValueError("Class name is empty")
        with self._lock:
            for c in self.config.classes:
                if c.name == name:
                    if description is not None:
                        c.description = description
                    self.save()
                    return c
            c = ClassDef(name=name, description=description or "")
            self.config.classes.append(c)
            self.save()
            return c

    def describe_classes(self, template: str, overwrite: bool = False) -> int:
        """Set descriptions from ``template`` ('{}' -> class name); returns how many changed."""
        if "{}" not in template:
            raise ValueError("Template must contain '{}'")
        n = 0
        with self._lock:
            for c in self.config.classes:
                if overwrite or not c.description.strip():
                    c.description = template.format(c.name.replace("_", " "))
                    n += 1
            self.save()
        return n

    def rename_class(self, old: str, new: str) -> None:
        new = new.strip()
        with self._lock:
            if new in self.class_names():
                raise ValueError(f"Class {new!r} already exists")
            c = self._class(old)
            c.name = new
            for e in self.config.examples:
                if e.label == old:
                    e.label = new
            self.save()

    def delete_class(self, name: str) -> int:
        """Delete a class and its examples; returns the number of examples removed."""
        with self._lock:
            self._class(name)
            gone = [e for e in self.config.examples if e.label == name]
            for e in gone:
                (self.path / e.file).unlink(missing_ok=True)
            self.config.examples = [e for e in self.config.examples if e.label != name]
            self.config.classes = [c for c in self.config.classes if c.name != name]
            self.save()
            return len(gone)

    def _class(self, name: str) -> ClassDef:
        for c in self.config.classes:
            if c.name == name:
                return c
        raise KeyError(f"No class {name!r}")

    # ---- examples ----------------------------------------------------
    def add_example(self, data: bytes, label: str, source: str = "") -> Example:
        load_image(data)  # validates before anything is written
        eid = image_id(data)
        with self._lock:
            if label not in self.class_names():
                self.upsert_class(label)
            for e in self.config.examples:
                if e.id == eid:
                    e.label = label
                    self.save()
                    return e
            fmt = (Image.open(io.BytesIO(data)).format or "png").lower()
            ext = {"jpeg": "jpg"}.get(fmt, fmt)
            rel = f"images/{eid}.{ext}"
            (self.path / "images").mkdir(exist_ok=True)
            (self.path / rel).write_bytes(data)
            ex = Example(id=eid, label=label, file=rel, source=source)
            self.config.examples.append(ex)
            self.save()
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
        for e in self.config.examples:
            if e.id == eid:
                return e
        raise KeyError(f"No example {eid!r}")

    def relabel(self, eid: str, label: str) -> Example:
        with self._lock:
            e = self.example(eid)
            if label not in self.class_names():
                self.upsert_class(label)
            e.label = label
            self.save()
            return e

    def delete_example(self, eid: str) -> None:
        with self._lock:
            e = self.example(eid)
            (self.path / e.file).unlink(missing_ok=True)
            self.config.examples = [x for x in self.config.examples if x.id != eid]
            self.save()

    def image_path(self, eid: str) -> Path:
        return self.path / self.example(eid).file

    def summary(self) -> dict[str, Any]:
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
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root).resolve() if root else default_root()
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def list_tasks(self) -> list[Task]:
        return [self.get(p.name) for p in sorted((self.root / "tasks").iterdir()) if (p / "task.json").exists()]

    def get(self, name: str) -> Task:
        with self._lock:
            if name in self._tasks:
                return self._tasks[name]
            p = self.root / "tasks" / name
            if not NAME_RE.match(name) or not (p / "task.json").exists():
                raise KeyError(f"No task {name!r}")
            t = Task(p, TaskConfig.model_validate_json((p / "task.json").read_text()))
            self._tasks[name] = t
            return t

    def create(self, name: str, **settings: Any) -> Task:
        if not NAME_RE.match(name):
            raise ValueError("Task names use letters, digits, '_', '-', '.' (max 64 chars)")
        p = self.root / "tasks" / name
        if p.exists():
            raise ValueError(f"Task {name!r} already exists")
        cfg = TaskConfig(name=name, **settings)
        if cfg.classifier not in CLASSIFIERS:
            raise ValueError(f"Unknown classifier {cfg.classifier!r}")
        (p / "images").mkdir(parents=True)
        t = Task(p, cfg)
        t.save()
        with self._lock:
            self._tasks[name] = t
        return t

    def delete(self, name: str) -> None:
        t = self.get(name)
        with self._lock:
            shutil.rmtree(t.path)
            self._tasks.pop(name, None)

    @property
    def cache_path(self) -> Path:
        return self.root / "cache.sqlite"
