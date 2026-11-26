"""What a task is: the pydantic models saved in ``task.json``, and reading them back."""

from __future__ import annotations

import copy
import logging
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from .embedders import DEFAULT_SPEC
from .fsutil import FileStamp
from .fsutil import stamp as _stamp
from .settings import STORED_DEFAULTS
from .text import check_tag

log = logging.getLogger(__name__)

# Example files are always written as images/<id>.<ext>; anything else in task.json was not
# written by us, and following it could read or delete files outside the task.
EXAMPLE_FILE_RE = re.compile(r"^images/[0-9a-f]{16}(?:[0-9a-f]{16})?\.[a-z0-9]{1,10}$")
MAX_NOTE = 2000
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
    return datetime.now(UTC).isoformat(timespec="seconds")


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
        cfg = TaskConfig.stored(data)
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
    classifier: str = "gda"
    classifier_params: dict[str, Any] = Field(default_factory=dict)
    text_weight: float = 1.0
    reject_below: float | None = None  # similarity under which predictions become "unknown"
    # The same for zero-shot predictions (no example images), whose image-to-text similarities are
    # far lower (about 0.2-0.35 for CLIP) than image-to-image ones; None never rejects them.
    reject_below_zero_shot: float | None = None
    flip: bool = False  # average each image's embedding with its mirror image's (test-time augmentation)
    # With a text-capable embedder, a class with examples but no description uses "a photo of a
    # <name>." as its text, unless the names don't match the examples (see Pipeline.class_data).
    names_as_text: bool = True
    temperature: float | None = None  # confidence calibration fitted by evaluation (None: 1, uncalibrated)
    classes: list[ClassDef] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    inbox: list[InboxItem] = Field(default_factory=list)  # unlabelled images waiting to be labelled
    created: str = Field(default_factory=_now)
    revision: int = 0
    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)  # tells a re-created task from the old one

    @model_validator(mode="before")
    @classmethod
    def _stored_defaults(cls, data: Any, info: ValidationInfo) -> Any:
        stored = info.context and info.context.get("stored") and isinstance(data, dict)
        return {**STORED_DEFAULTS, **data} if stored else data

    @classmethod
    def stored(cls, data: bytes | str) -> TaskConfig:
        """Parse a saved task.json (or a kept version or an export of one), reading settings it
        predates as ``STORED_DEFAULTS``. Raises ``ValidationError``."""
        return cls.model_validate_json(data, context={"stored": True})

    def files(self) -> set[str]:
        """Every image file this version uses: examples' and the inbox's."""
        return {e.file for e in self.examples} | {i.file for i in self.inbox}
