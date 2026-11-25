"""Export a task as a zip (task.json plus its images) and import it into a workspace."""

from __future__ import annotations

import shutil
import uuid
import zipfile
from pathlib import Path
from typing import IO

from pydantic import ValidationError

from .store import ClassDef, Task, TaskConfig, Workspace, check_class_name, check_settings, image_id

# Largest image and task.json accepted from an export zip (guards against zip bombs).
MAX_IMPORT_IMAGE = 200 * 1024 * 1024
MAX_IMPORT_CONFIG = 256 * 1024 * 1024


def export_task(ws: Workspace, name: str, out: IO[bytes]) -> int:
    """Write task ``name`` (task.json and images, not its history) as a zip to ``out``;
    returns the number of examples. Raises ``KeyError`` if there is no such task."""
    t = ws.get(name)
    cfg = t.config  # one snapshot
    examples = []
    with zipfile.ZipFile(out, "w") as z:
        for e in cfg.examples:
            try:
                data = (t.path / e.file).read_bytes()
            except FileNotFoundError:  # deleted since the snapshot
                continue
            z.writestr(e.file, data, zipfile.ZIP_STORED)  # images are compressed already
            examples.append(e)
        z.writestr("task.json", cfg.model_copy(update={"examples": examples}).model_dump_json(indent=2),
                   zipfile.ZIP_DEFLATED)
    return len(examples)


def import_task(ws: Workspace, src: IO[bytes] | Path, name: str | None = None) -> Task:
    """Create a task from a zip written by ``export_task``, named ``name`` or as exported.

    Every image is checked against its id. Raises ``ValueError`` if the zip isn't a valid
    export or the name is bad or taken.
    """
    try:
        with zipfile.ZipFile(src) as z:
            cfg, images = _read_export(z)
            cfg = cfg.model_copy(update={"name": name or cfg.name, "uid": uuid.uuid4().hex, "revision": 0})
            p = ws._reserve(cfg.name)  # pylint: disable=protected-access
            try:
                t = Task(p, cfg, blobs=ws.blobs)
                for e in cfg.examples:
                    t._store_image(e.file, _read_member(z, images[e.file]))  # pylint: disable=protected-access
                t._save(record=False)  # pylint: disable=protected-access
            except BaseException:
                shutil.rmtree(p, ignore_errors=True)
                raise
    except zipfile.BadZipFile as e:
        raise ValueError(f"Not a task export: {e}") from None
    return ws.get(cfg.name)


def _read_member(z: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int | None = None) -> bytes:
    """A zip member's bytes, refusing (``ValueError``) one that unpacks to more than ``limit``
    (default ``MAX_IMPORT_IMAGE``)."""
    limit = limit or MAX_IMPORT_IMAGE
    if info.file_size > limit:
        raise ValueError(f"{info.filename} is larger than {limit // 1024 // 1024} MB")
    return z.read(info)  # zipfile stops at the declared size (and checks the CRC), so this is bounded


def _read_export(z: zipfile.ZipFile) -> tuple[TaskConfig, dict[str, zipfile.ZipInfo]]:
    """The checked config of an export zip and its image members by file name.

    Each image must be present and hash to its example id; classes, settings and tags are checked
    as they would be when made through a ``Task``.
    """
    try:
        info = z.getinfo("task.json")
    except KeyError:
        raise ValueError("Not a task export: no task.json") from None
    try:
        cfg = TaskConfig.model_validate_json(_read_member(z, info, MAX_IMPORT_CONFIG))
    except ValidationError as err_:
        err = err_.errors()[0]
        raise ValueError(f"task.json is invalid: {'.'.join(map(str, err['loc']))}: {err['msg']}") from None
    check_settings(cfg)
    classes = [c.model_copy(update={"name": check_class_name(c.name)}) for c in cfg.classes]
    names = {c.name for c in classes}
    examples = list({x.id: x for x in cfg.examples}.values())  # one per id
    for ex in examples:
        if ex.label not in names:
            classes.append(ClassDef(name=check_class_name(ex.label)))
            names.add(ex.label)
    members = {m.filename: m for m in z.infolist()}
    for ex in examples:
        m = members.get(ex.file)
        if m is None:
            raise ValueError(f"Not a valid task export: {ex.file} is missing")
        if not image_id(_read_member(z, m)).startswith(ex.id):
            raise ValueError(f"Not a valid task export: {ex.file} doesn't match its id")
    return cfg.model_copy(update={"classes": classes, "examples": examples}), members
