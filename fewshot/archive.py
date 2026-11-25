"""Export a task as a zip (task.json plus its images) and import it into a workspace."""

from __future__ import annotations

import os
import shutil
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import IO

from pydantic import ValidationError

from .images import image_ext, image_id
from .store import ClassDef, Task, TaskConfig, Workspace, check_settings
from .text import check_class_name

# Limits on what an export zip may unpack to (guards against zip bombs): per image, for task.json,
# and for all images together.
MAX_IMPORT_IMAGE = 50 * 1024 * 1024
MAX_IMPORT_CONFIG = 64 * 1024 * 1024
MAX_IMPORT_TOTAL = 10 * 1024 * 1024 * 1024


def export_task(ws: Workspace, name: str, out: IO[bytes]) -> int:
    """Write task ``name`` (task.json and images, not its history) as a zip to ``out``;
    returns the number of examples. Raises ``KeyError`` if there is no such task.

    The task is locked meanwhile, so another process can't rename or change it half-way; an image
    file missing from the task (only if it was damaged) is left out with its example.
    """
    t = ws.get(name)
    examples = []
    with t._writing(), zipfile.ZipFile(out, "w") as z:  # pylint: disable=protected-access
        cfg = t.config
        for e in cfg.examples:
            try:
                data = (t.path / e.file).read_bytes()
            except FileNotFoundError:
                continue
            z.writestr(e.file, data, zipfile.ZIP_STORED)  # images are compressed already
            examples.append(e)
        z.writestr("task.json", cfg.model_copy(update={"examples": examples}).model_dump_json(indent=2),
                   zipfile.ZIP_DEFLATED)
    return len(examples)


def import_task(ws: Workspace, src: IO[bytes] | Path, name: str | None = None,
                embedder_ok: Callable[[str], None] | None = None) -> Task:
    """Create a task from a zip written by ``export_task``, named ``name`` or as exported.

    Every image must decode, have the extension of its real format, and hash to its id; the total
    unpacked size is limited. ``embedder_ok`` is called with the task's embedder spec before
    anything is written, and may raise to refuse it (the server uses this to apply its model
    allowlist). Raises ``ValueError`` if the zip isn't a valid export or the name is bad or taken.
    """
    try:
        with zipfile.ZipFile(src) as z:
            cfg, images = _read_export(z)
            if embedder_ok is not None:
                embedder_ok(cfg.embedder)
            cfg = cfg.model_copy(update={"name": name or cfg.name, "uid": uuid.uuid4().hex, "revision": 0})
            p = ws._reserve(cfg.name)  # pylint: disable=protected-access
            # Built in a hidden folder and moved into place at the end, so a crash part-way leaves
            # no half-imported task (Workspace.gc removes the leftover).
            build = ws.root / "tasks" / f".tmp-{uuid.uuid4().hex}"
            build.mkdir()
            try:
                t = Task(build, cfg, blobs=ws.blobs)
                for e in cfg.examples:
                    t._store_image(e.file, _read_member(z, images[e.file]))  # pylint: disable=protected-access
                t._save(record=False)  # pylint: disable=protected-access
                os.replace(build, p)  # replaces the empty reserved folder (POSIX)
            except BaseException:
                shutil.rmtree(build, ignore_errors=True)
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
    classes: dict[str, ClassDef] = {}  # by cleaned-up name; the first of any duplicates wins
    for c in cfg.classes:
        classes.setdefault(check_class_name(c.name), c.model_copy(update={"name": check_class_name(c.name)}))
    examples = [x.model_copy(update={"label": check_class_name(x.label)})
                for x in {x.id: x for x in cfg.examples}.values()]  # one per id
    for ex in examples:
        classes.setdefault(ex.label, ClassDef(name=ex.label))
    members = {m.filename: m for m in z.infolist()}
    missing = next((ex.file for ex in examples if ex.file not in members), None)
    if missing:
        raise ValueError(f"Not a valid task export: {missing} is missing")
    if sum(members[ex.file].file_size for ex in examples) > MAX_IMPORT_TOTAL:
        raise ValueError(f"Not imported: the images unpack to more than {MAX_IMPORT_TOTAL // 1024 ** 3} GB")
    for ex in examples:
        data = _read_member(z, members[ex.file])
        if not image_id(data).startswith(ex.id):
            raise ValueError(f"Not a valid task export: {ex.file} doesn't match its id")
        try:
            ext = image_ext(data)
        except ValueError as e:
            raise ValueError(f"Not a valid task export: {ex.file}: {e}") from None
        if not ex.file.endswith("." + ext):
            raise ValueError(f"Not a valid task export: {ex.file} is really a .{ext} image")
    return cfg.model_copy(update={"classes": list(classes.values()), "examples": examples}), members
