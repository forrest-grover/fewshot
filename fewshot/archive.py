"""Export a task as a zip (task.json plus its images) and import it into a workspace."""

from __future__ import annotations

import os
import struct
import uuid
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, Any

from pydantic import ValidationError

from .embedders import check_embedder
from .images import image_ext, image_id
from .models import ClassDef, TaskConfig
from .settings import check_settings, for_new_embedder
from .store import Task, Workspace
from .text import check_class_name, check_description

# Limits on what an export zip may unpack to (guards against zip bombs): per image, for task.json,
# and for all images together.
MAX_IMPORT_IMAGE = 50 * 1024 * 1024
MAX_IMPORT_CONFIG = 64 * 1024 * 1024
MAX_IMPORT_TOTAL = 10 * 1024 * 1024 * 1024
# Files an export zip may list (its images and task.json): checked before zipfile reads the zip's
# directory, which costs about half a kilobyte of memory per file.
MAX_IMPORT_FILES = 250_000

_EOCD, _EOCD64, _LOCATOR64, _CENTRAL = b"PK\x05\x06", b"PK\x06\x06", b"PK\x06\x07", b"PK\x01\x02"


def export_task(ws: Workspace, name: str, out: IO[bytes]) -> tuple[int, int]:
    """Write task ``name`` (task.json and images, not its history) as a zip to ``out``; returns
    the number of examples written and of examples left out because their image file was
    missing. Raises ``KeyError`` if there is no such task.

    The task is locked meanwhile (``Task.locked``, which needs no write access), so another
    process can't rename or change it half-way; an image file missing from the task (only if it
    was damaged) is left out with its example.
    """
    t = ws.get(name)
    with t.locked() as cfg, zipfile.ZipFile(out, "w") as z:
        kept: dict[str, list[Any]] = {"examples": [], "inbox": []}
        for field, items in (("examples", cfg.examples), ("inbox", cfg.inbox)):
            for e in items:
                try:
                    data = (t.path / e.file).read_bytes()
                except FileNotFoundError:
                    continue
                z.writestr(e.file, data, zipfile.ZIP_STORED)  # images are compressed already
                kept[field].append(e)
        z.writestr("task.json", cfg.model_copy(update=kept).model_dump_json(indent=2), zipfile.ZIP_DEFLATED)
        skipped = len(cfg.examples) + len(cfg.inbox) - len(kept["examples"]) - len(kept["inbox"])
    return len(kept["examples"]), skipped


def import_task(ws: Workspace, src: IO[bytes] | Path, name: str | None = None,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                embedder_ok: Callable[[str], None] | None = None, embedder: str | None = None) -> Task:
    """Create a task from a zip written by ``export_task``, named ``name`` or as exported.

    Every image must decode, have the extension of its real format, and hash to its id; the total
    unpacked size is limited. ``embedder_ok`` is called with the task's embedder spec before
    anything is written and raises to refuse it: by default ``check_embedder``, which loads the
    model (``ValueError`` if it can't be); the server passes one that also applies its model
    allowlist. ``embedder`` replaces the exported embedder (e.g. one this server doesn't allow).
    Raises ``ValueError`` if the zip isn't a valid export, its embedder can't be loaded, or the
    name is bad or taken.
    """
    if isinstance(src, Path):
        with open(src, "rb") as f:
            return import_task(ws, f, name, embedder_ok, embedder)
    _count_files(src)
    try:
        with zipfile.ZipFile(src) as z:
            cfg, images = _read_export(z)
            if embedder and embedder != cfg.embedder:
                cfg = cfg.model_copy(update={"embedder": embedder})
                for_new_embedder(cfg)
            (embedder_ok or check_embedder)(cfg.embedder)
            cfg = cfg.model_copy(update={"name": name or cfg.name, "uid": uuid.uuid4().hex, "revision": 0})

            def checked() -> Iterator[tuple[str, bytes]]:
                for e in [*cfg.examples, *cfg.inbox]:
                    data = _read_member(z, images[e.file])
                    if not image_id(data).startswith(e.id):  # not the bytes checked before
                        raise ValueError(f"Not a valid task export: {e.file} changed while importing")
                    yield e.file, data

            return ws.build_task(cfg, checked())
    except zipfile.BadZipFile as e:
        raise ValueError(f"Not a task export: {e}") from None


def _central_directory(f: IO[bytes]) -> tuple[int, int] | None:
    """Where the zip in ``f`` has its central directory, ``(start, size)``, found as zipfile
    (``_EndRecData``) finds it: from the end of central directory record, and the Zip64 one if
    there is one. None where zipfile would find none, or refuse the records it found."""
    size = f.seek(0, os.SEEK_END)
    if size < 22:
        return None
    f.seek(size - 22)
    rec, at = f.read(22), size - 22
    if not (rec[:4] == _EOCD and rec[-2:] == b"\0\0"):  # not a record without a comment: search
        tail_at = max(0, size - 22 - 0xFFFF)  # for one followed by a comment of up to 64 KB
        f.seek(tail_at)
        tail = f.read()
        i = tail.rfind(_EOCD)
        if i < 0 or len(tail) - i < 22:
            return None
        rec, at = tail[i : i + 22], tail_at + i
    (cd_size,) = struct.unpack_from("<L", rec, 12)
    end = at  # the directory ends where the record (or the Zip64 one) starts
    if at >= 20:
        f.seek(at - 20)
        locator = f.read(20)
        if locator[:4] == _LOCATOR64:
            # The Zip64 record is where the locator says, or else (data prepended) just before it.
            (where,) = struct.unpack_from("<Q", locator, 8)
            for end in (where, at - 20 - 56):
                f.seek(max(0, end))
                rec64 = f.read(56)
                if end >= 0 and len(rec64) == 56 and rec64[:4] == _EOCD64:
                    (cd_size,) = struct.unpack_from("<Q", rec64, 40)
                    break
            else:
                return None
    return (end - cd_size, cd_size) if end >= cd_size else None


def _count_files(f: IO[bytes]) -> None:
    """``ValueError`` if the zip in ``f`` lists more than ``MAX_IMPORT_FILES`` files, counted by
    walking its central directory's headers (zipfile trusts neither its declared count nor any
    limit, and builds an object per entry)."""
    where = _central_directory(f)
    if where is not None:
        pos, size = where
        end, n = pos + size, 0
        while pos < end:
            f.seek(pos)
            header = f.read(46)
            if len(header) < 46 or header[:4] != _CENTRAL:
                break  # damaged: zipfile reports it
            n += 1
            if n > MAX_IMPORT_FILES:
                raise ValueError(f"Not imported: the zip holds more than {MAX_IMPORT_FILES:,} files")
            name_len, extra_len, comment_len = struct.unpack_from("<HHH", header, 28)
            pos += 46 + name_len + extra_len + comment_len
    f.seek(0)


def _read_member(z: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int | None = None) -> bytes:
    """A zip member's bytes, refusing (``ValueError``) one that unpacks to more than ``limit``
    (default ``MAX_IMPORT_IMAGE``)."""
    limit = limit or MAX_IMPORT_IMAGE
    if info.file_size > limit:
        raise ValueError(f"{info.filename} is larger than {limit // 1024 // 1024} MB")
    return z.read(info)  # zipfile stops at the declared size (and checks the CRC), so this is bounded


def _check_member(z: zipfile.ZipFile, info: zipfile.ZipInfo) -> None:
    """``ValueError`` unless the member is an image whose content hashes to the id in its name and
    whose extension matches its real format."""
    data = _read_member(z, info)
    filename = info.filename
    name = filename.rsplit("/", 1)[-1]
    if not image_id(data).startswith(name.split(".", 1)[0]):
        raise ValueError(f"Not a valid task export: {filename} doesn't match its id")
    try:
        ext = image_ext(data)
    except ValueError as e:
        raise ValueError(f"Not a valid task export: {filename}: {e}") from None
    if not name.endswith("." + ext):
        raise ValueError(f"Not a valid task export: {filename} is really a .{ext} image")


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
        cfg = TaskConfig.stored(_read_member(z, info, MAX_IMPORT_CONFIG))
    except ValidationError as err_:
        err = err_.errors()[0]
        raise ValueError(f"task.json is invalid: {'.'.join(map(str, err['loc']))}: {err['msg']}") from None
    check_settings(cfg)
    classes: dict[str, ClassDef] = {}  # by cleaned-up name; the first of any duplicates wins
    for c in cfg.classes:
        check_description(c.description)
        classes.setdefault(check_class_name(c.name), c.model_copy(update={"name": check_class_name(c.name)}))
    examples = [x.model_copy(update={"label": check_class_name(x.label)})
                for x in {x.id: x for x in cfg.examples}.values()]  # one per id
    labelled = {x.id for x in examples}
    inbox = [x for x in {x.id: x for x in cfg.inbox}.values() if x.id not in labelled]
    for ex in examples:
        classes.setdefault(ex.label, ClassDef(name=ex.label))
    members = {m.filename: m for m in z.infolist()}
    files = [x.file for x in [*examples, *inbox]]
    missing = next((f for f in files if f not in members), None)
    if missing:
        raise ValueError(f"Not a valid task export: {missing} is missing")
    if sum(members[f].file_size for f in files) > MAX_IMPORT_TOTAL:
        raise ValueError(f"Not imported: the images unpack to more than {MAX_IMPORT_TOTAL // 1024 ** 3} GB")
    for f in files:
        _check_member(z, members[f])
    return cfg.model_copy(update={"classes": list(classes.values()), "examples": examples, "inbox": inbox}), members
