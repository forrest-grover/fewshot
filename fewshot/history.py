"""Undo and redo for a task: earlier versions of task.json kept in its history/ folder.

``history/<rev>.json`` is the task as it was at revision ``rev`` (a hard link to the replaced
task.json, so keeping it is free), ``<rev>.gone`` lists the image files the following change
stopped using, and ``<rev>.label`` describes that change. ``history/redo/`` holds undone versions
until another change is made. Image files stay while any kept version names them.

Saves that add images can share one undo step (an *undo group*: the chunks of one call, or the
requests of one upload a client sent in several): ``<rev>.group`` records the group, what it
added so far and the revision its last save made. A later save in the same group, while that
step is still the newest and nothing was saved since, adds to that step instead of making one.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from .fsutil import fsync_dir as _fsync_dir
from .fsutil import write_atomic as _write_atomic

if TYPE_CHECKING:
    from .models import TaskConfig

log = logging.getLogger(__name__)

HISTORY_KEPT = 20  # earlier task.json versions kept for undo
GROUP_NAMES = 4  # class names an undo group keeps for its description
MEMO_FILES = 2_000_000  # file names of kept versions remembered per process (see ``_SnapshotFiles``)

# How ``model_dump_json`` writes an example's or inbox image's file (see ``models.EXAMPLE_FILE_RE``).
_FILE_RE = re.compile(rb'"file"\s*:\s*"(images/[0-9a-f]+\.[a-z0-9]+)"')


def _parse_files(data: bytes) -> frozenset[str]:
    """The image files a task.json names, examples' and inbox's (read without full validation).

    A scan for the ``"file"`` fields is several times faster than parsing the JSON; if any
    ``"file"`` key isn't in the form the scan expects (a hand edit, say), the JSON is parsed."""
    found = _FILE_RE.findall(data)
    if len(found) == data.count(b'"file"'):
        # Interned, so the memo's sets of a task's versions (mostly the same files) share strings.
        return frozenset(map(sys.intern, map(bytes.decode, found)))
    try:
        cfg = json.loads(data)
        names = [e["file"] for e in [*cfg.get("examples", []), *cfg.get("inbox", [])]]
        return frozenset(sys.intern(f) for f in names if isinstance(f, str))
    except (ValueError, KeyError, TypeError, AttributeError):
        return frozenset()


class _SnapshotFiles:  # pylint: disable=too-few-public-methods  # a memo
    """The image files each kept version names, read once per process: kept versions never change
    once written, so a file's set is reused while it is the same file (device, inode, size and
    modification time; a hand edit changes those). Thread-safe, and bounded to about ``limit``
    names in all (least recently used versions are forgotten first)."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._sets: OrderedDict[tuple[str, int, int, int, int], frozenset[str]] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()

    def files(self, snap: Path) -> frozenset[str]:
        """The files named by kept version ``snap`` (none if it can't be read)."""
        try:
            with open(snap, "rb") as f:
                st = os.fstat(f.fileno())
                key = (str(snap), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
                with self._lock:
                    hit = self._sets.get(key)
                    if hit is not None:
                        self._sets.move_to_end(key)
                        return hit
                data = f.read()
        except OSError:
            return frozenset()
        files = _parse_files(data)
        with self._lock:
            # Consecutive versions often name the same files: keep one set for them all.
            same = next((s for s in reversed(self._sets.values()) if len(s) == len(files) and s == files), None)
            files = same if same is not None else files
            self._sets[key] = files
            self._size += len(files)
            while self._size > self._limit and len(self._sets) > 1:
                self._size -= len(self._sets.popitem(last=False)[1])
        return files


_snapshot_files = _SnapshotFiles(MEMO_FILES)


def _inode(path: Path) -> int | None:
    try:
        return path.stat().st_ino
    except OSError:
        return None


class TaskHistory:
    """The undo/redo half of ``store.Task``, which provides the attributes and methods declared here."""

    path: Path
    config: TaskConfig
    _commits: int
    _pending_ino: int | None
    # The kept version ``version`` parsed last, keyed by its file, for repeated predictions at it.
    _version_memo: tuple[tuple[str, int, int, int], TaskConfig] | None = None

    def _committed_since(self, commits: int) -> bool:  # pragma: no cover  # Task's
        raise NotImplementedError

    def _writing(self) -> contextlib.AbstractContextManager[None]:  # pragma: no cover  # Task's
        raise NotImplementedError

    def _save(self, gone: Iterable[str] = (), record: bool = True, label: str = "",
              keep_redo: bool = False) -> None:  # pragma: no cover  # Task's
        raise NotImplementedError

    def _drop_file(self, file: str) -> None:  # pragma: no cover  # Task's
        raise NotImplementedError

    def _snapshot(self, rev: int, gone: list[str], label: str = "", folder: str = "history") -> list[Path]:
        """Keep the current task.json (revision ``rev``) as ``<folder>/<rev>.json``, plus the image
        files the change being saved drops (``<rev>.gone``) and a description of the change
        (``<rev>.label``); returns the files written."""
        hist = self.path / folder
        hist.mkdir(parents=True, exist_ok=True)
        snap = hist / f"{rev:010d}.json"
        self._forget(snap)  # with any side files a crashed attempt at this revision left
        try:
            os.link(self.path / "task.json", snap)  # free: task.json is about to be replaced, not changed
        except OSError:  # pragma: no cover  # filesystems without hard links
            shutil.copyfile(self.path / "task.json", snap)
        made = [snap]
        try:
            for suffix, data in ((".gone", json.dumps(gone) if gone else ""), (".label", label)):
                if data:
                    _write_atomic(snap.with_suffix(suffix), data.encode())
                    made.append(snap.with_suffix(suffix))
            _fsync_dir(hist)  # kept before the save it belongs to commits
        except BaseException:
            for f in made:
                f.unlink(missing_ok=True)
            raise
        return made

    @contextlib.contextmanager
    def _new_files(self, created: list[str]) -> Iterator[None]:
        """Remove the image files listed in ``created`` (stored in the block, named by no saved
        version) if the block fails before a save commits."""
        commits = self._commits
        self._pending_ino = None  # a save before the block is no commit of the block's
        try:
            yield
        except BaseException:
            if not self._committed_since(commits):
                for f in created:
                    self._drop_file(f)
            raise

    @staticmethod
    def _forget(snap: Path) -> None:
        """Delete a kept version and its side files."""
        for suffix in (".json", ".gone", ".label", ".group"):
            snap.with_suffix(suffix).unlink(missing_ok=True)

    @staticmethod
    def _label(snap: Path) -> str:
        """The description of the change after version ``snap`` ("" if none was kept)."""
        try:
            return snap.with_suffix(".label").read_text().strip()
        except (OSError, UnicodeDecodeError):
            return ""

    def _save_new(self, created: list[str], record: bool, label: str) -> None:  # pragma: no cover  # Task's
        raise NotImplementedError

    def _save_in_group(self, created: list[str], group: tuple[str, str], count: int, names: list[str],  # pylint: disable=too-many-arguments,too-many-positional-arguments
                       describe: Callable[[int, list[str]], str]) -> None:
        """``_save_new`` for a change adding ``count`` images (to classes ``names``) in undo group
        ``group``, ``(kind, id)``. If the newest undo step is that group's and nothing was saved
        since, the change joins it: nothing new is kept for undo, and the step's description
        (``describe(count, names)``) covers the whole group. Call inside ``_writing``."""
        kind, gid = group
        snaps = self._history()
        info = self._group_of(snaps[-1][1]) if snaps else None
        joined = info is not None and (info["kind"], info["group"], info["revision"]) == (
            kind, gid, self.config.revision)
        if joined and info is not None:
            count += info["count"]
            names = list(dict.fromkeys([*info["names"], *names]))
        before = self.config.revision
        label = describe(count, names)
        self._save_new(created, not joined, label)
        snap = snaps[-1][1] if joined else self.path / "history" / f"{before:010d}.json"
        record = {"kind": kind, "group": gid, "revision": self.config.revision, "count": count,
                  "names": names[:GROUP_NAMES]}
        try:  # the change is saved: failing to note the group only ends it
            if joined:
                _write_atomic(snap.with_suffix(".label"), label.encode())
            if snap.exists():
                _write_atomic(snap.with_suffix(".group"), json.dumps(record).encode())
        except OSError:
            log.exception("%s: the change was saved, but its undo group was not noted", self.path)

    @staticmethod
    def _group_of(snap: Path) -> dict[str, Any] | None:
        """The undo group noted for the change after version ``snap``, if any (and readable)."""
        try:
            info = json.loads(snap.with_suffix(".group").read_text())
        except (OSError, ValueError):
            return None
        keys = {"kind": str, "group": str, "revision": int, "count": int, "names": list}
        if not isinstance(info, dict) or not all(isinstance(info.get(k), t) for k, t in keys.items()):
            return None
        return info

    def _redo_list(self) -> list[tuple[int, Path]]:
        """Undone versions that can be redone, oldest first (the last is redone next)."""
        folder = self.path / "history" / "redo"
        if not folder.is_dir():
            return []
        return sorted((int(f.stem), f) for f in folder.glob("*.json") if f.stem.isdigit())

    def _redo_files(self) -> set[str]:
        return set().union(*(self._files_in(p) for _, p in self._redo_list()))

    def _set_redo_aside(self) -> Path | None:
        """Move the undone versions out of reach (a new change is about to make them so); returns
        where they went, for ``_clear_redo`` once the change is saved, or None if there were none."""
        redo = self.path / "history" / "redo"
        if not redo.is_dir():
            return None
        aside = self.path / "history" / f".redo-{uuid.uuid4().hex}"
        os.rename(redo, aside)
        return aside

    def _clear_redo(self, aside: Path) -> None:
        """Forget undone versions moved ``aside`` and delete the images only they used."""
        files = set().union(*(self._files_in(p) for p in aside.glob("*.json")))
        shutil.rmtree(aside, ignore_errors=True)
        files -= self.config.files()
        files -= set().union(*(self._files_in(p) for _, p in self._history()))
        files -= self._redo_files()
        for f in files:
            self._drop_file(f)

    def _ahead(self, cfg: TaskConfig, ino: int) -> TaskConfig:
        """``cfg``, just read from task.json (inode ``ino``), with a revision above every kept and
        undone version's and at least the last evaluation's.

        Saves keep it so, but a task.json put back by hand (say, a kept version copied over it)
        carries an old revision: kept as it is, its version would sort below older ones for undo
        and pruning, a save would replace the kept version of the same revision, and an
        evaluation of another version would show as current. It is taken as a new version, with
        the next revision (written by the next save). A kept version that *is* task.json, left by
        a save interrupted before replacing it, doesn't count."""
        revs = [rev for rev, snap in [*self._history(), *self._redo_list()]
                if not (rev == cfg.revision and _inode(snap) == ino)]
        evaluated = (self.last_evaluation() or {}).get("revision")
        evaluated = evaluated if isinstance(evaluated, int) else -1
        top = max([*revs, evaluated], default=-1)
        if cfg.revision > top or cfg.revision == evaluated > max(revs, default=-1):
            return cfg
        log.warning("%s: task.json has revision %d, but a kept version or evaluation has a later one; taking it as "
                    "revision %d", self.path, cfg.revision, top + 1)
        return cfg.model_copy(update={"revision": top + 1})

    def last_evaluation(self) -> dict[str, Any] | None:  # pragma: no cover  # Task's
        """Task's."""
        raise NotImplementedError

    def _history(self) -> list[tuple[int, Path]]:
        """Kept versions as ``(revision, path)``, oldest first."""
        hist = self.path / "history"
        if not hist.is_dir():
            return []
        return sorted((int(f.stem), f) for f in hist.glob("*.json") if f.stem.isdigit())

    @staticmethod
    def _gone(snap: Path) -> set[str]:
        """Image files dropped by the change after version ``snap``. An unreadable list names
        none, so those files are kept (``Workspace.gc`` finds them later) rather than lost."""
        try:
            files = json.loads(snap.with_suffix(".gone").read_text())
        except FileNotFoundError:
            return set()
        except (OSError, ValueError):
            log.warning("Ignoring unreadable %s", snap.with_suffix(".gone"))
            return set()
        return {f for f in files if isinstance(f, str)} if isinstance(files, list) else set()

    def _prune_history(self) -> None:
        """Forget versions beyond ``HISTORY_KEPT`` and delete images that only they used."""
        snaps = self._history()
        cut = max(0, len(snaps) - HISTORY_KEPT)
        old, kept = snaps[:cut], snaps[cut:]
        if not old:
            return
        gone = set().union(*(self._gone(p) for _, p in old))
        for _, p in old:
            self._forget(p)
        # A file dropped again later is deleted when that later version goes. Most changes drop
        # no file, so the kept versions are only read while some file may still be deleted.
        if gone:
            gone -= self.config.files()
        for _, p in kept:
            if not gone:
                break
            gone -= self._gone(p) | self._files_in(p)
        if gone:
            gone -= self._redo_files()
        for f in gone:
            self._drop_file(f)

    def _drop_unused_images(self) -> int:
        """Delete image files that neither the task nor any kept version names; returns how many."""
        with self._writing():
            for stale in (self.path / "history").glob(".redo-*"):  # set aside by a save that crashed
                shutil.rmtree(stale, ignore_errors=True)
            for folder in (self.path, self.path / "images", self.path / "history", self.path / "history" / "redo"):
                for tmp in folder.glob(".*.tmp"):  # half-written by a write that crashed (writes hold the lock)
                    tmp.unlink(missing_ok=True)
            used = self.config.files() | self._redo_files()
            used |= set().union(*(self._files_in(p) for _, p in self._history()))
            images = self.path / "images"
            unused = [f"images/{f.name}" for f in images.iterdir() if f.is_file()] if images.is_dir() else []
            unused = [f for f in unused if f not in used]
            for f in unused:
                self._drop_file(f)
            return len(unused)

    def kept_files(self) -> set[str]:
        """Image files that undo or redo may bring back (named by a kept version, not the current one)."""
        files = self._redo_files() | set().union(*(self._files_in(p) for _, p in self._history()))
        return files - self.config.files()

    def version(self, revision: int) -> TaskConfig:
        """The task as it was at ``revision``: the current version, or one kept for undo (the last
        ``HISTORY_KEPT``, whose images are kept too). Raises ``KeyError`` for any other."""
        if revision == self.config.revision:
            return self.config
        for rev, snap in self._history():
            if rev == revision:
                try:
                    with open(snap, "rb") as f:
                        st = os.fstat(f.fileno())
                        key = (str(snap), st.st_ino, st.st_size, st.st_mtime_ns)
                        memo = self._version_memo
                        if memo is not None and memo[0] == key:  # predicting at it again: parsed already
                            cfg = memo[1]
                        else:
                            cfg = type(self.config).stored(f.read())
                            self._version_memo = (key, cfg)
                except (OSError, ValidationError):
                    break
                return cfg.model_copy(update={"name": self.config.name})
        raise KeyError(f"Revision {revision} of {self.config.name!r} is not kept (the last {HISTORY_KEPT} are)")

    def can_undo(self) -> bool:
        """Whether there is an earlier version to go back to."""
        return bool(self._history())

    def can_redo(self) -> bool:
        """Whether an undone change can be made again."""
        return bool(self._redo_list())

    def undo_label(self) -> str:
        """What the next undo takes back, e.g. "delete 3 examples" ("" if unknown or nothing)."""
        snaps = self._history()
        return self._label(snaps[-1][1]) if snaps else ""

    def undo_group(self) -> str | None:
        """The client's undo group id of the change the next undo takes back, if it was made in one
        (see ``_save_in_group``): a client that sent an upload in several groups' requests can undo
        while the next step is one of its own."""
        snaps = self._history()
        info = self._group_of(snaps[-1][1]) if snaps else None
        return info["group"] if info else None

    def redo_label(self) -> str:
        """What the next redo makes again ("" if unknown or nothing)."""
        redo = self._redo_list()
        return self._label(redo[-1][1]) if redo else ""

    def undo(self) -> int:
        """Go back to the version before the last change (of any kind); returns its revision.

        Deleted examples come back with their images. The undone change can be made again with
        ``redo`` until another change is made. The task keeps its name and its revision still
        increases. Raises ``ValueError`` if there is nothing to undo.
        """
        with self._writing():
            while True:
                snaps = self._history()
                if not snaps:
                    raise ValueError("Nothing to undo")
                rev, snap = snaps[-1]
                old = self._restorable(snap, "undo")
                if not self._same(old):
                    break
                # A step that changes nothing: left by a crash (after a version was kept but
                # before the save replaced task.json, or after an undo but before it forgot the
                # version it restored), or a save that changed nothing. Skip it.
                self._forget(snap)
            # The current version becomes the next redo, under the undone change's description.
            redo = self._snapshot(self.config.revision, [], self._label(snap), folder="history/redo")
            commits = self._commits
            try:
                self._restore(old)
            except BaseException:
                if not self._committed_since(commits):  # nothing was undone: nothing to redo
                    for f in redo:
                        f.unlink(missing_ok=True)
                raise
            self._forget(snap)
            _fsync_dir(snap.parent)  # else a power cut could bring the step back as a no-op one
            return rev

    def _same(self, cfg: TaskConfig) -> bool:
        """Whether kept version ``cfg`` is the current one, but for name, uid and revision."""
        return cfg.model_copy(update={k: getattr(self.config, k) for k in ("name", "uid", "revision")}) == self.config

    def redo(self) -> int:
        """Make the last undone change again; returns the revision it had. Raises ``ValueError``
        if there is nothing to redo (a change since the undo discards what was undone)."""
        with self._writing():
            redo = self._redo_list()
            if not redo:
                raise ValueError("Nothing to redo")
            rev, snap = redo[-1]
            new = self._restorable(snap, "redo")
            # The version redo replaces is kept for undo: name the files the redo stops using.
            self._restore(new, label=self._label(snap), gone=self.config.files() - new.files())
            self._forget(snap)
            return rev

    def _restorable(self, snap: Path, what: str) -> TaskConfig:
        """The version in ``snap``, if it can be restored; ``ValueError`` (after removing a damaged
        one, so the next try can go further) otherwise."""
        try:
            cfg = type(self.config).stored(snap.read_bytes())
        except ValidationError as e:
            self._forget(snap)
            raise ValueError(f"Can't {what}: {snap.name} is invalid and was removed; {what} again to go "
                             "further") from e
        missing = [f for f in sorted(cfg.files()) if not (self.path / f).exists()]
        if missing:
            raise ValueError(f"Can't {what}: {len(missing)} image files are gone, e.g. {missing[0]}")
        return cfg

    def _restore(self, cfg: TaskConfig, label: str | None = None, gone: Iterable[str] = ()) -> None:
        """Make ``cfg`` the current version, keeping the task's name, uid and rising revision. With
        a ``label`` (a redo) the replaced version is kept for undo, with ``gone`` the image files
        the change stops using; otherwise (an undo) it isn't."""
        self.config = cfg.model_copy(update={"name": self.config.name, "uid": self.config.uid,
                                             "revision": self.config.revision})
        self._save(gone=gone, record=label is not None, label=label or "", keep_redo=True)

    @staticmethod
    def _files_in(snap: Path) -> frozenset[str]:
        """Image files named by a kept version, examples' and inbox's (read without full
        validation, and once per process: see ``_SnapshotFiles``)."""
        return _snapshot_files.files(snap)
