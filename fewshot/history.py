"""Undo and redo for a task: earlier versions of task.json kept in its history/ folder.

``history/<rev>.json`` is the task as it was at revision ``rev`` (a hard link to the replaced
task.json, so keeping it is free), ``<rev>.gone`` lists the image files the following change
stopped using, and ``<rev>.label`` describes that change. ``history/redo/`` holds undone versions
until another change is made. Image files stay while any kept version names them.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from .fsutil import write_atomic as _write_atomic

if TYPE_CHECKING:
    from .store import TaskConfig

log = logging.getLogger(__name__)

HISTORY_KEPT = 20  # earlier task.json versions kept for undo


class TaskHistory:
    """The undo/redo half of ``store.Task``, which provides the attributes and methods declared here."""

    path: Path
    config: TaskConfig

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
        snap.unlink(missing_ok=True)
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
        except BaseException:
            for f in made:
                f.unlink(missing_ok=True)
            raise
        return made

    @staticmethod
    def _forget(snap: Path) -> None:
        """Delete a kept version and its side files."""
        for suffix in (".json", ".gone", ".label"):
            snap.with_suffix(suffix).unlink(missing_ok=True)

    @staticmethod
    def _label(snap: Path) -> str:
        """The description of the change after version ``snap`` ("" if none was kept)."""
        try:
            return snap.with_suffix(".label").read_text().strip()
        except (OSError, UnicodeDecodeError):
            return ""

    def _redo_list(self) -> list[tuple[int, Path]]:
        """Undone versions that can be redone, oldest first (the last is redone next)."""
        folder = self.path / "history" / "redo"
        if not folder.is_dir():
            return []
        return sorted((int(f.stem), f) for f in folder.glob("*.json") if f.stem.isdigit())

    def _redo_files(self) -> set[str]:
        return set().union(*(self._files_in(p) for _, p in self._redo_list()))

    def _clear_redo(self) -> None:
        """A new change makes undone ones unreachable: forget them and the images only they used."""
        redo = self._redo_list()
        if not redo:
            return
        files = self._redo_files()
        shutil.rmtree(self.path / "history" / "redo", ignore_errors=True)
        files -= self.config.files()
        files -= set().union(*(self._files_in(p) for _, p in self._history()))
        for f in files:
            self._drop_file(f)

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
        # A file dropped again later is deleted when that later version goes.
        gone -= self.config.files()
        gone -= set().union(*(self._gone(p) | self._files_in(p) for _, p in kept))
        gone -= self._redo_files()
        for f in gone:
            self._drop_file(f)

    def _drop_unused_images(self) -> int:
        """Delete image files that neither the task nor any kept version names; returns how many."""
        with self._writing():
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
                    cfg = type(self.config).stored(snap.read_bytes())
                except ValidationError:
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
            snaps = self._history()
            if not snaps:
                raise ValueError("Nothing to undo")
            rev, snap = snaps[-1]
            old = self._restorable(snap, "undo")
            # The current version becomes the next redo, under the undone change's description.
            self._snapshot(self.config.revision, [], self._label(snap), folder="history/redo")
            self._restore(old)
            self._forget(snap)
            return rev

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
        missing = [e.file for e in cfg.examples if not (self.path / e.file).exists()]
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
    def _files_in(snap: Path) -> set[str]:
        """Image files named by a kept version, examples' and inbox's (read without full validation)."""
        try:
            cfg = json.loads(snap.read_bytes())
            return {e["file"] for e in [*cfg.get("examples", []), *cfg.get("inbox", [])]}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return set()
