"""File helpers for the task store: change stamps, cross-process locks and durable atomic writes."""

from __future__ import annotations

import contextlib
import importlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover  # Windows: msvcrt below
    fcntl = None  # type: ignore[assignment]
msvcrt: Any = None
if fcntl is None:  # pragma: no cover  # Windows only
    msvcrt = importlib.import_module("msvcrt")


# (inode, mtime_ns, ctime_ns, size): changes whenever task.json is replaced. The ctime (set when the
# new file is renamed into place) guards against a freed inode being reused within one mtime tick.
FileStamp = tuple[int, int, int, int]


def stamp(st: os.stat_result) -> FileStamp:
    """The stamp of a file from its ``os.stat``."""
    return (st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size)


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` (created if missing) across processes: ``flock`` on
    Linux and macOS, ``msvcrt.locking`` on Windows.

    Raises ``FileNotFoundError`` if the folder holding ``path`` is gone.
    """
    while True:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _lock(fd)
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


def _lock(fd: int) -> None:
    """Block until this process holds the exclusive lock on open file ``fd``."""
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return
    while True:  # pragma: no cover  # Windows: LK_LOCK gives up after 10 one-second tries
        try:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            return
        except OSError:
            continue


@contextlib.contextmanager
def dir_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on folder ``path`` itself while a task is built in it, so that
    ``dir_locked`` (and so ``Workspace.gc``) can tell it from a crash's leftover however long the
    build takes. The lock follows the folder when it is renamed. No-op without ``fcntl``."""
    if fcntl is None:  # pragma: no cover  # Windows: gc falls back to the folder's age
        yield
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def dir_locked(path: Path) -> bool:
    """Whether some process (this one included) holds ``dir_lock`` on folder ``path``."""
    if fcntl is None:  # pragma: no cover  # Windows
        return False
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)  # releases the probe's own lock, never another holder's (flock is per open file)
    return False


def fsync_dir(path: Path) -> None:
    """Make the creation, renaming or removal of entries in folder ``path`` durable (POSIX; a
    no-op where folders can't be opened, such as Windows)."""
    if not hasattr(os, "O_DIRECTORY"):  # pragma: no cover  # Windows
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_synced(fd: int, data: bytes) -> None:
    """Write ``data`` to the new file ``fd`` (closing it) and flush it to disk, so that once it is
    renamed or linked into place a power cut can't leave it empty or partly written."""
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``path`` so that readers see either nothing or all of ``data``, and the content is on
    disk before the name points at it. (The rename itself is durable after ``fsync_dir``.)"""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        _write_synced(fd, data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def create_exclusive(path: Path, data: bytes) -> bool:
    """Create ``path`` with all of ``data`` unless it already exists; returns whether it was
    created. Readers never see it partly written, and its content is on disk before it is linked
    into place. Raises ``OSError`` where hard links aren't supported (the file is written aside
    and linked into place)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        _write_synced(fd, data)
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
