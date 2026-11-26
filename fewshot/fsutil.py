"""File helpers for the task store: change stamps, cross-process locks and durable atomic writes."""

from __future__ import annotations

import contextlib
import errno
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
def file_lock(path: Path, shared: bool = False) -> Iterator[None]:
    """Hold a lock on ``path`` (created if missing) across processes: ``flock`` on Linux and
    macOS, ``msvcrt.locking`` (always exclusive) on Windows. An exclusive lock waits for every
    other holder; a ``shared`` one only for exclusive holders, and needs no write access to an
    existing ``path`` (where it can't be created either, as in a read-only folder, nobody can be
    changing what it guards, and nothing is locked).

    Raises ``FileNotFoundError`` if the folder holding ``path`` is gone.
    """
    while True:
        try:
            fd = _open_lock(path, shared)
        except PermissionError:
            if not shared or path.exists():
                raise
            yield
            return
        try:
            _lock(fd, shared)
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


def _open_lock(path: Path, shared: bool) -> int:
    if shared:
        with contextlib.suppress(FileNotFoundError):
            return os.open(path, os.O_RDONLY)  # flock needs no write access
    try:
        return os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        if e.errno == errno.EROFS:  # a read-only file system: as for a folder without write access
            raise PermissionError(e.errno, e.strerror, e.filename) from e
        raise


def _lock(fd: int, shared: bool = False) -> None:
    """Block until this process holds the lock (exclusive unless ``shared``) on open file ``fd``."""
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
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


@contextlib.contextmanager
def claim_dir(path: Path) -> Iterator[bool]:
    """Try to take ``dir_lock`` on folder ``path`` without waiting; yields whether this caller now
    holds it on the folder still at ``path`` (False if another process holds it, or the folder is
    gone or was replaced meanwhile). Without ``fcntl`` it yields True and locks nothing."""
    if fcntl is None:  # pragma: no cover  # Windows
        yield True
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        yield False
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            st, now = os.fstat(fd), os.stat(path)
            held = (st.st_dev, st.st_ino) == (now.st_dev, now.st_ino)
        except OSError:  # BlockingIOError: someone else holds it; FileNotFoundError: gone since
            held = False
        yield held
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


def os_reason(e: Exception) -> str:
    """The message of ``e``: an OSError's description (with the file), not its repr."""
    if isinstance(e, OSError) and e.strerror:
        return f"{e.filename}: {e.strerror}" if e.filename else e.strerror
    return str(e)
