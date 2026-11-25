"""File helpers for the task store: change stamps, a cross-process lock and atomic writes."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover  # Windows: no cross-process lock
    fcntl = None  # type: ignore[assignment]


FileStamp = tuple[int, int, int]  # (inode, mtime_ns, size): changes whenever task.json is replaced


def stamp(st: os.stat_result) -> FileStamp:
    """The stamp of a file from its ``os.stat``."""
    return (st.st_ino, st.st_mtime_ns, st.st_size)


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
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


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``path`` so that readers see either nothing or all of ``data``."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def create_exclusive(path: Path, data: bytes) -> bool:
    """Create ``path`` with all of ``data`` unless it already exists; returns whether it was
    created. Readers never see it partly written. Raises ``OSError`` where hard links aren't
    supported (the file is written aside and linked into place)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
