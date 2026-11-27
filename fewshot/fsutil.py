"""File helpers for the task store: change stamps, cross-process locks and durable atomic writes."""

from __future__ import annotations

import contextlib
import errno
import functools
import importlib
import os
import shutil
import tempfile
import time
import uuid
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
def file_lock(path: Path, shared: bool = False, guard: Path | None = None) -> Iterator[None]:
    """Hold a lock on ``path`` (created if missing) across processes: ``flock`` on Linux and
    macOS, ``msvcrt.locking`` (always exclusive) on Windows. An exclusive lock waits for every
    other holder; a ``shared`` one only for exclusive holders. Neither needs write access to an
    existing ``path`` (where it can't be created either, as in a read-only folder, nobody can be
    changing what it guards, and a shared lock locks nothing). A new lock file gets the
    permissions an ordinary new file does, so the group can lock it in a shared workspace.

    With ``guard``, a missing ``path`` is only created while ``guard`` exists (a task's lock
    only in a folder holding its task.json): a lock taken late on a deleted task must not leave
    a file in the empty folder a new task of that name is about to replace.

    Raises ``FileNotFoundError`` if the folder holding ``path`` (or ``guard``) is gone.
    """
    while True:
        try:
            fd = _open_lock(path, shared, guard)
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


def _open_lock(path: Path, shared: bool, guard: Path | None = None) -> int:
    # flock needs no write access; msvcrt.locking (exclusive) wants a file open for writing.
    mode = os.O_RDONLY if shared or fcntl is not None else os.O_RDWR
    with contextlib.suppress(FileNotFoundError):
        return os.open(path, mode)
    if guard is not None and not guard.exists():
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(guard))
    try:
        return os.open(path, mode | os.O_CREAT, 0o666)  # the umask applies, as to any new file
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


def _read_umask() -> int:
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


_UMASK = _read_umask()  # read once: setting it to read it isn't thread-safe


def mkstemp(path: Path) -> tuple[int, str]:
    """A new temporary file next to ``path`` (``.<name>.*.tmp``), open for writing, with the
    permissions an ordinary new file gets (``tempfile`` makes it 0600 whatever the umask, which
    would lock the group out of a shared workspace). Returns its descriptor and path."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    if hasattr(os, "fchmod"):  # pragma: no branch  # POSIX
        try:
            os.fchmod(fd, 0o666 & ~_UMASK)
        except BaseException:
            os.close(fd)
            os.unlink(tmp)
            raise
    return fd, tmp


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``path`` so that readers see either nothing or all of ``data``, and the content is on
    disk before the name points at it. (The rename itself is durable after ``fsync_dir``.)"""
    fd, tmp = mkstemp(path)
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
    fd, tmp = mkstemp(path)
    try:
        _write_synced(fd, data)
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def create_lock_file(folder: Path) -> None:
    """Create the lock file of a task being built in ``folder``: from then on, locking the task
    never creates one (see ``file_lock``'s ``guard``)."""
    os.close(os.open(folder / ".lock", os.O_RDONLY | os.O_CREAT, 0o666))


def replace_reserved(build: Path, p: Path, tries: int = 3) -> None:
    """Move the finished task folder ``build`` onto its empty reserved folder ``p`` (POSIX
    replaces an empty folder). A lock file in ``p``, left by a lock on the deleted task of that
    name taken just as its folder was reserved (a lock is only created beside a task.json, so the
    window is tiny), is no part of anything: it is removed, and the move tried again."""
    for _ in range(tries - 1):
        try:
            os.replace(build, p)
            return
        except OSError as e:
            if e.errno not in (errno.ENOTEMPTY, errno.EEXIST) or [q.name for q in p.iterdir()] != [".lock"]:
                raise
            (p / ".lock").unlink(missing_ok=True)
    os.replace(build, p)


def move_back(dest: Path, src: Path) -> bool:
    """Move folder ``dest`` back to ``src``, a name it had just given up; returns whether it did.
    Another process may have reserved ``src`` (an empty folder) for a new task meanwhile, which a
    plain rename would replace (and whose build would then fail and remove what was moved there):
    the move only goes onto a folder reserved again by this caller."""
    try:
        src.mkdir()
    except FileExistsError:
        return False
    os.replace(dest, src)  # onto our own empty reservation (POSIX)
    return True


@functools.lru_cache(maxsize=64)
def folds_case(folder: Path) -> bool:
    """Whether the file system holding ``folder`` ignores case in names (macOS and Windows by
    default): whether ``folder`` is found under its name in capitals too. Found once per folder."""
    try:
        return os.path.samefile(folder, folder.with_name(folder.name.upper()))
    except OSError:
        return False


def spelt_as_listed(folder: Path, name: str) -> bool:
    """Whether entry ``name`` of ``folder``, if there is one, is spelt as ``name``. Where case is
    ignored, opening ``FOO`` would open ``foo``: a task folder would then be opened as a second
    task under another name (and a save would write that name into its task.json)."""
    if not folds_case(folder):
        return True
    try:
        return name in os.listdir(folder)
    except OSError:
        return True  # the open that follows reports it


def os_reason(e: Exception) -> str:
    """The message of ``e``: an OSError's description (with the file), not its repr."""
    if isinstance(e, OSError) and e.strerror:
        return f"{e.filename}: {e.strerror}" if e.filename else e.strerror
    return str(e)


def abandoned(folder: Path, stale_seconds: float, locked: bool = False) -> bool:
    """Whether ``folder`` is a leftover of a crash: no task.json (or hidden), untouched for
    ``stale_seconds`` (so it isn't a folder just made, before its builder locked it) and not
    locked (``dir_lock``) by a create, duplicate or import still building it, however long that
    takes (unless ``locked``: the caller holds that lock itself)."""
    try:
        age = time.time() - folder.stat().st_mtime
    except FileNotFoundError:
        return False
    return (age > stale_seconds and (folder.name.startswith(".") or not (folder / "task.json").exists())
            and (locked or not dir_locked(folder)))


def reclaim(folder: Path, stale_seconds: float) -> bool:
    """Delete ``folder`` if it is a crash's leftover (see ``abandoned``); returns whether it went.

    Judged again while holding its lock (taken without waiting) and moved aside to a ``.trash-``
    name before it is removed, so a folder that another process took back and started building
    in meanwhile is never the one deleted."""
    if not abandoned(folder, stale_seconds):
        return False
    with claim_dir(folder) as held:
        if not held or not abandoned(folder, stale_seconds, locked=True):
            return False
        trash = folder.parent / f".trash-{uuid.uuid4().hex}"
        try:
            os.rename(folder, trash)
        except FileNotFoundError:  # pragma: no cover  # removed by a process not taking the lock
            return False
    shutil.rmtree(trash, ignore_errors=True)
    return True
