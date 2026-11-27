"""The workspace's shared image store: each image stored once in ``blobs/<id[:2]>/<id>.<ext>`` and
hard-linked into every task folder that uses it (copied where hard links aren't possible).

A blob is removed when the last task file linking to it goes: its link count says how many
task files still share it.
"""

from __future__ import annotations

import contextlib
import os
import stat
import time
from pathlib import Path

from .fsutil import create_exclusive, write_atomic


def blob_path(blobs: Path | None, file: str) -> Path | None:
    """The shared-store file for a task's ``images/<name>``, or None without a store."""
    name = file.rsplit("/", 1)[-1]
    # Ids from before 128-bit ids are too short to rule out a crafted collision, which in a
    # shared store would let one task's image stand in for another's: those stay task-local.
    if blobs is None or len(name.split(".", 1)[0]) != 32:
        return None
    return blobs / name[:2] / name


def intact(path: Path, data: bytes) -> bool:
    """Whether the stored file ``path``, named by the hash of ``data``, exists at full length.

    A power cut can leave a file short; files are now synced before they are linked into place,
    so only one written by an older version (or a file system that zero-fills, such as XFS after
    a crash) could still be damaged at full length, which this doesn't detect."""
    try:
        return path.stat().st_size == len(data)
    except FileNotFoundError:
        return False


def store_image(task_dir: Path, file: str, data: bytes, blobs: Path | None, unsynced: set[Path]) -> bool:
    """Write ``data`` (whose id the name of ``file`` carries) as ``task_dir/file``: a hard link to
    the workspace's copy, made if needed, or a plain copy where links aren't possible. Folders
    that gained an entry are added to ``unsynced``, to be synced before a task.json names it.
    Returns whether this call created the file (False if it was there already)."""
    dest = task_dir / file
    dest.parent.mkdir(exist_ok=True)
    if intact(dest, data):  # kept for undo, or re-added: same content, since the name is its hash
        return False
    dest.unlink(missing_ok=True)  # cut short by a power cut: store it again
    unsynced.add(dest.parent)
    blob = blob_path(blobs, file)
    if blob is not None:
        made = False
        try:
            blob.parent.mkdir(parents=True, exist_ok=True)
            made = create_exclusive(blob, data)
            if not made and not intact(blob, data):  # a damaged copy must not spread to this task
                write_atomic(blob, data)
                made = True
            if made:
                unsynced.add(blob.parent)
            os.link(blob, dest)
            return True
        except FileExistsError:  # another writer stored this file in this task first
            return False
        except OSError:  # no hard links here, or the blob vanished under us: copy instead
            if made:  # a blob nothing can link to would only waste space
                blob.unlink(missing_ok=True)
    write_atomic(dest, data)
    return True


def link_image(task_dir: Path, file: str, src: Path, blobs: Path | None, unsynced: set[Path]) -> None:
    """Store another task's image file ``src`` as ``task_dir/file``: as another hard link to the
    same file (and so to its shared copy, if it has one) where possible, as ``store_image`` does
    otherwise."""
    dest = task_dir / file
    dest.parent.mkdir(exist_ok=True)
    try:
        os.link(src, dest)
    except FileExistsError:
        return
    except OSError:  # no hard links here
        store_image(task_dir, file, src.read_bytes(), blobs, unsynced)
        return
    unsynced.add(dest.parent)


def release(blob: Path | None) -> None:
    """Delete ``blob`` if no task file links to it any more."""
    if blob is None:
        return
    with contextlib.suppress(FileNotFoundError):
        if blob.stat().st_nlink == 1:
            blob.unlink()


def drop_file(task_dir: Path, file: str, blobs: Path | None) -> None:
    """Delete a task image file, and the shared copy once no task links to it."""
    (task_dir / file).unlink(missing_ok=True)
    release(blob_path(blobs, file))


# A shared copy written this recently may be one being stored right now, not yet linked into its
# task (or the temporary file it is written to): gc leaves it for next time.
GC_MIN_AGE = 60.0


def gc_blobs(blobs: Path, min_age: float = GC_MIN_AGE, dry_run: bool = False) -> int:
    """Delete shared image copies that no task links to any more, and temporary files a crashed
    write left, if written over ``min_age`` seconds ago; returns how many (with ``dry_run``, how
    many would go, deleting nothing)."""
    n = 0
    now = time.time()
    for f in blobs.glob("*/*") if blobs.is_dir() else []:
        with contextlib.suppress(FileNotFoundError):
            st = f.stat()
            if stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and now - st.st_mtime > min_age:
                if not dry_run:
                    f.unlink()
                n += 1
    return n
