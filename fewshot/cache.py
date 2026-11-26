"""SQLite cache of embeddings keyed by (embedder spec, content key).

Image keys are the example id (a hash of the file bytes); text keys are
``"text:" + string``. Switching a task's embedder therefore only re-embeds
images that the new model has not seen before.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .embedders import Vectors
from .images import CONVERSION_KEY

if TYPE_CHECKING:
    from .store import Workspace

log = logging.getLogger(__name__)

BUSY_TIMEOUT = 30.0  # seconds to wait for another process's write to finish
# Suffix of the key a flip-averaged image embedding is cached under (as pipeline.FLIP_KEY, which
# writes them); pruning keeps every key of each image in use (see ``image_keys``).
FLIP_KEY = ":flip"


def image_keys(image_id: str) -> tuple[str, ...]:
    """Every key an image's vectors may be cached under: its id, plus ``CONVERSION_KEY`` if its
    conversion to RGB has changed (see ``images.conversion_key``), each plus ``FLIP_KEY`` too."""
    return tuple(image_id + c + f for c in ("", CONVERSION_KEY) for f in ("", FLIP_KEY))


# SQLite's primary result codes for a file that isn't a sound database.
_SQLITE_CORRUPT, _SQLITE_NOTADB = 11, 26


def _damaged(e: sqlite3.DatabaseError) -> bool:
    """Whether ``e`` says the database file itself is damaged, rather than busy or unwritable."""
    code = getattr(e, "sqlite_errorcode", None)
    return code is not None and code & 0xFF in (_SQLITE_CORRUPT, _SQLITE_NOTADB)


class EmbeddingCache:
    """Thread-safe embedding store; vectors are float32 blobs."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._ino: int | None = None  # of the file the connection opened
        try:
            self._conn = self._open()
        except sqlite3.DatabaseError as e:
            if not _damaged(e):  # busy, full, unreadable: the file may be fine, and in use
                raise
            # It is only a cache: move a damaged file aside (for inspection) and start afresh,
            # rather than leaving the server and CLI unable to start.
            self._move_aside(e)
            self._conn = self._open()

    def _move_aside(self, e: sqlite3.DatabaseError) -> None:
        aside = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}")
        log.warning("Embedding cache %s is unreadable (%s); moved it to %s and starting a new one", self.path, e, aside)
        for suffix in ("", "-wal", "-shm"):
            with contextlib.suppress(FileNotFoundError):
                os.replace(f"{self.path}{suffix}", f"{aside}{suffix}")

    def _recover(self, e: sqlite3.DatabaseError) -> None:
        """After ``e``, damage found in the open cache (a read or write reached a corrupt page),
        move the file aside and continue with a new one; raise ``e`` if it isn't damage. Another
        process that found it first may have done so already: then only reopen. Call holding
        ``_lock``."""
        if not _damaged(e):
            raise e
        self._conn.close()
        try:
            same = os.stat(self.path).st_ino == self._ino
        except FileNotFoundError:
            same = False
        if same:
            self._move_aside(e)
        self._conn = self._open()

    def _open(self) -> sqlite3.Connection:
        # The server and CLI may share this file: wait up to BUSY_TIMEOUT for the other's write
        # instead of failing with "database is locked".
        conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=BUSY_TIMEOUT)
        try:
            # WAL: readers never block on a writer, and a writer blocks only other writers.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")  # durable enough for a cache, much faster
            conn.execute("CREATE TABLE IF NOT EXISTS emb (spec TEXT, key TEXT, vec BLOB, PRIMARY KEY (spec, key))")
            conn.commit()
            self._ino = os.stat(self.path).st_ino
        except sqlite3.DatabaseError:
            conn.close()
            raise
        return conn

    def get_many(self, spec: str, keys: list[str]) -> dict[str, Vectors]:
        """Return the cached vectors among ``keys`` (missing keys are simply absent).

        A damaged cache file is moved aside and a new one started (see ``_recover``): every key
        is then missing."""
        found: dict[str, Vectors] = {}
        with self._lock:
            try:
                for i in range(0, len(keys), 500):
                    chunk = keys[i : i + 500]
                    q = f"SELECT key, vec FROM emb WHERE spec=? AND key IN ({','.join('?' * len(chunk))})"
                    for k, blob in self._conn.execute(q, [spec, *chunk]):
                        found[k] = np.frombuffer(blob, dtype=np.float32)
            except sqlite3.DatabaseError as e:
                self._recover(e)
                return {}
        return found

    def put_many(self, spec: str, items: dict[str, Vectors]) -> None:
        """Insert or replace vectors for ``spec`` (into a new cache file if the current one turns
        out to be damaged: see ``_recover``)."""
        if not items:
            return
        rows = [(spec, k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()]
        with self._lock:
            try:
                self._insert(rows)
            except sqlite3.DatabaseError as e:
                self._recover(e)
                self._insert(rows)

    def _insert(self, rows: list[tuple[str, str, bytes]]) -> None:
        try:
            self._conn.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)", rows)
            self._conn.commit()
        except sqlite3.DatabaseError:
            self._conn.rollback()
            raise

    def stats(self) -> list[tuple[str, int, int]]:
        """``(namespace, rows, bytes of vectors)`` per namespace, largest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT spec, COUNT(*), SUM(LENGTH(vec)) FROM emb GROUP BY spec ORDER BY 3 DESC").fetchall()
        return [(ns, n, size or 0) for ns, n, size in rows]

    def prune(self, keep: set[str], dry_run: bool = False) -> int:
        """Delete image vectors whose key isn't in ``keep`` (text vectors are kept); returns how
        many rows were (or, with ``dry_run``, would be) deleted. SQLite reuses the space they took;
        ``vacuum`` gives it back to the file system."""
        with self._lock:
            self._conn.execute("CREATE TEMP TABLE IF NOT EXISTS keep (key TEXT PRIMARY KEY)")
            self._conn.execute("DELETE FROM keep")
            self._conn.executemany("INSERT OR IGNORE INTO keep VALUES (?)", [(k,) for k in keep])
            where = "key NOT LIKE 'text:%' AND key NOT IN (SELECT key FROM keep)"
            if dry_run:
                n: int = self._conn.execute(f"SELECT COUNT(*) FROM emb WHERE {where}").fetchone()[0]
            else:
                n = self._conn.execute(f"DELETE FROM emb WHERE {where}").rowcount
            self._conn.execute("DROP TABLE keep")
            self._conn.commit()
        return n

    def vacuum(self) -> None:
        """Rewrite the cache file without its free space. This can take a while for a large cache,
        and other processes can't write to it meanwhile (they wait up to ``BUSY_TIMEOUT``)."""
        with self._lock:
            self._conn.execute("VACUUM")


def prune_unused(ws: Workspace, dry_run: bool = False, vacuum: bool = False) -> tuple[int, int]:
    """Delete the vectors in ``ws``'s cache that no task uses: neither its examples and inbox nor
    the images undo or redo may bring back (text vectors are kept). Unless ``dry_run``, stored
    images no task links to go too. Returns how many vectors were (or, with ``dry_run``, would be)
    deleted, and how many stored images. ``vacuum`` then gives the freed space back to the file
    system, which rewrites the whole cache and keeps other writers waiting; without it SQLite
    reuses the space.

    Every task is locked while the vectors to keep are worked out and the others deleted, so an
    image added meanwhile can't lose its vector. Raises ``ValueError`` if a task can't be read,
    since its vectors would be lost."""
    tasks, unreadable = [], []
    for p in ws.task_folders():  # one listing, so a task made meanwhile isn't taken for unreadable
        try:
            tasks.append(ws.get(p.name))
        except KeyError:
            if (p / "task.json").exists():
                unreadable.append(p.name)
    if unreadable:
        raise ValueError(f"Can't read task(s) {', '.join(unreadable)}; fix or remove them before pruning")
    cache = EmbeddingCache(ws.cache_path)
    with contextlib.ExitStack() as locks:
        for t in list(tasks):
            try:
                locks.enter_context(t.locked())  # one consistent view of every task
            except KeyError:  # deleted meanwhile: its vectors needn't be kept
                tasks.remove(t)
        ids = {e.id for t in tasks for e in [*t.config.examples, *t.config.inbox]}
        ids |= {f.split("/", 1)[1].rsplit(".", 1)[0] for t in tasks for f in t.kept_files()}
        n = cache.prune({k for i in ids for k in image_keys(i)}, dry_run=dry_run)
    if dry_run:
        return n, 0
    if vacuum:
        cache.vacuum()
    return n, ws.gc_blobs()
