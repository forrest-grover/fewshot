"""SQLite cache of embeddings keyed by (embedder spec, content key).

Image keys are the example id (a hash of the file bytes); text keys are
``"text:" + string``. Switching a task's embedder therefore only re-embeds
images that the new model has not seen before.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np

from .embedders import Vectors

BUSY_TIMEOUT = 30.0  # seconds to wait for another process's write to finish


class EmbeddingCache:
    """Thread-safe embedding store; vectors are float32 blobs."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # The server and CLI may share this file: wait up to BUSY_TIMEOUT for the other's write
        # instead of failing with "database is locked".
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=BUSY_TIMEOUT)
        self._lock = threading.Lock()
        with self._lock:
            # WAL: readers never block on a writer, and a writer blocks only other writers.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")  # durable enough for a cache, much faster
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS emb (spec TEXT, key TEXT, vec BLOB, PRIMARY KEY (spec, key))"
            )
            self._conn.commit()

    def get_many(self, spec: str, keys: list[str]) -> dict[str, Vectors]:
        """Return the cached vectors among ``keys`` (missing keys are simply absent)."""
        found: dict[str, Vectors] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i : i + 500]
                q = f"SELECT key, vec FROM emb WHERE spec=? AND key IN ({','.join('?' * len(chunk))})"
                for k, blob in self._conn.execute(q, [spec, *chunk]):
                    found[k] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, spec: str, items: dict[str, Vectors]) -> None:
        """Insert or replace vectors for ``spec``."""
        if not items:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO emb VALUES (?, ?, ?)",
                [(spec, k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()],
            )
            self._conn.commit()

    def stats(self) -> list[tuple[str, int, int]]:
        """``(namespace, rows, bytes of vectors)`` per namespace, largest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT spec, COUNT(*), SUM(LENGTH(vec)) FROM emb GROUP BY spec ORDER BY 3 DESC").fetchall()
        return [(ns, n, size or 0) for ns, n, size in rows]

    def prune(self, keep: set[str], dry_run: bool = False) -> int:
        """Delete image vectors whose key isn't in ``keep`` (text vectors are kept); returns how
        many rows were (or, with ``dry_run``, would be) deleted. Frees the disk space afterwards."""
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
            if n and not dry_run:
                self._conn.execute("VACUUM")
        return n
