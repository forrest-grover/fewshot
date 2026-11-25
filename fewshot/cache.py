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


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS emb (spec TEXT, key TEXT, vec BLOB, PRIMARY KEY (spec, key))"
            )
            self._conn.commit()

    def get_many(self, spec: str, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i : i + 500]
                q = f"SELECT key, vec FROM emb WHERE spec=? AND key IN ({','.join('?' * len(chunk))})"
                for k, blob in self._conn.execute(q, [spec, *chunk]):
                    found[k] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, spec: str, items: dict[str, np.ndarray]) -> None:
        if not items:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO emb VALUES (?, ?, ?)",
                [(spec, k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()],
            )
            self._conn.commit()
