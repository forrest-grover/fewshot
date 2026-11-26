"""EmbeddingCache: SQLite persistence of vectors per (spec, key)."""

import sqlite3

import numpy as np
import pytest

from fewshot.cache import EmbeddingCache, prune_unused
from fewshot.store import Workspace
from tests.conftest import two_each


def _rows(path):
    with sqlite3.connect(str(path)) as conn:
        return conn.execute("SELECT spec, key FROM emb ORDER BY spec, key").fetchall()


def test_creates_parent_folder_and_round_trips(tmp_path):
    """Vectors come back as float32 with the same values; unknown keys are absent."""
    path = tmp_path / "deep" / "dir" / "cache.sqlite"
    c = EmbeddingCache(path)
    assert path.exists()
    c.put_many("s", {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([0.5, 0.25], np.float64)})
    got = c.get_many("s", ["a", "b", "missing"])
    assert set(got) == {"a", "b"}
    assert got["a"].dtype == np.float32
    assert got["a"].tolist() == [1.0, 2.0, 3.0]
    assert got["b"].tolist() == [0.5, 0.25]


def test_specs_are_separate_and_put_replaces(tmp_path):
    """The same key under two specs holds two vectors; re-putting overwrites."""
    c = EmbeddingCache(tmp_path / "c.sqlite")
    c.put_many("s1", {"k": np.ones(2, np.float32)})
    c.put_many("s2", {"k": np.zeros(2, np.float32)})
    c.put_many("s1", {"k": np.full(2, 7.0, np.float32)})
    assert c.get_many("s1", ["k"])["k"].tolist() == [7.0, 7.0]
    assert c.get_many("s2", ["k"])["k"].tolist() == [0.0, 0.0]
    assert not c.get_many("s3", ["k"])


def test_empty_inputs(tmp_path):
    """No keys returns nothing; putting nothing writes nothing."""
    path = tmp_path / "c.sqlite"
    c = EmbeddingCache(path)
    c.put_many("s", {})
    assert not _rows(path)
    assert not c.get_many("s", [])


def test_lookup_of_more_keys_than_one_query_chunk(tmp_path):
    """Lookups beyond the 500-parameter chunk size find keys in every chunk."""
    c = EmbeddingCache(tmp_path / "c.sqlite")
    keys = [f"k{i}" for i in range(1203)]
    c.put_many("s", {k: np.array([float(i)]) for i, k in enumerate(keys) if i % 2 == 0})
    got = c.get_many("s", keys)
    assert len(got) == 602
    assert got["k0"].tolist() == [0.0]
    assert got["k600"].tolist() == [600.0]
    assert got["k1202"].tolist() == [1202.0]
    assert "k1" not in got


def test_persists_across_instances(tmp_path):
    """A second cache on the same file sees what the first wrote."""
    path = tmp_path / "c.sqlite"
    EmbeddingCache(path).put_many("s", {"k": np.array([4.0])})
    assert EmbeddingCache(path).get_many("s", ["k"])["k"].tolist() == [4.0]
    assert _rows(path) == [("s", "k")]


def test_wal_mode_and_busy_timeout(tmp_path):
    """The cache uses WAL journaling so the server and CLI can share it."""
    c = EmbeddingCache(tmp_path / "c.sqlite")
    with sqlite3.connect(str(c.path)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_stats_and_prune(tmp_path):
    """prune drops image vectors not in ``keep`` (texts stay); stats sums per namespace."""
    c = EmbeddingCache(tmp_path / "c.sqlite")
    c.put_many("s1", {"keep": np.ones(4, np.float32), "old": np.ones(4, np.float32), "text:x": np.ones(4, np.float32)})
    c.put_many("s2", {"old2": np.ones(2, np.float32)})
    assert c.stats() == [("s1", 3, 48), ("s2", 1, 8)]
    assert c.prune({"keep"}, dry_run=True) == 2
    assert len(_rows(c.path)) == 4
    assert c.prune({"keep"}) == 2
    assert _rows(c.path) == [("s1", "keep"), ("s1", "text:x")]
    assert c.prune({"keep"}) == 0


# ---- damaged and busy caches ---------------------------------------------
def test_corrupt_cache_is_moved_aside(tmp_path, caplog):
    """A damaged cache file is renamed (with its WAL files) and a new cache started."""
    path = tmp_path / "cache.sqlite"
    path.write_bytes(b"this is not a database" * 100)
    (tmp_path / "cache.sqlite-wal").write_bytes(b"junk")
    cache = EmbeddingCache(path)
    cache.put_many("s", {"k": np.ones(3, np.float32)})
    assert set(cache.get_many("s", ["k"])) == {"k"}
    names = sorted(p.name for p in tmp_path.iterdir())
    assert any(n.startswith("cache.sqlite.corrupt-") and not n.endswith(("-wal", "-shm")) for n in names)
    assert "unreadable" in caplog.text


def test_busy_cache_is_not_moved_aside(tmp_path, monkeypatch):
    """A cache that is only locked or unwritable is reported, never mistaken for a damaged one."""
    path = tmp_path / "cache.sqlite"
    EmbeddingCache(path).put_many("s", {"k": np.ones(3, np.float32)})

    def locked(_self):
        e = sqlite3.OperationalError("database is locked")
        e.sqlite_errorcode = 5  # SQLITE_BUSY
        raise e

    monkeypatch.setattr(EmbeddingCache, "_open", locked)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        EmbeddingCache(path)
    assert not [p for p in tmp_path.iterdir() if "corrupt" in p.name]


# ---- pruning -------------------------------------------------------------
def test_prune_cache_holds_the_task_locks(ws, monkeypatch):
    """Pruning decides what to keep and deletes the rest with every task locked."""
    t = two_each(ws)
    seen = []
    real = EmbeddingCache.prune

    def prune(self, keep, dry_run=False):
        """Record whether the task lock is held."""
        seen.append(t._lock._is_owned())  # pylint: disable=protected-access
        return real(self, keep, dry_run)

    monkeypatch.setattr(EmbeddingCache, "prune", prune)
    assert prune_unused(ws, ) == (0, 0)
    assert seen == [True]


def test_prune_cache_vacuums_only_when_asked(ws, monkeypatch):
    """VACUUM, which blocks other writers, runs only with vacuum=True."""
    two_each(ws)
    cache = EmbeddingCache(ws.cache_path)
    cache.put_many("fake:color", {"gone": np.array([1.0, 0.0, 0.0], np.float32)})
    vacuums: list[int] = []
    monkeypatch.setattr(EmbeddingCache, "vacuum", lambda self: vacuums.append(1))
    assert prune_unused(ws, dry_run=True) == (1, 0)
    assert prune_unused(ws, ) == (1, 0)
    assert not vacuums
    cache.put_many("fake:color", {"gone": np.array([1.0, 0.0, 0.0], np.float32)})
    monkeypatch.undo()
    assert prune_unused(ws, vacuum=True) == (1, 0)
    with sqlite3.connect(ws.cache_path) as db:
        assert db.execute("SELECT COUNT(*) FROM emb WHERE key='gone'").fetchone() == (0,)


def test_prune_skips_a_task_deleted_meanwhile(ws, monkeypatch):
    """A task deleted between listing and locking is left out, not reported as an error."""
    ws.create("a")
    ws.create("b")
    (ws.root / "tasks" / "empty").mkdir()  # a folder without task.json is no task, not unreadable
    real = ws.get

    def get_then_delete(name):
        t = real(name)
        if name == "b":
            Workspace(ws.root).delete("b")  # another process, just after this one read it
        return t

    monkeypatch.setattr(ws, "get", get_then_delete)
    assert prune_unused(ws) == (0, 0)
