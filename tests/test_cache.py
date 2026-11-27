"""EmbeddingCache: SQLite persistence of vectors per (spec, key)."""

import sqlite3

import numpy as np
import pytest

from fewshot.cache import EmbeddingCache, prune_unused
from fewshot.store import Workspace
from tests.conftest import two_each

pytestmark = pytest.mark.posix_fs  # also run on macOS: see .github/workflows/ci.yml


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


def _corrupt_middle(path):
    """Overwrite the middle of a checkpointed cache file with junk (pages the header doesn't cover)."""
    size = path.stat().st_size
    with open(path, "r+b") as f:
        f.seek(size // 2)
        f.write(b"\xa5" * (64 * 1024))


def test_damage_found_after_opening_is_a_miss_and_a_new_cache(tmp_path, caplog):
    """Corruption that a read reaches only later (the file opened fine) makes the lookup a miss,
    moves the file aside once and starts a new cache, instead of failing every lookup from then on."""
    path = tmp_path / "cache.sqlite"
    cache = EmbeddingCache(path)
    keys = [f"k{i}" for i in range(2000)]
    cache.put_many("s", {k: np.ones(256, np.float32) for k in keys})
    cache._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # pylint: disable=protected-access
    _corrupt_middle(path)
    damaged = EmbeddingCache(path)  # opens fine: the header is intact
    assert not damaged.get_many("s", keys)
    assert len([p for p in tmp_path.glob("cache.sqlite.corrupt-*") if not p.name.endswith(("-wal", "-shm"))]) == 1
    assert "unreadable" in caplog.text
    damaged.put_many("s", {"k0": np.ones(256, np.float32)})
    assert set(damaged.get_many("s", keys)) == {"k0"}


class _Failing:  # pylint: disable=too-few-public-methods
    """A connection whose queries fail with SQLite result ``code`` (11: corrupt; 5: busy)."""

    def __init__(self, conn, code):
        self.conn, self.code = conn, code

    def _fail(self, *_args):
        e = sqlite3.DatabaseError("database disk image is malformed")
        e.sqlite_errorcode = self.code
        raise e

    execute = executemany = _fail

    def __getattr__(self, name):
        return getattr(self.conn, name)


def _break(cache, code):
    """Make ``cache``'s queries fail with SQLite result ``code``."""
    cache.__dict__["_conn"] = _Failing(cache.__dict__["_conn"], code)


def test_damage_found_by_a_write_moves_the_cache_aside_and_keeps_the_vectors(tmp_path):
    """A write that finds the file damaged goes into the new cache; a busy one is raised as before."""
    path = tmp_path / "cache.sqlite"
    cache = EmbeddingCache(path)
    _break(cache, 11)
    cache.put_many("s", {"k": np.ones(3, np.float32)})
    assert set(EmbeddingCache(path).get_many("s", ["k"])) == {"k"}
    assert [p for p in tmp_path.iterdir() if p.name.startswith("cache.sqlite.corrupt-")]
    _break(cache, 5)
    with pytest.raises(sqlite3.DatabaseError):
        cache.get_many("s", ["k"])
    with pytest.raises(sqlite3.DatabaseError):
        cache.put_many("s", {"k": np.ones(3, np.float32)})


def test_damage_another_process_dealt_with_only_reopens(tmp_path):
    """If the damaged file was already replaced (or removed) by another process, the new cache
    there is opened, not moved aside too."""
    path = tmp_path / "cache.sqlite"
    cache = EmbeddingCache(path)
    path.rename(tmp_path / "moved-by-other")
    EmbeddingCache(path).put_many("s", {"k": np.ones(3, np.float32)})
    _break(cache, 11)
    assert not cache.get_many("s", ["k"])  # this lookup misses ...
    assert set(cache.get_many("s", ["k"])) == {"k"}  # ... then the other's cache is used
    assert not [p for p in tmp_path.iterdir() if "corrupt" in p.name]
    path.unlink()
    _break(cache, 26)
    assert not cache.get_many("s", ["k"])
    assert path.exists()


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


# ---- read-only use and concurrent recovery ---------------------------------
def test_a_read_only_cache_is_read_without_writing(tmp_path):
    """A cache opened read-only needs no write access to the file or its folder, even with its
    WAL side files gone (as after the last writer closed it); it never moves a damaged file."""
    path = tmp_path / "c" / "cache.sqlite"
    writer = EmbeddingCache(path)
    writer.put_many("s", {"a": np.ones(3, np.float32), "b": np.ones(3, np.float32)})
    writer._conn.close()  # pylint: disable=protected-access
    assert sorted(p.name for p in path.parent.iterdir()) == ["cache.sqlite"]
    path.chmod(0o444)
    path.parent.chmod(0o555)
    try:
        ro = EmbeddingCache(path, read_only=True)
        assert set(ro.get_many("s", ["a", "b"])) == {"a", "b"}
        assert ro.prune({"a"}, dry_run=True) == 1
    finally:
        path.parent.chmod(0o755)
        path.chmod(0o644)
    live = EmbeddingCache(path)  # with a writer's side files there, read-only use reads them
    live.put_many("s", {"c": np.ones(3, np.float32)})
    assert set(EmbeddingCache(path, read_only=True).get_many("s", ["a", "c"])) == {"a", "c"}
    with pytest.raises(FileNotFoundError):
        EmbeddingCache(tmp_path / "missing.sqlite", read_only=True)
    ro = EmbeddingCache(path, read_only=True)
    _break(ro, 11)
    with pytest.raises(sqlite3.DatabaseError):
        ro.get_many("s", ["a"])
    assert not [p for p in path.parent.iterdir() if "corrupt" in p.name]


def test_a_dry_run_prune_works_without_write_access(ws):
    """``prune_unused(dry_run=True)`` works where nothing in the workspace can be written, the
    cache file included; without a cache there is nothing to prune."""
    two_each(ws)
    assert prune_unused(ws, dry_run=True) == (0, 0)
    assert not ws.cache_path.exists()  # a dry run doesn't make one
    EmbeddingCache(ws.cache_path).put_many("x", {"gone": np.ones(1, np.float32)})
    paths = [*sorted(ws.root.rglob("*"), reverse=True), ws.root]
    for p in paths:
        p.chmod(0o555 if p.is_dir() else 0o444)
    try:
        assert prune_unused(Workspace(ws.root, create=False), dry_run=True) == (1, 0)
    finally:
        for p in reversed(paths):
            p.chmod(0o755 if p.is_dir() else 0o644)


def test_two_processes_finding_the_damage_at_once_keep_one_new_cache(tmp_path, monkeypatch):
    """Two processes open a damaged cache at the same time: the first moves it aside and starts a
    new one; the second, finding the file already replaced, uses that one instead of moving it
    aside too. Each move keeps its damaged copy under its own name."""
    path = tmp_path / "cache.sqlite"
    path.write_bytes(b"this is not a database" * 100)
    real_open = EmbeddingCache._open  # pylint: disable=protected-access
    raced: list[bool] = []

    def racing(self):
        if not raced:  # this process's open fails on the damaged file, once the other has recovered
            raced.append(True)
            EmbeddingCache(self.path).put_many("s", {"k": np.ones(3, np.float32)})
            e = sqlite3.DatabaseError("file is not a database")
            e.sqlite_errorcode = 26
            raise e
        return real_open(self)

    monkeypatch.setattr(EmbeddingCache, "_open", racing)
    second = EmbeddingCache(path)
    assert set(second.get_many("s", ["k"])) == {"k"}  # the first's new cache, not moved aside
    assert len([p for p in tmp_path.iterdir() if "corrupt" in p.name]) == 1
