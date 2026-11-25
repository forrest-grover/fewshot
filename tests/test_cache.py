"""EmbeddingCache: SQLite persistence of vectors per (spec, key)."""

import sqlite3

import numpy as np

from fewshot.cache import EmbeddingCache


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
