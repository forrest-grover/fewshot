"""Second-review improvements: cache recovery, model memory, device checks, batch sizes, fits,
duplicate images and the fast leave-one-out path."""

import sqlite3

import numpy as np
import pytest
import torch
from PIL import Image

import fewshot.evaluate as ev
from fewshot import embedders
from fewshot.cache import EmbeddingCache
from fewshot.classifiers import ClassData
from fewshot.models import TaskConfig
from fewshot.pipeline import Pipeline
from tests.conftest import ColorEmbedder, png

RED, BLUE = (220, 30, 30), (30, 30, 220)


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


def test_models_in_use_are_not_unloaded(monkeypatch):
    """Eviction skips a model that is embedding right now, even past the limit."""
    busy, idle, new = ColorEmbedder(), ColorEmbedder(), ColorEmbedder()
    monkeypatch.setattr(embedders, "_loaded", {"m:busy": busy, "m:idle": idle, "m:new": new})
    monkeypatch.setenv("FEWSHOT_MAX_MODELS", "1")
    busy.users = 1
    with embedders._load_lock:  # pylint: disable=protected-access
        gone = embedders._evict_locked(keep="m:new")  # pylint: disable=protected-access
    assert gone == ["m:idle"]
    assert list(embedders._loaded) == ["m:busy", "m:new"]  # pylint: disable=protected-access


def test_use_count_follows_embedding():
    """users counts embed calls in progress, and drops back on errors too."""
    seen = []

    class Watch(ColorEmbedder):
        """Records the use count during each batch; fails on batches of 2."""

        def _embed_image_batch(self, images):
            seen.append(self.users)
            if len(images) == 2:
                raise RuntimeError("boom")
            return super()._embed_image_batch(images)

    emb = Watch()
    emb.embed_images([Image.new("RGB", (2, 2))])
    with pytest.raises(RuntimeError):
        emb.embed_images([Image.new("RGB", (2, 2))] * 2)
    assert seen == [1, 1]
    assert emb.users == 0


def test_batch_size_grows_back(monkeypatch):
    """After GROW_AFTER batches that fit, a halved batch size doubles again, up to the original."""
    monkeypatch.setattr(embedders, "_free_gpu_memory", lambda: None)
    monkeypatch.setattr(embedders, "GROW_AFTER", 3)
    fail = {"next": True}

    class Once(ColorEmbedder):
        """Runs out of memory once."""

        def _embed_prepared(self, batch):
            if fail["next"] and len(batch) > 1:
                fail["next"] = False
                raise RuntimeError("CUDA out of memory")
            return super()._embed_prepared(batch)

    emb = Once()
    emb.batch_size = 4
    images = [Image.new("RGB", (2, 2), (i, 0, 0)) for i in range(4)]
    emb.embed_images(images)
    assert emb.batch_size == 2
    emb.embed_images(images)  # with the two halves before, three batches have fitted
    assert emb.batch_size == 4
    emb.embed_images(images)
    assert emb.batch_size == 4  # never above the original


def test_device_and_model_limit_checks(monkeypatch):
    """A device that isn't there, or a bad FEWSHOT_MAX_MODELS, fails with a clear message."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("FEWSHOT_DEVICE", "cuda:1")
    with pytest.raises(ValueError, match="has 1 CUDA devices"):
        embedders._device()  # pylint: disable=protected-access
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setenv("FEWSHOT_DEVICE", "mps")
    with pytest.raises(ValueError, match=r"MPS\) is not available"):
        embedders._device()  # pylint: disable=protected-access
    monkeypatch.setenv("FEWSHOT_DEVICE", "cuda:0")
    assert embedders._device() == "cuda:0"  # pylint: disable=protected-access
    monkeypatch.setenv("FEWSHOT_MAX_MODELS", "two")
    with pytest.raises(ValueError, match="must be a whole number"):
        embedders.max_models()


def test_free_gpu_memory_on_mps(monkeypatch):
    """Unloading a model empties Apple's GPU cache too."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
    embedders._free_gpu_memory()  # pylint: disable=protected-access
    assert calls == ["mps"]


def test_an_older_fit_does_not_replace_a_newer_one(ws):
    """A slow fit of an older revision, finishing late, leaves the newer fit cached."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    old = t.config
    t.add_example(png(BLUE), "blue")
    pipe = Pipeline(ws)
    newer = pipe.fitted(t)
    pipe.fitted(t, cfg=old)  # the late, older fit
    assert pipe.fitted(t) is newer


def test_repeated_images_are_embedded_once(ws):
    """The same uncached image twice in one call goes through the model once; a repeated bad
    image gets the error at every position."""
    count = []

    class Count(ColorEmbedder):
        """Counts the images it embeds."""

        def _embed_image_batch(self, images):
            count.append(len(images))
            return super()._embed_image_batch(images)

    embedders.register_embedder("fake:count", Count())
    try:
        pipe = Pipeline(ws)
        v = pipe.embed_images("fake:count", [("a", png(RED)), ("b", png(BLUE)), ("a", png(RED))])
        assert sum(count) == 2
        assert np.allclose(v[0], v[2])
        errors: dict[int, str] = {}
        pipe.embed_images("fake:count", [("bad", b"junk"), ("bad", b"junk")], errors)
        assert set(errors) == {0, 1}
    finally:
        embedders.unregister_embedder("fake:count")


def test_fast_leave_one_out_with_unused_descriptions(monkeypatch):
    """With text_weight 0 and every class keeping an example, descriptions don't matter, so the
    closed form is used; a class that would be left with only its description refits."""
    X = np.eye(4, dtype=np.float32)[[0, 0, 1, 1, 2]] + 0.01
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    text = {0: np.eye(4, dtype=np.float32)[3]}
    two_each = ClassData(["a", "b"], X[:4], np.array([0, 0, 1, 1]), text)
    used: list[int] = []
    real = ev._fast_leave_one_out  # pylint: disable=protected-access
    def counted(*args):
        used.append(1)
        return real(*args)

    monkeypatch.setattr(ev, "_fast_leave_one_out", counted)
    ev._cross_val_predict(two_each, TaskConfig(name="t", classifier="prototype", text_weight=0.0), 4, 0)  # pylint: disable=protected-access
    assert used == [1]
    ev._cross_val_predict(two_each, TaskConfig(name="t", classifier="prototype", text_weight=1.0), 4, 0)  # pylint: disable=protected-access
    single = ClassData(["a", "b", "c"], X, np.array([0, 0, 1, 1, 2]), text)
    ev._cross_val_predict(single, TaskConfig(name="t", classifier="prototype", text_weight=0.0), 5, 0)  # pylint: disable=protected-access
    assert used == [1]
