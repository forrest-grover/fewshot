"""Devices and GPU memory: device choice, out-of-memory batch halving and growing back, and which
models stay loaded."""

import numpy as np
import pytest
import torch
from PIL import Image

from fewshot import embedders
from fewshot.pipeline import Pipeline
from tests.conftest import ColorEmbedder, TextColor


# ---- devices -------------------------------------------------------------
def test_device_from_environment_and_mps(monkeypatch):
    """FEWSHOT_DEVICE wins (and must be a torch device); otherwise CUDA, then MPS, then CPU."""
    monkeypatch.setenv("FEWSHOT_DEVICE", "cpu")
    assert embedders._device() == "cpu"  # pylint: disable=protected-access
    monkeypatch.setenv("FEWSHOT_DEVICE", "warp-drive")
    with pytest.raises(ValueError, match="not a torch device"):
        embedders._device()  # pylint: disable=protected-access
    monkeypatch.delenv("FEWSHOT_DEVICE")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert embedders._device() == "mps"  # pylint: disable=protected-access
    with embedders._autocast("mps"):  # pylint: disable=protected-access
        assert not torch.is_autocast_enabled("cpu")


def test_a_bad_device_is_not_reported_as_a_model_that_wont_load(monkeypatch):
    """get_embedder reports a bad FEWSHOT_DEVICE as such, before trying to build the model;
    check_device names the setting it is given."""
    monkeypatch.setenv("FEWSHOT_DEVICE", "warp-drive")
    monkeypatch.setattr(embedders, "_build", lambda spec: pytest.fail("built"))
    with pytest.raises(ValueError, match=r"^FEWSHOT_DEVICE='warp-drive' is not a torch device"):
        embedders.get_embedder("timm:not_loaded_yet")
    with pytest.raises(ValueError, match=r"^--device='warp-drive' is not"):
        embedders.check_device("warp-drive", "--device")


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


# ---- out of memory and batch sizes ---------------------------------------
class Flaky(ColorEmbedder):
    """Runs out of memory on batches larger than ``fits``, raising ``error``."""

    def __init__(self, fits, error):
        super().__init__()
        self.fits, self.error, self.batches = fits, error, []

    def _embed_prepared(self, batch):
        if len(batch) > self.fits:
            raise self.error
        self.batches.append(len(batch))
        return super()._embed_prepared(batch)


@pytest.mark.parametrize("error", [torch.cuda.OutOfMemoryError("CUDA out of memory"),
                                   RuntimeError("MPS backend out of memory"),
                                   RuntimeError("[enforce fail at alloc_cpu.cpp:117] . DefaultCPUAllocator: "
                                                "can't allocate memory: you tried to allocate 3221225472 bytes.")])
def test_out_of_memory_halves_the_batch(error, monkeypatch):
    """On OOM the batch is split in halves and later batches stay smaller; results are unchanged."""
    freed = []
    monkeypatch.setattr(embedders, "_free_gpu_memory", lambda: freed.append(1))
    emb = Flaky(3, error)
    emb.batch_size = 8
    images = [Image.new("RGB", (4, 4), (i * 20, 50, 100)) for i in range(10)]
    got = emb.embed_images(images)
    assert np.allclose(got, ColorEmbedder().embed_images(images))
    assert emb.batch_size <= 3
    assert max(emb.batches) <= 3
    assert freed


def test_other_errors_and_single_image_oom_are_raised(monkeypatch):
    """Errors other than OOM, and OOM on a single image, propagate."""
    monkeypatch.setattr(embedders, "_free_gpu_memory", lambda: None)
    with pytest.raises(RuntimeError, match="boom"):
        Flaky(0, RuntimeError("boom")).embed_images([Image.new("RGB", (4, 4))] * 2)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        Flaky(0, torch.cuda.OutOfMemoryError("out of memory")).embed_images([Image.new("RGB", (4, 4))])


class _Activation:  # pylint: disable=too-few-public-methods
    """Stands in for a forward pass's GPU tensors."""


class Leaky(ColorEmbedder):
    """Runs out of memory on batches over 2; records weak references to each pass's 'activations'."""

    def __init__(self):
        super().__init__()
        self.refs = []
        self.alive_at_retry = []

    def _embed_prepared(self, batch):
        import gc  # pylint: disable=import-outside-toplevel
        import weakref  # pylint: disable=import-outside-toplevel

        gc.collect()
        self.alive_at_retry.append(any(r() is not None for r in self.refs))
        act = _Activation()  # held by this frame, so by a traceback through it
        self.refs.append(weakref.ref(act))
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory")
        return super()._embed_prepared(batch)


def test_out_of_memory_retry_frees_the_failed_pass(monkeypatch):
    """The halves run after the failed pass's memory is released, not while its traceback holds it."""
    monkeypatch.setattr(embedders, "_free_gpu_memory", lambda: None)
    emb = Leaky()
    emb.batch_size = 8
    emb.embed_images([Image.new("RGB", (4, 4), (i * 30, 0, 0)) for i in range(8)])
    assert not any(emb.alive_at_retry)


def test_texts_are_embedded_in_batches_with_oom_halving(monkeypatch):
    """Texts go through the model batch_size at a time, and halve on OOM like images."""
    monkeypatch.setattr(embedders, "_free_gpu_memory", lambda: None)
    sizes = []

    class Texts(TextColor):
        """Text embedder that runs out of memory above 2 texts."""

        def _embed_text_batch(self, texts):
            if len(texts) > 2:
                raise RuntimeError("CUDA out of memory")
            sizes.append(len(texts))
            return super()._embed_text_batch(texts)

    emb = Texts()
    emb.batch_size = 4
    v = emb.embed_texts(["red", "green", "blue", "red", "blue"])
    assert v.shape == (5, 3)
    assert sizes == [2, 2, 1]
    assert emb.batch_size == 2


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


def test_batch_size_changes_under_the_model_lock(monkeypatch):
    """Out-of-memory halving and growing back change ``batch_size`` only while holding the model's
    lock, so concurrent jobs can't undo each other's changes."""
    monkeypatch.setattr(embedders, "_free_gpu_memory", lambda: None)
    monkeypatch.setattr(embedders, "GROW_AFTER", 1)

    class Guarded(ColorEmbedder):
        """Fails batches over 2 with out-of-memory; checks the lock on every batch size change."""

        batch_size = 4
        armed = False

        def __setattr__(self, name, value):
            if self.armed and name in {"batch_size", "_good_batches", "_full_batch"}:
                assert self._lock.locked(), name
            super().__setattr__(name, value)

        def _embed_image_batch(self, images):
            if len(images) > 2:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return super()._embed_image_batch(images)

    emb = Guarded()
    emb.armed = True
    emb.embed_images([Image.new("RGB", (2, 2))] * 8)
    assert emb._full_batch == 4  # pylint: disable=protected-access
    assert emb.batch_size == 4


# ---- loaded models -------------------------------------------------------
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


def test_model_is_held_between_batches(ws, monkeypatch):
    """A model stays loaded for a whole pipeline operation, not just while a batch is in the model:
    another model loaded between two batches (here, from the progress callback) doesn't unload it."""
    emb = ColorEmbedder()
    emb.batch_size = 1
    monkeypatch.setattr(embedders, "_loaded", {"m:a": emb})
    monkeypatch.setenv("FEWSHOT_MAX_MODELS", "1")
    monkeypatch.setattr(embedders, "_free_gpu_memory", lambda: None)
    evicted = []

    def load_another(done, _total):
        assert emb.users == 0  # between batches: no embed call is running
        with embedders._load_lock:  # pylint: disable=protected-access
            embedders._loaded[f"m:other{done}"] = ColorEmbedder()  # pylint: disable=protected-access
            evicted.extend(embedders._evict_locked(keep=f"m:other{done}"))  # pylint: disable=protected-access
            assert "m:a" in embedders._loaded  # pylint: disable=protected-access

    items = [(None, Image.new("RGB", (2, 2), (i * 50, 0, 0))) for i in range(3)]
    Pipeline(ws).embed_images("m:a", items, progress=load_another)
    assert "m:a" not in evicted
    assert not embedders._held  # pylint: disable=protected-access
    assert list(embedders._loaded) == ["m:other3"]  # pylint: disable=protected-access  # m:a went when its hold ended


def test_models_over_the_limit_go_when_free(monkeypatch):
    """A model kept past the limit only because it was held or embedding is unloaded, and GPU
    memory freed, as soon as it is free; one still held by another operation stays."""
    freed = []
    monkeypatch.setattr(embedders, "_free_gpu_memory", lambda: freed.append(1))
    monkeypatch.setattr(embedders, "_loaded", {"m:a": ColorEmbedder()})
    monkeypatch.setenv("FEWSHOT_MAX_MODELS", "1")
    with embedders.hold("m:a"):
        with embedders.hold("m:a"), embedders._load_lock:  # pylint: disable=protected-access
            embedders._loaded["m:b"] = ColorEmbedder()  # pylint: disable=protected-access  # loaded meanwhile
            assert not embedders._evict_locked(keep="m:b")  # pylint: disable=protected-access
        assert set(embedders._loaded) == {"m:a", "m:b"}  # pylint: disable=protected-access
        assert not freed
    assert list(embedders._loaded) == ["m:b"]  # pylint: disable=protected-access
    assert freed == [1]

    class LoadsAnother(ColorEmbedder):
        """Another model is loaded while this one embeds."""

        def _embed_image_batch(self, images):
            with embedders._load_lock:  # pylint: disable=protected-access
                embedders._loaded["m:c"] = ColorEmbedder()  # pylint: disable=protected-access
                assert not embedders._evict_locked(keep="m:c")  # pylint: disable=protected-access
            return super()._embed_image_batch(images)

    busy = LoadsAnother()
    monkeypatch.setattr(embedders, "_loaded", {"m:busy": busy})
    busy.embed_images([Image.new("RGB", (2, 2))])
    assert list(embedders._loaded) == ["m:c"]  # pylint: disable=protected-access
    assert freed == [1, 1]


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
