"""Tests for fewshot.embedders using tiny fake OpenCLIP/timm models (no downloads, no GPU required)."""

import contextlib
import threading
from types import SimpleNamespace

import numpy as np
import open_clip
import pytest
import timm
import timm.data
import torch
from PIL import Image

import fewshot
from fewshot import embedders
from fewshot.embedders import (
    DEFAULT_SPEC,
    PRESETS,
    EmbedderInfo,
    OpenCLIPEmbedder,
    TimmEmbedder,
    get_embedder,
    register_embedder,
    spec_supports_text,
    unregister_embedder,
)
from tests.conftest import ColorEmbedder

CLIP_DIM = 4
TIMM_DIM = 5


def solid(rgb, mode="RGB"):
    """A small solid-colour image, optionally converted to another PIL mode."""
    return Image.new("RGB", (4, 4), tuple(rgb)).convert(mode)


def mean_rgb(im):
    """Preprocess stand-in: an image becomes its mean RGB as a float tensor."""
    assert im.mode == "RGB", "embedders must convert images to RGB before preprocessing"
    return torch.from_numpy(np.asarray(im, np.float32).reshape(-1, 3).mean(0))


def unit(rows):
    """Row-wise L2 normalisation as float32, the reference for embedder outputs."""
    a = np.asarray(rows, np.float64)
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


def clip_image_ref(rgb):
    """What FakeCLIP.encode_image produces for one solid-colour image, before normalisation."""
    r, g, b = (float(c) for c in rgb)
    return [r, g, b, 1.0]


def timm_ref(rgb):
    """What FakeTimm produces for one solid-colour image, before normalisation."""
    r, g, b = (float(c) for c in rgb)
    return [r, g, b, r + g, 2.0]


class FakeCLIP(torch.nn.Module):
    """Minimal OpenCLIP model: records batch sizes and checks it runs under inference mode."""

    def __init__(self):
        super().__init__()
        self.visual = SimpleNamespace(output_dim=CLIP_DIM)  # only output_dim is read
        self.image_batches = []
        self.text_batches = []

    def forward(self, x):
        """Calling the model directly embeds images, like OpenCLIP's ``encode_image``."""
        return self.encode_image(x)

    def encode_image(self, x):
        """Mean RGB plus a constant column, returned as float64 to check the float32 cast."""
        assert torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu")
        self.image_batches.append(x.shape[0])
        return torch.cat([x, torch.ones(x.shape[0], 1)], dim=1).double()

    def encode_text(self, tok):
        """Token rows padded with a constant column."""
        assert torch.is_inference_mode_enabled()
        self.text_batches.append(tok.shape[0])
        return torch.cat([tok.float(), torch.full((tok.shape[0], 1), 3.0)], dim=1)


def fake_tokenizer(texts):
    """Tokenizer stand-in: (length, vowel count, 1) per text, so outputs are distinct and predictable."""
    return torch.tensor([[len(t), sum(c in "aeiou" for c in t), 1] for t in texts], dtype=torch.long)


def text_ref(text):
    """What FakeCLIP.encode_text produces for one text, before normalisation."""
    return [len(text), sum(c in "aeiou" for c in text), 1.0, 3.0]


class FakeTimm(torch.nn.Module):
    """Minimal timm backbone with ``num_features`` and a recorded ``.to`` device."""

    num_features = TIMM_DIM

    def __init__(self):
        super().__init__()
        self.moved_to = None
        self.batches = []

    def to(self, *args, **kwargs):  # pylint: disable=arguments-differ  # nn.Module.to is overloaded
        self.moved_to = args[0] if args else kwargs.get("device")
        return self

    def forward(self, x):
        """Mean RGB, R+G, and a constant column."""
        assert torch.is_inference_mode_enabled()
        self.batches.append(x.shape[0])
        return torch.cat([x, (x[:, 0] + x[:, 1]).unsqueeze(1), torch.full((x.shape[0], 1), 2.0)], dim=1)


@pytest.fixture(name="calls")
def fixture_calls():
    """Records the arguments passed to the patched model factories."""
    return {}


@pytest.fixture(name="fake_backends")
def fixture_fake_backends(monkeypatch, calls):
    """Replace open_clip and timm model factories with tiny fakes and start with an empty embedder cache."""

    def create_model_and_transforms(model_name, pretrained=None, device=None):
        model = FakeCLIP()
        calls.setdefault("clip", []).append((model_name, pretrained, device))
        calls["clip_model"] = model
        return model, None, mean_rgb

    def get_tokenizer(model_name):
        calls["tokenizer"] = model_name
        return fake_tokenizer

    def create_model(model_name, pretrained=False, num_classes=None):
        model = FakeTimm()
        calls.setdefault("timm", []).append((model_name, pretrained, num_classes))
        calls["timm_model"] = model
        return model

    def resolve_data_config(args, model=None):
        calls["data_config"] = (args, model)
        return {"input_size": (3, 4, 4)}

    def create_transform(**cfg):
        calls["transform_cfg"] = cfg
        return mean_rgb

    monkeypatch.setattr(open_clip, "create_model_and_transforms", create_model_and_transforms)
    monkeypatch.setattr(open_clip, "get_tokenizer", get_tokenizer)
    monkeypatch.setattr(timm, "create_model", create_model)
    monkeypatch.setattr(timm.data, "resolve_data_config", resolve_data_config)
    monkeypatch.setattr(timm.data, "create_transform", create_transform)
    monkeypatch.setattr(embedders, "_loaded", {})


@pytest.fixture(name="cpu")
def fixture_cpu(monkeypatch):
    """Pretend no GPU is present."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(name="fake_cuda")
def fixture_fake_cuda(monkeypatch):
    """Pretend a GPU is present without needing one.

    ``Tensor.to("cuda")`` is redirected to a no-op and ``torch.autocast`` is replaced by a recorder, so the CUDA code
    path runs on any machine. Returns the lists of tensor-move devices and autocast calls.
    """
    moves, autocasts = [], []
    real_to = torch.Tensor.to

    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get("device")
        if isinstance(device, str) and device.startswith("cuda"):
            moves.append(device)
            return self
        return real_to(self, *args, **kwargs)

    @contextlib.contextmanager
    def autocast(device_type, dtype=None, enabled=True):
        autocasts.append((device_type, dtype, enabled))
        yield

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch, "autocast", autocast)
    return moves, autocasts


# --- module-level helpers -------------------------------------------------------------------------------------------


def test_device_follows_cuda_availability(monkeypatch):
    """_device picks CUDA exactly when torch reports it available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert embedders._device() == "cuda"  # pylint: disable=protected-access  # testing the private helper
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert embedders._device() == "cpu"  # pylint: disable=protected-access  # testing the private helper


def test_autocast_is_noop_on_cpu():
    """On CPU no autocast is enabled for any device."""
    with embedders._autocast("cpu"):  # pylint: disable=protected-access  # testing the private helper
        assert not torch.is_autocast_enabled("cpu")
        assert not torch.is_autocast_enabled("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real CUDA device")
def test_autocast_enables_fp16_on_real_cuda():
    """On a real GPU the CUDA autocast context computes in float16."""
    with embedders._autocast("cuda"):  # pylint: disable=protected-access  # testing the private helper
        assert torch.is_autocast_enabled("cuda")
        assert torch.get_autocast_dtype("cuda") == torch.float16
        y = torch.ones(2, 2, device="cuda") @ torch.ones(2, 2, device="cuda")
    assert y.dtype == torch.float16


def test_autocast_requests_fp16_on_cuda(fake_cuda):
    """The CUDA branch asks torch for float16 autocast."""
    _, autocasts = fake_cuda
    with embedders._autocast("cuda"):  # pylint: disable=protected-access  # testing the private helper
        pass
    assert autocasts == [("cuda", torch.float16, True)]


def test_l2_normalises_to_float32():
    """_l2 returns unit-length float32 rows and leaves all-zero rows at zero."""
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float64)
    out = embedders._l2(x)  # pylint: disable=protected-access  # testing the private helper
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])  # zero rows stay zero instead of NaN


def test_presets_and_spec_supports_text():
    """Text support is decided by backend and agrees with every preset."""
    assert PRESETS[0].spec == DEFAULT_SPEC
    assert all(isinstance(p, EmbedderInfo) for p in PRESETS)
    for p in PRESETS:
        assert spec_supports_text(p.spec) == p.supports_text
    assert spec_supports_text("openclip:anything")
    assert not spec_supports_text("timm:vit_small_patch14_dinov2.lvd142m")
    assert not spec_supports_text("fake:color")


# --- OpenCLIP -------------------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_backends", "cpu")
def test_openclip_construction_on_cpu(calls):
    """OpenCLIPEmbedder loads the model on CPU, in eval mode, with its tokenizer."""
    emb = OpenCLIPEmbedder("openclip:Tiny@weights", "Tiny", "weights")
    assert calls["clip"] == [("Tiny", "weights", "cpu")]
    assert calls["tokenizer"] == "Tiny"
    assert not calls["clip_model"].training  # eval() was called
    assert emb.spec == "openclip:Tiny@weights"
    assert emb.device == "cpu"
    assert emb.dim == CLIP_DIM
    assert emb.supports_text


@pytest.mark.usefixtures("fake_backends", "cpu")
def test_openclip_embed_images_batches_in_order(calls):
    """Images are converted to RGB, split by batch_size and concatenated in input order."""
    emb = OpenCLIPEmbedder("openclip:Tiny", "Tiny", None)
    emb.batch_size = 2
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (10, 20, 30), (200, 100, 50)]
    images = [solid(colours[0], "L"), *map(solid, colours[1:])]  # first image is greyscale: must be converted

    out = emb.embed_images(images)

    assert calls["clip_model"].image_batches == [2, 2, 1]
    assert out.dtype == np.float32
    assert out.shape == (5, CLIP_DIM)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)
    grey = np.asarray(images[0].convert("RGB"), np.float32)[0, 0]
    expected = unit([clip_image_ref(grey), *map(clip_image_ref, colours[1:])])
    np.testing.assert_allclose(out, expected, rtol=1e-6)


@pytest.mark.usefixtures("fake_backends", "cpu")
def test_openclip_embed_texts(calls):
    """Texts are tokenised and embedded into normalised float32 rows."""
    emb = OpenCLIPEmbedder("openclip:Tiny", "Tiny", None)
    texts = ("a cat", "a photo of a dog")  # any sequence works, not just lists

    out = emb.embed_texts(texts)

    assert calls["clip_model"].text_batches == [2]
    assert out.dtype == np.float32
    assert out.shape == (2, CLIP_DIM)
    np.testing.assert_allclose(out, unit([text_ref(t) for t in texts]), rtol=1e-6)


@pytest.mark.usefixtures("fake_backends", "cpu")
def test_openclip_empty_inputs(calls):
    """Empty inputs return (0, dim) float32 arrays without touching the model."""
    emb = OpenCLIPEmbedder("openclip:Tiny", "Tiny", None)
    for out in (emb.embed_images([]), emb.embed_texts([])):
        assert out.shape == (0, CLIP_DIM)
        assert out.dtype == np.float32
    model = calls["clip_model"]
    assert not model.image_batches
    assert not model.text_batches


@pytest.mark.usefixtures("fake_backends")
def test_openclip_on_cuda_moves_inputs_and_uses_fp16_autocast(calls, fake_cuda):
    """With a GPU, inputs go to CUDA and both encoders run under float16 autocast."""
    moves, autocasts = fake_cuda
    emb = OpenCLIPEmbedder("openclip:Tiny", "Tiny", None)
    assert emb.device == "cuda"
    assert calls["clip"] == [("Tiny", None, "cuda")]

    img = emb.embed_images([solid((1, 2, 3))])
    txt = emb.embed_texts(["hello"])

    assert moves == ["cuda", "cuda"]  # image batch, then tokens
    assert autocasts == [("cuda", torch.float16, True)] * 2
    np.testing.assert_allclose(img, unit([clip_image_ref((1, 2, 3))]), rtol=1e-6)
    np.testing.assert_allclose(txt, unit([text_ref("hello")]), rtol=1e-6)


# --- timm -----------------------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_backends", "cpu")
def test_timm_construction_on_cpu(calls):
    """TimmEmbedder builds a headless pretrained model and its matching transform."""
    emb = TimmEmbedder("timm:tiny", "tiny")
    model = calls["timm_model"]
    assert calls["timm"] == [("tiny", True, 0)]  # pretrained weights, classifier head removed
    assert model.moved_to == "cpu"
    assert not model.training
    assert calls["data_config"] == ({}, model)
    assert calls["transform_cfg"] == {"input_size": (3, 4, 4)}
    assert emb.spec == "timm:tiny"
    assert emb.device == "cpu"
    assert emb.dim == TIMM_DIM
    assert not emb.supports_text


@pytest.mark.usefixtures("fake_backends", "cpu")
def test_timm_embed_images_batches_in_order(calls):
    """timm images are converted to RGB, batched and concatenated in order."""
    emb = TimmEmbedder("timm:tiny", "tiny")
    emb.batch_size = 3
    colours = [(i * 30, 255 - i * 20, 7) for i in range(7)]

    out = emb.embed_images([solid(c, "RGBA") for c in colours])

    assert calls["timm_model"].batches == [3, 3, 1]
    assert out.dtype == np.float32
    assert out.shape == (7, TIMM_DIM)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(out, unit([timm_ref(c) for c in colours]), rtol=1e-6)


@pytest.mark.usefixtures("fake_backends", "cpu")
def test_timm_texts_unsupported():
    """Image-only embedders reject non-empty text but accept an empty list."""
    emb = TimmEmbedder("timm:tiny", "tiny")
    empty = emb.embed_texts([])
    assert empty.shape == (0, TIMM_DIM)
    assert empty.dtype == np.float32
    with pytest.raises(NotImplementedError, match="timm:tiny does not support text"):
        emb.embed_texts(["a cat"])


@pytest.mark.usefixtures("fake_backends")
def test_timm_on_cuda_moves_model_and_inputs_and_uses_fp16_autocast(calls, fake_cuda):
    """With a GPU, the timm model and inputs go to CUDA under float16 autocast."""
    moves, autocasts = fake_cuda
    emb = TimmEmbedder("timm:tiny", "tiny")
    assert emb.device == "cuda"
    assert calls["timm_model"].moved_to == "cuda"

    out = emb.embed_images([solid((9, 8, 7)), solid((1, 1, 1))])

    assert moves == ["cuda"]
    assert autocasts == [("cuda", torch.float16, True)]
    np.testing.assert_allclose(out, unit([timm_ref((9, 8, 7)), timm_ref((1, 1, 1))]), rtol=1e-6)


# --- get_embedder ---------------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_backends", "cpu")
def test_get_embedder_builds_openclip_and_caches(calls):
    """openclip specs build once per spec; the '@pretrained' part is optional."""
    emb = get_embedder("openclip:Tiny@weights")
    assert isinstance(emb, OpenCLIPEmbedder)
    assert calls["clip"] == [("Tiny", "weights", "cpu")]
    assert get_embedder("openclip:Tiny@weights") is emb
    assert len(calls["clip"]) == 1  # second call served from the cache

    bare = get_embedder("openclip:Tiny")
    assert bare is not emb
    assert calls["clip"][-1] == ("Tiny", None, "cpu")  # no '@' means no pretrained tag


@pytest.mark.usefixtures("fake_backends", "cpu")
def test_get_embedder_default_spec(calls):
    """With no argument the default preset is loaded."""
    emb = get_embedder()
    assert emb.spec == DEFAULT_SPEC
    model, _, pretrained = DEFAULT_SPEC.partition(":")[2].partition("@")
    assert calls["clip"] == [(model, pretrained, "cpu")]


@pytest.mark.usefixtures("fake_backends", "cpu")
def test_get_embedder_builds_timm_and_caches(calls):
    """timm specs build once and are then served from the cache."""
    emb = get_embedder("timm:vit_tiny.dino")
    assert isinstance(emb, TimmEmbedder)
    assert calls["timm"] == [("vit_tiny.dino", True, 0)]
    assert get_embedder("timm:vit_tiny.dino") is emb
    assert len(calls["timm"]) == 1


@pytest.mark.usefixtures("fake_backends")
@pytest.mark.parametrize("spec", ["openclip", "timm:", "", "ViT-B-32"])
def test_get_embedder_rejects_bad_spec(spec):
    """Specs without '<backend>:<model>' are rejected."""
    with pytest.raises(ValueError, match="Bad embedder spec"):
        get_embedder(spec)
    assert not embedders._loaded  # pylint: disable=protected-access  # nothing cached on failure


@pytest.mark.usefixtures("fake_backends")
def test_get_embedder_rejects_unknown_backend():
    """Unknown backends are rejected."""
    with pytest.raises(ValueError, match="Unknown embedder backend 'hf'"):
        get_embedder("hf:some/model")
    assert not embedders._loaded  # pylint: disable=protected-access  # nothing cached on failure


def test_get_embedder_returns_registered_instance():
    """A registered embedder is returned as is, whatever its backend."""
    # conftest registers a fake embedder under a spec whose backend get_embedder could not build itself.
    emb = get_embedder("fake:color")
    assert isinstance(emb, ColorEmbedder)
    out = emb.embed_images([solid((255, 0, 0))])
    assert out.shape == (1, 3)


def test_register_and_unregister_embedder():
    """register_embedder adds or replaces a spec; unregister_embedder removes it and ignores unknown specs."""
    first, second = ColorEmbedder(), ColorEmbedder()
    register_embedder("custom:mine", first)
    assert get_embedder("custom:mine") is first
    register_embedder("custom:mine", second)
    assert get_embedder("custom:mine") is second
    unregister_embedder("custom:mine")
    with pytest.raises(ValueError, match="Unknown embedder backend 'custom'"):
        get_embedder("custom:mine")
    unregister_embedder("custom:mine")  # already gone: no error
    unregister_embedder("never:registered")


def test_register_embedder_is_exported_lazily():
    """The top-level package re-exports register_embedder."""
    assert fewshot.register_embedder is register_embedder


# --- output checks, locking and the loaded-model limit -----------------------------------------------------------


@pytest.mark.usefixtures("fake_backends", "cpu")
def test_builtin_embed_image_batch_matches_embed_images():
    """The built-ins' direct batch hook gives the same rows as the batched path."""
    images = [solid((255, 0, 0)), solid((0, 0, 255))]
    for emb in (OpenCLIPEmbedder("openclip:Tiny", "Tiny", None), TimmEmbedder("timm:t", "t")):
        with torch.inference_mode():
            direct = emb._embed_image_batch(images)  # pylint: disable=protected-access
        np.testing.assert_allclose(direct, emb.embed_images(images), rtol=1e-6)


class Raw(ColorEmbedder):
    """Returns whatever ``out`` holds, to test the output checks."""

    supports_text = True

    def __init__(self, out):
        super().__init__()
        self.out = out

    def _embed_image_batch(self, images):
        return self.out

    def _embed_text_batch(self, texts):  # pylint: disable=unused-argument  # returns out whatever the input
        return self.out


def test_embedder_output_is_checked():
    """Wrong shapes and non-finite values are errors naming the spec; other rows are normalised."""
    img = [solid((1, 2, 3))]
    with pytest.raises(ValueError, match=r"'fake:color' returned shape \(1, 2\), expected \(1, 3\)"):
        Raw(np.ones((1, 2))).embed_images(img)
    with pytest.raises(ValueError, match="NaN or infinite"):
        Raw(np.array([[np.nan, 1, 1]])).embed_texts(["x"])
    out = Raw(np.array([[3.0, 0, 4]], np.float64)).embed_images(img)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.6, 0, 0.8]], rtol=1e-6)
    already = np.array([[0.6, 0, 0.8]], np.float32)
    assert Raw(already).embed_texts(["x"]).tolist() == already.tolist()


def test_lock_is_held_only_for_the_forward_pass():
    """Preprocessing runs without the model lock; the forward pass holds it."""
    seen = []

    class Watch(ColorEmbedder):
        """Records whether the model lock is held in each step."""

        def _prepare_images(self, images):
            seen.append(("prepare", self._lock.locked()))
            return images

        def _embed_image_batch(self, images):
            seen.append(("forward", self._lock.locked()))
            return super()._embed_image_batch(images)

    emb = Watch()
    emb.batch_size = 1
    emb.embed_images([solid((1, 2, 3)), solid((3, 2, 1))])
    assert seen == [("prepare", False), ("forward", True)] * 2


@pytest.mark.usefixtures("fake_backends", "cpu")
def test_least_recently_used_model_is_unloaded(monkeypatch):
    """Beyond FEWSHOT_MAX_MODELS built-in models, the least recently used one is dropped and GPU
    memory released; registered embedders are never dropped."""
    freed = []
    monkeypatch.setattr(embedders, "_free_gpu_memory", lambda: freed.append(True))
    monkeypatch.setenv("FEWSHOT_MAX_MODELS", "2")
    register_embedder("custom:mine", ColorEmbedder())
    try:
        a, b = get_embedder("timm:a"), get_embedder("timm:b")
        assert get_embedder("timm:a") is a  # a is now the most recently used
        get_embedder("timm:c")
        assert embedders.loaded_embedders() == ["custom:mine", "timm:a", "timm:c"]
        assert freed == [True]
        assert get_embedder("timm:b") is not b  # reloaded
    finally:
        unregister_embedder("custom:mine")
    monkeypatch.setenv("FEWSHOT_MAX_MODELS", "0")
    assert embedders.max_models() == 1  # at least one


def test_free_gpu_memory_empties_the_cuda_cache(monkeypatch):
    """With CUDA, freeing memory also empties torch's CUDA cache."""
    emptied = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    embedders._free_gpu_memory()  # pylint: disable=protected-access
    assert emptied == [True]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    embedders._free_gpu_memory()  # pylint: disable=protected-access
    assert emptied == [True]


def _blocking_build(monkeypatch):
    """Make model building wait for the returned event; returns (event, started, builds)."""
    release, started, builds = threading.Event(), threading.Event(), []

    def build(spec):
        builds.append(spec)
        started.set()
        assert release.wait(5)
        return ColorEmbedder()

    monkeypatch.setattr(embedders, "_build", build)
    return release, started, builds


def test_one_model_loading_does_not_block_others(monkeypatch):
    """While one spec loads, another loaded spec is still served, and a second request for the
    loading spec waits for that load instead of starting its own."""
    only = {"fake:color": get_embedder("fake:color")}
    monkeypatch.setattr(embedders, "_loaded", only)
    release, started, builds = _blocking_build(monkeypatch)
    results = []
    threads = [threading.Thread(target=lambda: results.append(get_embedder("slow:x"))) for _ in range(2)]
    threads[0].start()
    assert started.wait(5)
    threads[1].start()
    assert isinstance(get_embedder("fake:color"), ColorEmbedder)  # not blocked by the load
    release.set()
    for t in threads:
        t.join(5)
    assert builds == ["slow:x"]
    assert results[0] is results[1]


def test_registration_during_a_load_wins(monkeypatch):
    """An embedder registered while the same spec is loading replaces the loaded model."""
    release, started, _ = _blocking_build(monkeypatch)
    result = []
    t = threading.Thread(target=lambda: result.append(get_embedder("slow:y")))
    t.start()
    assert started.wait(5)
    mine = ColorEmbedder()
    register_embedder("slow:y", mine)
    try:
        assert embedders.is_registered("slow:y")
        release.set()
        t.join(5)
        assert result == [mine]
    finally:
        unregister_embedder("slow:y")
    assert not embedders.is_registered("slow:y")
