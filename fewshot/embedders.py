"""Image (and optionally text) embedders.

Every embedder returns L2-normalised float32 vectors so downstream classifiers
can use dot products as cosine similarity. Embedders are identified by a spec
string of the form ``<backend>:<model>[@<pretrained>]``:

    openclip:ViT-B-32@laion2b_s34b_b79k
    timm:vit_small_patch14_dinov2.lvd142m
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from PIL import Image

# torch is imported inside functions so that importing this module (for PRESETS)
# stays fast for the CLI.
if TYPE_CHECKING:
    import torch

Vectors = npt.NDArray[np.float32]  # L2-normalised embeddings, one per row


@dataclass(frozen=True)
class EmbedderInfo:
    """Description of an embedder preset for display in the UI and CLI."""

    spec: str
    label: str
    supports_text: bool
    notes: str


# Presets shown in the UI/CLI. Any valid spec string also works.
PRESETS: list[EmbedderInfo] = [
    EmbedderInfo(
        "openclip:ViT-B-32@laion2b_s34b_b79k",
        "CLIP ViT-B/32 (LAION-2B)",
        True,
        "Fast default. Supports text class descriptions, so classes work with zero images.",
    ),
    EmbedderInfo(
        "openclip:ViT-L-14@laion2b_s32b_b82k",
        "CLIP ViT-L/14 (LAION-2B)",
        True,
        "Higher accuracy than ViT-B/32, about 4x slower. Supports text descriptions.",
    ),
    EmbedderInfo(
        "timm:vit_small_patch14_dinov2.lvd142m",
        "DINOv2 ViT-S/14",
        False,
        "Image-only. Often better than CLIP on fine-grained visual differences.",
    ),
    EmbedderInfo(
        "timm:vit_base_patch14_dinov2.lvd142m",
        "DINOv2 ViT-B/14",
        False,
        "Image-only. Larger DINOv2.",
    ),
    EmbedderInfo(
        "timm:vit_large_patch14_dinov2.lvd142m",
        "DINOv2 ViT-L/14",
        False,
        "Image-only. Largest DINOv2 preset: most accurate on fine-grained classes, about 3x slower than ViT-B.",
    ),
    EmbedderInfo(
        "openclip:ViT-B-16-SigLIP2@webli",
        "SigLIP 2 ViT-B/16 (WebLI)",
        True,
        "Usually more accurate than CLIP, with multilingual descriptions. Needs `pip install transformers` "
        "for its tokenizer.",
    ),
]

DEFAULT_SPEC = PRESETS[0].spec


def _device() -> str:
    """``$FEWSHOT_DEVICE`` if set (e.g. ``cpu``, ``cuda:1``, ``mps``), else CUDA, then Apple's GPU
    (MPS), then the CPU, whichever is available first."""
    import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

    if forced := os.environ.get("FEWSHOT_DEVICE", "").strip():
        try:
            torch.device(forced)
        except RuntimeError as e:
            raise ValueError(f"FEWSHOT_DEVICE={forced!r} is not a torch device: {e}") from None
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _autocast(device: str) -> AbstractContextManager[Any]:
    """Half-precision autocast on CUDA; a no-op context on other devices."""
    if not device.startswith("cuda"):
        return contextlib.nullcontext()
    import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

    return torch.autocast("cuda", dtype=torch.float16)


def _is_oom(e: BaseException) -> bool:
    """Whether ``e`` is the GPU running out of memory (CUDA's own error, or MPS's RuntimeError)."""
    import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower())


def _l2(x: torch.Tensor) -> Vectors:
    x = x.float()
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    out: Vectors = x.cpu().numpy().astype(np.float32)
    return out


class Embedder(ABC):
    """Base class: batches images/texts and returns L2-normalised float32 rows."""

    spec: str
    supports_text: bool = False
    # Text hook: embedders that set ``supports_text`` provide this; ``embed_texts`` checks the flag first.
    _embed_text_batch: Callable[[list[str]], Vectors]
    batch_size: int = 64
    # For a registered embedder: a name for its weights in the embedding cache, changed whenever the
    # model changes. None (the default) uses a fingerprint of the model's output instead.
    cache_version: str | None = None

    def __init__(self) -> None:
        # Models are shared across API request threads; serialise GPU use. The lock is taken per batch
        # and only around the forward pass, so one long job doesn't hold off other requests.
        self._lock = threading.Lock()

    @abstractmethod
    def _embed_image_batch(self, images: Sequence[Image.Image]) -> Vectors: ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Length of each embedding vector."""

    def _prepare_images(self, images: list[Image.Image]) -> Any:
        """CPU-side preprocessing of one batch, run without holding the model lock.

        The default passes the images through to ``_embed_image_batch`` unchanged.
        """
        return images

    def _embed_prepared(self, batch: Any) -> Vectors:
        """Embed one ``_prepare_images`` result; runs under the model lock."""
        return self._embed_image_batch(batch)

    def embed_images(self, images: Sequence[Image.Image]) -> Vectors:
        """Embed images (converted to RGB) in batches of ``batch_size``; returns an ``(n, dim)`` array.

        Raises ``ValueError`` if the model returns the wrong shape or non-finite values; rows that
        aren't unit length are normalised.
        """
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)

        out: list[Vectors] = []
        i = 0
        while i < len(images):
            batch = [im.convert("RGB") for im in images[i : i + self.batch_size]]
            out.append(self._embed_halving(self._prepare_images(batch), len(batch)))
            i += len(batch)
        return np.concatenate(out)

    def _embed_halving(self, prepared: Any, n: int) -> Vectors:
        """Embed a prepared batch of ``n`` images. If the GPU runs out of memory, free what it can,
        halve ``batch_size`` for the rest of this embedder's life, and embed the batch in halves."""
        import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

        try:
            with self._lock, torch.inference_mode():
                vecs = self._embed_prepared(prepared)
        except Exception as e:  # pylint: disable=broad-exception-caught  # only OOM is handled; re-raised
            if n == 1 or not _is_oom(e):
                raise
            del e  # the traceback holds the failed batch's GPU tensors
            _free_gpu_memory()
            half = n // 2
            self.batch_size = max(1, min(self.batch_size, half))
            return np.concatenate([self._embed_halving(prepared[:half], half),
                                   self._embed_halving(prepared[half:], n - half)])
        return self._checked(vecs, n)

    def embed_texts(self, texts: Sequence[str]) -> Vectors:
        """Embed texts into the image embedding space; raises NotImplementedError if unsupported."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not self.supports_text:
            raise NotImplementedError(f"{self.spec} does not support text embeddings")
        import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

        with self._lock, torch.inference_mode():
            vecs = self._embed_text_batch(list(texts))
        return self._checked(vecs, len(texts))

    def _checked(self, vecs: Any, n: int) -> Vectors:
        """``vecs`` as ``(n, dim)`` float32 unit rows; ``ValueError`` names the embedder if it can't be."""
        v = np.asarray(vecs, dtype=np.float32)
        if v.shape != (n, self.dim):
            raise ValueError(f"Embedder {self.spec!r} returned shape {v.shape}, expected {(n, self.dim)}")
        if not np.isfinite(v).all():
            raise ValueError(f"Embedder {self.spec!r} returned NaN or infinite values")
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0, atol=1e-3):
            v = v / np.maximum(norms, 1e-12)
        return v


class OpenCLIPEmbedder(Embedder):
    """OpenCLIP image/text model; image and text vectors share one space."""

    supports_text = True

    def __init__(self, spec: str, model_name: str, pretrained: str | None) -> None:
        super().__init__()
        import open_clip  # pylint: disable=import-outside-toplevel  # lazy: heavy import, keeps CLI startup fast

        self.spec = spec
        self.device = _device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        dim = getattr(self.model.visual, "output_dim", None)
        if dim is None:  # e.g. SigLIP's timm vision tower: measure it on a blank image
            import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

            with torch.inference_mode():
                dim = self._embed_image_batch([Image.new("RGB", (32, 32))]).shape[1]
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def _prepare_images(self, images: list[Image.Image]) -> torch.Tensor:
        import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

        return torch.stack([self.preprocess(im) for im in images])

    def _embed_prepared(self, batch: torch.Tensor) -> Vectors:
        with _autocast(self.device):
            return _l2(self.model.encode_image(batch.to(self.device)))

    def _embed_image_batch(self, images: Sequence[Image.Image]) -> Vectors:
        return self._embed_prepared(self._prepare_images(list(images)))

    def _embed_text_batch(self, texts: list[str]) -> Vectors:
        tok = self.tokenizer(texts).to(self.device)
        with _autocast(self.device):
            return _l2(self.model.encode_text(tok))


class TimmEmbedder(Embedder):
    """Image-only timm backbone (e.g. DINOv2) with its classifier head removed."""

    def __init__(self, spec: str, model_name: str) -> None:
        super().__init__()
        import timm  # pylint: disable=import-outside-toplevel  # lazy: heavy import, keeps CLI startup fast

        self.spec = spec
        self.device = _device()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device).eval()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.preprocess = timm.data.create_transform(**cfg)
        self._dim = int(self.model.num_features)

    @property
    def dim(self) -> int:
        return self._dim

    def _prepare_images(self, images: list[Image.Image]) -> torch.Tensor:
        import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

        return torch.stack([self.preprocess(im) for im in images])

    def _embed_prepared(self, batch: torch.Tensor) -> Vectors:
        with _autocast(self.device):
            return _l2(self.model(batch.to(self.device)))

    def _embed_image_batch(self, images: Sequence[Image.Image]) -> Vectors:
        return self._embed_prepared(self._prepare_images(list(images)))


_loaded: dict[str, Embedder] = {}  # in least- to most-recently used order
_load_lock = threading.Lock()  # guards the dicts here; never held while a model loads
_spec_locks: dict[str, threading.Lock] = {}  # one per spec: two loads of one model wait for each other
_registered: set[str] = set()  # specs whose embedder came from register_embedder
_namespaces: dict[str, str] = {}  # registered spec -> its cache namespace, once computed


def max_models() -> int:
    """Most built-in models kept loaded at once (``$FEWSHOT_MAX_MODELS``, default 2); loading
    another unloads the least recently used. Registered embedders don't count and stay loaded."""
    return max(1, int(os.environ.get("FEWSHOT_MAX_MODELS", "2")))


def _probe_image() -> Image.Image:
    """A fixed, deterministic image used to fingerprint a model."""
    a = np.indices((32, 32)).sum(0) * 4
    return Image.fromarray(np.stack([a, a[::-1], a.T], -1).astype(np.uint8))


def _fingerprint(emb: Embedder) -> str:
    v = emb.embed_images([_probe_image()])
    return hashlib.sha256(np.round(v, 4).astype(np.float32).tobytes()).hexdigest()[:12]


def cache_namespace(spec: str) -> str:
    """The name under which embeddings for ``spec`` are cached.

    A built-in spec names its weights, so it is its own namespace. A registered embedder can be
    replaced under the same spec, so its namespace adds its ``cache_version`` or, by default, a
    fingerprint of its output on a fixed image: a new model never gets the old one's vectors, and
    the same model gets them back in a later process.
    """
    with _load_lock:
        emb = _loaded.get(spec) if spec in _registered else None
        ns = _namespaces.get(spec)
    if emb is None:
        return spec
    if ns is None:
        ns = f"{spec}#{emb.cache_version or _fingerprint(emb)}"
        with _load_lock:
            if _loaded.get(spec) is emb:
                _namespaces[spec] = ns
    return ns


def get_embedder(spec: str = DEFAULT_SPEC) -> Embedder:
    """Load (once per process) and return the embedder for ``spec``.

    Loading one model (which may mean downloading it) doesn't block requests for other models.
    """
    with _load_lock:
        if spec in _loaded:
            _loaded[spec] = emb = _loaded.pop(spec)  # now most recently used
            return emb
        spec_lock = _spec_locks.setdefault(spec, threading.Lock())
    with spec_lock:
        with _load_lock:
            if spec in _loaded:  # another thread loaded it while we waited
                return _loaded[spec]
        emb = _build(spec)
        with _load_lock:
            if spec in _loaded:  # registered while we were loading: the registration wins
                return _loaded[spec]
            _loaded[spec] = emb
            evicted = _evict_locked()
    if evicted:
        _free_gpu_memory()
    return emb


def _build(spec: str) -> Embedder:
    backend, _, rest = spec.partition(":")
    if not rest:
        raise ValueError(f"Bad embedder spec {spec!r}; expected '<backend>:<model>'")
    if backend == "openclip":
        model, _, pretrained = rest.partition("@")
        return OpenCLIPEmbedder(spec, model, pretrained or None)
    if backend == "timm":
        return TimmEmbedder(spec, rest)
    raise ValueError(f"Unknown embedder backend {backend!r} (use 'openclip' or 'timm')")


def _evict_locked() -> list[str]:
    """Drop least recently used built-in models beyond ``max_models()``; returns their specs."""
    built_in = [s for s in _loaded if s not in _registered]
    gone = built_in[: max(0, len(built_in) - max_models())]
    for spec in gone:
        del _loaded[spec]
    return gone


def _free_gpu_memory() -> None:
    """Return unloaded models' GPU memory to the driver (a model still in use is freed after its job)."""
    import gc  # pylint: disable=import-outside-toplevel  # only needed when a model is unloaded

    import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def loaded_embedders() -> list[str]:
    """Specs of the embedders currently loaded, least recently used first."""
    with _load_lock:
        return list(_loaded)


def check_embedder(spec: str) -> None:
    """Load the embedder for ``spec`` now, raising ``ValueError`` with a short reason if it can't be.

    Used where a spec is chosen (task create and settings) so a bad one fails there, not at
    predict time. A preset may be downloaded on first use.
    """
    if not spec.strip():
        raise ValueError("Embedder spec is empty")
    try:
        get_embedder(spec)
    # Model loaders (open_clip, timm, hub downloads) raise many unrelated exception types for a bad name.
    except Exception as e:  # pylint: disable=broad-exception-caught
        msg = str(e).split(". Available:", maxsplit=1)[0]  # open_clip appends every known model name
        raise ValueError(f"Could not load embedder {spec!r}: {msg[:300]}") from e


def register_embedder(spec: str, embedder: Embedder) -> None:
    """Make ``get_embedder(spec)`` return ``embedder``, replacing any embedder loaded under ``spec``.

    Use this to plug in your own ``Embedder`` subclass under any spec string.
    """
    with _load_lock:
        _loaded[spec] = embedder
        _registered.add(spec)
        _namespaces.pop(spec, None)


def unregister_embedder(spec: str) -> None:
    """Forget the embedder registered or loaded under ``spec``; does nothing if there is none."""
    with _load_lock:
        _loaded.pop(spec, None)
        _registered.discard(spec)
        _namespaces.pop(spec, None)


def is_registered(spec: str) -> bool:
    """Whether ``spec`` names an embedder added with ``register_embedder``."""
    with _load_lock:
        return spec in _registered


def spec_supports_text(spec: str) -> bool:
    """Whether the embedder for ``spec`` can embed text.

    A registered or already-loaded embedder answers through its ``supports_text`` flag; for any other
    spec, only the OpenCLIP backend can.
    """
    with _load_lock:
        emb = _loaded.get(spec)
    return emb.supports_text if emb is not None else spec.startswith("openclip:")
