"""Image (and optionally text) embedders.

Every embedder returns L2-normalised float32 vectors so downstream classifiers
can use dot products as cosine similarity. Embedders are identified by a spec
string of the form ``<backend>:<model>[@<pretrained>]``:

    openclip:PE-Core-B-16@meta
    timm:vit_small_patch14_dinov2.lvd142m
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
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
GROW_AFTER = 20  # batches that must fit before a batch size halved by an out-of-memory error doubles


@dataclass(frozen=True)
class EmbedderInfo:
    """Description of an embedder preset for display in the UI and CLI."""

    spec: str
    label: str
    supports_text: bool
    notes: str


# Presets shown in the UI/CLI. Any valid spec string also works. Accuracy notes come from the
# all-classes benchmark over six datasets in the README (GDA classifier, 3 seeds).
PRESETS: list[EmbedderInfo] = [
    EmbedderInfo(
        "openclip:PE-Core-B-16@meta",
        "PE Core B/16 (Meta)",
        True,
        "Default: fast, and the most accurate base-size model here (78% at 1 example per class, 89% at "
        "16, averaged over six datasets). Supports text descriptions.",
    ),
    EmbedderInfo(
        "openclip:PE-Core-L-14-336@meta",
        "PE Core L/14, 336 px (Meta)",
        True,
        "Most accurate (83% at 1 example per class, 92% at 16), about 4x slower to embed than B/16. "
        "Supports text descriptions.",
    ),
    EmbedderInfo(
        "openclip:ViT-B-16-SigLIP2-256@webli",
        "SigLIP 2 B/16, 256 px (WebLI)",
        True,
        "Close to PE Core B/16, and reads descriptions in many languages. Needs `pip install transformers` "
        "for its tokenizer.",
    ),
    EmbedderInfo(
        "openclip:ViT-B-32@laion2b_s34b_b79k",
        "CLIP ViT-B/32 (LAION-2B)",
        True,
        "Smallest download; the previous default. About 10 points less accurate than PE Core B/16.",
    ),
    EmbedderInfo(
        "timm:vit_small_patch14_dinov2.lvd142m",
        "DINOv2 ViT-S/14",
        False,
        "Image-only and small. On the benchmark photos it trails PE Core even without descriptions; it may "
        "suit images unlike web photos.",
    ),
    EmbedderInfo(
        "timm:vit_large_patch14_dinov2.lvd142m",
        "DINOv2 ViT-L/14",
        False,
        "Image-only. Larger DINOv2; on the benchmark it trails PE Core B/16 by 2 to 3 points at 16 examples.",
    ),
]

DEFAULT_SPEC = PRESETS[0].spec


def _device() -> str:
    """``$FEWSHOT_DEVICE`` if set (e.g. ``cpu``, ``cuda:1``, ``mps``), else CUDA, then Apple's GPU
    (MPS), then the CPU, whichever is available first."""
    import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

    if forced := os.environ.get("FEWSHOT_DEVICE", "").strip():
        try:
            dev = torch.device(forced)
        except RuntimeError as e:
            raise ValueError(f"FEWSHOT_DEVICE={forced!r} is not a torch device: {e}") from None
        # Checked now, so a missing device fails with a clear message rather than inside a model load.
        if dev.type == "cuda" and (dev.index or 0) >= torch.cuda.device_count():
            raise ValueError(f"FEWSHOT_DEVICE={forced!r}: this machine has {torch.cuda.device_count()} CUDA devices")
        if dev.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError(f"FEWSHOT_DEVICE={forced!r}: Apple's GPU (MPS) is not available here")
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
        self.users = 0  # embed calls in progress; a model in use is never unloaded
        self._full_batch: int | None = None  # batch_size before out-of-memory halving
        self._good_batches = 0  # batches that fitted since the last halving

    @contextlib.contextmanager
    def _in_use(self) -> Iterator[None]:
        with _load_lock:
            self.users += 1
        try:
            yield
        finally:
            with _load_lock:
                self.users -= 1

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
        with self._in_use():
            while i < len(images):
                batch = [im.convert("RGB") for im in images[i : i + self.batch_size]]
                out.append(self._embed_halving(self._prepare_images(batch), len(batch), self._embed_prepared))
                i += len(batch)
        return np.concatenate(out)

    def _embed_halving(self, batch: Any, n: int, forward: Callable[[Any], Vectors]) -> Vectors:
        """Run ``forward`` (under the model lock) on a batch of ``n`` prepared images or texts. If
        the GPU runs out of memory, free what it can, halve ``batch_size`` for the rest of this
        embedder's life, and run the batch in halves."""
        import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

        try:
            with self._lock, torch.inference_mode():
                vecs = self._checked(forward(batch), n)
                self._fitted_batch()
            return vecs
        except Exception as e:  # pylint: disable=broad-exception-caught  # only OOM is handled; re-raised
            if n == 1 or not _is_oom(e):
                raise
        # Retry only here, outside the except block: inside it, the exception's traceback would keep
        # the failed forward pass's tensors alive, so freeing memory couldn't help the retry.
        _free_gpu_memory()
        half = n // 2
        with self._lock:  # like ``_fitted_batch``: another job's update mustn't interleave
            self._full_batch = self._full_batch or self.batch_size
            self._good_batches = 0
            self.batch_size = max(1, min(self.batch_size, half))
        return np.concatenate([self._embed_halving(batch[:half], half, forward),
                               self._embed_halving(batch[half:], n - half, forward)])

    def _fitted_batch(self) -> None:
        """Count a batch that fitted; after ``GROW_AFTER`` in a row, double a halved ``batch_size``
        again (up to its original size), since an out-of-memory error may have been caused by
        another job that has finished since. Called holding the model lock."""
        if self._full_batch is None or self.batch_size >= self._full_batch:
            return
        self._good_batches += 1
        if self._good_batches >= GROW_AFTER:
            self.batch_size = min(self._full_batch, self.batch_size * 2)
            self._good_batches = 0

    def embed_texts(self, texts: Sequence[str]) -> Vectors:
        """Embed texts into the image embedding space, ``batch_size`` at a time; raises
        NotImplementedError if unsupported."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not self.supports_text:
            raise NotImplementedError(f"{self.spec} does not support text embeddings")
        texts = list(texts)
        out = []
        i = 0
        with self._in_use():
            while i < len(texts):  # in batches, like images, so a long prompt ensemble fits in memory
                batch = texts[i : i + self.batch_size]
                out.append(self._embed_halving(batch, len(batch), self._embed_text_batch))
                i += len(batch)
        return np.concatenate(out)

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
_held: dict[str, int] = {}  # spec -> operations in progress that may embed with it (see ``hold``)
_namespaces: dict[str, str] = {}  # registered spec -> its cache namespace, once computed


def max_models() -> int:
    """Most built-in models kept loaded at once (``$FEWSHOT_MAX_MODELS``, default 2); loading
    another unloads the least recently used. Registered embedders don't count and stay loaded."""
    raw = os.environ.get("FEWSHOT_MAX_MODELS", "2")
    try:
        return max(1, int(raw))
    except ValueError:
        raise ValueError(f"FEWSHOT_MAX_MODELS must be a whole number, got {raw!r}") from None


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
    Raises ``ValueError`` with a short reason if the model can't be loaded (a bad spec, no network,
    a model that doesn't exist).
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
        try:
            emb = _build(spec)
        # Model loaders (open_clip, timm, hub downloads) raise many unrelated exception types.
        except Exception as e:  # pylint: disable=broad-exception-caught
            msg = str(e).split(". Available:", maxsplit=1)[0]  # open_clip appends every known model name
            raise ValueError(f"Could not load embedder {spec!r}: {msg[:300]}") from e
        with _load_lock:
            if spec in _loaded:  # registered while we were loading: the registration wins
                return _loaded[spec]
            _loaded[spec] = emb
            evicted = _evict_locked(keep=spec)
    if evicted:
        _free_gpu_memory()
    return emb


@contextlib.contextmanager
def hold(spec: str) -> Iterator[None]:
    """Keep ``spec``'s model loaded (if it is, or once it is) until the block ends.

    ``Embedder.users`` only counts calls into the model, so an operation that embeds batch by
    batch, decoding images or reading files in between, would leave it idle between calls; held
    for the whole operation (embedding a task, predicting, evaluating), the model can't be
    unloaded for another one halfway and loaded again for the next batch. Doesn't load anything."""
    with _load_lock:
        _held[spec] = _held.get(spec, 0) + 1
    try:
        yield
    finally:
        with _load_lock:
            _held[spec] -= 1
            if not _held[spec]:
                del _held[spec]


def _build(spec: str) -> Embedder:
    backend, _, rest = spec.partition(":")
    if not rest:
        raise ValueError(f"Bad embedder spec {spec!r}; expected '<backend>:<model>'")
    if backend == "openclip":
        model, _, pretrained = rest.partition("@")
        if not pretrained:  # open_clip would quietly build the model with random weights
            raise ValueError(f"{spec!r} names no pretrained weights; add them as 'openclip:{model}@<pretrained>' "
                             "(open_clip.list_pretrained() lists them)")
        return OpenCLIPEmbedder(spec, model, pretrained)
    if backend == "timm":
        return TimmEmbedder(spec, rest)
    raise ValueError(f"Unknown embedder backend {backend!r} (use 'openclip' or 'timm')")


def _evict_locked(keep: str | None = None) -> list[str]:
    """Drop least recently used built-in models beyond ``max_models()``; returns their specs.

    A model in use (embedding right now, or held by an operation, see ``hold``) is skipped, so a
    later request can't load a second copy of it while the first still holds its memory; the limit
    may be exceeded until it is free.
    """
    built_in = [s for s in _loaded if s not in _registered]
    idle = [s for s in built_in  # never the model just loaded
            if _loaded[s].users == 0 and not _held.get(s) and s != keep]
    gone = idle[: max(0, len(built_in) - max_models())]
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
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


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
    get_embedder(spec)


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
