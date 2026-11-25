"""Image (and optionally text) embedders.

Every embedder returns L2-normalised float32 vectors so downstream classifiers
can use dot products as cosine similarity. Embedders are identified by a spec
string of the form ``<backend>:<model>[@<pretrained>]``:

    openclip:ViT-B-32@laion2b_s34b_b79k
    timm:vit_small_patch14_dinov2.lvd142m
"""

from __future__ import annotations

import contextlib
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
]

DEFAULT_SPEC = PRESETS[0].spec


def _device() -> str:
    import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

    return "cuda" if torch.cuda.is_available() else "cpu"


def _autocast(device: str) -> AbstractContextManager[Any]:
    """Half-precision autocast on CUDA; a no-op context on CPU."""
    if device != "cuda":
        return contextlib.nullcontext()
    import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

    return torch.autocast("cuda", dtype=torch.float16)


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

    def __init__(self) -> None:
        # Models are shared across API request threads; serialise GPU use.
        self._lock = threading.Lock()

    @abstractmethod
    def _embed_image_batch(self, images: Sequence[Image.Image]) -> Vectors: ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Length of each embedding vector."""

    def embed_images(self, images: Sequence[Image.Image]) -> Vectors:
        """Embed images (converted to RGB) in batches of ``batch_size``; returns an ``(n, dim)`` array."""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

        out: list[Vectors] = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                batch = [im.convert("RGB") for im in images[i : i + self.batch_size]]
                out.append(self._embed_image_batch(batch))
        return np.concatenate(out)

    def embed_texts(self, texts: Sequence[str]) -> Vectors:
        """Embed texts into the image embedding space; raises NotImplementedError if unsupported."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not self.supports_text:
            raise NotImplementedError(f"{self.spec} does not support text embeddings")
        import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

        with self._lock, torch.inference_mode():
            return self._embed_text_batch(list(texts))


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
        self._dim = int(self.model.visual.output_dim)

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_image_batch(self, images: Sequence[Image.Image]) -> Vectors:
        import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with _autocast(self.device):
            return _l2(self.model.encode_image(x))

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

    def _embed_image_batch(self, images: Sequence[Image.Image]) -> Vectors:
        import torch  # pylint: disable=import-outside-toplevel  # lazy: keeps CLI startup fast

        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with _autocast(self.device):
            return _l2(self.model(x))


_loaded: dict[str, Embedder] = {}
_load_lock = threading.Lock()


def get_embedder(spec: str = DEFAULT_SPEC) -> Embedder:
    """Load (once per process) and return the embedder for ``spec``."""
    with _load_lock:
        if spec in _loaded:
            return _loaded[spec]
        backend, _, rest = spec.partition(":")
        if not rest:
            raise ValueError(f"Bad embedder spec {spec!r}; expected '<backend>:<model>'")
        if backend == "openclip":
            model, _, pretrained = rest.partition("@")
            emb: Embedder = OpenCLIPEmbedder(spec, model, pretrained or None)
        elif backend == "timm":
            emb = TimmEmbedder(spec, rest)
        else:
            raise ValueError(f"Unknown embedder backend {backend!r} (use 'openclip' or 'timm')")
        _loaded[spec] = emb
        return emb


def register_embedder(spec: str, embedder: Embedder) -> None:
    """Make ``get_embedder(spec)`` return ``embedder``, replacing any embedder loaded under ``spec``.

    Use this to plug in your own ``Embedder`` subclass under any spec string.
    """
    with _load_lock:
        _loaded[spec] = embedder


def unregister_embedder(spec: str) -> None:
    """Forget the embedder registered or loaded under ``spec``; does nothing if there is none."""
    with _load_lock:
        _loaded.pop(spec, None)


def spec_supports_text(spec: str) -> bool:
    """Whether the embedder for ``spec`` can embed text.

    A registered or already-loaded embedder answers through its ``supports_text`` flag; for any other
    spec, only the OpenCLIP backend can.
    """
    with _load_lock:
        emb = _loaded.get(spec)
    return emb.supports_text if emb is not None else spec.startswith("openclip:")
