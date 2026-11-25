"""Image (and optionally text) embedders.

Every embedder returns L2-normalised float32 vectors so downstream classifiers
can use dot products as cosine similarity. Embedders are identified by a spec
string of the form ``<backend>:<model>[@<pretrained>]``:

    openclip:ViT-B-32@laion2b_s34b_b79k
    timm:vit_small_patch14_dinov2.lvd142m
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

# torch is imported inside functions so that importing this module (for PRESETS)
# stays fast for the CLI.


@dataclass(frozen=True)
class EmbedderInfo:
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
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _l2(x) -> np.ndarray:
    x = x.float()
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return x.cpu().numpy().astype(np.float32)


class Embedder(ABC):
    spec: str
    supports_text: bool = False
    batch_size: int = 64

    def __init__(self) -> None:
        # Models are shared across API request threads; serialise GPU use.
        self._lock = threading.Lock()

    @abstractmethod
    def _embed_image_batch(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def _embed_text_batch(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.spec} does not support text embeddings")

    @property
    @abstractmethod
    def dim(self) -> int: ...

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        import torch

        out = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                batch = [im.convert("RGB") for im in images[i : i + self.batch_size]]
                out.append(self._embed_image_batch(batch))
        return np.concatenate(out)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        import torch

        with self._lock, torch.inference_mode():
            return self._embed_text_batch(list(texts))


class OpenCLIPEmbedder(Embedder):
    supports_text = True

    def __init__(self, spec: str, model_name: str, pretrained: str | None) -> None:
        super().__init__()
        import open_clip

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

    def _autocast(self):
        import torch

        return torch.autocast("cuda", dtype=torch.float16) if self.device == "cuda" else torch.autocast("cpu", enabled=False)

    def _embed_image_batch(self, images):
        import torch

        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with self._autocast():
            return _l2(self.model.encode_image(x))

    def _embed_text_batch(self, texts):
        tok = self.tokenizer(texts).to(self.device)
        with self._autocast():
            return _l2(self.model.encode_text(tok))


class TimmEmbedder(Embedder):
    def __init__(self, spec: str, model_name: str) -> None:
        super().__init__()
        import timm

        self.spec = spec
        self.device = _device()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device).eval()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.preprocess = timm.data.create_transform(**cfg)
        self._dim = int(self.model.num_features)

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_image_batch(self, images):
        import torch

        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        if self.device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                return _l2(self.model(x))
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


def spec_supports_text(spec: str) -> bool:
    return spec.startswith("openclip:")
