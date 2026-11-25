"""Shared fixtures: a deterministic, model-free embedder and an empty workspace."""

import io

import numpy as np
import pytest
from PIL import Image

from fewshot import Workspace, embedders
from fewshot.embedders import Embedder

# Tests inject into the embedder registry directly; there is no public registration API.
REGISTRY = embedders._loaded  # pylint: disable=protected-access


# Image-only on purpose: the inherited text path raises NotImplementedError, like DINOv2.
class ColorEmbedder(Embedder):  # pylint: disable=abstract-method
    """Embeds an image as its normalised mean RGB colour: deterministic and model-free."""

    spec = "fake:color"

    @property
    def dim(self):
        return 3

    def _embed_image_batch(self, images):
        v = np.stack([np.asarray(im, np.float32).reshape(-1, 3).mean(0) + 1.0 for im in images])
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def png(rgb, size=8, noise=0, seed=0):
    """PNG bytes of a ``size``x``size`` image of colour ``rgb`` with optional Gaussian noise."""
    rng = np.random.default_rng(seed)
    a = np.clip(np.array(rgb, float) + rng.normal(0, noise, (size, size, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def fake_embedder():
    """Register the "fake:color" spec so no real model is ever loaded."""
    REGISTRY["fake:color"] = ColorEmbedder()
    yield
    REGISTRY.pop("fake:color", None)


@pytest.fixture
def ws(tmp_path):
    """An empty workspace in a temporary folder."""
    return Workspace(tmp_path / "ws")
