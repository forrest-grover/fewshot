import io

import numpy as np
import pytest
from PIL import Image

from fewshot import embedders
from fewshot.embedders import Embedder


class ColorEmbedder(Embedder):
    """Embeds an image as its normalised mean RGB colour: deterministic and model-free."""

    spec = "fake:color"

    @property
    def dim(self):
        return 3

    def _embed_image_batch(self, images):
        v = np.stack([np.asarray(im, np.float32).reshape(-1, 3).mean(0) + 1.0 for im in images])
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def png(rgb, size=8, noise=0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.clip(np.array(rgb, float) + rng.normal(0, noise, (size, size, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def fake_embedder():
    embedders._loaded["fake:color"] = ColorEmbedder()
    yield
    embedders._loaded.pop("fake:color", None)


@pytest.fixture
def ws(tmp_path):
    from fewshot import Workspace
    return Workspace(tmp_path / "ws")
