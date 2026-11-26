"""Smoke test of a real model, which every other test replaces with a fake: opt-in, as it downloads
the weights (into HF_HOME) unless they are cached. Run with FEWSHOT_TEST_MODELS=1 pytest -m model."""

import numpy as np
import pytest
from PIL import Image

from fewshot.embedders import get_embedder
from tests.conftest import BLUE, RED

pytestmark = pytest.mark.model

SMALLEST = "openclip:ViT-B-32@laion2b_s34b_b79k"  # the smallest preset that embeds text too


def test_the_smallest_preset_embeds_images_and_texts():
    """Images and texts come out as unit-length float32 rows of the model's size, in one space:
    a red image is nearer "red" than "blue"."""
    emb = get_embedder(SMALLEST)
    images = emb.embed_images([Image.new("RGB", (96, 64), RED), Image.new("RGB", (40, 40), BLUE)])
    texts = emb.embed_texts(["a plain red square", "a plain blue square"])
    for v, n in ((images, 2), (texts, 2)):
        assert v.shape == (n, emb.dim)
        assert v.dtype == np.float32
        assert np.isfinite(v).all()
        assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4)
    sims = images @ texts.T
    assert sims[0, 0] > sims[0, 1]
    assert sims[1, 1] > sims[1, 0]
