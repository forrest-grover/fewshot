"""Smoke tests of the real models, which every other test replaces with a fake: opt-in, as they
download the weights (into HF_HOME) unless they are cached. Run with
FEWSHOT_TEST_MODELS=1 pytest -m model; the scheduled workflow runs them weekly against both the
pinned and the newest dependencies, so a release that breaks loading a preset is caught."""

import numpy as np
import pytest
from PIL import Image

from fewshot.embedders import DEFAULT_SPEC, PRESETS, EmbedderInfo, get_embedder
from tests.conftest import BLUE, RED

pytestmark = pytest.mark.model


def _unit_rows(v: np.ndarray, n: int, dim: int) -> None:
    assert v.shape == (n, dim)
    assert v.dtype == np.float32
    assert np.isfinite(v).all()
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4)


def test_the_default_is_a_preset():
    """The default embedder is one of the presets tested below."""
    assert DEFAULT_SPEC in {p.spec for p in PRESETS}


@pytest.mark.parametrize("preset", PRESETS, ids=[p.spec for p in PRESETS])
def test_every_preset_embeds_images_and_texts(preset: EmbedderInfo):
    """Each preset loads and embeds images (and texts, if it supports them) as unit-length
    float32 rows of its size. Text models put a red image nearer "red" than "blue"; image-only
    ones tell a red image from a blue one."""
    emb = get_embedder(preset.spec)  # FEWSHOT_MAX_MODELS bounds how many stay loaded
    images = emb.embed_images([Image.new("RGB", (96, 64), RED), Image.new("RGB", (40, 40), BLUE),
                               Image.new("RGB", (96, 64), RED)])
    _unit_rows(images, 3, emb.dim)
    assert images[0] @ images[2] > images[0] @ images[1]
    assert emb.supports_text == preset.supports_text
    if preset.supports_text:
        texts = emb.embed_texts(["a plain red square", "a plain blue square"])
        _unit_rows(texts, 2, emb.dim)
        sims = images[:2] @ texts.T
        assert sims[0, 0] > sims[0, 1]
        assert sims[1, 1] > sims[1, 0]
