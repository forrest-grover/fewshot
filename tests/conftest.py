"""Shared fixtures: a deterministic, model-free embedder and an empty workspace."""

import dataclasses
import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# TestClient sends "Host: testserver"; tests of the host check set their own list.
os.environ.setdefault("FEWSHOT_ALLOWED_HOSTS", "*")

from fastapi.testclient import TestClient  # pylint: disable=wrong-import-position

from fewshot import Workspace  # pylint: disable=wrong-import-position
from fewshot.api import create_app  # pylint: disable=wrong-import-position
from fewshot.embedders import Embedder, register_embedder, unregister_embedder  # pylint: disable=wrong-import-position
from fewshot.server_settings import ServerSettings  # pylint: disable=wrong-import-position


# Image-only on purpose: the inherited text path raises NotImplementedError, like DINOv2.
class ColorEmbedder(Embedder):  # pylint: disable=abstract-method
    """Embeds an image as its normalised mean RGB colour: deterministic and model-free."""

    spec = "fake:color"
    cache_version: str | None = "test"  # skip the fingerprint, which would embed an extra probe image

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


@pytest.fixture(autouse=True, scope="session")
def no_workspace_in_the_working_directory():
    """Fail the run if a test made ./fewshot_data (the default workspace) where pytest runs."""
    existed = Path("fewshot_data").exists()
    yield
    assert existed or not Path("fewshot_data").exists(), "a test created ./fewshot_data"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """Tests never touch a real workspace: FEWSHOT_HOME is a fresh temporary folder."""
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path_factory.mktemp("home")))


def api_client(tmp_path, base_url="http://testserver", **settings):
    """A TestClient on a new server for ``settings`` (any host allowed, no warm-up, workspace in
    ``tmp_path/ws`` unless given); the server is ``client.srv``."""
    settings = {"home": tmp_path / "ws", "allowed_hosts": frozenset({"*"}), "warm": False, **settings}
    app = create_app(ServerSettings(**settings))
    client = TestClient(app, base_url=base_url)
    client.srv = app.state.server  # type: ignore[attr-defined]
    return client


def set_settings(monkeypatch, client, **changes):
    """Change the settings of ``client``'s server for this test."""
    monkeypatch.setattr(client.srv, "settings", dataclasses.replace(client.srv.settings, **changes))


@pytest.fixture(autouse=True)
def restore_environment():
    """Undo environment changes a test makes (``fewshot serve`` exports its settings)."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def fake_embedder():
    """Register the "fake:color" spec so no real model is ever loaded."""
    register_embedder("fake:color", ColorEmbedder())
    yield
    unregister_embedder("fake:color")


@pytest.fixture
def ws(tmp_path):
    """An empty workspace in a temporary folder."""
    return Workspace(tmp_path / "ws")
