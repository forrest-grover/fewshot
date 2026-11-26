"""Shared fixtures and helpers: a deterministic, model-free embedder, an empty workspace, API and CLI
runners, a local HTTP server and a zip builder; and guards that keep tests out of the working directory."""

import contextlib
import dataclasses
import http.server
import io
import os
import re
import threading
import time
import zipfile
from collections.abc import Iterator

import numpy as np
import pytest
from PIL import Image

# TestClient sends "Host: testserver"; tests of the host check set their own list.
os.environ.setdefault("FEWSHOT_ALLOWED_HOSTS", "*")

from fastapi.testclient import TestClient  # pylint: disable=wrong-import-position
from typer.testing import CliRunner  # pylint: disable=wrong-import-position

from fewshot import Workspace, cli  # pylint: disable=wrong-import-position
from fewshot.api import create_app  # pylint: disable=wrong-import-position
from fewshot.embedders import Embedder, register_embedder, unregister_embedder  # pylint: disable=wrong-import-position
from fewshot.server_settings import ServerSettings  # pylint: disable=wrong-import-position

RED, GREEN, BLUE = (220, 30, 30), (30, 220, 30), (30, 30, 220)


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


class TextColor(ColorEmbedder):
    """Colour embedder that also embeds texts naming a colour (red, green or blue)."""

    supports_text = True

    def _embed_text_batch(self, texts):
        words = {"red": RED, "green": GREEN, "blue": BLUE}
        v = np.array([[float(c) + 1.0 for c in next(rgb for w, rgb in words.items() if w in t)] for t in texts])
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def png(rgb, size=8, noise=0, seed=0):
    """PNG bytes of a ``size``x``size`` image of colour ``rgb`` with optional Gaussian noise."""
    rng = np.random.default_rng(seed)
    a = np.clip(np.array(rgb, float) + rng.normal(0, noise, (size, size, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


def pytest_collection_modifyitems(config, items):  # pylint: disable=unused-argument  # pytest's hook signature
    """Skip the tests marked ``model`` (they load real models) unless FEWSHOT_TEST_MODELS=1."""
    if os.environ.get("FEWSHOT_TEST_MODELS") == "1":
        return
    skip = pytest.mark.skip(reason="loads a real model: set FEWSHOT_TEST_MODELS=1 to run it")
    for item in items:
        if item.get_closest_marker("model"):
            item.add_marker(skip)


# Made by the tools that run the tests, not by the tests themselves.
_RUNNER_FILES = re.compile(r"\.coverage.*|\.pytest_cache|__pycache__|\.mypy_cache|\.ruff_cache|\.hypothesis")


@pytest.fixture(autouse=True)
def nothing_written_to_the_working_directory(request):
    """Fail a test that leaves a new file or folder (say ./fewshot_data, the default workspace, or a
    stray tmpXXXXXXXX) in the folder pytest runs from: tests write only under tmp_path."""
    before = set(os.listdir())
    yield
    new = sorted(n for n in set(os.listdir()) - before if not _RUNNER_FILES.fullmatch(n))
    assert not new, f"{request.node.nodeid} left {new} in the working directory {os.getcwd()}"


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


@pytest.fixture(name="ws")
def ws_fixture(tmp_path):
    """An empty workspace in a temporary folder."""
    return Workspace(tmp_path / "ws")


# ---- small tasks --------------------------------------------------------------------------------
def red_task(ws, name="t", n=2):
    """Task ``name`` in ``ws`` with ``n`` red examples."""
    t = ws.create(name, embedder="fake:color")
    for i in range(n):
        t.add_example(png(RED, noise=10, seed=i), "red")
    return t


def red_green_task(ws, name="t", n=2):
    """Task ``name`` in ``ws`` with ``n`` red and ``n`` green examples, added one at a time."""
    t = ws.create(name, embedder="fake:color")
    for i in range(n):
        t.add_example(png(RED, noise=10, seed=i), "red")
        t.add_example(png(GREEN, noise=10, seed=i), "green")
    return t


def two_each(ws):
    """Task 't' with two red and two blue examples (r0.png, r1.png, b0.png, b1.png), added in one change."""
    t = ws.create("t", embedder="fake:color")
    t.add_examples([(png(RED, noise=10, seed=i), "red", f"r{i}.png") for i in range(2)]
                   + [(png(BLUE, noise=10, seed=i), "blue", f"b{i}.png") for i in range(2)])
    return t


def red_blue_task(ws, name="t"):
    """Task ``name`` in ``ws`` with two red and two blue examples."""
    t = ws.create(name, embedder="fake:color")
    for i in range(2):
        t.add_example(png(RED, noise=5, seed=i), "red")
        t.add_example(png(BLUE, noise=5, seed=i), "blue")
    return t


@pytest.fixture(name="text_spec")
def fixture_text_spec():
    """A text-capable colour embedder registered under a CLIP-like spec."""
    register_embedder("openclip:fake-text", TextColor())
    yield "openclip:fake-text"
    unregister_embedder("openclip:fake-text")


def zip_of(files):
    """A zip (as a file object at position 0) holding ``files``, a {name: data} dict."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    buf.seek(0)
    return buf


@contextlib.contextmanager
def serve_http(handler: type[http.server.BaseHTTPRequestHandler]) -> Iterator[str]:
    """Run a local threaded HTTP server with ``handler``; yields its base URL ("http://127.0.0.1:PORT")."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


# ---- API ----------------------------------------------------------------------------------------
@pytest.fixture(name="client")
def client_fixture(tmp_path):
    """A TestClient on a new server with default settings, whose workspace lives in tmp_path."""
    return api_client(tmp_path)


@pytest.fixture(name="task_t")
def task_t_fixture(client):
    """Task 't' on ``client``'s server: two red and two blue examples."""
    client.post("/api/tasks", json={"name": "t", "embedder": "fake:color"})
    for label, rgb in (("red", RED), ("blue", BLUE)):
        files = [("files", (f"{label}{i}.png", png(rgb, noise=10, seed=i), "image/png")) for i in range(2)]
        assert client.post("/api/tasks/t/examples", data={"label": label}, files=files).status_code == 201
    return "t"


def host_client(tmp_path, monkeypatch, hosts=None, base_url="http://testserver"):  # pylint: disable=unused-argument
    """A TestClient on a new server that answers the host names ``hosts`` (comma-separated), by default
    the server's default list."""
    allowed = ServerSettings().allowed_hosts if hosts is None else frozenset(hosts.split(","))
    return api_client(tmp_path, base_url=base_url, allowed_hosts=allowed)


def api_task(client, name="c", n=3):
    """Create a task with ``n`` red and ``n`` blue examples over the API."""
    assert client.post("/api/tasks", json={"name": name, "embedder": "fake:color"}).status_code == 201
    for label, rgb in (("red", RED), ("blue", BLUE)):
        files = [("files", (f"{label}{i}.png", png(rgb, noise=15, seed=i), "image/png")) for i in range(n)]
        assert client.post(f"/api/tasks/{name}/examples", data={"label": label}, files=files).status_code == 201
    return name


def query_files(n, rgb=RED):
    """``n`` upload fields of noisy ``rgb`` images (q0.png, q1.png, ...)."""
    return [("files", (f"q{i}.png", png(rgb, noise=10, seed=50 + i), "image/png")) for i in range(n)]


def wait_for_job(client, jid, timeout=10.0):
    """Poll job ``jid`` until it is neither running nor cancelling (or ``timeout`` passes); returns it."""
    deadline = time.time() + timeout
    while True:
        job = client.get(f"/api/jobs/{jid}").json()
        if job["status"] not in {"running", "cancelling"} or time.time() > deadline:
            return job
        time.sleep(0.02)


# ---- CLI ----------------------------------------------------------------------------------------
ANSI = re.compile(r"\x1b\[[0-9;]*m")


class PlainResult:  # pylint: disable=too-few-public-methods  # stands in for click's Result
    """A click Result with ANSI styling removed from ``output``.

    Typer forces coloured error output when GITHUB_ACTIONS, FORCE_COLOR or PY_COLORS is set,
    and the escape codes split text such as ``--shots`` that tests look for.
    """

    def __init__(self, result):
        self._result = result
        self.output = ANSI.sub("", result.output)

    def __getattr__(self, name):
        return getattr(self._result, name)


@pytest.fixture(name="root")
def root_fixture(tmp_path):
    """Workspace directory for the test."""
    return tmp_path / "ws"


@pytest.fixture(name="run")
def run_fixture(root):
    """Invoke the CLI against the temporary workspace; returns the Result with plain-text output."""
    runner = CliRunner()

    def invoke(*args, stdin=None):
        return PlainResult(runner.invoke(cli.app, [str(a) for a in args] + ["--root", str(root)], input=stdin))

    return invoke


def served(monkeypatch, *args):
    """Run `fewshot serve ARGS` with uvicorn stubbed out; returns (result, the server, uvicorn's options)."""
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.append((app, kw)))
    r = CliRunner().invoke(cli.app, ["serve", *args])
    assert r.exit_code == 0, r.output
    app, kw = calls[-1]
    return PlainResult(r), app.state.server, kw


def write_images(folder, rgb, n, start=0):
    """Write ``n`` noisy PNGs of colour ``rgb`` into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + n):
        (folder / f"{i}.png").write_bytes(png(rgb, noise=20, seed=i))
