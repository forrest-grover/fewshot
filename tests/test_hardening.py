"""Limits that keep a server safe from hostile input: redirects, slow downloads, huge images,
host names, unchecked thresholds, unfitted model weights and overconfident calibration."""

import http.server
import importlib
import io
import threading
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import fewshot.api
from fewshot import evaluate as ev
from fewshot import fetch, images, pipeline
from fewshot.archive import export_task, import_task
from fewshot.guard import host_allowed
from tests.conftest import png

RED = (220, 30, 30)


class _Hostile(http.server.BaseHTTPRequestHandler):
    """/ftp redirects to an ftp:// URL; /drip sends its body a byte at a time."""

    def do_GET(self):  # pylint: disable=invalid-name  # http.server's naming
        """Answer as the path says."""
        if self.path == "/ftp":
            self.send_response(302)
            self.send_header("Location", "ftp://127.0.0.1:1/secret")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", "20")
        self.end_headers()
        for _ in range(20):
            self.wfile.write(b"x")
            self.wfile.flush()
            time.sleep(0.1)

    def log_message(self, *_args):
        pass


@pytest.fixture(name="hostile")
def fixture_hostile():
    """The URL of a local server running ``_Hostile``."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hostile)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_redirects_only_to_http(hostile, monkeypatch):
    """A redirect to ftp:// (which no address check covers) is refused, with or without public_only."""
    with pytest.raises(ValueError, match="isn't http"):
        fetch.fetch_url(hostile + "/ftp", 10_000)
    monkeypatch.setattr(fetch, "public_addresses", lambda host, port: ["127.0.0.1"])
    with pytest.raises(ValueError, match="isn't http"):
        fetch.fetch_url(hostile + "/ftp", 10_000, public_only=True)


def test_timeout_bounds_the_whole_download(hostile):
    """A body sent a byte at a time fails once the timeout has passed, though no single read is slow."""
    t0 = time.monotonic()
    with pytest.raises(ValueError, match=r"took longer than 0\.5 s"):
        fetch.fetch_url(hostile + "/drip", 10_000, timeout=0.5)
    assert time.monotonic() - t0 < 1.5


def _png_of(mode, size):
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, "PNG")
    return buf.getvalue()


def _jpeg_of(size):
    buf = io.BytesIO()
    Image.new("RGB", size, RED).save(buf, "JPEG")
    return buf.getvalue()


def test_pixel_limit(monkeypatch):
    """An image over MAX_PIXELS is refused before decoding; a JPEG counts at its draft scale."""
    monkeypatch.setattr(images, "MAX_PIXELS", 10_000)
    with pytest.raises(ValueError, match=r"200x200 is over the 0\.01 megapixel limit"):
        images.load_image(_png_of("1", (200, 200)))
    with pytest.raises(ValueError, match="megapixel limit"):
        images.image_ext(_png_of("1", (200, 200)))
    monkeypatch.setattr(images, "VERIFY_SIDE", 50)
    assert images.image_ext(_jpeg_of((400, 400))) == "jpg"  # verified at 1/8 scale: 50x50


def test_images_are_shrunk_before_embedding(monkeypatch):
    """Decoded images, and PIL images passed in, are scaled to fit EMBED_MAX_SIDE."""
    assert images.load_image(_png_of("RGB", (300, 150)), max_side=100).size == (100, 50)
    assert images.load_image(_png_of("RGB", (30, 15)), max_side=100).size == (30, 15)
    monkeypatch.setattr(pipeline, "EMBED_MAX_SIDE", 64)
    assert pipeline._to_pil(_png_of("RGB", (256, 128))).size == (64, 32)  # pylint: disable=protected-access
    assert pipeline._to_pil(Image.new("L", (128, 256))).size == (32, 64)  # pylint: disable=protected-access
    assert pipeline._to_pil(Image.new("L", (16, 16))).mode == "RGB"  # pylint: disable=protected-access


def test_host_allowed():
    """Listed names (any port), '*', and IP addresses pass; other names don't."""
    allowed = {"box.lan"}
    for host in ("box.lan", "BOX.LAN:8000", "10.0.0.5:8000", "[fe80::1]:80", "203.0.113.9"):
        assert host_allowed(host, allowed), host
    for host in ("evil.example", "box.lan.evil.example", "", "10.0.0.5.nip.io"):
        assert not host_allowed(host, allowed), host
    assert host_allowed("anything", {"*"})


def test_allowed_host_entries_may_have_ports(tmp_path, monkeypatch):
    """An entry written with a port matches its name; the health check by IP always works."""
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path / "ws"))
    monkeypatch.setenv("FEWSHOT_ALLOWED_HOSTS", "box.lan:8443")
    c = TestClient(importlib.reload(fewshot.api).app)
    assert c.get("/api/health", headers={"Host": "box.lan:8443"}).status_code == 200
    assert c.get("/healthz", headers={"Host": "127.0.0.1:8000"}).status_code == 200
    assert c.get("/api/health", headers={"Host": "other.lan"}).status_code == 400


def test_import_checks_class_thresholds(ws, tmp_path):
    """An export whose task.json gives a class a threshold outside -1..1 is refused."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    t.set_class_thresholds({"red": 0.5})
    zpath = tmp_path / "t.zip"
    with zpath.open("wb") as f:
        export_task(ws, "t", f)
    edited = tmp_path / "bad.zip"
    with zipfile.ZipFile(zpath) as src, zipfile.ZipFile(edited, "w") as dst:
        for item in src.infolist():
            data = src.read(item)
            if item.filename == "task.json":
                data = data.replace(b'"reject_below": 0.5', b'"reject_below": 5.0')
                assert b"5.0" in data
            dst.writestr(item, data)
    with pytest.raises(ValueError, match=r"threshold of 'red' must be between -1 and 1, got 5\.0"):
        import_task(ws, edited, name="t2")


def _held_out(P, truth):
    h = ev.HeldOut.empty(len(truth), P.shape[1])
    h.record(np.arange(len(truth)), P, np.ones(len(truth)), truth, np.ones(len(truth), bool))
    return h


def test_calibration_when_held_out_predictions_are_all_right():
    """All-correct predictions suggest a sharper but bounded temperature, not the grid's minimum,
    and the suggested absolute temperature stays within the range settings accept."""
    n = 60
    truth = np.arange(n) % 3
    P = np.full((n, 3), 0.14)
    P[np.arange(n), truth] = 0.72
    cal = ev._calibration(_held_out(P, truth), truth, 1.0)  # pylint: disable=protected-access
    assert cal is not None
    assert 0.05 < cal["temperature"] < 1
    assert cal["ece_calibrated"] < cal["ece"]
    cal = ev._calibration(_held_out(P, truth), truth, 0.02)  # pylint: disable=protected-access
    assert cal is not None
    assert cal["temperature"] == 0.01  # 0.02 times the fitted one, clamped
