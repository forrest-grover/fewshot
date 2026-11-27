"""Tests for serving the built web UI (fewshot.api.mount_ui)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fewshot import errors
from fewshot.api import mount_ui


@pytest.fixture(name="dist")
def dist_fixture(tmp_path):
    """A fake Vite build: index.html, a hashed asset, a top-level file, and a secret outside it."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>INDEX</html>")
    (dist / "assets" / "app.js").write_text("console.log('app')")
    (dist / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("SECRET")
    return dist


@pytest.fixture(name="ui")
def ui_fixture(dist):
    """A client for an app with only the UI mounted."""
    app = FastAPI()
    errors.install(app)
    mount_ui(app, dist)
    return TestClient(app)


def test_index_and_spa_fallback(ui):
    """The root and unknown client-side routes get index.html."""
    for path in ["/", "/tasks/pets", "/does/not/exist.js", "/assets"]:
        r = ui.get(path)
        assert r.status_code == 200, path
        assert r.text == "<html>INDEX</html>", path
        assert r.headers["content-type"].startswith("text/html")


def test_ui_pages_cant_be_framed(ui):
    """The UI's pages carry a CSP (no other site may frame them; only the UI's own scripts and
    styles) and X-Frame-Options; index.html is revalidated, so a new build is picked up."""
    for path in ["/", "/tasks/pets", "/favicon.svg"]:
        r = ui.get(path)
        csp = r.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in csp, path
        assert "script-src 'self';" in csp, path
        assert "img-src 'self' blob: data:" in csp, path  # thumbnails, and previews of picked files
        assert (r.headers["x-frame-options"], r.headers["x-content-type-options"]) == ("DENY", "nosniff"), path
    assert ui.get("/").headers["cache-control"] == "no-cache"


def test_real_files(ui):
    """Files in dist and dist/assets are served as-is."""
    assert ui.get("/favicon.svg").text == "<svg/>"
    r = ui.get("/assets/app.js")
    assert r.status_code == 200
    assert r.text == "console.log('app')"
    assert ui.get("/assets/missing.js").status_code == 404


def test_path_traversal_gets_index(ui):
    """Paths escaping dist never serve files outside it."""
    for path in ["/..%2fsecret.txt", "/%2e%2e/secret.txt", "/assets/..%2f..%2fsecret.txt"]:
        r = ui.get(path)
        assert "SECRET" not in r.text, path
    assert ui.get("/..%2fsecret.txt").text == "<html>INDEX</html>"


def test_unknown_api_route_is_404(ui):
    """Unmatched /api paths are a JSON 404, not the HTML page."""
    assert ui.get("/api").status_code == 404
    r = ui.get("/api/nope")
    assert r.status_code == 404
    assert r.json() == {"type": "about:blank", "title": "Not Found", "status": 404, "code": "ROUTE_NOT_FOUND",
                        "detail": "No route /api/nope; see /docs for the API's routes", "instance": "/api/nope"}
    assert ui.get("/apiary").text == "<html>INDEX</html>"


def test_missing_dist_mounts_nothing(tmp_path):
    """Without a build, no UI routes are registered."""
    app = FastAPI()
    before = len(app.routes)
    mount_ui(app, tmp_path / "nope")
    assert len(app.routes) == before
    assert TestClient(app).get("/").status_code == 404
