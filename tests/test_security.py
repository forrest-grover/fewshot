"""Security checks: host names, cross-site writes, image responses and task imports."""

import importlib
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

import fewshot.api
from fewshot import archive, models
from fewshot.archive import import_task
from fewshot.embedders import DEFAULT_SPEC
from fewshot.images import image_id
from tests.conftest import png

RED, BLUE = (220, 30, 30), (30, 30, 220)


def _client(tmp_path, monkeypatch, hosts=None, base_url="http://testserver"):
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path / "ws"))
    if hosts is None:
        monkeypatch.delenv("FEWSHOT_ALLOWED_HOSTS", raising=False)
    else:
        monkeypatch.setenv("FEWSHOT_ALLOWED_HOSTS", hosts)
    return TestClient(importlib.reload(fewshot.api).app, base_url=base_url)


# ---- host names (DNS rebinding) -----------------------------------------
def test_only_local_host_names_by_default(tmp_path, monkeypatch):
    """By default only localhost names are answered; anything else is 400, the UI included."""
    c = _client(tmp_path, monkeypatch, base_url="http://localhost:8000")
    assert c.get("/api/tasks").status_code == 200
    for host in ("127.0.0.1:8000", "[::1]:8000", "LOCALHOST"):
        assert c.get("/api/tasks", headers={"Host": host}).status_code == 200
    for host in ("evil.example", "evil.example:8000", "localhost.evil.example", ""):
        r = c.get("/api/health", headers={"Host": host})
        assert (r.status_code, r.json()["detail"]) == (400, "Unknown host; see FEWSHOT_ALLOWED_HOSTS")


def test_configured_host_names(tmp_path, monkeypatch):
    """FEWSHOT_ALLOWED_HOSTS lists the names; '*' allows any."""
    c = _client(tmp_path, monkeypatch, hosts="box.lan, [fe80::1]")
    assert c.get("/api/tasks", headers={"Host": "box.lan:8000"}).status_code == 200
    assert c.get("/api/tasks", headers={"Host": "[fe80::1]:8000"}).status_code == 200
    assert c.get("/api/tasks", headers={"Host": "localhost"}).status_code == 400
    c = _client(tmp_path, monkeypatch, hosts="*")
    assert c.get("/api/tasks", headers={"Host": "anything"}).status_code == 200


# ---- cross-site writes (CSRF) -------------------------------------------
def test_cross_site_writes_are_refused(tmp_path, monkeypatch):
    """A write whose Origin is another site, or that the browser marks cross-site, is 403;
    same-origin writes, reads, and clients that send neither header go through."""
    c = _client(tmp_path, monkeypatch, hosts="*", base_url="http://localhost:8000")
    body = {"name": "t", "embedder": "fake:color"}
    for headers in ({"Origin": "https://evil.example"}, {"Origin": "null"}, {"Origin": "http://localhost:9999"},
                    {"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"}):
        r = c.post("/api/tasks", json=body, headers=headers)
        assert (r.status_code, r.json()["detail"]) == (403, "Cross-site requests are not allowed"), headers
    files = [("files", ("a.png", png(RED), "image/png"))]
    assert c.post("/api/tasks/t/examples", data={"label": "red"}, files=files,
                  headers={"Origin": "https://evil.example"}).status_code == 403
    assert c.post("/api/tasks", json=body, headers={"Origin": "http://localhost:8000"}).status_code == 201
    same = {"Sec-Fetch-Site": "same-origin"}
    assert c.post("/api/tasks", json={**body, "name": "u"}, headers=same).status_code == 201
    assert c.post("/api/tasks", json={**body, "name": "v"}).status_code == 201  # curl, the CLI
    assert c.get("/api/tasks", headers={"Origin": "https://evil.example"}).status_code == 200


# ---- image responses ----------------------------------------------------
def test_images_are_served_only_as_images(tmp_path, monkeypatch):
    """Example images carry nosniff and a sandboxing CSP; a file whose name isn't an image type is
    served as an opaque download, never as a page."""
    c = _client(tmp_path, monkeypatch, hosts="*")
    c.post("/api/tasks", json={"name": "t", "embedder": "fake:color"})
    eid = c.post("/api/tasks/t/examples", data={"label": "red"},
                 files=[("files", ("a.png", png(RED), "image/png"))]).json()["added"][0]["id"]
    for url in (f"/api/tasks/t/examples/{eid}/image", f"/api/tasks/t/examples/{eid}/image?size=32"):
        r = c.get(url)
        assert r.headers["x-content-type-options"] == "nosniff"
        assert "sandbox" in r.headers["content-security-policy"]
        assert r.headers["content-type"].startswith("image/")

    # A file named .html (as a crafted import used to allow) is not served as HTML.
    page = b"<script>alert(1)</script>"
    hid = image_id(page)
    t = fewshot.api.ws.get("t")
    (t.path / f"images/{hid}.html").write_bytes(page)
    with t._writing():  # pylint: disable=protected-access
        t.config.examples.append(models.Example(id=hid, label="red", file=f"images/{hid}.html"))
        t._save()  # pylint: disable=protected-access
    r = c.get(f"/api/tasks/t/examples/{hid}/image")
    assert r.headers["content-type"] == "application/octet-stream"


# ---- imports ------------------------------------------------------------
def _zip(examples, files, **cfg):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("task.json", json.dumps({"name": "t", "classes": [{"name": "red"}], "examples": examples, **cfg}))
        for name, data in files.items():
            z.writestr(name, data)
    buf.seek(0)
    return buf


def _one(data, ext):
    eid = image_id(data)
    return _zip([{"id": eid, "label": "red", "file": f"images/{eid}.{ext}"}], {f"images/{eid}.{ext}": data})


def test_import_refuses_files_that_are_not_images(ws):
    """Every imported file must decode as an image with the extension of its real format."""
    with pytest.raises(ValueError, match="Not a readable image"):
        import_task(ws, _one(b"<script>alert(1)</script>", "html"))
    with pytest.raises(ValueError, match=r"is really a \.png image"):
        import_task(ws, _one(png(RED), "jpg"))
    assert import_task(ws, _one(png(RED), "png")).name == "t"


def test_import_stores_only_the_bytes_it_checked(ws, monkeypatch):
    """An image whose bytes differ when read again to be stored is refused, and nothing is left."""
    real = archive._read_member  # pylint: disable=protected-access
    reads = {"n": 0}

    def changing(z, info, limit=None):
        data = real(z, info, limit)
        if info.filename.startswith("images/"):
            reads["n"] += 1
            if reads["n"] > 1:  # the read that stores it, after the one that checked it
                return png(BLUE)
        return data

    monkeypatch.setattr(archive, "_read_member", changing)
    with pytest.raises(ValueError, match="changed while importing"):
        import_task(ws, _one(png(RED), "png"))
    assert not list((ws.root / "tasks").iterdir())


def test_import_total_size_limit(ws, monkeypatch):
    """The images together may not unpack beyond MAX_IMPORT_TOTAL."""
    monkeypatch.setattr(archive, "MAX_IMPORT_TOTAL", 10)
    with pytest.raises(ValueError, match="unpack to more than"):
        import_task(ws, _one(png(RED), "png"))
    assert not list((ws.root / "tasks").iterdir())


def test_import_checks_the_embedder_first(ws):
    """embedder_ok sees the spec before anything is written, and can refuse it."""
    seen = []

    def refuse(spec):
        seen.append(spec)
        raise PermissionError("not allowed")

    with pytest.raises(PermissionError):
        import_task(ws, _one(png(RED), "png"), embedder_ok=refuse)
    assert seen == [DEFAULT_SPEC]
    assert not list((ws.root / "tasks").iterdir())


def test_api_import_applies_the_model_allowlist(tmp_path, monkeypatch):
    """Importing over the API can't bring in a model the server wouldn't otherwise load."""
    c = _client(tmp_path, monkeypatch, hosts="*")
    loads: list[str] = []
    monkeypatch.setattr(fewshot.api, "check_embedder", loads.append)
    data = png(RED)
    eid = image_id(data)
    zip_ = _zip([{"id": eid, "label": "red", "file": f"images/{eid}.png"}], {f"images/{eid}.png": data},
                embedder="timm:hf-hub:attacker/model")
    r = c.post("/api/tasks/import", files={"file": ("t.zip", zip_.getvalue(), "application/zip")})
    assert r.status_code == 403
    assert "--allow-any-model" in r.json()["detail"]
    assert c.get("/api/tasks").json() == []
    assert not loads
