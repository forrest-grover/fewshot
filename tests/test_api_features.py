"""API endpoints for undo, rename, export/import, example pages, tags, URLs, duplicates and top_k."""

import importlib
import io
import zipfile

import pytest
from fastapi.testclient import TestClient

import fewshot.api
import fewshot.embedders
import fewshot.store
from tests.conftest import png

RED, BLUE = (220, 30, 30), (30, 30, 220)


@pytest.fixture(name="client")
def client_fixture(tmp_path, monkeypatch):
    """A TestClient on a freshly loaded API whose workspace lives in tmp_path."""
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path / "ws"))
    monkeypatch.delenv("FEWSHOT_ALLOW_URLS", raising=False)
    importlib.reload(fewshot.api)
    return TestClient(fewshot.api.app)


@pytest.fixture(name="task")
def task_fixture(client):
    """Task 'c' with three red and three blue examples."""
    assert client.post("/api/tasks", json={"name": "c", "embedder": "fake:color"}).status_code == 201
    for label, rgb in (("red", RED), ("blue", BLUE)):
        files = [("files", (f"{label}{i}.png", png(rgb, noise=15, seed=i), "image/png")) for i in range(3)]
        assert client.post("/api/tasks/c/examples", data={"label": label}, files=files).status_code == 201
    return "c"


def test_undo(client, task):
    """Undo restores the version before the last change and reports whether more undo is possible."""
    t = client.get(f"/api/tasks/{task}").json()
    assert t["can_undo"]
    eid = t["examples"][0]["id"]
    assert len(client.delete(f"/api/tasks/{task}/examples/{eid}").json()["examples"]) == 5
    t = client.post(f"/api/tasks/{task}/undo").json()
    assert len(t["examples"]) == 6
    assert client.get(f"/api/tasks/{task}/examples/{eid}/image").status_code == 200
    client.post("/api/tasks", json={"name": "fresh", "embedder": "fake:color"})
    r = client.post("/api/tasks/fresh/undo")
    assert (r.status_code, r.json()["detail"]) == (400, "Nothing to undo")
    assert client.post("/api/tasks/nope/undo").status_code == 404


def test_undo_of_a_task_deleted_meanwhile(client, task, monkeypatch):
    """A task that disappears between lookup and undo is a 404."""

    def gone(_self):
        raise KeyError("No task 'c'")

    monkeypatch.setattr(fewshot.store.Task, "undo", gone)
    assert client.post(f"/api/tasks/{task}/undo").status_code == 404


def test_rename(client, task):
    """Renaming returns the task under its new name; bad or taken names are 400."""
    t = client.post(f"/api/tasks/{task}/rename", json={"name": "d"}).json()
    assert t["name"] == "d"
    assert client.get("/api/tasks/c").status_code == 404
    assert len(client.get("/api/tasks/d").json()["examples"]) == 6
    client.post("/api/tasks", json={"name": "e", "embedder": "fake:color"})
    assert client.post("/api/tasks/d/rename", json={"name": "e"}).status_code == 400
    assert client.post("/api/tasks/d/rename", json={"name": "a b"}).status_code == 400
    assert client.post("/api/tasks/nope/rename", json={"name": "x"}).status_code == 404


def test_rename_race(client, task, monkeypatch):
    """A task deleted between the lookup and the rename is a 404."""

    def gone(*_args):
        raise KeyError("No task 'c'")

    monkeypatch.setattr(fewshot.api.ws, "rename", gone)
    assert client.post(f"/api/tasks/{task}/rename", json={"name": "d"}).status_code == 404


def test_export_and_import(client, task):
    """A task downloads as a zip and uploads back as a new task."""
    r = client.get(f"/api/tasks/{task}/export")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == 'attachment; filename="c.zip"'
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        assert "task.json" in z.namelist()
    up = client.post("/api/tasks/import", files={"file": ("c.zip", r.content, "application/zip")},
                     data={"name": "copy"})
    assert up.status_code == 201
    assert up.json()["name"] == "copy"
    assert len(up.json()["examples"]) == 6
    dup = client.post("/api/tasks/import", files={"file": ("c.zip", r.content, "application/zip")})
    assert (dup.status_code, dup.json()["detail"]) == (400, "Task 'c' already exists")
    assert client.post("/api/tasks/import", files={"file": ("x.zip", b"nope", "application/zip")}).status_code == 400
    assert client.get("/api/tasks/nope/export").status_code == 404


def test_export_race(client, task, monkeypatch):
    """A task deleted after the lookup is a 404 (and the buffer is closed)."""

    def gone(*_args):
        raise KeyError("No task 'c'")

    monkeypatch.setattr(fewshot.api, "write_export", gone)
    assert client.get(f"/api/tasks/{task}/export").status_code == 404


def test_example_pages(client, task):
    """Examples are paged and filtered by label and tag."""
    page = client.get(f"/api/tasks/{task}/examples", params={"limit": 4}).json()
    assert (page["total"], page["offset"], len(page["examples"])) == (6, 0, 4)
    page = client.get(f"/api/tasks/{task}/examples", params={"offset": 4}).json()
    assert len(page["examples"]) == 2
    assert client.get(f"/api/tasks/{task}/examples", params={"label": "red"}).json()["total"] == 3
    assert client.get(f"/api/tasks/{task}/examples", params={"limit": 0}).status_code == 422


def test_tags_and_notes(client, task):
    """PATCH on one example sets tags and a note; the bulk PATCH adds and removes tags."""
    ids = [e["id"] for e in client.get(f"/api/tasks/{task}").json()["examples"]]
    t = client.patch(f"/api/tasks/{task}/examples/{ids[0]}", json={"tags": ["hard"], "note": "blurry"}).json()
    assert (t["examples"][0]["tags"], t["examples"][0]["note"]) == (["hard"], "blurry")
    t = client.patch(f"/api/tasks/{task}/examples", json={"ids": ids[:2], "add_tags": ["x"],
                                                          "remove_tags": ["hard"]}).json()
    assert [e["tags"] for e in t["examples"][:3]] == [["x"], ["x"], []]
    assert client.get(f"/api/tasks/{task}/examples", params={"tag": "x"}).json()["total"] == 2
    t = client.patch(f"/api/tasks/{task}/examples/{ids[0]}", json={"label": "blue", "note": ""}).json()
    assert (t["examples"][0]["label"], t["examples"][0]["note"]) == ("blue", "")
    assert client.patch(f"/api/tasks/{task}/examples/{ids[0]}", json={"tags": ["a,b"]}).status_code == 400
    assert client.patch(f"/api/tasks/{task}/examples", json={"ids": ["nope"], "add_tags": ["x"]}).status_code == 404
    assert client.patch(f"/api/tasks/{task}/examples/{ids[0]}", json={"note": "x" * 3000}).status_code == 422


def test_settings_for_zero_shot_threshold_and_flip(client, task):
    """The zero-shot threshold can be set and cleared, and flip switched, through PATCH."""
    t = client.patch(f"/api/tasks/{task}", json={"reject_below_zero_shot": 0.25, "flip": True}).json()
    assert (t["reject_below_zero_shot"], t["flip"]) == (0.25, True)
    t = client.patch(f"/api/tasks/{task}", json={"clear_reject_zero_shot": True}).json()
    assert t["reject_below_zero_shot"] is None
    r = client.patch(f"/api/tasks/{task}", json={"clear_reject_zero_shot": True, "reject_below_zero_shot": 0.2})
    assert r.status_code == 400
    assert "either reject_below_zero_shot or clear_reject_zero_shot" in r.json()["detail"]


def test_predict_top_k_and_labels(client, task):
    """top_k trims the scores; every prediction lists its labels."""
    files = [("files", ("q.png", png(RED, seed=9), "image/png"))]
    p = client.post(f"/api/tasks/{task}/predict", params={"top_k": 1}, files=files).json()[0]
    assert list(p["scores"]) == ["red"]
    assert p["labels"] == ["red"]


def test_duplicates(client, task):
    """Near-duplicate pairs are listed; a bad threshold is 422."""
    pairs = client.get(f"/api/tasks/{task}/duplicates", params={"threshold": 0.99}).json()
    assert pairs
    assert all(p["same_label"] for p in pairs)
    assert client.get(f"/api/tasks/{task}/duplicates", params={"threshold": 2}).status_code == 422
    assert client.get("/api/tasks/nope/duplicates").status_code == 404


def test_duplicates_with_bad_embeddings(client, task, monkeypatch):
    """An embedding failure is a 400."""

    def broken(*_args):
        raise ValueError("Embedder 'fake:color' returned NaN or infinite values")

    monkeypatch.setattr(fewshot.api.pipe, "near_duplicates", broken)
    assert client.get(f"/api/tasks/{task}/duplicates").status_code == 400


def test_urls_need_the_server_option(client, task):
    """Without FEWSHOT_ALLOW_URLS the URL endpoints are 403; health says so."""
    assert client.get("/api/health").json()["url_fetch"] is False
    r = client.post(f"/api/tasks/{task}/predict/urls", json={"urls": ["http://x/a.png"]})
    assert r.status_code == 403
    assert "--allow-urls" in r.json()["detail"]


def test_urls(client, task, monkeypatch):
    """With URLs allowed, images are downloaded (public only) to predict or add; failures are per URL."""
    monkeypatch.setattr(fewshot.api, "ALLOW_URLS", True)
    fetched = []

    def fake_fetch(url, max_bytes, public_only=False):
        fetched.append((url, max_bytes, public_only))
        if "bad" in url:
            raise ValueError(f"Could not fetch {url}: HTTP 404 Not Found")
        return png(BLUE, seed=len(fetched))

    monkeypatch.setattr(fewshot.api, "fetch_url", fake_fetch)
    preds = client.post(f"/api/tasks/{task}/predict/urls",
                        json={"urls": ["http://h/b.png", "http://h/bad.png"]}).json()
    assert [(p["file"], p["label"]) for p in preds] == [("http://h/b.png", "blue"), ("http://h/bad.png", None)]
    assert preds[1]["error"].endswith("HTTP 404 Not Found")
    assert fetched[0][2] is True
    r = client.post(f"/api/tasks/{task}/examples/urls", json={"label": "blue", "urls": ["http://h/new.png",
                                                                                          "http://h/bad.png"]})
    assert r.status_code == 201
    assert [a["source"] for a in r.json()["added"]] == ["http://h/new.png"]
    assert r.json()["errors"][0]["file"] == "http://h/bad.png"
    r = client.post(f"/api/tasks/{task}/examples/urls", json={"label": "blue", "urls": ["http://h/bad.png"]})
    assert r.status_code == 400
    assert client.post(f"/api/tasks/{task}/predict/urls", json={"urls": []}).status_code == 422


def test_openapi_has_typed_responses(client):
    """The API schema names the response models."""
    spec = client.get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    assert {"TaskDetail", "PredictionOut", "EvalReport", "JobOut", "Health"} <= set(schemas)
    ok = spec["paths"]["/api/tasks/{name}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert ok == {"$ref": "#/components/schemas/TaskDetail"}


# ---- second review: API correctness -------------------------------------
def test_model_that_cannot_load_is_a_400(client, task, monkeypatch):
    """If a task's model can't be loaded (offline, removed), predict, duplicates and evaluate say
    so with 400, and an evaluation job fails with the reason, instead of a 500."""
    def offline(spec):
        raise RuntimeError("couldn't reach huggingface.co")

    t = fewshot.api.ws.get(task)
    t.update_settings(embedder="timm:gone")
    monkeypatch.setattr(fewshot.embedders, "_build", offline)
    files = [("files", ("q.png", png(RED), "image/png"))]
    for r in (client.post(f"/api/tasks/{task}/predict", files=files), client.get(f"/api/tasks/{task}/duplicates"),
              client.post(f"/api/tasks/{task}/evaluate")):
        assert r.status_code == 400
        assert r.json()["detail"] == "Could not load embedder 'timm:gone': couldn't reach huggingface.co"


def test_streamed_body_over_the_limit_is_413(tmp_path, monkeypatch):
    """A body without Content-Length that grows past the limit is refused with 413, not 400."""
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path / "ws"))
    monkeypatch.setenv("FEWSHOT_MAX_REQUEST_MB", "0.01")
    c = TestClient(importlib.reload(fewshot.api).app)

    def chunks():
        for _ in range(20):
            yield b" " * 1024

    r = c.post("/api/tasks", content=chunks(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    assert r.json()["detail"] == "Request is larger than the 0.01 MB limit"


def test_patch_checks_everything_before_changing_anything(client, task):
    """A bad tag fails the whole edit (the label isn't changed either), and an unknown id is 404
    even when the edit is empty."""
    t = client.get(f"/api/tasks/{task}").json()
    eid, rev = t["examples"][0]["id"], t["revision"]
    r = client.patch(f"/api/tasks/{task}/examples/{eid}", json={"label": "green", "tags": ["x" * 60]})
    assert r.status_code == 400
    r = client.patch(f"/api/tasks/{task}/examples", json={"ids": [eid], "label": "purple", "add_tags": [""]})
    assert r.status_code == 400
    t = client.get(f"/api/tasks/{task}").json()
    assert (t["examples"][0]["label"], t["revision"]) == ("red", rev)
    assert client.patch(f"/api/tasks/{task}/examples/nope", json={}).status_code == 404
    assert client.patch(f"/api/tasks/{task}/examples", json={"ids": ["nope"]}).status_code == 404


def test_urls_check_the_label_before_downloading(client, task, monkeypatch):
    """A bad label is refused before any URL is fetched."""
    monkeypatch.setattr(fewshot.api, "ALLOW_URLS", True)
    fetched = []
    monkeypatch.setattr(fewshot.api, "fetch_url", lambda url, *a, **k: fetched.append(url))
    for label in ("  ", "a/b"):
        r = client.post(f"/api/tasks/{task}/examples/urls", json={"label": label, "urls": ["http://h/x.png"] * 5})
        assert r.status_code == 400
    assert not fetched
