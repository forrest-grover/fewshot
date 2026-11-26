"""Tests for the REST API (fewshot.api) using the fake colour embedder."""

import contextlib
import io
from typing import Any

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

import fewshot.api
import fewshot.history
from fewshot import models
from fewshot.images import image_id
from fewshot.store import Task, Workspace
from fewshot.text import MAX_DESCRIPTION, MAX_PROMPTS
from tests.conftest import BLUE, RED, api_client, host_client, png, query_files, red_blue_task, set_settings


def _upload(client, task, label, rgb, seeds=range(3)):
    files = [("files", (f"{label}{i}.png", png(rgb, noise=15, seed=i), "image/png")) for i in seeds]
    return client.post(f"/api/tasks/{task}/examples", data={"label": label}, files=files)


def _query(rgb, seed=50):
    return [("files", ("q.png", png(rgb, seed=seed), "image/png"))]


@pytest.fixture(name="task")
def task_fixture(client):
    """A task 'c' with three red and three blue examples."""
    assert client.post("/api/tasks", json={"name": "c", "embedder": "fake:color"}).status_code == 201
    assert _upload(client, "c", "red", RED).status_code == 201
    assert _upload(client, "c", "blue", BLUE, seeds=range(10, 13)).status_code == 201
    return "c"


def test_end_to_end(client):
    """Create, upload, predict, evaluate, relabel, configure and delete a task."""
    assert client.post("/api/tasks", json={"name": "c", "embedder": "fake:color"}).status_code == 201
    assert client.post("/api/tasks", json={"name": "c"}).status_code == 400
    for label, rgb in [("red", RED), ("blue", BLUE)]:
        r = _upload(client, "c", label, rgb)
        assert r.status_code == 201
        assert len(r.json()["added"]) == 3
    junk = [("files", ("x.png", b"junk", "image/png"))]
    bad = client.post("/api/tasks/c/examples", data={"label": "red"}, files=junk)
    assert bad.status_code == 400
    assert bad.json()["detail"].startswith("No images added: x.png: Not a readable image")

    r = client.post("/api/tasks/c/predict", files=_query((210, 40, 40)))
    assert r.status_code == 200
    assert r.json()[0]["label"] == "red"
    assert r.json()[0]["file"] == "q.png"

    ev = client.post("/api/tasks/c/evaluate").json()
    assert ev["accuracy"] == 1.0

    t = client.get("/api/tasks/c").json()
    eid = t["examples"][0]["id"]
    img = client.get(f"/api/tasks/c/examples/{eid}/image")
    assert img.headers["content-type"] == "image/png"
    assert "immutable" in img.headers["cache-control"]
    t = client.patch(f"/api/tasks/c/examples/{eid}", json={"label": "blue"}).json()
    assert {c["name"]: c["count"] for c in t["classes"]} == {"red": 2, "blue": 4}
    t = client.patch("/api/tasks/c", json={"classifier": "knn", "reject_below": 0.5}).json()
    assert t["classifier"] == "knn"
    assert t["reject_below"] == 0.5
    assert client.patch("/api/tasks/c", json={"classifier": "nope"}).status_code == 400
    assert client.get("/api/tasks/missing").status_code == 404
    assert client.delete("/api/tasks/c").status_code == 204
    assert client.get("/api/tasks").json() == []


def test_models(client):
    """The metadata endpoint lists embedder presets and classifiers."""
    body = client.get("/api/models").json()
    assert body["embedders"]
    assert all("spec" in e for e in body["embedders"])
    assert body["classifiers"] == sorted(body["classifiers"])
    assert "prototype" in body["classifiers"]


def test_create_task_default_embedder_and_bad_name(client):
    """Default embedder is used when omitted; invalid names/classifiers are 400."""
    r = client.post("/api/tasks", json={"name": "plain"})
    assert r.status_code == 201
    assert r.json()["embedder"] == client.srv.ws.get("plain").config.embedder
    assert r.json()["classes"] == []
    assert r.json()["examples"] == []
    r = client.post("/api/tasks", json={"name": "bad/name", "embedder": "fake:color"})
    assert r.status_code == 400
    assert "Task names" in r.json()["detail"]
    r = client.post("/api/tasks", json={"name": "x", "embedder": "fake:color", "classifier": "nope"})
    assert r.status_code == 400
    assert "nope" in r.json()["detail"]
    assert [t["name"] for t in client.get("/api/tasks").json()] == ["plain"]


def test_create_task_bad_embedder(client, monkeypatch):
    """An unloadable embedder spec is rejected before the task is created."""
    set_settings(monkeypatch, client, allow_any_model=True)  # these specs aren't presets
    r = client.post("/api/tasks", json={"name": "x", "embedder": "bogus:x"})
    assert r.status_code == 400
    assert r.json()["detail"].startswith("Could not load embedder 'bogus:x': Unknown embedder backend")
    r = client.post("/api/tasks", json={"name": "x", "embedder": "nocolon"})
    assert r.status_code == 400
    assert "Bad embedder spec" in r.json()["detail"]
    assert client.get("/api/tasks").json() == []


def test_check_embedder_trims_model_list(client, monkeypatch):
    """open_clip's long 'Available:' list is cut from the error message."""
    set_settings(monkeypatch, client, allow_any_model=True)  # these specs aren't presets
    def boom(spec):
        raise RuntimeError(f"Model {spec} not found. Available: " + ", ".join(["m"] * 500))

    monkeypatch.setattr(fewshot.embedders, "_build", boom)
    r = client.post("/api/tasks", json={"name": "x", "embedder": "openclip:NotAModel"})
    assert r.status_code == 400
    assert r.json()["detail"] == "Could not load embedder 'openclip:NotAModel': Model openclip:NotAModel not found"


def test_update_task_settings(client, task, monkeypatch, request):
    """PATCH settings: classifier params reset, clear_reject, embedder validation."""
    set_settings(monkeypatch, client, allow_any_model=True)  # these specs aren't presets
    t = client.patch(f"/api/tasks/{task}", json={"classifier": "knn", "classifier_params": {"k": 3}}).json()
    assert t["classifier"] == "knn"
    assert t["classifier_params"] == {"k": 3}
    # same classifier again keeps its params
    t = client.patch(f"/api/tasks/{task}", json={"classifier": "knn"}).json()
    assert t["classifier_params"] == {"k": 3}
    # switching classifier without params resets them
    t = client.patch(f"/api/tasks/{task}", json={"classifier": "prototype"}).json()
    assert t["classifier"] == "prototype"
    assert t["classifier_params"] == {}

    t = client.patch(f"/api/tasks/{task}", json={"reject_below": 0.7, "text_weight": 0.5}).json()
    assert t["reject_below"] == 0.7
    assert t["text_weight"] == 0.5
    t = client.patch(f"/api/tasks/{task}", json={"clear_reject": True}).json()
    assert t["reject_below"] is None
    assert t["text_weight"] == 0.5

    # unchanged embedder is not reloaded; a bad one is rejected; a good new one is accepted
    calls = []
    real = fewshot.embedders.get_embedder

    def recording_get_embedder(spec):
        calls.append(spec)
        return real(spec)

    monkeypatch.setattr(fewshot.embedders, "get_embedder", recording_get_embedder)
    assert client.patch(f"/api/tasks/{task}", json={"embedder": "fake:color"}).status_code == 200
    assert not calls
    r = client.patch(f"/api/tasks/{task}", json={"embedder": "bogus:x"})
    assert r.status_code == 400
    assert "bogus:x" in r.json()["detail"]
    assert client.get(f"/api/tasks/{task}").json()["embedder"] == "fake:color"
    fewshot.embedders.register_embedder("fake:other", real("fake:color"))
    request.addfinalizer(lambda: fewshot.embedders.unregister_embedder("fake:other"))
    t = client.patch(f"/api/tasks/{task}", json={"embedder": "fake:other"}).json()
    assert t["embedder"] == "fake:other"
    assert calls == ["bogus:x", "fake:other"]
    assert t["supports_text"] is False

    assert client.patch("/api/tasks/missing", json={}).status_code == 404


def test_delete_missing_task(client):
    """Deleting an unknown task is 404."""
    assert client.delete("/api/tasks/missing").status_code == 404


def test_add_class(client, task):
    """Adding a class with a description; blank names are 400."""
    t = client.post(f"/api/tasks/{task}/classes", json={"name": "green", "description": "a green thing"}).json()
    assert {"name": "green", "description": "a green thing", "count": 0, "reject_below": None} in t["classes"]
    r = client.post(f"/api/tasks/{task}/classes", json={"name": "  "})
    assert r.status_code == 400
    assert r.json()["detail"] == "Class name is empty"
    assert client.post("/api/tasks/missing/classes", json={"name": "a"}).status_code == 404


def test_patch_class(client, task):
    """Rename and describe classes; failures are atomic and unknown classes are 404."""
    url = f"/api/tasks/{task}/classes"
    t = client.patch(f"{url}/red", json={"description": "reddish"}).json()
    assert {"name": "red", "description": "reddish", "count": 3, "reject_below": None} in t["classes"]

    t = client.patch(f"{url}/red", json={"name": "crimson", "description": "deep red"}).json()
    classes = {c["name"]: c for c in t["classes"]}
    assert "red" not in classes
    assert classes["crimson"] == {"name": "crimson", "description": "deep red", "count": 3, "reject_below": None}
    assert sum(e["label"] == "crimson" for e in t["examples"]) == 3

    # the same name (after trimming) is a no-op rename; an empty one is an error
    before = client.get(f"/api/tasks/{task}").json()["classes"]
    assert client.patch(f"{url}/crimson", json={"name": " crimson "}).json()["classes"] == before
    r = client.patch(f"{url}/crimson", json={"name": ""})
    assert (r.status_code, r.json()["detail"]) == (400, "Class name is empty")

    r = client.patch(f"{url}/crimson", json={"name": "blue", "description": "clobbered?"})
    assert r.status_code == 400
    assert "already exists" in r.json()["detail"]
    r = client.patch(f"{url}/crimson", json={"name": "   "})
    assert r.status_code == 400
    assert r.json()["detail"] == "Class name is empty"
    assert client.get(f"/api/tasks/{task}").json()["classes"] == before  # failed patches change nothing

    r = client.patch(f"{url}/nope", json={"description": "x"})
    assert r.status_code == 404
    assert r.json()["detail"] == "No class 'nope'"
    assert "nope" not in {c["name"] for c in client.get(f"/api/tasks/{task}").json()["classes"]}
    assert client.patch("/api/tasks/missing/classes/red", json={}).status_code == 404


def test_describe_classes(client, task):
    """Template descriptions respect overwrite; templates need '{}'."""
    url = f"/api/tasks/{task}/describe"
    client.patch(f"/api/tasks/{task}/classes/red", json={"description": "keep me"})
    t = client.post(url, json={"template": "a {} square"}).json()
    assert {c["name"]: c["description"] for c in t["classes"]} == {"red": "keep me", "blue": "a blue square"}
    t = client.post(url, json={"overwrite": True}).json()
    descriptions = {c["name"]: c["description"] for c in t["classes"]}
    assert descriptions == {"red": "a photo of a red", "blue": "a photo of a blue"}
    r = client.post(url, json={"template": "no placeholder"})
    assert r.status_code == 400
    assert "{}" in r.json()["detail"]
    assert client.post("/api/tasks/missing/describe", json={}).status_code == 404


def test_delete_class(client, task, tmp_path, monkeypatch):
    """Deleting a class removes its examples and (with no undo history kept) image files."""
    monkeypatch.setattr(fewshot.history, "HISTORY_KEPT", 0)
    t = client.delete(f"/api/tasks/{task}/classes/red").json()
    assert [c["name"] for c in t["classes"]] == ["blue"]
    assert {e["label"] for e in t["examples"]} == {"blue"}
    assert len(list((tmp_path / "ws" / "tasks" / task / "images").iterdir())) == 3
    r = client.delete(f"/api/tasks/{task}/classes/red")
    assert r.status_code == 404
    assert r.json()["detail"] == "No class 'red'"
    assert client.delete("/api/tasks/missing/classes/red").status_code == 404


def test_examples_errors(client, task):
    """Unknown tasks/examples are 404 for upload, relabel and image endpoints."""
    assert _upload(client, "missing", "red", RED).status_code == 404
    # re-uploading an identical image moves it rather than duplicating it
    r = _upload(client, task, "blue", RED, seeds=[0])
    assert r.status_code == 201
    assert r.json()["added"][0]["label"] == "blue"
    assert len(r.json()["task"]["examples"]) == 6

    r = client.patch(f"/api/tasks/{task}/examples/nope", json={"label": "red"})
    assert r.status_code == 404
    assert r.json()["detail"] == "No example 'nope'"
    assert client.patch("/api/tasks/missing/examples/nope", json={"label": "red"}).status_code == 404
    assert client.get(f"/api/tasks/{task}/examples/nope/image").status_code == 404
    assert client.get("/api/tasks/missing/examples/nope/image").status_code == 404


def test_bulk_relabel_and_delete(client, task, tmp_path, monkeypatch):
    """Several examples move or go in one request; an unknown id or bad label changes nothing."""
    ex = client.get(f"/api/tasks/{task}").json()["examples"]
    red = [e["id"] for e in ex if e["label"] == "red"]
    rev = client.get(f"/api/tasks/{task}").json()["revision"]
    t = client.patch(f"/api/tasks/{task}/examples", json={"ids": red[:2] + red[:1], "label": "pink"}).json()
    assert {c["name"]: c["count"] for c in t["classes"]} == {"red": 1, "blue": 3, "pink": 2}
    assert t["revision"] == rev + 1  # one save creates "pink" and moves both examples

    r = client.patch(f"/api/tasks/{task}/examples", json={"ids": [red[2], "nope"], "label": "pink"})
    assert r.status_code == 404
    assert r.json()["detail"] == "No example 'nope'"
    assert client.patch(f"/api/tasks/{task}/examples", json={"ids": [red[2]], "label": " "}).status_code == 400
    assert client.patch(f"/api/tasks/{task}/examples", json={"ids": [], "label": "x"}).status_code == 422
    assert client.patch("/api/tasks/missing/examples", json={"ids": ["a"], "label": "x"}).status_code == 404

    images = tmp_path / "ws" / "tasks" / task / "images"
    monkeypatch.setattr(fewshot.history, "HISTORY_KEPT", 0)  # delete image files at once
    assert client.post(f"/api/tasks/{task}/examples/delete", json={"ids": [red[0], "nope"]}).status_code == 404
    assert len(list(images.iterdir())) == 6
    t = client.post(f"/api/tasks/{task}/examples/delete", json={"ids": red[:2]}).json()
    assert {c["name"]: c["count"] for c in t["classes"]} == {"red": 1, "blue": 3, "pink": 0}
    assert len(list(images.iterdir())) == 4
    assert client.post("/api/tasks/missing/examples/delete", json={"ids": ["a"]}).status_code == 404


def test_image_thumbnails(client, task, tmp_path):
    """``?size=`` serves a JPEG scaled down to fit; bad sizes, missing and corrupt files are errors."""
    big = io.BytesIO()
    Image.new("RGB", (900, 300), RED).save(big, "JPEG")
    r = client.post(f"/api/tasks/{task}/examples", data={"label": "red"},
                    files=[("files", ("big.jpg", big.getvalue(), "image/jpeg"))])
    eid = r.json()["added"][0]["id"]
    url = f"/api/tasks/{task}/examples/{eid}/image"
    thumb = client.get(url, params={"size": 120})
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"
    assert "immutable" in thumb.headers["cache-control"]
    assert Image.open(io.BytesIO(thumb.content)).size == (120, 40)
    small = client.get(f"/api/tasks/{task}/examples/{r.json()['task']['examples'][0]['id']}/image",
                       params={"size": 120})
    assert Image.open(io.BytesIO(small.content)).size == (8, 8)  # never enlarged
    assert client.get(url, params={"size": 8}).status_code == 422
    assert client.get(url, params={"size": 5000}).status_code == 422
    assert client.get(f"/api/tasks/{task}/examples/nope/image", params={"size": 120}).status_code == 404

    path = tmp_path / "ws" / "tasks" / task / r.json()["added"][0]["file"]
    path.write_bytes(b"junk")
    bad = client.get(url, params={"size": 120})
    assert bad.status_code == 400
    assert bad.json()["detail"].startswith("Not a readable image")
    path.unlink()
    assert client.get(url, params={"size": 120}).status_code == 404


def test_relabel_to_new_class(client, task):
    """Relabelling to an unknown label creates that class."""
    eid = client.get(f"/api/tasks/{task}").json()["examples"][0]["id"]
    t = client.patch(f"/api/tasks/{task}/examples/{eid}", json={"label": "pink"}).json()
    assert {c["name"]: c["count"] for c in t["classes"]} == {"red": 2, "blue": 3, "pink": 1}


def test_delete_example(client, task):
    """Deleting an example removes it and its image."""
    eid = client.get(f"/api/tasks/{task}").json()["examples"][0]["id"]
    t = client.delete(f"/api/tasks/{task}/examples/{eid}").json()
    assert eid not in {e["id"] for e in t["examples"]}
    assert len(t["examples"]) == 5
    assert client.get(f"/api/tasks/{task}/examples/{eid}/image").status_code == 404
    r = client.delete(f"/api/tasks/{task}/examples/{eid}")
    assert r.status_code == 404
    assert eid in r.json()["detail"]
    assert client.delete(f"/api/tasks/missing/examples/{eid}").status_code == 404


def test_predict(client, task):
    """Predictions are returned per file, in upload order."""
    files = [*_query(BLUE, seed=1), ("files", ("r.png", png(RED, seed=2), "image/png"))]
    r = client.post(f"/api/tasks/{task}/predict", files=files)
    assert [(p["file"], p["label"]) for p in r.json()] == [("q.png", "blue"), ("r.png", "red")]


def test_predict_errors(client):
    """Predict is 404 for unknown tasks and 400 with no classes; unreadable images get a per-file error."""
    assert client.post("/api/tasks/missing/predict", files=_query(RED)).status_code == 404
    client.post("/api/tasks", json={"name": "one", "embedder": "fake:color"})
    r = client.post("/api/tasks/one/predict", files=_query(RED))
    assert r.status_code == 400
    assert r.json()["detail"] == "Need at least 1 class with examples or a description"
    _upload(client, "one", "red", RED)
    assert client.post("/api/tasks/one/predict", files=_query(BLUE)).json()[0]["label"] == "red"  # one class
    _upload(client, "one", "blue", BLUE, seeds=range(10, 13))
    r = client.post("/api/tasks/one/predict", files=[("files", ("x.png", b"junk", "image/png")), *_query(BLUE)])
    assert r.status_code == 200
    bad, good = r.json()
    assert (bad["file"], bad["label"]) == ("x.png", None)
    assert bad["error"] == "Not a readable image: unrecognised file format"
    assert (good["label"], good["error"]) == ("blue", None)


def test_predict_ignores_empty_class(client, task):
    """A freshly created class with no examples doesn't break predict."""
    client.post(f"/api/tasks/{task}/classes", json={"name": "green"})
    r = client.post(f"/api/tasks/{task}/predict", files=_query(BLUE))
    assert r.status_code == 200
    assert r.json()[0]["label"] == "blue"


@pytest.mark.parametrize(("settings", "message"), [
    ({"classifier_params": {"bogus": 1}}, "Unknown parameter 'bogus' for gda"),
    ({"classifier": "knn", "classifier_params": {"k": "abc"}}, "'k' must be an integer"),
    ({"text_weight": -1}, "'text_weight' must be >= 0"),
    ({"reject_below": 99}, "reject_below must be between -1 and 1"),
])
def test_update_task_rejects_bad_settings(client, task, settings, message):
    """Invalid settings are a 400 and leave the task predicting as before."""
    r = client.patch(f"/api/tasks/{task}", json=settings)
    assert r.status_code == 400
    assert message in r.json()["detail"]
    assert client.post(f"/api/tasks/{task}/predict", files=_query(BLUE)).status_code == 200


def test_bad_input_is_400_not_500(client, task):
    """Blank labels, folds below 2 and odd templates are client errors or handled."""
    eid = client.get(f"/api/tasks/{task}").json()["examples"][0]["id"]
    r = client.patch(f"/api/tasks/{task}/examples/{eid}", json={"label": "  "})
    assert (r.status_code, r.json()["detail"]) == (400, "Class name is empty")
    r = client.post(f"/api/tasks/{task}/examples", data={"label": " "}, files=_query(RED))
    assert (r.status_code, r.json()["detail"]) == (400, "Class name is empty")
    for folds in (1, 0, -1):
        r = client.post(f"/api/tasks/{task}/evaluate", params={"folds": folds})
        assert (r.status_code, r.json()["detail"]) == (400, f"folds must be at least 2, got {folds}")
    r = client.post(f"/api/tasks/{task}/describe", json={"template": "{x} {} {}", "overwrite": True})
    assert r.status_code == 200
    assert r.json()["classes"][0]["description"] == "{x} red red"


def test_evaluate(client, task):
    """Evaluate with k folds; unknown task is 404 and an empty task is 400."""
    ev = client.post(f"/api/tasks/{task}/evaluate", params={"folds": 3})
    assert ev.status_code == 200
    assert ev.json()["accuracy"] == 1.0
    assert client.post("/api/tasks/missing/evaluate").status_code == 404
    client.post("/api/tasks", json={"name": "empty", "embedder": "fake:color"})
    r = client.post("/api/tasks/empty/evaluate")
    assert r.status_code == 400
    assert r.json()["detail"] == "Task has no examples to evaluate"


@pytest.mark.parametrize("label", ["..", ".", "a/b"])
def test_path_like_class_names_are_refused(client, task, label):
    """A class named '..' would make DELETE .../classes/.. resolve to the task itself."""
    r = client.post(f"/api/tasks/{task}/examples", data={"label": label}, files=_query(RED))
    assert r.status_code == 400
    assert "can't be '.' or '..'" in r.json()["detail"]
    r = client.post(f"/api/tasks/{task}/classes", json={"name": label})
    assert r.status_code == 400
    r = client.patch(f"/api/tasks/{task}/classes/red", json={"name": label})
    assert r.status_code == 400
    assert client.get(f"/api/tasks/{task}").status_code == 200


def test_corrupt_task_is_skipped_and_reported(client, task, tmp_path):
    """A broken task.json is left out of the list and is a 404 with the reason."""
    (tmp_path / "ws" / "tasks" / task / "task.json").write_text("{not json")
    assert client.get("/api/tasks").json() == []
    r = client.get(f"/api/tasks/{task}")
    assert r.status_code == 404
    assert "can't be loaded" in r.json()["detail"]


def test_settings_bodies_are_strict(client, task):
    """Unknown fields, contradictory options and an empty embedder are errors, not silent no-ops."""
    r = client.patch(f"/api/tasks/{task}", json={"reject_bellow": 0.5})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "reject_bellow"]
    r = client.patch(f"/api/tasks/{task}", json={"clear_reject": True, "reject_below": 0.5})
    assert (r.status_code, r.json()["detail"]) == (400, "Give either reject_below or clear_reject, not both")
    for body in ({"embedder": ""}, {"embedder": "  "}):
        r = client.patch(f"/api/tasks/{task}", json=body)
        assert (r.status_code, r.json()["detail"]) == (400, "Embedder spec is empty")
    r = client.post("/api/tasks", json={"name": "d", "embedder": ""})
    assert (r.status_code, r.json()["detail"]) == (400, "Embedder spec is empty")
    assert client.post(f"/api/tasks/{task}/classes", json={"name": "x", "colour": "red"}).status_code == 422
    assert client.get(f"/api/tasks/{task}").json()["embedder"] == "fake:color"


def test_uploads_over_the_size_limit_are_refused(client, task, monkeypatch):
    """Files over the limit are reported per file and never fully read."""
    set_settings(monkeypatch, client, max_upload_mb=(100) / 2**20)
    big = png(RED, size=64, noise=50)
    assert len(big) > 100
    files = [("files", ("big.png", big, "image/png")), ("files", ("small.png", png(RED, size=2), "image/png"))]
    r = client.post(f"/api/tasks/{task}/examples", data={"label": "red"}, files=files)
    assert r.status_code == 201
    assert r.json()["errors"] == [{"file": "big.png", "error": "File is larger than the 9.53674e-05 MB upload limit"}]
    r = client.post(f"/api/tasks/{task}/predict", files=files)
    big_pred, small_pred = r.json()
    assert big_pred["error"].startswith("File is larger than")
    assert small_pred["error"] is None


# ---- one request, one change ---------------------------------------------
@pytest.mark.usefixtures("task_t")
def test_patch_requests_are_one_undo_step(client):
    """Renaming and describing a class, or moving and tagging an example, is one undo step."""
    t = client.get("/api/tasks/t").json()
    rev = t["revision"]
    t = client.patch("/api/tasks/t/classes/red", json={"name": "crimson", "description": "deep red"}).json()
    assert t["revision"] == rev + 1
    eid = t["examples"][0]["id"]
    t = client.patch(f"/api/tasks/t/examples/{eid}", json={"label": "blue", "tags": ["x"], "note": "n"}).json()
    assert t["revision"] == rev + 2
    t = client.patch("/api/tasks/t/examples", json={"ids": [eid], "label": "crimson", "add_tags": ["y"]}).json()
    assert t["revision"] == rev + 3
    t = client.post("/api/tasks/t/undo").json()
    assert next(e for e in t["examples"] if e["id"] == eid)["tags"] == ["x"]
    t = client.post("/api/tasks/t/undo").json()
    assert next(e for e in t["examples"] if e["id"] == eid)["label"] == "crimson"
    t = client.post("/api/tasks/t/undo").json()
    assert [c["name"] for c in t["classes"]] == ["red", "blue"]
    assert client.patch("/api/tasks/t/classes/nope", json={"description": "x"}).status_code == 404


@pytest.mark.usefixtures("task_t")
def test_empty_edits_change_nothing(client):
    """An edit with nothing in it is accepted and changes nothing."""
    t = client.get("/api/tasks/t").json()
    eid = t["examples"][0]["id"]
    assert client.patch(f"/api/tasks/t/examples/{eid}", json={}).json()["revision"] == t["revision"]
    assert client.patch("/api/tasks/t/examples", json={"ids": [eid]}).json()["revision"] == t["revision"]


@pytest.mark.usefixtures("task_t")
def test_descriptions_are_capped_over_the_api(client):
    """A description too long is refused by the schema; too many prompts by the store."""
    r = client.post("/api/tasks/t/classes", json={"name": "red", "description": "x" * (MAX_DESCRIPTION + 1)})
    assert r.status_code == 422
    r = client.patch("/api/tasks/t/classes/red", json={"description": "a\n" * (MAX_PROMPTS + 1)})
    assert (r.status_code, "prompts" in r.json()["detail"]) == (400, True)
    r = client.post("/api/tasks/t/classes", json={"name": "red", "description": "a\n" * (MAX_PROMPTS + 1)})
    assert r.status_code == 400


@pytest.mark.usefixtures("task_t")
def test_a_task_deleted_meanwhile_is_404_not_500(client, monkeypatch):
    """Every change to a task that another process deleted after it was looked up gets 404."""
    eid = client.get("/api/tasks/t").json()["examples"][0]["id"]
    assert client.post("/api/tasks/t/inbox", files=query_files(1, BLUE)).status_code == 201

    @contextlib.contextmanager
    def gone(_self):
        raise KeyError("No task 't'")
        yield  # pylint: disable=unreachable

    monkeypatch.setattr(Task, "_writing", gone)
    def delete(_self, name):
        raise KeyError(f"No task {name!r}")

    monkeypatch.setattr(Workspace, "delete", delete)
    calls: list[tuple[str, str, dict[str, Any]]] = [
        ("patch", "/api/tasks/t", {"json": {"flip": True}}),
        ("delete", "/api/tasks/t", {}),
        ("post", "/api/tasks/t/classes", {"json": {"name": "green"}}),
        ("post", "/api/tasks/t/describe", {"json": {"template": "a {}"}}),
        ("post", "/api/tasks/t/examples", {"data": {"label": "red"}, "files": query_files(1)}),
        ("post", "/api/tasks/t/inbox", {"files": query_files(1, (30, 220, 30))}),
        ("post", "/api/tasks/t/inbox/accept-confident", {"json": {"min_confidence": 0}}),
        ("patch", f"/api/tasks/t/examples/{eid}", {"json": {"label": "blue"}}),
    ]
    for method, url, kw in calls:
        r = getattr(client, method)(url, **kw)
        assert (r.status_code, r.json()["detail"]) == (404, "No task 't'"), url


@pytest.mark.usefixtures("task_t")
def test_accept_confident_suggestions(client):
    """The inbox's confident suggestions can be accepted in one request, as `inbox accept --min-confidence`."""
    assert client.post("/api/tasks/t/inbox", files=query_files(2) + query_files(1, BLUE)[:0]).status_code == 201
    t = client.post("/api/tasks/t/inbox/accept-confident", json={"min_confidence": 0.0}).json()
    assert (t["inbox_count"], len(t["examples"])) == (0, 6)
    assert client.post("/api/tasks/t/inbox/accept-confident", json={"min_confidence": 0.5}).status_code == 200
    assert client.post("/api/tasks/t/inbox/accept-confident", json={"min_confidence": 2}).status_code == 422
    assert client.post("/api/tasks/nope/inbox/accept-confident", json={"min_confidence": 0.5}).status_code == 404


def test_quotes_in_uploaded_file_names(tmp_path):
    """A '"' that a browser sent as %22 comes back as a quote."""
    c = api_client(tmp_path)
    red_blue_task(c.srv.ws)
    r = c.post("/api/tasks/t/predict", files=[("files", ('say "cheese".png', png(RED, seed=3), "image/png"))])
    assert r.json()[0]["file"] == 'say "cheese".png'


def test_images_are_served_only_as_images(tmp_path, monkeypatch):
    """Example images carry nosniff and a sandboxing CSP; a file whose name isn't an image type is
    served as an opaque download, never as a page."""
    c = host_client(tmp_path, monkeypatch, hosts="*")
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
    t = c.srv.ws.get("t")
    (t.path / f"images/{hid}.html").write_bytes(page)
    with t._writing():  # pylint: disable=protected-access
        t.config.examples.append(models.Example(id=hid, label="red", file=f"images/{hid}.html"))
        t._save()  # pylint: disable=protected-access
    r = c.get(f"/api/tasks/t/examples/{hid}/image")
    assert r.headers["content-type"] == "application/octet-stream"


def test_an_upload_whose_files_are_closed_stops_cleanly(client, monkeypatch, caplog):
    """If the request is abandoned while its upload is being stored (the client went away), the
    files it was reading are closed: the upload stops with a 400 and a log line, the batch being
    read isn't stored, and batches stored before stay."""
    assert client.post("/api/tasks", json={"name": "t", "embedder": "fake:color"}).status_code == 201
    monkeypatch.setattr(Task.add_examples, "__defaults__", (2, None))  # batches of 2 images
    files = [UploadFile(io.BytesIO(png(RED, noise=10, seed=i)), filename=f"r{i}.png") for i in range(5)]
    files[3].file.close()
    with pytest.raises(HTTPException) as e:
        fewshot.api._add_examples(client.srv, "t", "red", fewshot.api._uploads(client.srv, files))  # pylint: disable=protected-access
    assert (e.value.status_code, e.value.detail) == (
        400, "The upload stopped at file 4 of 5 (r3.png): it could no longer be read; images before it may have "
             "been stored")
    assert len(client.get("/api/tasks/t").json()["examples"]) == 2  # the first batch; not r2.png's
    assert any("Stopped an upload at file 4 of 5 (r3.png)" in r.getMessage() for r in caplog.records)
