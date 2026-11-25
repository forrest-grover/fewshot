"""Tests for the REST API (fewshot.api) using the fake colour embedder."""

import importlib

import pytest
from fastapi.testclient import TestClient

import fewshot.api
from tests.conftest import png

RED, BLUE = (220, 30, 30), (30, 30, 220)


@pytest.fixture(name="client")
def client_fixture(tmp_path, monkeypatch):
    """A TestClient on a freshly loaded API whose workspace lives in tmp_path."""
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path / "ws"))
    importlib.reload(fewshot.api)
    return TestClient(fewshot.api.app)


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
    assert r.json()["embedder"] == fewshot.api.ws.get("plain").config.embedder
    assert r.json()["classes"] == []
    assert r.json()["examples"] == []
    r = client.post("/api/tasks", json={"name": "bad/name", "embedder": "fake:color"})
    assert r.status_code == 400
    assert "Task names" in r.json()["detail"]
    r = client.post("/api/tasks", json={"name": "x", "embedder": "fake:color", "classifier": "nope"})
    assert r.status_code == 400
    assert "nope" in r.json()["detail"]
    assert [t["name"] for t in client.get("/api/tasks").json()] == ["plain"]


def test_create_task_bad_embedder(client):
    """An unloadable embedder spec is rejected before the task is created."""
    r = client.post("/api/tasks", json={"name": "x", "embedder": "bogus:x"})
    assert r.status_code == 400
    assert r.json()["detail"].startswith("Could not load embedder 'bogus:x': Unknown embedder backend")
    r = client.post("/api/tasks", json={"name": "x", "embedder": "nocolon"})
    assert r.status_code == 400
    assert "Bad embedder spec" in r.json()["detail"]
    assert client.get("/api/tasks").json() == []


def test_check_embedder_trims_model_list(client, monkeypatch):
    """open_clip's long 'Available:' list is cut from the error message."""
    def boom(spec):
        raise RuntimeError(f"Model {spec} not found. Available: " + ", ".join(["m"] * 500))

    monkeypatch.setattr(fewshot.api, "get_embedder", boom)
    r = client.post("/api/tasks", json={"name": "x", "embedder": "openclip:NotAModel"})
    assert r.status_code == 400
    assert r.json()["detail"] == "Could not load embedder 'openclip:NotAModel': Model openclip:NotAModel not found"


def test_update_task_settings(client, task, monkeypatch, request):
    """PATCH settings: classifier params reset, clear_reject, embedder validation."""
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

    monkeypatch.setattr(fewshot.api, "get_embedder", recording_get_embedder)
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
    assert {"name": "green", "description": "a green thing", "count": 0} in t["classes"]
    r = client.post(f"/api/tasks/{task}/classes", json={"name": "  "})
    assert r.status_code == 400
    assert r.json()["detail"] == "Class name is empty"
    assert client.post("/api/tasks/missing/classes", json={"name": "a"}).status_code == 404


def test_patch_class(client, task):
    """Rename and describe classes; failures are atomic and unknown classes are 404."""
    url = f"/api/tasks/{task}/classes"
    t = client.patch(f"{url}/red", json={"description": "reddish"}).json()
    assert {"name": "red", "description": "reddish", "count": 3} in t["classes"]

    t = client.patch(f"{url}/red", json={"name": "crimson", "description": "deep red"}).json()
    classes = {c["name"]: c for c in t["classes"]}
    assert "red" not in classes
    assert classes["crimson"] == {"name": "crimson", "description": "deep red", "count": 3}
    assert sum(e["label"] == "crimson" for e in t["examples"]) == 3

    # same name (after trimming) or empty-string name is a no-op rename
    before = client.get(f"/api/tasks/{task}").json()["classes"]
    assert client.patch(f"{url}/crimson", json={"name": " crimson "}).json()["classes"] == before
    assert client.patch(f"{url}/crimson", json={"name": ""}).json()["classes"] == before

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


def test_delete_class(client, task, tmp_path):
    """Deleting a class removes its examples and image files."""
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
    """Predict is 404 for unknown tasks and 400 for one class; unreadable images get a per-file error."""
    assert client.post("/api/tasks/missing/predict", files=_query(RED)).status_code == 404
    client.post("/api/tasks", json={"name": "one", "embedder": "fake:color"})
    _upload(client, "one", "red", RED)
    r = client.post("/api/tasks/one/predict", files=_query(RED))
    assert r.status_code == 400
    assert r.json()["detail"] == "Need at least 2 classes with examples or descriptions"
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
    ({"classifier_params": {"bogus": 1}}, "Unknown parameter 'bogus' for prototype"),
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
