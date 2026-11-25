import importlib

import pytest
from fastapi.testclient import TestClient

from conftest import png


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path / "ws"))
    import fewshot.api
    importlib.reload(fewshot.api)
    return TestClient(fewshot.api.app)


def test_end_to_end(client):
    assert client.post("/api/tasks", json={"name": "c", "embedder": "fake:color"}).status_code == 201
    assert client.post("/api/tasks", json={"name": "c"}).status_code == 400
    for label, rgb in [("red", (220, 30, 30)), ("blue", (30, 30, 220))]:
        files = [("files", (f"{i}.png", png(rgb, noise=15, seed=i), "image/png")) for i in range(3)]
        r = client.post("/api/tasks/c/examples", data={"label": label}, files=files)
        assert r.status_code == 201 and len(r.json()["added"]) == 3
    bad = client.post("/api/tasks/c/examples", data={"label": "red"}, files=[("files", ("x.png", b"junk", "image/png"))])
    assert bad.json()["errors"]

    r = client.post("/api/tasks/c/predict", files=[("files", ("q.png", png((210, 40, 40), seed=50), "image/png"))])
    assert r.status_code == 200 and r.json()[0]["label"] == "red"

    ev = client.post("/api/tasks/c/evaluate").json()
    assert ev["accuracy"] == 1.0

    t = client.get("/api/tasks/c").json()
    eid = t["examples"][0]["id"]
    assert client.get(f"/api/tasks/c/examples/{eid}/image").headers["content-type"] == "image/png"
    t = client.patch(f"/api/tasks/c/examples/{eid}", json={"label": "blue"}).json()
    assert {c["name"]: c["count"] for c in t["classes"]} == {"red": 2, "blue": 4}
    t = client.patch("/api/tasks/c", json={"classifier": "knn", "reject_below": 0.5}).json()
    assert t["classifier"] == "knn" and t["reject_below"] == 0.5
    assert client.patch("/api/tasks/c", json={"classifier": "nope"}).status_code == 400
    assert client.get("/api/tasks/missing").status_code == 404
    assert client.delete("/api/tasks/c").status_code == 204
    assert client.get("/api/tasks").json() == []
