"""Server operations: health, token auth, request size limit, model allowlist, warm-up and jobs."""

import asyncio
import importlib
import threading
import time

import pytest
from fastapi.testclient import TestClient

import fewshot.api
from fewshot.embedders import DEFAULT_SPEC, PRESETS
from tests.conftest import png

RED, BLUE = (220, 30, 30), (30, 30, 220)


def _reload(tmp_path, monkeypatch, **env):
    """The API module reloaded with FEWSHOT_HOME in tmp_path and the given environment."""
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path / "ws"))
    for k in ("FEWSHOT_TOKEN", "FEWSHOT_ALLOW_ANY_MODEL", "FEWSHOT_WARM", "FEWSHOT_MAX_REQUEST_MB"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(fewshot.api)


@pytest.fixture(name="client")
def client_fixture(tmp_path, monkeypatch):
    """A TestClient on a freshly loaded API with default settings."""
    return TestClient(_reload(tmp_path, monkeypatch).app)


def _task(client, name="c", n=3):
    """Create a task with ``n`` red and ``n`` blue examples."""
    assert client.post("/api/tasks", json={"name": name, "embedder": "fake:color"}).status_code == 201
    for label, rgb in (("red", RED), ("blue", BLUE)):
        files = [("files", (f"{label}{i}.png", png(rgb, noise=15, seed=i), "image/png")) for i in range(n)]
        assert client.post(f"/api/tasks/{name}/examples", data={"label": label}, files=files).status_code == 201
    return name


def _wait(client, jid, timeout=10.0):
    deadline = time.time() + timeout
    while True:
        job = client.get(f"/api/jobs/{jid}").json()
        if job["status"] != "running" or time.time() > deadline:
            return job
        time.sleep(0.02)


# ---- health -------------------------------------------------------------
def test_health(client):
    """/api/health and /healthz report status, loaded models and running jobs."""
    for path in ("/api/health", "/healthz"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["auth_required"] is False
        assert r.json()["jobs_running"] == 0
    assert "fake:color" in client.get("/api/health").json()["models_loaded"]


# ---- auth ---------------------------------------------------------------
def test_token_auth(tmp_path, monkeypatch):
    """With a token set, /api needs it as a Bearer header or the login cookie."""
    client = TestClient(_reload(tmp_path, monkeypatch, FEWSHOT_TOKEN="s3cret").app)
    assert client.get("/api/tasks").status_code == 401
    assert client.get("/api/tasks").headers["www-authenticate"] == "Bearer"
    assert client.get("/api/tasks", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/tasks", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["auth_required"] is True
    assert client.get("/docs").status_code == 200  # only /api is protected

    assert client.post("/api/login", json={"token": "nope"}).status_code == 401
    r = client.post("/api/login", json={"token": "s3cret"})
    assert r.status_code == 204
    assert "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/api/tasks").status_code == 200  # the cookie now signs requests in
    assert client.post("/api/logout").status_code == 204
    assert client.get("/api/tasks").status_code == 401


def test_login_without_token_configured(client):
    """With no token configured, login succeeds and sets no cookie."""
    r = client.post("/api/login", json={"token": "anything"})
    assert r.status_code == 204
    assert "set-cookie" not in r.headers


# ---- request size -------------------------------------------------------
def test_request_size_limit_by_content_length(tmp_path, monkeypatch):
    """A body over FEWSHOT_MAX_REQUEST_MB is refused with 413 before it is read."""
    client = TestClient(_reload(tmp_path, monkeypatch, FEWSHOT_MAX_REQUEST_MB="0.5").app)
    _task(client, n=1)  # small requests still work
    big = [("files", ("x.png", b"x" * 600_000, "image/png"))]
    r = client.post("/api/tasks/c/examples", data={"label": "red"}, files=big)
    assert r.status_code == 413
    assert r.json()["detail"] == "Request is larger than the 0.5 MB limit"


def _asgi(chunks, inner):
    """Run ``inner`` behind a 10-byte LimitBody on a POST without Content-Length whose body
    arrives in ``chunks``; returns the messages sent."""
    sent = []
    queue = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1} for i, c in enumerate(chunks)]

    async def receive():
        return queue.pop(0)

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""}
    asyncio.run(fewshot.api.LimitBody(inner, 10)(scope, receive, send))
    return sent


def test_request_size_limit_while_streaming():
    """Without Content-Length the body is counted as it arrives; over the limit, 413."""

    async def reads_all(_scope, receive, send):
        while (await receive()).get("more_body"):
            pass
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    ok = _asgi([b"12345", b"6789"], reads_all)
    assert ok[0]["status"] == 200
    refused = _asgi([b"123456", b"789012"], reads_all)
    assert refused[0]["status"] == 413

    async def responds_then_reads(_scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        while (await receive()).get("more_body"):
            pass

    late = _asgi([b"123456", b"789012"], responds_then_reads)
    assert [m["type"] for m in late] == ["http.response.start"]  # too late for a 413: just stop


def test_request_size_limit_passes_other_scopes():
    """Non-HTTP scopes (lifespan, websockets) go straight through."""
    seen = []

    async def inner(scope, _receive, _send):
        seen.append(scope["type"])

    async def unused(*_args):
        raise AssertionError("not called")

    asyncio.run(fewshot.api.LimitBody(inner, 10)({"type": "lifespan"}, unused, unused))
    assert seen == ["lifespan"]


# ---- model allowlist ----------------------------------------------------
def test_only_known_models_by_default(client, monkeypatch):
    """Non-preset specs are refused (403) without loading anything, unless a task already uses
    them or the server allows any model."""
    loads: list[str] = []
    monkeypatch.setattr(fewshot.api, "check_embedder", loads.append)
    r = client.post("/api/tasks", json={"name": "x", "embedder": "timm:some_model"})
    assert r.status_code == 403
    assert "--allow-any-model" in r.json()["detail"]
    assert not loads
    preset = PRESETS[2].spec
    assert client.post("/api/tasks", json={"name": "p", "embedder": preset}).status_code == 201
    assert client.post("/api/tasks", json={"name": "f", "embedder": "fake:color"}).status_code == 201  # registered
    fewshot.api.ws.create("cli", embedder="timm:some_model")  # e.g. made with the CLI
    assert client.patch("/api/tasks/f", json={"embedder": "timm:some_model"}).status_code == 200
    assert loads == [preset, "fake:color", "timm:some_model"]  # allowed specs are still loaded to check them
    monkeypatch.setattr(fewshot.api, "ALLOW_ANY_MODEL", True)
    assert client.post("/api/tasks", json={"name": "y", "embedder": "timm:other"}).status_code == 201


# ---- warm-up ------------------------------------------------------------
def test_warm_models(client, monkeypatch):  # pylint: disable=unused-argument  # loads a fresh API
    """Warm-up loads the most used embedders (default spec with no tasks), skipping failures."""
    loaded = []

    def fake_get(spec):
        if spec == "bad:x":
            raise RuntimeError("no such model")
        loaded.append(spec)

    monkeypatch.setattr(fewshot.api, "get_embedder", fake_get)
    assert fewshot.api.warm_models() == [DEFAULT_SPEC]
    ws = fewshot.api.ws
    ws.create("a", embedder="fake:color")
    ws.create("b", embedder="fake:color")
    ws.create("c", embedder="bad:x")
    ws.create("d", embedder="other:y")
    monkeypatch.setenv("FEWSHOT_MAX_MODELS", "2")
    assert fewshot.api.warm_models() == ["fake:color"]  # bad:x is second most used, and fails
    assert loaded == [DEFAULT_SPEC, "fake:color"]


def test_warm_at_startup(tmp_path, monkeypatch):
    """With FEWSHOT_WARM=1, starting the app warms models in a background thread."""
    api = _reload(tmp_path, monkeypatch, FEWSHOT_WARM="1")
    done = threading.Event()
    monkeypatch.setattr(api, "warm_models", done.set)
    with TestClient(api.app):
        assert done.wait(5)


# ---- jobs ---------------------------------------------------------------
def test_evaluation_job(client):
    """An evaluation job reports progress and ends with the same report as the direct endpoint."""
    task = _task(client)
    r = client.post(f"/api/tasks/{task}/evaluate/jobs")
    assert r.status_code == 202
    job = _wait(client, r.json()["id"])
    assert job["status"] == "done"
    assert (job["task"], job["kind"], job["stage"]) == (task, "evaluate", "evaluating")
    assert job["done"] == job["total"] == 6  # leave-one-out: one fold per example
    assert job["result"] == client.post(f"/api/tasks/{task}/evaluate").json()
    assert client.get("/api/health").json()["jobs_running"] == 0


def test_evaluation_job_errors(client):
    """Bad arguments fail at once; errors during the work are reported on the job."""
    assert client.post("/api/tasks/nope/evaluate/jobs").status_code == 404
    client.post("/api/tasks", json={"name": "empty", "embedder": "fake:color"})
    assert client.post("/api/tasks/empty/evaluate/jobs", params={"folds": 1}).status_code == 400
    job = _wait(client, client.post("/api/tasks/empty/evaluate/jobs").json()["id"])
    assert (job["status"], job["error"]) == ("failed", "Task has no examples to evaluate")
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.delete("/api/jobs/nope").status_code == 404


def test_job_internal_error_and_cancel(client, monkeypatch):
    """An unexpected exception fails the job; cancelling stops it at its next progress step."""
    task = _task(client, n=1)

    def broken(*_args, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(fewshot.api, "evaluate_task", broken)
    job = _wait(client, client.post(f"/api/tasks/{task}/evaluate/jobs").json()["id"])
    assert (job["status"], job["error"]) == ("failed", "Internal error: boom")

    started = threading.Event()

    def slow(_pipe, _task, _folds, progress):
        for i in range(1000):
            progress("evaluating", i, 1000)
            started.set()
            time.sleep(0.01)

    monkeypatch.setattr(fewshot.api, "evaluate_task", slow)
    jid = client.post(f"/api/tasks/{task}/evaluate/jobs").json()["id"]
    assert started.wait(5)
    assert client.get("/api/health").json()["jobs_running"] == 1
    assert client.delete(f"/api/jobs/{jid}").status_code == 200
    job = _wait(client, jid)
    assert job["status"] == "cancelled"
    assert job["done"] < 999  # stopped long before the end


def test_old_jobs_are_forgotten(client, monkeypatch):
    """Only the most recent JOBS_KEPT finished jobs are kept."""
    task = _task(client, n=1)
    monkeypatch.setattr(fewshot.api, "JOBS_KEPT", 2)
    ids = []
    for _ in range(3):
        ids.append(client.post(f"/api/tasks/{task}/evaluate/jobs").json()["id"])
        _wait(client, ids[-1])
    assert client.get(f"/api/jobs/{ids[0]}").status_code == 404
    assert all(client.get(f"/api/jobs/{j}").status_code == 200 for j in ids[1:])


def test_no_warm_by_default(client):  # pylint: disable=unused-argument  # loads a fresh API
    """Without FEWSHOT_WARM, startup loads nothing in the background."""
    with TestClient(fewshot.api.app):
        assert not any(t.name == "fewshot-warm" for t in threading.enumerate())


def test_request_size_limit_cuts_off_the_app():
    """Past the limit the app sees a disconnect (and keeps seeing it); if it then fails, that is
    not an error, while an app error before the limit still is."""

    async def fails_after_disconnect(_scope, receive, send):
        while (await receive())["type"] != "http.disconnect":
            pass
        assert (await receive())["type"] == "http.disconnect"
        await send({"type": "http.response.start", "status": 400, "headers": []})  # dropped
        raise RuntimeError("body cut off")

    sent = _asgi([b"123456", b"789012"], fails_after_disconnect)
    assert [m.get("status") for m in sent if m["type"] == "http.response.start"] == [413]

    async def broken(_scope, _receive, _send):
        raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        _asgi([b"1"], broken)
