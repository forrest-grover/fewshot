"""Server operations: settings and the app factory, health, threads for heavy work, request and
upload limits, the model allowlist and warm-up."""

import asyncio
import importlib
import secrets
import subprocess
import sys
import threading
import time

import anyio.to_thread
import pydantic
import pytest

import fewshot.api
import fewshot.guard
import fewshot.responses
import fewshot.server
import fewshot.server_settings
import fewshot.services
from fewshot.embedders import DEFAULT_SPEC, PRESETS
from fewshot.pipeline import Prediction
from fewshot.server_settings import MAX_BULK_JSON_BYTES, MAX_JSON_BYTES, ServerSettings
from fewshot.store import Task
from tests.conftest import RED, api_client, api_task, png, query_files, set_settings


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


# ---- request size -------------------------------------------------------
def test_request_size_limit_by_content_length(tmp_path):
    """A body over FEWSHOT_MAX_REQUEST_MB is refused with 413 before it is read."""
    client = api_client(tmp_path, max_request_mb=0.5)
    api_task(client, n=1)  # small requests still work
    big = [("files", ("x.png", b"x" * 600_000, "image/png"))]
    r = client.post("/api/tasks/c/examples", data={"label": "red"}, files=big)
    assert r.status_code == 413
    assert r.json()["detail"] == "Request is larger than the 0.5 MB limit"


def test_the_body_limit_comes_from_the_route_not_the_content_type(tmp_path):
    """Only an upload route takes a large multipart body: a body labelled as an upload gets the
    JSON limit anywhere else (as on /api/login, which needs no token), and so does a body to an
    upload route that isn't multipart."""
    client = api_client(tmp_path, token="0123456789abcdef-token")
    multipart = {"Content-Type": "multipart/form-data; boundary=x"}
    body = b"x" * (MAX_JSON_BYTES + 10)
    for path in ("/api/login", "/api/tasks", "/api/tasks/t/examples/urls", "/api/nowhere", "/"):
        r = client.post(path, content=body, headers=multipart)
        assert (r.status_code, r.json()["detail"]) == (413, "Request is larger than the 1 MB limit"), path
    for path in ("/api/tasks/t/examples", "/api/tasks/t/inbox", "/api/tasks/t/predict", "/api/tasks/import"):
        r = client.post(path, content=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
        assert r.status_code == 413, path
        assert client.post(path, content=body, headers=multipart).status_code == 401, path  # past the limit
    assert client.request("GET", "/api/tasks/t/examples", content=body, headers=multipart).status_code == 413
    scope = {"path": "/app/api/tasks/import", "root_path": "/app", "method": "POST",
             "headers": [(b"content-type", b"multipart/form-data")]}
    root = fewshot.guard.body_limit(ServerSettings().max_request_bytes, scope)
    assert root == ServerSettings().max_request_bytes


# Every route in the API's schema, by the body it may send: an upload (multipart: the request
# limit), a JSON list of ids (16 MB) or anything else (1 MB). A new route must be added here.
UPLOAD, BULK, SMALL = "upload", "bulk", "small"
ROUTE_BODIES = {
    ("GET", "/api/health"): SMALL, ("POST", "/api/login"): SMALL, ("POST", "/api/logout"): SMALL,
    ("GET", "/api/models"): SMALL, ("GET", "/api/tasks"): SMALL, ("POST", "/api/tasks"): SMALL,
    ("GET", "/api/tasks/{name}"): SMALL, ("PATCH", "/api/tasks/{name}"): SMALL,
    ("DELETE", "/api/tasks/{name}"): SMALL, ("POST", "/api/tasks/{name}/rename"): SMALL,
    ("POST", "/api/tasks/{name}/duplicate"): SMALL, ("POST", "/api/tasks/{name}/undo"): SMALL,
    ("POST", "/api/tasks/{name}/redo"): SMALL, ("GET", "/api/tasks/{name}/export"): SMALL,
    ("POST", "/api/tasks/import"): UPLOAD, ("POST", "/api/tasks/{name}/classes"): SMALL,
    ("PATCH", "/api/tasks/{name}/classes/{label}"): SMALL, ("DELETE", "/api/tasks/{name}/classes/{label}"): SMALL,
    ("POST", "/api/tasks/{name}/describe"): SMALL, ("PUT", "/api/tasks/{name}/class-thresholds"): SMALL,
    ("GET", "/api/tasks/{name}/examples"): SMALL, ("POST", "/api/tasks/{name}/examples"): UPLOAD,
    ("PATCH", "/api/tasks/{name}/examples"): BULK, ("POST", "/api/tasks/{name}/examples/urls"): SMALL,
    ("PATCH", "/api/tasks/{name}/examples/{eid}"): SMALL, ("DELETE", "/api/tasks/{name}/examples/{eid}"): SMALL,
    ("POST", "/api/tasks/{name}/examples/delete"): BULK, ("GET", "/api/tasks/{name}/examples/{eid}/image"): SMALL,
    ("POST", "/api/tasks/{name}/inbox"): UPLOAD, ("GET", "/api/tasks/{name}/inbox"): SMALL,
    ("POST", "/api/tasks/{name}/inbox/urls"): SMALL, ("POST", "/api/tasks/{name}/inbox/accept"): BULK,
    ("POST", "/api/tasks/{name}/inbox/accept-confident"): BULK, ("POST", "/api/tasks/{name}/inbox/dismiss"): BULK,
    ("POST", "/api/tasks/{name}/predict"): UPLOAD, ("POST", "/api/tasks/{name}/predict/urls"): SMALL,
    ("GET", "/api/tasks/{name}/duplicates"): SMALL, ("POST", "/api/tasks/{name}/evaluate"): SMALL,
    ("POST", "/api/tasks/{name}/evaluate/jobs"): SMALL, ("POST", "/api/tasks/{name}/compare/jobs"): SMALL,
    ("GET", "/api/tasks/{name}/jobs"): SMALL, ("GET", "/api/tasks/{name}/evaluation"): SMALL,
    ("GET", "/api/jobs/{jid}"): SMALL, ("DELETE", "/api/jobs/{jid}"): SMALL,
}


def test_every_route_has_the_body_limit_of_its_kind(client):
    """Each route in the OpenAPI schema is classified above, and gets its kind's limit for a
    multipart and for a JSON body (path parameters filled in with ordinary values)."""
    schema = client.app.openapi()
    routes = {(m.upper(), p) for p, ops in schema["paths"].items() for m in ops}
    assert routes == set(ROUTE_BODIES)
    big = ServerSettings().max_request_bytes
    for (method, template), kind in ROUTE_BODIES.items():
        path = template.format(name="t", label="red", eid="e1", jid="j1")
        for ctype in (b"multipart/form-data; boundary=x", b"application/json"):
            scope = {"path": path, "method": method, "headers": [(b"content-type", ctype)]}
            want = {UPLOAD: big if ctype.startswith(b"multipart") else MAX_JSON_BYTES,
                    BULK: MAX_BULK_JSON_BYTES, SMALL: MAX_JSON_BYTES}[kind]
            assert fewshot.guard.body_limit(big, scope) == want, (method, template, ctype)


@pytest.mark.usefixtures("task_t")
def test_lists_of_ids_have_a_larger_limit(client):
    """Bulk edits, deletes and inbox decisions may list tens of thousands of ids (16 MB)."""
    ids = {"ids": [f"{i:032x}" for i in range(60_000)]}  # about 2.6 MB
    assert client.patch("/api/tasks/t/examples", json={**ids, "label": "red"}).status_code == 404
    for path in ("examples/delete", "inbox/dismiss"):
        assert client.post(f"/api/tasks/t/{path}", json=ids).status_code == 404, path
    accept = {"labels": {i: "red" for i in ids["ids"]}}
    assert client.post("/api/tasks/t/inbox/accept", json=accept).status_code == 404
    assert client.post("/api/tasks/t/inbox/accept-confident", json={**ids, "min_confidence": 0.5}).status_code == 200
    r = client.post("/api/tasks/t/examples/delete", content=b" " * (MAX_BULK_JSON_BYTES + 1),
                    headers={"Content-Type": "application/json"})
    assert (r.status_code, r.json()["detail"]) == (413, "Request is larger than the 16 MB limit")


def test_validation_errors_dont_echo_the_body(client):
    """A 422 says what is wrong without sending the bad input back."""
    r = client.post("/api/tasks", json={"name": "x", "junk": "y" * 500_000})
    assert r.status_code == 422
    assert len(r.content) < 1000
    assert r.json()["detail"] == "1 problem with the request: unknown field 'junk'"
    assert r.json()["errors"] == [{"pointer": "/junk", "detail": "unknown field 'junk'", "code": "extra_forbidden",
                                   "loc": ["body", "junk"], "msg": "Extra inputs are not permitted"}]


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
    asyncio.run(fewshot.guard.LimitBody(inner, 10)(scope, receive, send))
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

    asyncio.run(fewshot.guard.LimitBody(inner, 10)({"type": "lifespan"}, unused, unused))
    assert seen == ["lifespan"]


# ---- model allowlist ----------------------------------------------------
def test_only_known_models_by_default(client, monkeypatch):
    """Non-preset specs are refused (403) without loading anything, unless a task already uses
    them or the server allows any model."""
    loads: list[str] = []
    monkeypatch.setattr("fewshot.store.check_embedder", lambda spec: None)  # tasks whose model no longer loads
    monkeypatch.setattr(fewshot.server, "check_embedder", loads.append)
    r = client.post("/api/tasks", json={"name": "x", "embedder": "timm:some_model"})
    assert r.status_code == 403
    assert "--allow-any-model" in r.json()["detail"]
    assert not loads
    preset = PRESETS[2].spec
    assert client.post("/api/tasks", json={"name": "p", "embedder": preset}).status_code == 201
    assert client.post("/api/tasks", json={"name": "f", "embedder": "fake:color"}).status_code == 201  # registered
    client.srv.ws.create("cli", embedder="timm:some_model")  # e.g. made with the CLI
    assert client.patch("/api/tasks/f", json={"embedder": "timm:some_model"}).status_code == 200
    assert loads == [preset, "fake:color", "timm:some_model"]  # allowed specs are still loaded to check them
    set_settings(monkeypatch, client, allow_any_model=True)
    assert client.post("/api/tasks", json={"name": "y", "embedder": "timm:other"}).status_code == 201


# ---- warm-up ------------------------------------------------------------
def test_warm_models(client, monkeypatch):
    """Warm-up loads the most used embedders (default spec with no tasks), skipping failures."""
    loaded = []
    monkeypatch.setattr("fewshot.store.check_embedder", lambda spec: None)  # tasks whose model no longer loads

    def fake_get(spec):
        if spec == "bad:x":
            raise RuntimeError("no such model")
        loaded.append(spec)

    monkeypatch.setattr(fewshot.server, "get_embedder", fake_get)
    assert client.srv.warm_models() == [DEFAULT_SPEC]
    ws = client.srv.ws
    ws.create("a", embedder="fake:color")
    ws.create("b", embedder="fake:color")
    ws.create("c", embedder="bad:x")
    ws.create("d", embedder="other:y")
    monkeypatch.setenv("FEWSHOT_MAX_MODELS", "2")
    assert client.srv.warm_models() == ["fake:color"]  # bad:x is second most used, and fails
    assert loaded == [DEFAULT_SPEC, "fake:color"]


def test_warm_at_startup(tmp_path, monkeypatch):
    """With warm-up on, starting the app warms models in a background thread."""
    client = api_client(tmp_path, warm=True)
    done = threading.Event()
    monkeypatch.setattr(client.srv, "warm_models", done.set)
    with client:
        assert done.wait(5)


def test_no_warm_when_off(client):
    """With warm-up off, startup loads nothing in the background."""
    with client:
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


# ---- URL budget, export headers ---------------------------------------------
def test_url_fetches_have_a_budget(client, monkeypatch):
    """URLs share a time and size budget per request; what doesn't fit is reported per URL."""
    task = api_task(client, n=1)
    set_settings(monkeypatch, client, allow_urls=True)
    monkeypatch.setattr(fewshot.services, "fetch_url", lambda url, max_bytes, timeout=20.0, public_only=False: png(RED))
    set_settings(monkeypatch, client, max_request_mb=(len(png(RED)) + 1) / 2**20)
    preds = client.post(f"/api/tasks/{task}/predict/urls", json={"urls": ["http://h/a.png", "http://h/b.png"]}).json()
    assert sorted(p["error"] is None for p in preds) == [False, True]
    assert any("exceed the" in (p["error"] or "") for p in preds)
    monkeypatch.setattr(fewshot.api, "URL_FETCH_SECONDS", 0.0)
    preds = client.post(f"/api/tasks/{task}/predict/urls", json={"urls": ["http://h/a.png"]}).json()
    assert "budget for URLs ran out" in preds[0]["error"]


def test_export_headers_and_import_with_another_embedder(client):
    """The export says its size and how many examples were left out; import can switch embedder."""
    task = api_task(client, n=1)
    t = client.srv.ws.get(task)
    (t.path / t.config.examples[0].file).unlink()
    r = client.get(f"/api/tasks/{task}/export")
    assert r.headers["content-length"] == str(len(r.content))
    assert r.headers["x-fewshot-skipped"] == "1"
    up = client.post("/api/tasks/import", files={"file": ("c.zip", r.content, "application/zip")},
                     data={"name": "copy", "embedder": "fake:color"})
    assert (up.status_code, up.json()["embedder"]) == (201, "fake:color")


# ---- settings and the app factory ----------------------------------------
def test_settings_from_the_environment_are_checked():
    """Every setting is read and checked in one place, with a message naming the bad variable."""
    s = ServerSettings.from_env({"FEWSHOT_TOKEN": "0123456789abcdef", "FEWSHOT_WARM": "0", "FEWSHOT_ALLOW_URLS": "yes",
                                 "FEWSHOT_ALLOWED_HOSTS": "a.lan, B.lan:8000", "FEWSHOT_MAX_UPLOAD_MB": "2",
                                 "FEWSHOT_WORKERS": "3", "FEWSHOT_HOME": "/w",
                                 "FEWSHOT_FORWARDED_ALLOW_IPS": "10.0.0.1"})
    assert (s.token, s.warm, s.allow_urls) == ("0123456789abcdef", False, True)
    assert s.allowed_hosts == {"a.lan", "b.lan"}
    assert (s.max_upload_bytes, s.workers, str(s.home), s.forwarded_allow_ips) == (2 * 2**20, 3, "/w", "10.0.0.1")
    d = ServerSettings.from_env({})
    assert (d.token, d.warm, d.max_request_mb) == (None, True, 1024)
    assert d.allowed_hosts == {"localhost", "127.0.0.1", "::1"}
    for env, message in [({"FEWSHOT_MAX_UPLOAD_MB": "abc"}, "FEWSHOT_MAX_UPLOAD_MB must be a number"),
                         ({"FEWSHOT_MAX_REQUEST_MB": "-5"}, "FEWSHOT_MAX_REQUEST_MB must be a positive number"),
                         ({"FEWSHOT_MAX_UPLOAD_MB": "inf"}, "must be a positive number"),
                         ({"FEWSHOT_WARM": "maybe"}, "FEWSHOT_WARM must be 1 or 0"),
                         ({"FEWSHOT_WORKERS": "0.5"}, "FEWSHOT_WORKERS must be a whole number from 1 to 256"),
                         ({"FEWSHOT_WORKERS": "300"}, "FEWSHOT_WORKERS must be a whole number"),
                         ({"FEWSHOT_TOKEN": "s3cret"}, "FEWSHOT_TOKEN is too weak to be safe: it has 6 characters")]:
        with pytest.raises(ValueError, match=message):
            ServerSettings.from_env(env)
    with pytest.raises(ValueError, match="max_upload_mb must be a positive"):
        ServerSettings(max_upload_mb=0)
    with pytest.raises(ValueError, match="workers must be between"):
        ServerSettings(workers=0)
    assert ServerSettings(token="  ").token is None
    assert ServerSettings.from_env({"FEWSHOT_TOKEN": " "}).token is None
    assert ServerSettings.from_env({"FEWSHOT_WARM": "", "FEWSHOT_ALLOW_URLS": " "}).warm  # empty: unset
    assert ServerSettings.from_env({"FEWSHOT_ALLOW_NO_TOKEN": "1"}).allow_no_token
    assert not ServerSettings.from_env({}).allow_no_token


def test_tokens_that_could_be_guessed_are_refused():
    """A token must have at least 16 characters, of at least 5 different ones; random ones pass."""
    for weak, why in [("short-token", "it has 11 characters"), ("a" * 40, "it repeats 1 different characters"),
                      ("1234" * 8, "it repeats 4 different characters")]:
        with pytest.raises(ValueError, match=f"The token is too weak to be safe: {why}"):
            ServerSettings(token=weak)
    for good in ("0123456789abcdef", secrets.token_hex(16), secrets.token_urlsafe(12), "correct horse battery"):
        assert ServerSettings(token=good).token == good


def test_importing_the_api_reads_no_settings_and_makes_no_workspace(tmp_path, monkeypatch):
    """`import fewshot.api` has no side effects; `fewshot.api.app` is built from the environment
    on first use, and a bad setting stops it with a message instead of a traceback."""
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path / "ws"))
    monkeypatch.setenv("FEWSHOT_MAX_UPLOAD_MB", "abc")
    api = importlib.reload(fewshot.api)
    assert not (tmp_path / "ws").exists()
    with pytest.raises(SystemExit, match="fewshot: FEWSHOT_MAX_UPLOAD_MB must be a number"):
        _ = api.app
    monkeypatch.setenv("FEWSHOT_MAX_UPLOAD_MB", "3")
    api = importlib.reload(fewshot.api)
    first = api.app
    assert api.app is first  # built once
    assert api.app.state.server.settings.max_upload_mb == 3
    assert (tmp_path / "ws" / "tasks").is_dir()
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = api.nope
    importlib.reload(fewshot.api)


def test_uvicorn_app_string_loads(tmp_path):
    """`uvicorn fewshot.api:app` finds the app, as `fewshot serve` would build it."""
    code = ("import uvicorn.importer as i; app = i.import_from_string('fewshot.api:app'); "
            "print(type(app).__name__, app.state.server.settings.warm)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120,
                         env={"PATH": "", "FEWSHOT_HOME": str(tmp_path / "ws"), "FEWSHOT_WARM": "0"})
    assert out.stdout.split() == ["FastAPI", "False"]


# ---- threads -------------------------------------------------------------
@pytest.mark.usefixtures("task_t")
def test_healthz_answers_while_every_thread_is_busy(client, monkeypatch):
    """With Starlette's thread pool used up by slow requests, /healthz still answers."""
    entered, release = threading.Event(), threading.Event()
    real = Task.summary

    def slow_summary(self):
        entered.set()
        return release.wait(10) and real(self)

    monkeypatch.setattr(Task, "summary", slow_summary)
    with client:  # one event loop, whose thread limit is set to one
        client.portal.call(lambda: setattr(anyio.to_thread.current_default_thread_limiter(), "total_tokens", 1))
        stuck = threading.Thread(target=lambda: client.get("/api/tasks"))
        stuck.start()
        try:
            assert entered.wait(5)  # the one thread is taken
            answered = []
            probe = threading.Thread(target=lambda: answered.append(client.get("/healthz").status_code))
            probe.start()
            probe.join(5)
            assert answered == [200]  # answered while the thread was still taken
            assert stuck.is_alive()
        finally:
            release.set()
            stuck.join(10)


def test_heavy_work_waits_on_its_own_threads(tmp_path, monkeypatch):
    """Predictions queue on the server's worker threads; cheap requests are still answered."""
    c = api_client(tmp_path, workers=1)
    c.post("/api/tasks", json={"name": "t", "embedder": "fake:color"})
    c.post("/api/tasks/t/examples", data={"label": "red"}, files=query_files(1))
    entered, release, calls = threading.Event(), threading.Event(), []

    def slow_predict(_pipe, _t, blobs, **_kw):
        calls.append(1)
        entered.set()
        return release.wait(10) and [Prediction.failed("not really") for _ in blobs]

    monkeypatch.setattr(fewshot.services, "predict", slow_predict)
    with c:  # one event loop for every request
        slow = [threading.Thread(target=lambda: c.post("/api/tasks/t/predict", files=query_files(1))) for _ in range(3)]
        for th in slow:
            th.start()
        try:
            assert entered.wait(5)  # the one worker is busy
            queue = c.srv.heavy._work_queue  # pylint: disable=protected-access
            deadline = time.monotonic() + 5
            while queue.qsize() < 2:  # and the other two wait for it
                assert time.monotonic() < deadline, "the other predictions never queued"
                time.sleep(0.01)
            done = []
            probe = threading.Thread(target=lambda: done.append(c.get("/api/tasks/t").status_code))
            probe.start()
            probe.join(5)
            assert done == [200]  # answered while the heavy work was still waiting
            assert len(calls) == 1
            assert queue.qsize() == 2
        finally:
            release.set()
            for th in slow:
                th.join(10)


def test_the_app_can_be_started_again(tmp_path):
    """Stopping the app stops its heavy-work threads, and starting it again makes new ones, so an
    app can be served (or tested) more than once."""
    c = api_client(tmp_path)
    c.post("/api/tasks", json={"name": "t", "embedder": "fake:color"})
    c.post("/api/tasks/t/examples", data={"label": "red"}, files=query_files(1))
    for _ in range(2):
        with c:
            assert c.post("/api/tasks/t/predict", files=query_files(1)).status_code == 200
    c.srv.stop_heavy()  # stopping twice is harmless
    assert c.post("/api/tasks/t/predict", files=query_files(1)).status_code == 200


# ---- uploads -------------------------------------------------------------
@pytest.mark.usefixtures("task_t")
def test_uploads_are_read_and_classified_a_chunk_at_a_time(client, monkeypatch):
    """Uploaded images are read lazily and classified PREDICT_CHUNK at a time, with one version."""
    monkeypatch.setattr(fewshot.api, "PREDICT_CHUNK", 2)
    seen = []
    real = fewshot.services.predict

    def spy(pipe, task, blobs, **kw):
        seen.append((len(blobs), kw["cfg"].revision))
        return real(pipe, task, blobs, **kw)

    monkeypatch.setattr(fewshot.services, "predict", spy)
    preds = client.post("/api/tasks/t/predict", files=query_files(5)).json()
    assert [p["label"] for p in preds] == ["red"] * 5
    assert [n for n, _ in seen] == [2, 2, 1]
    assert len({rev for _, rev in seen}) == 1


@pytest.mark.usefixtures("task_t")
def test_added_images_are_handed_over_lazily(client, monkeypatch):
    """The store gets the uploads as a generator, so it holds one chunk of images at a time."""
    kinds = []
    real = Task.add_examples

    def spy(self, items, chunk=256, **kw):
        kinds.append(isinstance(items, list))
        return real(self, items, chunk, **kw)

    monkeypatch.setattr(Task, "add_examples", spy)
    assert client.post("/api/tasks/t/examples", data={"label": "red"}, files=query_files(3)).status_code == 201
    assert kinds == [False]


@pytest.mark.usefixtures("task_t")
def test_json_bodies_have_a_small_limit(client):
    """Bodies that aren't uploads are refused past 1 MB, though uploads may be far larger."""
    big = {"name": "x" * (MAX_JSON_BYTES + 10)}
    r = client.post("/api/tasks", json=big)
    assert (r.status_code, r.json()["detail"]) == (413, "Request is larger than the 1 MB limit")
    assert client.post("/api/tasks/t/examples", data={"label": "red"},
                       files=[("files", ("big.png", png(RED, size=800, noise=60), "image/png"))]).status_code == 201


# ---- responses -------------------------------------------------------------
@pytest.mark.usefixtures("task_t")
def test_responses_are_built_off_the_event_loop(client, monkeypatch):
    """Validating and serialising a response (which for a large task takes long enough to stall
    every other request) happens in the request's thread, never on the event loop."""
    loops = []
    real = fewshot.responses.render

    def spy(model, content):
        try:
            asyncio.get_running_loop()
            loops.append(model)
        except RuntimeError:
            pass
        return real(model, content)

    monkeypatch.setattr(fewshot.responses, "render", spy)
    with client:
        assert client.get("/api/tasks/t").status_code == 200
        assert client.get("/api/tasks").status_code == 200
        assert client.post("/api/tasks/t/predict", files=query_files(1)).status_code == 200
        assert client.post("/api/tasks/t/examples", data={"label": "red"}, files=query_files(1)).status_code == 201
        jid = client.post("/api/tasks/t/evaluate/jobs").json()["id"]
        assert client.get(f"/api/jobs/{jid}").status_code == 200
    assert not loops


@pytest.mark.usefixtures("task_t")
def test_long_lists_are_rendered_in_chunks_with_the_same_result(client, monkeypatch):
    """A response with a long list is validated and serialised a chunk at a time (so the GIL is
    let go in between), giving the same JSON as rendering it whole."""
    client.post("/api/tasks/t/examples", data={"label": "blue"}, files=query_files(5, rgb=(0, 0, 255)))
    whole = client.get("/api/tasks/t").content
    preds = client.post("/api/tasks/t/predict", files=query_files(5)).content
    monkeypatch.setattr(fewshot.responses, "RENDER_CHUNK", 2)
    assert client.get("/api/tasks/t").content == whole
    assert client.post("/api/tasks/t/predict", files=query_files(5)).content == preds

    class Listing(pydantic.BaseModel):  # pylint: disable=missing-class-docstring
        xs: list[int]
        note: str = "none"

    class Strict(pydantic.BaseModel):  # pylint: disable=missing-class-docstring
        xs: list[int]
        note: str

    assert fewshot.responses.render(Listing, {"xs": [1, 2, 3]}) == b'{"xs":[1,2,3],"note":"none"}'
    with pytest.raises(pydantic.ValidationError, match="note"):
        fewshot.responses.render(Strict, {"xs": [1, 2, 3]})


def test_choosing_a_model_loads_it_on_the_heavy_work_threads(client, monkeypatch):
    """Creating a task or switching its embedder, which loads the model, waits on the server's
    own threads, like other heavy work."""
    threads = []
    monkeypatch.setattr("fewshot.store.check_embedder", lambda spec: None)  # the server's own check is enough here
    monkeypatch.setattr(client.srv, "check_embedder", lambda _spec: threads.append(threading.current_thread().name))
    assert client.post("/api/tasks", json={"name": "t", "embedder": "fake:color"}).status_code == 201
    assert client.patch("/api/tasks/t", json={"embedder": "fake:other"}).status_code == 200
    assert len(threads) == 2
    assert all(name.startswith("fewshot-work") for name in threads)


def test_several_processes_are_noticed_and_warned_about(client, monkeypatch, caplog):
    """uvicorn's or gunicorn's --workers (or WEB_CONCURRENCY) count as processes, fewshot serve's
    --workers (threads) doesn't; a server started with several warns, as its state is per process."""
    count = fewshot.server_settings.requested_processes
    uvicorn = ["/venv/bin/uvicorn", "fewshot.api:app"]
    assert count([*uvicorn, "--workers", "4"], {}) == 4
    assert count([*uvicorn, "--workers=3"], {}) == 3
    assert count(["/venv/lib/uvicorn/__main__.py", "fewshot.api:app", "--workers", "x"], {}) == 1
    assert count(["/venv/bin/python", "/venv/bin/gunicorn", "-w", "2"], {}) == 2
    assert count([*uvicorn, "--workers"], {"WEB_CONCURRENCY": "5"}) == 5
    assert count(["/venv/bin/fewshot", "serve", "--workers", "8"], {}) == 1
    assert count([], {}) == 1
    assert count() == 1  # these tests
    monkeypatch.setattr(fewshot.api, "requested_processes", lambda: 2)
    with caplog.at_level("WARNING", logger="fewshot.api"), client:
        pass
    assert "started with 2 processes, which isn't supported" in caplog.text
