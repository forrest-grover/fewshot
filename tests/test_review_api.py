"""Fixes from the interface review, server side: settings and the app factory, threads for heavy
work, upload streaming and limits, jobs, tokens and sessions, one change per request."""

import contextlib
import importlib
import subprocess
import sys
import threading
import time
from typing import Any

import anyio.to_thread
import pytest
from fastapi.testclient import TestClient

import fewshot.api
import fewshot.auth
from fewshot import services
from fewshot.auth import Sessions, WrongTokens
from fewshot.folders import read_or_empty
from fewshot.jobs import Job, Jobs
from fewshot.pipeline import Prediction
from fewshot.server_settings import MAX_JSON_BYTES, ServerSettings
from fewshot.store import Task, Workspace
from fewshot.text import MAX_DESCRIPTION, MAX_PROMPTS
from tests.conftest import api_client, png

RED, BLUE = (220, 30, 30), (30, 30, 220)


@pytest.fixture(name="client")
def client_fixture(tmp_path):
    """A server with task 't': two red and two blue examples."""
    c = api_client(tmp_path)
    c.post("/api/tasks", json={"name": "t", "embedder": "fake:color"})
    for label, rgb in (("red", RED), ("blue", BLUE)):
        files = [("files", (f"{label}{i}.png", png(rgb, noise=10, seed=i), "image/png")) for i in range(2)]
        assert c.post("/api/tasks/t/examples", data={"label": label}, files=files).status_code == 201
    return c


def _files(n, rgb=RED):
    return [("files", (f"q{i}.png", png(rgb, noise=10, seed=50 + i), "image/png")) for i in range(n)]


# ---- settings and the app factory ------------------------------------------------
def test_settings_from_the_environment_are_checked():
    """Every setting is read and checked in one place, with a message naming the bad variable."""
    s = ServerSettings.from_env({"FEWSHOT_TOKEN": "x", "FEWSHOT_WARM": "0", "FEWSHOT_ALLOW_URLS": "yes",
                                 "FEWSHOT_ALLOWED_HOSTS": "a.lan, B.lan:8000", "FEWSHOT_MAX_UPLOAD_MB": "2",
                                 "FEWSHOT_WORKERS": "3", "FEWSHOT_HOME": "/w",
                                 "FEWSHOT_FORWARDED_ALLOW_IPS": "10.0.0.1"})
    assert (s.token, s.warm, s.allow_urls, s.allowed_hosts) == ("x", False, True, frozenset({"a.lan", "b.lan"}))
    assert (s.max_upload_bytes, s.workers, str(s.home), s.forwarded_allow_ips) == (2 * 2**20, 3, "/w", "10.0.0.1")
    d = ServerSettings.from_env({})
    assert (d.token, d.warm, d.max_request_mb) == (None, True, 1024)
    assert d.allowed_hosts == {"localhost", "127.0.0.1", "::1"}
    for env, message in [({"FEWSHOT_MAX_UPLOAD_MB": "abc"}, "FEWSHOT_MAX_UPLOAD_MB must be a number"),
                         ({"FEWSHOT_MAX_REQUEST_MB": "-5"}, "FEWSHOT_MAX_REQUEST_MB must be a positive number"),
                         ({"FEWSHOT_MAX_UPLOAD_MB": "inf"}, "must be a positive number"),
                         ({"FEWSHOT_WARM": "maybe"}, "FEWSHOT_WARM must be 1 or 0")]:
        with pytest.raises(ValueError, match=message):
            ServerSettings.from_env(env)
    with pytest.raises(ValueError, match="max_upload_mb must be a positive"):
        ServerSettings(max_upload_mb=0)
    with pytest.raises(ValueError, match="workers must be between"):
        ServerSettings(workers=0)
    assert ServerSettings(token="  ").token is None


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


# ---- threads ------------------------------------------------------------------------
def test_healthz_answers_while_every_thread_is_busy(client, monkeypatch):
    """With Starlette's thread pool used up by slow requests, /healthz still answers."""
    release = threading.Event()
    real = Task.summary
    monkeypatch.setattr(Task, "summary", lambda self: release.wait(10) and real(self))
    with client:  # one event loop, whose thread limit is set to one
        client.portal.call(lambda: setattr(anyio.to_thread.current_default_thread_limiter(), "total_tokens", 1))
        stuck = threading.Thread(target=lambda: client.get("/api/tasks"))
        stuck.start()
        time.sleep(0.2)
        answered = []
        probe = threading.Thread(target=lambda: answered.append(client.get("/healthz").status_code))
        probe.start()
        probe.join(5)
        release.set()
        stuck.join(10)
    assert answered == [200]


def test_heavy_work_waits_on_its_own_threads(tmp_path, monkeypatch):
    """Predictions queue on the server's worker threads; cheap requests are still answered."""
    c = api_client(tmp_path, workers=1)
    c.post("/api/tasks", json={"name": "t", "embedder": "fake:color"})
    c.post("/api/tasks/t/examples", data={"label": "red"}, files=_files(1))
    release = threading.Event()
    monkeypatch.setattr(services, "predict", lambda _pipe, _t, blobs, **k: release.wait(10) and
                        [Prediction.failed("not really") for _ in blobs])
    with c:  # one event loop for every request
        slow = [threading.Thread(target=lambda: c.post("/api/tasks/t/predict", files=_files(1))) for _ in range(3)]
        for th in slow:
            th.start()
        time.sleep(0.2)
        done = []
        probe = threading.Thread(target=lambda: done.append(c.get("/api/tasks/t").status_code))
        probe.start()
        probe.join(5)
        release.set()
        for th in slow:
            th.join(10)
    assert done == [200]


# ---- uploads --------------------------------------------------------------------------
def test_uploads_are_read_and_classified_a_chunk_at_a_time(client, monkeypatch):
    """Uploaded images are read lazily and classified PREDICT_CHUNK at a time, with one version."""
    monkeypatch.setattr(fewshot.api, "PREDICT_CHUNK", 2)
    seen = []
    real = services.predict

    def spy(pipe, task, blobs, **kw):
        seen.append((len(blobs), kw["cfg"].revision))
        return real(pipe, task, blobs, **kw)

    monkeypatch.setattr(services, "predict", spy)
    preds = client.post("/api/tasks/t/predict", files=_files(5)).json()
    assert [p["label"] for p in preds] == ["red"] * 5
    assert [n for n, _ in seen] == [2, 2, 1]
    assert len({rev for _, rev in seen}) == 1


def test_added_images_are_handed_over_lazily(client, monkeypatch):
    """The store gets the uploads as a generator, so it holds one chunk of images at a time."""
    kinds = []
    real = Task.add_examples

    def spy(self, items, chunk=256):
        kinds.append(isinstance(items, list))
        return real(self, items, chunk)

    monkeypatch.setattr(Task, "add_examples", spy)
    assert client.post("/api/tasks/t/examples", data={"label": "red"}, files=_files(3)).status_code == 201
    assert kinds == [False]


def test_json_bodies_have_a_small_limit(client):
    """Bodies that aren't uploads are refused past 1 MB, though uploads may be far larger."""
    big = {"name": "x" * (MAX_JSON_BYTES + 10)}
    r = client.post("/api/tasks", json=big)
    assert (r.status_code, r.json()["detail"]) == (413, "Request is larger than the 1 MB limit")
    assert client.post("/api/tasks/t/examples", data={"label": "red"},
                       files=[("files", ("big.png", png(RED, size=800, noise=60), "image/png"))]).status_code == 201


# ---- jobs --------------------------------------------------------------------------------
def test_the_same_job_is_shared_under_the_lock():
    """Jobs.start itself returns the running job doing the same work, so racing starts share it."""
    jobs, gate = Jobs(), threading.Event()
    got = []

    def start():
        got.append(jobs.start(Job(task="t", kind="evaluate", revision=3), lambda _p: gate.wait(5), limit=4).id)

    threads = [threading.Thread(target=start) for _ in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(set(got)) == 1
    other = jobs.start(Job(task="t", kind="evaluate", revision=4), lambda _p: gate.wait(5))
    assert other.id != got[0]  # another version is other work
    gate.set()


def test_a_cancelled_job_frees_its_place_at_once():
    """A job asked to stop is "cancelling" and no longer counts towards the limit."""
    jobs, gate = Jobs(), threading.Event()
    first = jobs.start(Job(task="t", kind="a"), lambda _p: gate.wait(5), limit=1)
    first.request_cancel()
    assert first.to_dict()["status"] == "cancelling"
    second = jobs.start(Job(task="t", kind="b"), lambda _p: gate.wait(5), limit=1)
    assert second.status == "running"
    assert jobs.running() == [second]
    gate.set()
    deadline = time.time() + 5
    while first.finished is None and time.time() < deadline:
        time.sleep(0.01)
    assert first.status == "done"  # it finished without another progress step: its result stands
    first.request_cancel()  # too late: nothing changes
    assert first.status == "done"


def test_only_recent_results_are_kept():
    """Finished jobs past kept_results keep their status but not their (possibly large) result."""
    jobs = Jobs(kept=5, kept_results=2)
    def work(i):
        return lambda _p: {"n": i}

    done = [jobs.start(Job(task="t", kind=str(i)), work(i)) for i in range(4)]
    deadline = time.time() + 5
    while any(j.finished is None for j in done) and time.time() < deadline:
        time.sleep(0.01)
    gate = threading.Event()
    jobs.start(Job(task="t", kind="last"), lambda _p: gate.wait(5))  # starting a job forgets old results too
    kept = [jobs.get(j.id) for j in done]
    gate.set()
    assert [j.expired for j in kept if j] == [True, True, False, False]
    assert kept[0] is not None
    assert kept[0].result is None
    assert kept[0].to_dict()["expired"] is True


def test_unknown_jobs_say_why(client):
    """A job id the server doesn't know may be from before a restart; the message says so."""
    r = client.get("/api/jobs/abc")
    assert r.status_code == 404
    assert "server was restarted" in r.json()["detail"]


# ---- tokens and sessions --------------------------------------------------------------------
def test_wrong_tokens_are_bounded_and_expire(monkeypatch):
    """Only wrong tokens make entries; old ones expire and the map has a size limit."""
    monkeypatch.setattr(fewshot.auth, "LOGIN_CLIENTS", 3)
    monkeypatch.setattr(fewshot.auth, "LOGIN_FAILURES", 2)
    w = WrongTokens()
    assert not w.blocked("a")
    assert not w._seen  # pylint: disable=protected-access  # asking made no entry
    for client in "abcd":
        w.wrong(client)
    assert list(w._seen) == ["b", "c", "d"]  # pylint: disable=protected-access
    w.wrong("d")
    assert w.blocked("d")
    monkeypatch.setattr(fewshot.auth, "LOGIN_WINDOW", 0.0)
    time.sleep(0.01)
    assert not w.blocked("d")
    assert "d" not in w._seen  # pylint: disable=protected-access  # expired entries go


def test_a_signed_in_browser_keeps_working_while_its_address_is_blocked(tmp_path, monkeypatch):
    """Someone guessing tokens from the same address (say, behind the same proxy) gets 429 for
    token checks, but a session cookie, which can't be guessed, still works."""
    monkeypatch.setattr(fewshot.auth, "LOGIN_FAILURES", 2)
    c = api_client(tmp_path, token="s3cret")
    assert c.post("/api/login", json={"token": "s3cret"}).status_code == 204
    other = TestClient(c.app)
    for guess in ("a", "b"):
        assert other.get("/api/tasks", headers={"Authorization": f"Bearer {guess}"}).status_code == 401
    assert other.post("/api/login", json={"token": "s3cret"}).status_code == 429
    assert c.get("/api/tasks").status_code == 200


def _cookie(value):
    return {"Cookie": f"{fewshot.api.SESSION_COOKIE}={value}"}


def test_the_session_cookie_is_signed_not_the_token(tmp_path):
    """The cookie holds a signed session id that expires and that logout ends; tampering, the
    raw token, or a cookie from a server with another token is refused."""
    c = api_client(tmp_path, token="s3cret")
    r = c.post("/api/login", json={"token": "s3cret"})
    cookie = r.cookies[fewshot.api.SESSION_COOKIE]
    assert "s3cret" not in r.headers["set-cookie"]
    assert "samesite=strict" in r.headers["set-cookie"].lower()
    fresh = TestClient(c.app)
    assert fresh.get("/api/tasks", headers=_cookie(cookie)).status_code == 200
    for bad in ("s3cret", cookie[:-1] + ("0" if cookie[-1] != "0" else "1"), "v1.1.2.3", "junk"):
        assert fresh.get("/api/tasks", headers=_cookie(bad)).status_code == 401
    other = api_client(tmp_path / "other", token="changed")
    assert other.get("/api/tasks", headers=_cookie(cookie)).status_code == 401
    assert c.post("/api/logout").status_code == 204
    assert fresh.get("/api/tasks", headers=_cookie(cookie)).status_code == 401  # revoked
    sessions = Sessions("s3cret")
    expired = sessions.issue(seconds=-1)
    assert not sessions.valid(expired)
    sessions.revoke(expired)  # forgotten at once: it has expired anyway
    sessions.revoke(None)
    assert not sessions._revoked  # pylint: disable=protected-access


def test_logout_without_a_token_configured(client):
    """Logging out of a server without a token just clears the cookie."""
    assert client.post("/api/logout").status_code == 204


# ---- one request, one change ----------------------------------------------------------------
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


def test_descriptions_are_capped_over_the_api(client):
    """A description too long is refused by the schema; too many prompts by the store."""
    r = client.post("/api/tasks/t/classes", json={"name": "red", "description": "x" * (MAX_DESCRIPTION + 1)})
    assert r.status_code == 422
    r = client.patch("/api/tasks/t/classes/red", json={"description": "a\n" * (MAX_PROMPTS + 1)})
    assert (r.status_code, "prompts" in r.json()["detail"]) == (400, True)
    r = client.post("/api/tasks/t/classes", json={"name": "red", "description": "a\n" * (MAX_PROMPTS + 1)})
    assert r.status_code == 400


# ---- shared operations ----------------------------------------------------------------------
def test_accept_confident_suggestions(client):
    """The inbox's confident suggestions can be accepted in one request, as `inbox accept --min-confidence`."""
    assert client.post("/api/tasks/t/inbox", files=_files(2) + _files(1, BLUE)[:0]).status_code == 201
    t = client.post("/api/tasks/t/inbox/accept-confident", json={"min_confidence": 0.0}).json()
    assert (t["inbox_count"], len(t["examples"])) == (0, 6)
    assert client.post("/api/tasks/t/inbox/accept-confident", json={"min_confidence": 0.5}).status_code == 200
    assert client.post("/api/tasks/t/inbox/accept-confident", json={"min_confidence": 2}).status_code == 422
    assert client.post("/api/tasks/nope/inbox/accept-confident", json={"min_confidence": 0.5}).status_code == 404


def test_a_task_deleted_meanwhile_is_404_not_500(client, monkeypatch):
    """Every change to a task that another process deleted after it was looked up gets 404."""
    eid = client.get("/api/tasks/t").json()["examples"][0]["id"]
    assert client.post("/api/tasks/t/inbox", files=_files(1, BLUE)).status_code == 201

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
        ("post", "/api/tasks/t/examples", {"data": {"label": "red"}, "files": _files(1)}),
        ("post", "/api/tasks/t/inbox", {"files": _files(1, (30, 220, 30))}),
        ("post", "/api/tasks/t/inbox/accept-confident", {"json": {"min_confidence": 0}}),
        ("patch", f"/api/tasks/t/examples/{eid}", {"json": {"label": "blue"}}),
    ]
    for method, url, kw in calls:
        r = getattr(client, method)(url, **kw)
        assert (r.status_code, r.json()["detail"]) == (404, "No task 't'"), url


def test_empty_edits_change_nothing(client):
    """An edit with nothing in it is accepted and changes nothing."""
    t = client.get("/api/tasks/t").json()
    eid = t["examples"][0]["id"]
    assert client.patch(f"/api/tasks/t/examples/{eid}", json={}).json()["revision"] == t["revision"]
    assert client.patch("/api/tasks/t/examples", json={"ids": [eid]}).json()["revision"] == t["revision"]


def test_an_unreadable_file_reads_as_empty(tmp_path):
    """A file that can't be read gives no bytes (and so an unreadable image), not an exception."""
    assert read_or_empty(tmp_path) == b""  # a folder
