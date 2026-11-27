"""fewshot.client against a real server, and the API's operation ids."""

import http.server
import socket
import threading
import time
from typing import ClassVar

import pytest
import uvicorn

import fewshot.api
import fewshot.client
from fewshot.api import create_app
from fewshot.client import (
    BATCH_BYTES,
    APIConnectionError,
    ApiError,
    AuthenticationError,
    BadRequestError,
    Client,
    ConflictError,
    FewshotError,
    JobCancelledError,
    JobFailedError,
    JobResultExpiredError,
    NotFoundError,
    PermissionDeniedError,
    ServerError,
    ValidationError,
    _batches,
)
from fewshot.server_settings import ServerSettings
from tests.conftest import BLUE, RED, png


def _serve(tmp_path, **settings):
    """The API served by uvicorn on a free local port, with token 'secret-0123456789'; yields its URL."""
    app = create_app(ServerSettings(home=tmp_path / "ws", token="secret-0123456789", warm=False, **settings))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}/"
    srv.should_exit = True
    thread.join(10)


@pytest.fixture(name="server")
def server_fixture(tmp_path):
    """A server with token 'secret-0123456789'; yields its URL."""
    yield from _serve(tmp_path)


@pytest.fixture(name="small_server")
def small_server_fixture(tmp_path):
    """A server taking requests of at most 64 KB; yields its URL."""
    yield from _serve(tmp_path, max_request_mb=1 / 16)


def test_client_round_trip(server, tmp_path):
    """Create, add, predict (at a revision), inbox, evaluate and delete through the client."""
    fs = Client(server, token="secret-0123456789")
    assert fs.health()["auth_required"] is True
    fs.create_task("my-pets", embedder="fake:color")
    f = tmp_path / "r.png"
    f.write_bytes(png(RED, noise=15, seed=1))
    fs.add_examples("my-pets", "red", [f, ("r2.png", png(RED, noise=15, seed=2))])
    added = fs.add_examples("my-pets", "blue", [(f'b"{i}.png', png(BLUE, noise=15, seed=i)) for i in range(2)])
    assert [x["source"] for x in added["added"]] == ['b"0.png', 'b"1.png']  # sent as %22, like browsers do
    rev = fs.task("my-pets")["revision"]
    [p] = fs.predict("my-pets", [("q.png", png(BLUE, seed=9))], top_k=1, evidence=1)
    assert (p["label"], list(p["scores"]), len(p["evidence"]), p["revision"]) == ("blue", ["blue"], 1, rev)
    assert fs.predict("my-pets", [f], revision=rev)[0]["label"] == "red"
    fs.add_to_inbox("my-pets", [("u.png", png(RED, seed=5))])
    [item] = fs.inbox("my-pets", sort="confident")["items"]
    assert item["prediction"] is not None
    assert item["prediction"]["label"] == "red"
    assert len(fs.accept_inbox("my-pets", {item["id"]: "red"})["examples"]) == 5
    assert fs.evaluate("my-pets")["n_examples"] == 5
    assert [t["name"] for t in fs.tasks()] == ["my-pets"]
    fs.delete_task("my-pets")
    assert fs.tasks() == []


def test_client_errors(server):
    """Error responses raise the ApiError of their status, with the server's code and detail."""
    with pytest.raises(AuthenticationError) as e:
        Client(server, token="wrong").tasks()
    assert (e.value.status, e.value.status_code, e.value.code) == (401, 401, "WRONG_TOKEN")
    assert not issubclass(PermissionDeniedError, AuthenticationError)  # a 403 isn't about the token
    fs = Client(server, token="secret-0123456789")
    fs.create_task("pets", embedder="fake:color")
    with pytest.raises(NotFoundError, match=r"^TASK_NOT_FOUND: No task 'pest'") as nf:
        fs.task("pest")
    assert (nf.value.fields["similar_to"], nf.value.body["instance"]) == (["pets"], "/api/tasks/pest")
    with pytest.raises(ConflictError, match="TASK_EXISTS"):
        fs.create_task("pets")
    with pytest.raises(ValidationError) as v:
        fs.create_task("x", classifier="nope", colour="red")
    assert [x["pointer"] for x in v.value.errors] == ["/classifier", "/colour"]
    assert isinstance(v.value, ApiError)
    assert isinstance(v.value, FewshotError)
    fs.add_examples("pets", "Persian", [("p.png", png(RED))])
    with pytest.raises(ValidationError) as v:
        fs.add_examples("pets", "Persain", [("q.png", png(BLUE))])
    assert (v.value.code, v.value.fields["similar_to"]) == ("CLASS_NAME_NEAR_MISS", ["Persian"])
    assert fs.add_examples("pets", "Persain", [("q.png", png(BLUE))], new_class=True)["created_classes"] == ["Persain"]
    with pytest.raises(ValidationError, match="unknown parameter 'sortt'"):
        fs.request("GET", "/tasks/pets/inbox", params={"sortt": "novel"})


def test_non_json_error_body(monkeypatch):
    """An error body that isn't JSON (a proxy's HTML page, say) becomes the detail as text."""
    import io  # pylint: disable=import-outside-toplevel
    from urllib.error import HTTPError  # pylint: disable=import-outside-toplevel

    def fail(*_a, **_k):
        raise HTTPError("http://x", 502, "Bad Gateway", {}, io.BytesIO(b"<html>bad gateway</html>"))  # type: ignore[arg-type]

    monkeypatch.setattr(fewshot.client._OPENER, "open", fail)  # pylint: disable=protected-access
    with pytest.raises(ServerError, match="HTTP_502: <html>bad gateway</html>"):
        Client().health()


def test_operation_ids_are_function_names():
    """Every operation id is its handler's name, so generated clients get tidy method names."""
    ops = [op for path in fewshot.api.app.openapi()["paths"].values() for op in path.values()]
    ids = [op["operationId"] for op in ops]
    assert len(ids) == len(set(ids))
    assert {"predict", "add_examples", "list_tasks", "health"} <= set(ids)


def test_client_sends_many_images_in_batches(server, monkeypatch):
    """Uploads and predictions go BATCH images per request; the results are combined."""
    monkeypatch.setattr("fewshot.client.BATCH", 2)
    fs = Client(server, token="secret-0123456789")
    fs.create_task("batched", embedder="fake:color")
    r = fs.add_examples("batched", "red", [(f"r{i}.png", png(RED, noise=15, seed=i)) for i in range(3)]
                        + [("bad.png", b"not an image")])
    assert (len(r["added"]), [e["file"] for e in r["errors"]], len(r["task"]["examples"])) == (3, ["bad.png"], 3)
    fs.add_examples("batched", "blue", [(f"b{i}.png", png(BLUE, noise=15, seed=i)) for i in range(2)])
    preds = fs.predict("batched", (("q.png", png(c, seed=9)) for c in (RED, BLUE, RED, BLUE, RED)))
    assert [p["label"] for p in preds] == ["red", "blue", "red", "blue", "red"]
    assert fs.predict("batched", []) == []
    inbox = fs.add_to_inbox("batched", [(f"u{i}.png", png(RED, noise=15, seed=20 + i)) for i in range(3)])
    assert (len(inbox["added"]), inbox["unchanged"]) == (3, [])
    with pytest.raises(ValueError, match="No images given"):
        fs.add_to_inbox("batched", [])
    again = fs.add_examples("batched", "red", [(f"r{i}.png", png(RED, noise=15, seed=i)) for i in range(3)])
    assert (again["added"], len(again["unchanged"]), again["moved"]) == ([], 3, [])  # combined over 2 batches
    moved = fs.add_examples("batched", "blue", [(f"r{i}.png", png(RED, noise=15, seed=i)) for i in range(3)])
    assert [m["previous_label"] for m in moved["moved"]] == ["red"] * 3
    fs.delete_task("batched")


def test_a_refused_batch_doesnt_lose_the_others(server, monkeypatch):
    """A batch the server refuses (nothing in it could be added) becomes an error; only if every
    batch is refused does the call raise, and other errors still raise at once."""
    monkeypatch.setattr("fewshot.client.BATCH", 2)
    fs = Client(server, token="secret-0123456789")
    fs.create_task("refused", embedder="fake:color")
    bad = [("x.png", b"junk"), ("y.png", b"junk")]
    r = fs.add_examples("refused", "red", [*bad, ("r.png", png(RED))])
    assert (len(r["added"]), [e["file"] for e in r["errors"]], len(r["task"]["examples"])) == (1, ["x.png", "y.png"], 1)
    assert "Not a readable image" in r["errors"][0]["error"]  # each file's reason, as the server lists them
    i = fs.add_to_inbox("refused", [("r.png", png(RED)), ("r2.png", png(RED)), ("u.png", png(BLUE))])
    assert (len(i["added"]), [e["file"] for e in i["errors"]]) == (1, ["r.png", "r2.png"])  # examples already
    with pytest.raises(ApiError) as e:
        fs.add_examples("refused", "red", [*bad, *bad])
    assert e.value.status == 400
    with pytest.raises(ApiError) as e:
        fs.add_examples("nope", "red", bad)
    assert e.value.status == 404


def test_uploads_are_batched_by_size(small_server, monkeypatch):
    """Images are sent in batches of at most half the server's request limit, so many large
    images don't make one request the server refuses."""
    fs = Client(small_server, token="secret-0123456789")
    assert fs.batch_bytes() == 32 * 1024
    fs.create_task("t", embedder="fake:color")
    images = [(f"{i}.png", png(RED, size=64, noise=40, seed=i)) for i in range(8)]
    assert 8 * len(images[0][1]) > 64 * 1024  # all at once would be refused
    sent = []
    real = Client.request

    def spy(self, method, path, **kw):
        sent.append(len(kw.get("files", ())))
        return real(self, method, path, **kw)

    monkeypatch.setattr(Client, "request", spy)
    r = fs.add_examples("t", "red", images)
    assert len(r["added"]) == 8
    assert len([n for n in sent if n]) > 1
    assert r["task"]["undo_label"] == "add 8 examples to red"  # the batches are one undo step
    assert len(fs.predict("t", images)) == 8


def test_batches_without_a_limit_from_the_server(monkeypatch, tmp_path):
    """An older server that doesn't report its limit gets BATCH_BYTES per request; a file that
    can't be read counts as empty here, and fails when it is sent."""
    monkeypatch.setattr(Client, "health", lambda self: {"status": "ok"})
    assert Client().batch_bytes() == BATCH_BYTES
    big = ("big.png", b"x" * 10)
    assert [len(b) for b in _batches([big, big, big], max_bytes=15)] == [1, 1, 1]
    assert [len(b) for b in _batches([tmp_path / "missing.png"] * 3, max_bytes=15)] == [3]


def test_connection_failures_raise_api_connection_error():
    """A refused connection raises APIConnectionError naming the URL and how to start a server."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(APIConnectionError, match=r"Connection to .* failed.*`fewshot serve`") as e:
        Client(f"http://127.0.0.1:{port}", timeout=5).health()
    assert e.value.url == f"http://127.0.0.1:{port}"
    assert not isinstance(e.value, ApiError)


def test_a_reset_upload_hints_at_the_size_limit(monkeypatch):
    """A connection dropped during a large upload (as a server refusing it does) says so."""
    def reset(*_a, **_k):
        raise ConnectionResetError(104, "Connection reset by peer")

    monkeypatch.setattr(fewshot.client._OPENER, "open", reset)  # pylint: disable=protected-access
    monkeypatch.setattr(Client, "batch_bytes", lambda self: 10**9)
    with pytest.raises(APIConnectionError, match="larger than the server accepts"):
        Client().predict("t", [("a.png", b"x" * (2 * 1024 * 1024))])


def test_evaluate_runs_as_a_job(server, monkeypatch):
    """evaluate polls a background job; one that can't start raises at once, and a failed,
    cancelled or expired one raises its JobError."""
    fs = Client(server, token="secret-0123456789")
    fs.create_task("empty", embedder="fake:color")
    with pytest.raises(BadRequestError, match="NO_EXAMPLES: Task has no examples") as e:
        fs.evaluate("empty", poll=0.01)
    assert e.value.status == 400
    replies = iter([{"id": "j", "status": "running"}, {"id": "j", "status": "cancelled", "error": None}])
    monkeypatch.setattr(Client, "request", lambda self, method, path, **kw: next(replies))
    with pytest.raises(JobCancelledError, match="Job j was cancelled") as c:
        fs.evaluate("empty", poll=0)
    assert c.value.job_id == "j"
    replies = iter([{"id": "j", "status": "done", "error": None, "result": None, "expired": True}])
    with pytest.raises(JobResultExpiredError, match="no longer keeps its result"):
        fs.evaluate("empty", poll=0)
    replies = iter([{"id": "j", "status": "failed", "error": "boom", "problem": {"code": "INTERNAL"}}])
    with pytest.raises(JobFailedError, match="INTERNAL: job j failed: boom") as f:
        fs.evaluate("empty", poll=0)
    assert (f.value.code, f.value.detail) == ("INTERNAL", "boom")


class _Recorder(http.server.BaseHTTPRequestHandler):
    """Answers every request with ``reply`` (status, headers, body), noting each one's method and
    Authorization header in ``seen``."""

    reply: ClassVar[tuple[int, dict[str, str], bytes]] = (200, {}, b"")
    seen: ClassVar[list[tuple[str, str | None]]] = []

    def _answer(self) -> None:
        type(self).seen.append((self.command, self.headers.get("Authorization")))
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        status, headers, body = self.reply
        self.send_response(status)
        for k, v in {**headers, "Content-Length": str(len(body))}.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _answer  # http.server's names

    def log_message(self, *_args) -> None:  # pylint: disable=arguments-differ  # quiet
        pass


def _recorder(reply):
    """A local HTTP server answering ``reply``; returns (its URL, the requests it saw, a stop function)."""
    seen: list[tuple[str, str | None]] = []
    handler = type("Handler", (_Recorder,), {"reply": reply, "seen": seen})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", seen, srv.shutdown


def test_redirects_are_refused_so_the_token_stays():
    """A redirect raises ApiError naming the target: the token isn't sent on to it, and a POST
    isn't repeated there as a GET (which would return the task list as if a task were made)."""
    other, other_seen, stop_other = _recorder((200, {"Content-Type": "application/json"}, b"[]"))
    url, seen, stop = _recorder((301, {"Location": f"{other}/api/tasks"}, b""))
    try:
        for call in (lambda fs: fs.tasks(), lambda fs: fs.create_task("new")):
            with pytest.raises(ApiError, match=f"redirected to {other}/api/tasks") as e:
                call(Client(url, token="s3cret-token"))
            assert e.value.status == 301
        assert [m for m, _ in seen] == ["GET", "POST"]
        assert not other_seen
    finally:
        stop()
        stop_other()


def test_a_response_that_isnt_json_raises_api_error():
    """A 200 page that isn't JSON (an auth proxy's login page, say) raises ApiError, not JSONDecodeError."""
    url, _, stop = _recorder((200, {"Content-Type": "text/html"}, b"<html>sign in</html>"))
    try:
        with pytest.raises(ApiError, match="isn't JSON: '<html>sign in</html>'") as e:
            Client(url).health()
        assert e.value.status == 200
    finally:
        stop()


def test_files_that_cant_be_read_are_errors_not_exceptions(server, tmp_path):
    """An unreadable path among the images is listed in ``errors`` (or as an error result, in its
    place, by predict) while the others are sent; with none readable, ApiError 400."""
    fs = Client(server, token="secret-0123456789")
    fs.create_task("t", embedder="fake:color")
    good, missing = tmp_path / "r.png", tmp_path / "missing.png"
    good.write_bytes(png(RED, noise=15, seed=1))
    r = fs.add_examples("t", "red", [missing, good])
    assert (len(r["added"]), r["errors"][0]["file"]) == (1, str(missing))
    assert "No such file" in r["errors"][0]["error"]
    preds = fs.predict("t", [good, missing, ("q.png", png(RED, seed=3))])
    assert [p.get("label") for p in preds] == ["red", None, "red"]
    assert preds[1]["file"] == str(missing)
    assert fs.predict("t", [missing])[0]["file"] == str(missing)
    assert preds[1] == {"file": str(missing), "label": None, "labels": [], "top_label": None, "scores": {},
                        "confidence": None, "similarity": None, "rejected": False, "evidence": [],
                        "error": preds[1]["error"], "revision": None}  # every key, as the server's
    with pytest.raises(BadRequestError, match="No image could be read") as e:
        fs.add_to_inbox("t", [missing])
    assert e.value.status == 400
    with pytest.raises(FewshotError, match="Can't read"):
        fs.request("POST", "/tasks/t/inbox", files=[missing])


def test_a_batch_too_large_for_the_server_doesnt_lose_the_others(small_server):
    """An image larger than the server's request limit goes alone and is refused (413): that
    becomes an error, and the other images are still added."""
    fs = Client(small_server, token="secret-0123456789")
    fs.create_task("t", embedder="fake:color")
    r = fs.add_examples("t", "red", [("small.png", png(RED)), ("huge.png", b"x" * (100 * 1024))])
    assert (len(r["added"]), r["errors"][0]["file"]) == (1, None)


def test_names_that_arent_utf8_are_sent():
    """A file name that isn't UTF-8 (escaped as lone surrogates) goes into the form with U+FFFD."""
    body, _ = fewshot.client._multipart({"label": "caf\udce9"}, [("caf\udce9.png", b"x")])  # pylint: disable=protected-access
    assert 'filename="caf�.png"'.encode() in body
    assert "caf�\r\n".encode() in body
    assert fewshot.client._text("\ud800") == "?"  # pylint: disable=protected-access  # not from a file name


# ---- configuration, images, paging, jobs and types -------------------------------------------
def test_client_reads_url_and_token_from_the_environment(monkeypatch):
    """Client() uses $FEWSHOT_URL and $FEWSHOT_TOKEN; arguments win over them."""
    monkeypatch.delenv("FEWSHOT_URL", raising=False)
    monkeypatch.delenv("FEWSHOT_TOKEN", raising=False)
    assert (Client().url, Client().token) == ("http://127.0.0.1:8000", None)
    monkeypatch.setenv("FEWSHOT_URL", "http://example.test:9000/")
    monkeypatch.setenv("FEWSHOT_TOKEN", "from-env")
    assert (Client().url, Client().token) == ("http://example.test:9000", "from-env")
    assert (Client("http://other", token="given").url, Client(token="given").token) == ("http://other", "given")


def test_images_can_be_bytes_files_or_pil_images(server):
    """bytes, file objects and PIL images are sent as images; anything else is a TypeError
    naming its index before anything is sent."""
    import io  # pylint: disable=import-outside-toplevel

    from PIL import Image  # pylint: disable=import-outside-toplevel

    fs = Client(server, token="secret-0123456789")
    fs.create_task("kinds", embedder="fake:color")
    pil = Image.new("RGB", (8, 8), BLUE)
    r = fs.add_examples("kinds", "red", [png(RED, noise=15, seed=1), io.BytesIO(png(RED, noise=15, seed=2))])
    assert [e["source"] for e in r["added"]] == ["image-0", "image-1"]
    named = io.BytesIO(png(RED, noise=15, seed=3))
    named.name = "/somewhere/named.png"
    assert fs.add_examples("kinds", "red", [named])["added"][0]["source"] == "named.png"
    assert fs.add_examples("kinds", "blue", [pil])["added"][0]["source"] == "image-0.png"
    assert [p["label"] for p in fs.predict("kinds", [pil, png(RED, seed=9)])] == ["blue", "red"]
    with pytest.raises(TypeError, match=r"image 1: expected a path, bytes, file object, \(name, bytes\) or "
                                        "PIL.Image, got int"):
        fs.add_examples("kinds", "red", [png(RED), 3])  # type: ignore[list-item]
    with pytest.raises(TypeError, match=r"image 0: .* got NoneType"):
        fs.predict("kinds", iter([None]))  # type: ignore[arg-type]
    assert len(fs.task("kinds")["examples"]) == 4  # nothing was sent


def test_a_file_object_that_cant_be_read_is_an_error():
    """A file object whose read fails is listed with the others that can't be read."""
    class Broken:  # pylint: disable=too-few-public-methods
        """A file that can't be read."""

        name = "broken.png"

        def read(self):
            """Fail, as a disk error would."""
            raise OSError(5, "Input/output error")

    loaded, unread = fewshot.client._read([Broken(), b"x"])  # pylint: disable=protected-access
    assert (loaded, unread) == ([("image-1", b"x")], [(0, "broken.png", "Input/output error")])
    assert [len(b) for b in _batches([b"xx", bytearray(b"yy"), Broken()], max_bytes=3)] == [1, 2]


def test_paging_helpers(server):
    """inbox() and examples() give one page; iter_inbox() and iter_examples() walk them all."""
    fs = Client(server, token="secret-0123456789")
    fs.create_task("paged", embedder="fake:color")
    fs.add_examples("paged", "red", [png(RED, noise=15, seed=i) for i in range(5)])
    fs.add_examples("paged", "blue", [png(BLUE, noise=15, seed=i) for i in range(2)], new_class=True)
    fs.add_to_inbox("paged", [png(RED, noise=15, seed=50 + i) for i in range(5)])
    page = fs.inbox("paged", "added", offset=1, limit=2)
    assert (page["total"], page["offset"], len(page["items"])) == (5, 1, 2)
    assert len(list(fs.iter_inbox("paged", "added", page_size=2))) == 5
    assert fs.examples("paged", label="red", limit=2)["total"] == 5
    assert [e["label"] for e in fs.iter_examples("paged", page_size=3)] == ["red"] * 5 + ["blue"] * 2
    assert len(list(fs.iter_examples("paged", label="blue", page_size=1))) == 2
    assert not list(fs.iter_examples("paged", tag="none"))
    with pytest.raises(NotFoundError, match="did you mean 'red'"):
        fs.examples("paged", label="Red")


def test_jobs(server):
    """start_evaluate and start_compare give Job handles to wait for, poll, cancel or come back to."""
    fs = Client(server, token="secret-0123456789")
    fs.create_task("jobs", embedder="fake:color")
    fs.add_examples("jobs", "red", [png(RED, noise=15, seed=i) for i in range(3)])
    fs.add_examples("jobs", "blue", [png(BLUE, noise=15, seed=i) for i in range(3)])
    job = fs.start_evaluate("jobs", folds=3)
    seen: list[str] = []
    report = job.wait(timeout=30, poll=0.01, on_progress=lambda info: seen.append(info["status"]))
    assert (report["method"], job.status(), job.result()["accuracy"]) == ("3-fold", "done", 1.0)
    assert set(seen) <= {"running"}
    assert fs.job(job.id).result() == report
    assert fs.evaluate("jobs", None, 0.01, timeout=30)["n_examples"] == 6  # poll stays positional, as it was
    assert fs.compare("jobs", ["fake:color"], 2, 0.01)["best"] == "fake:color"
    assert fs.start_compare("jobs", ["fake:color"]).wait(poll=0.01)["results"][0]["embedder"] == "fake:color"
    with pytest.raises(NotFoundError, match="JOB_NOT_FOUND"):
        fs.job("nope")


def test_job_timeouts_cancels_and_interrupts(monkeypatch):
    """A job still running after the timeout raises APITimeoutError with its id; cancel() asks
    it to stop; Ctrl-C while waiting says how to come back to it."""
    running = {"id": "j", "status": "running", "stage": "embedding", "done": 1, "total": 2}
    fs = Client("http://unused")
    monkeypatch.setattr(Client, "request", lambda self, method, path, **kw: (
        {**running, "status": "cancelling"} if method == "DELETE" else running))
    job = fs.start_evaluate("t")
    with pytest.raises(fewshot.client.APITimeoutError, match=r"Job j is still running after 0\.05 s") as e:
        job.wait(timeout=0.05, poll=0.01)
    assert (e.value.job_id, e.value.url) == ("j", "http://unused")
    with pytest.raises(fewshot.client.JobError, match="still running; wait"):
        job.result()
    assert job.cancel()["status"] == "cancelling"

    def interrupted(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr("fewshot.client.time.sleep", interrupted)
    with pytest.raises(KeyboardInterrupt, match=r"Stopped waiting for job j.*Client.job\('j'\).wait\(\)"):
        fs.evaluate("t")


def test_request_timeouts_and_rate_limits(monkeypatch):
    """A server that doesn't answer in time raises APITimeoutError; a 429 says when to retry."""
    import io  # pylint: disable=import-outside-toplevel
    from urllib.error import HTTPError, URLError  # pylint: disable=import-outside-toplevel

    def slow(*_a, **_k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(fewshot.client._OPENER, "open", slow)  # pylint: disable=protected-access
    with pytest.raises(fewshot.client.APITimeoutError, match="didn't answer GET /health within 300 s"):
        Client().health()

    def wrapped(*_a, **_k):
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr(fewshot.client._OPENER, "open", wrapped)  # pylint: disable=protected-access
    with pytest.raises(fewshot.client.APITimeoutError):
        Client().health()

    def limited(*_a, headers="5", **_k):
        raise HTTPError("http://x", 429, "Too Many", {"Retry-After": headers},  # type: ignore[arg-type]
                        io.BytesIO(b'{"code": "TOO_MANY_JOBS", "detail": "4 jobs"}'))

    monkeypatch.setattr(fewshot.client._OPENER, "open", limited)  # pylint: disable=protected-access
    with pytest.raises(fewshot.client.RateLimitError, match="TOO_MANY_JOBS: 4 jobs") as e:
        Client().health()
    assert e.value.retry_after == 5.0
    monkeypatch.setattr(fewshot.client._OPENER, "open", lambda *a, **k: limited(headers="soon"))  # pylint: disable=protected-access
    with pytest.raises(fewshot.client.RateLimitError) as e:
        Client().health()
    assert e.value.retry_after is None


def test_an_older_servers_422_list_becomes_the_detail(monkeypatch):
    """A 422 whose detail is a list (FastAPI's own) is still one readable string."""
    import io  # pylint: disable=import-outside-toplevel
    from urllib.error import HTTPError  # pylint: disable=import-outside-toplevel

    def old(*_a, **_k):
        raise HTTPError("http://x", 422, "Unprocessable", {},  # type: ignore[arg-type]
                        io.BytesIO(b'{"detail": [{"loc": ["body", "x"], "msg": "bad"}]}'))

    monkeypatch.setattr(fewshot.client._OPENER, "open", old)  # pylint: disable=protected-access
    with pytest.raises(ValidationError, match=r'HTTP_422: .*"msg": "bad"') as e:
        Client().health()
    assert e.value.errors == []


def test_if_match_refuses_a_change_to_a_task_changed_since(server):
    """With if_match, a change to a task no longer at that revision raises PreconditionFailedError."""
    fs = Client(server, token="secret-0123456789")
    rev = fs.create_task("etag", embedder="fake:color")["revision"]
    fs.add_examples("etag", "red", [png(RED, noise=15, seed=1)], if_match=rev)
    with pytest.raises(fewshot.client.PreconditionFailedError, match="PRECONDITION_FAILED") as e:
        fs.add_examples("etag", "red", [png(RED, noise=15, seed=2)], if_match=rev)
    assert e.value.fields["revision"] == rev + 1
    with pytest.raises(fewshot.client.PreconditionFailedError):
        fs.delete_task("etag", if_match=rev)
    fs.delete_task("etag", if_match=rev + 1)


def test_unknown_response_fields_are_kept(monkeypatch):
    """A newer server's extra fields don't break the client: the dict has them too."""
    reply = {"status": "ok", "max_request_bytes": 1000, "added_in_api_v2": {"x": 1}}
    monkeypatch.setattr(Client, "request", lambda self, method, path, **kw: reply)
    fs = Client()
    assert fs.health()["added_in_api_v2"] == {"x": 1}  # type: ignore[typeddict-item]
    assert fs.batch_bytes() == 500


def test_typed_dicts_mirror_the_schemas():
    """Each TypedDict the client returns has exactly the fields of the API model it mirrors."""
    from fewshot import responses, schemas  # pylint: disable=import-outside-toplevel

    pairs = {"Health": "Health", "LastEvaluation": "LastEvaluation", "TaskSummary": "TaskSummary",
             "ClassInfo": "ClassOut", "Example": "ExampleOut", "InboxItem": "InboxItemOut",
             "TaskDetail": "TaskDetail", "TaskEdited": "TaskEdited", "ExamplePage": "ExamplePage",
             "FileError": "FileError",
             "MovedExample": "MovedExample", "AddResult": "AddResult", "InboxAdded": "InboxAdded",
             "Evidence": "Evidence", "Prediction": "PredictionOut", "InboxEntry": "InboxEntry",
             "InboxPage": "InboxPage", "DuplicatePair": "DuplicatePair", "ClassMetrics": "ClassMetrics",
             "HeldOutPrediction": "HeldOutPrediction", "ReliabilityBin": "ReliabilityBin",
             "Calibration": "Calibration", "EvalReport": "EvalReport", "CompareResult": "CompareResult",
             "CompareReport": "CompareReport", "JobInfo": "JobOut"}
    for typed, model in pairs.items():
        keys = set(getattr(fewshot.client, typed).__annotations__)
        assert keys == set(getattr(schemas, model).model_fields), typed
    assert set(responses.Problem.model_fields) >= {"type", "title", "status", "detail", "instance", "code"}
