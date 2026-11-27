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
from fewshot.client import BATCH_BYTES, ApiError, Client, _batches
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
    assert item["prediction"]["label"] == "red"
    assert len(fs.accept_inbox("my-pets", {item["id"]: "red"})["examples"]) == 5
    assert fs.evaluate("my-pets")["n_examples"] == 5
    assert [t["name"] for t in fs.tasks()] == ["my-pets"]
    fs.delete_task("my-pets")
    assert fs.tasks() == []


def test_client_errors(server):
    """Error responses raise ApiError with the status and the server's detail."""
    with pytest.raises(ApiError) as e:
        Client(server, token="wrong").tasks()
    assert e.value.status == 401
    with pytest.raises(ApiError, match="HTTP 404") as e:
        Client(server, token="secret-0123456789").task("nope")
    assert "nope" in e.value.detail


def test_non_json_error_body(monkeypatch):
    """An error body that isn't JSON (a proxy's HTML page, say) becomes the detail as text."""
    import io  # pylint: disable=import-outside-toplevel
    from urllib.error import HTTPError  # pylint: disable=import-outside-toplevel

    def fail(*_a, **_k):
        raise HTTPError("http://x", 502, "Bad Gateway", {}, io.BytesIO(b"<html>bad gateway</html>"))  # type: ignore[arg-type]

    monkeypatch.setattr(fewshot.client._OPENER, "open", fail)  # pylint: disable=protected-access
    with pytest.raises(ApiError, match="HTTP 502: <html>bad gateway</html>"):
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
    assert len(fs.add_to_inbox("batched", [(f"u{i}.png", png(RED, seed=20 + i)) for i in range(3)])["added"]) == 3
    with pytest.raises(ValueError, match="No images given"):
        fs.add_to_inbox("batched", [])
    fs.delete_task("batched")


def test_a_refused_batch_doesnt_lose_the_others(server, monkeypatch):
    """A batch the server refuses (nothing in it could be added) becomes an error; only if every
    batch is refused does the call raise, and other errors still raise at once."""
    monkeypatch.setattr("fewshot.client.BATCH", 2)
    fs = Client(server, token="secret-0123456789")
    fs.create_task("refused", embedder="fake:color")
    bad = [("x.png", b"junk"), ("y.png", b"junk")]
    r = fs.add_examples("refused", "red", [*bad, ("r.png", png(RED))])
    assert (len(r["added"]), r["errors"][0]["file"], len(r["task"]["examples"])) == (1, None, 1)
    assert "No images added" in r["errors"][0]["error"]
    r = fs.add_to_inbox("refused", [("r.png", png(RED)), ("r2.png", png(RED)), ("u.png", png(BLUE))])
    assert (len(r["added"]), len(r["errors"])) == (1, 1)  # the first batch is all examples already
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


def test_connection_failures_raise_api_error():
    """A refused or reset connection raises ApiError (status 0), like an error response."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(ApiError, match=r"Connection to .* failed") as e:
        Client(f"http://127.0.0.1:{port}", timeout=5).health()
    assert e.value.status == 0


def test_a_reset_upload_hints_at_the_size_limit(monkeypatch):
    """A connection dropped during a large upload (as a server refusing it does) says so."""
    def reset(*_a, **_k):
        raise ConnectionResetError(104, "Connection reset by peer")

    monkeypatch.setattr(fewshot.client._OPENER, "open", reset)  # pylint: disable=protected-access
    monkeypatch.setattr(Client, "batch_bytes", lambda self: 10**9)
    with pytest.raises(ApiError, match="larger than the server accepts"):
        Client().predict("t", [("a.png", b"x" * (2 * 1024 * 1024))])


def test_evaluate_runs_as_a_job(server, monkeypatch):
    """evaluate polls a background job; a failed or cancelled one raises ApiError."""
    fs = Client(server, token="secret-0123456789")
    fs.create_task("empty", embedder="fake:color")
    with pytest.raises(ApiError, match="no examples") as e:
        fs.evaluate("empty", poll=0.01)
    assert e.value.status == 400
    replies = iter([{"id": "j", "status": "running"}, {"id": "j", "status": "cancelled", "error": None}])
    monkeypatch.setattr(Client, "request", lambda self, method, path, **kw: next(replies))
    with pytest.raises(ApiError, match="Evaluation cancelled") as e:
        fs.evaluate("empty", poll=0)
    assert e.value.status == 409
    replies = iter([{"id": "j", "status": "done", "error": None, "result": None, "expired": True}])
    with pytest.raises(ApiError, match="no longer keeps its report") as e:
        fs.evaluate("empty", poll=0)
    assert e.value.status == 410


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
    with pytest.raises(ApiError, match="No image could be read") as e:
        fs.add_to_inbox("t", [missing])
    assert e.value.status == 400
    with pytest.raises(ApiError, match="Can't read") as e:
        fs.request("POST", "/tasks/t/inbox", files=[missing])
    assert e.value.status == 0


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
