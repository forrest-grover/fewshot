"""Fetching images from URLs (fewshot.fetch): size limits, public addresses only, pinned
connections, redirects and timeouts, against local HTTP servers."""

import http.server
import socket
import threading
import time

import pytest

from fewshot import fetch
from tests.conftest import RED, png


# ---- fetching ------------------------------------------------------------
class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves /img, /redirect and 404s."""

    def do_GET(self):  # pylint: disable=invalid-name  # http.server's naming
        """Answer one GET."""
        if self.path == "/img":
            body = png(RED)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/img")
            self.end_headers()
        else:
            self.send_error(404, "Not Found")

    def log_message(self, *_args):
        pass


@pytest.fixture(name="server")
def fixture_server():
    """A local HTTP server: /img is a PNG, /redirect goes to /img, anything else is 404."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_fetch_url(server):
    """Bodies are fetched up to a size limit; HTTP and network errors become ValueErrors."""
    assert fetch.fetch_url(server + "/img", 10_000) == png(RED)
    assert fetch.fetch_url(server + "/redirect", 10_000) == png(RED)
    with pytest.raises(ValueError, match="larger than"):
        fetch.fetch_url(server + "/img", 10)
    with pytest.raises(ValueError, match="HTTP 404"):
        fetch.fetch_url(server + "/nope", 10_000)
    with pytest.raises(ValueError, match="Not an http"):
        fetch.fetch_url("file:///etc/passwd", 10_000)
    with pytest.raises(ValueError, match="Could not fetch"):
        fetch.fetch_url("http://127.0.0.1:1/x", 10_000)


def test_fetch_public_only(server, monkeypatch):
    """public_only refuses private hosts, and redirects are checked too."""
    with pytest.raises(ValueError, match="not a public address"):
        fetch.fetch_url(server + "/img", 10_000, public_only=True)
    real_check = fetch.check_public
    monkeypatch.setattr(fetch, "check_public", lambda url: None if url.endswith("/redirect") else real_check(url))
    with pytest.raises(ValueError, match="not a public address"):
        fetch.fetch_url(server + "/redirect", 10_000, public_only=True)


def test_check_public(monkeypatch):
    """Only http(s) URLs whose host resolves to public addresses pass."""
    real = socket.getaddrinfo

    def no_dns(host, *args, **kw):  # IP literals resolve without DNS; names fail at once
        if host[0].isdigit() or ":" in host:
            return real(host, *args, **kw)
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    fetch.check_public("http://8.8.8.8/x.png")
    for url, msg in (("ftp://8.8.8.8/x", "Not an http"), ("http:///x", "Not an http"),
                     ("http://10.0.0.1/x", "not a public"), ("http://[::1]/x", "not a public"),
                     ("http://no-such-host.invalid/x", "Can't resolve")):
        with pytest.raises(ValueError, match=msg):
            fetch.check_public(url)
    assert fetch.is_url("HTTPS://x")
    assert not fetch.is_url("/tmp/x.png")


def test_nat64_and_mapped_private_addresses_are_refused(monkeypatch):
    """An IPv6 address that reaches a private IPv4 one through NAT64 is not public."""
    def answer(ip):
        return lambda host, port, *a, **k: [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, port, 0, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", answer("64:ff9b::7f00:1"))
    with pytest.raises(ValueError, match="not a public address"):
        fetch.public_addresses("nat64.test", 80)
    monkeypatch.setattr(socket, "getaddrinfo", answer("64:ff9b::808:808"))  # 8.8.8.8
    assert fetch.public_addresses("nat64.test", 80) == ["64:ff9b::808:808"]


# ---- pinned connections --------------------------------------------------
def test_fetch_connects_to_the_checked_address(server, monkeypatch):
    """Each connection goes to the address that passed the check, looked up once: the host name
    itself is never resolved again (here it can't be resolved at all), and redirects are pinned too."""
    port = int(server.rsplit(":", 1)[1])
    looked_up = []

    def checked(host, _port):
        looked_up.append(host)
        return ["127.0.0.1"]  # pretend this passed the public-address check

    monkeypatch.setattr(fetch, "public_addresses", checked)
    url = f"http://pinned.invalid:{port}"
    assert fetch.fetch_url(url + "/redirect", 10_000, public_only=True) == png(RED)
    assert looked_up == ["pinned.invalid"] * 3  # the check, the first connection, the redirect's


def test_fetch_refuses_dns_rebinding(server, monkeypatch):
    """A host that answers the check with a public address and then with a private one is refused."""
    port = int(server.rsplit(":", 1)[1])
    answers = iter(["93.184.216.34", "127.0.0.1"])

    def rebinding(_host, p, *_args, **_kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), p))]

    monkeypatch.setattr(socket, "getaddrinfo", rebinding)
    with pytest.raises(ValueError, match="not a public address"):
        fetch.fetch_url(f"http://rebind.test:{port}/img", 10_000, public_only=True)


def test_fetch_https_is_pinned_too(server, monkeypatch):
    """HTTPS connections also go to the checked address (here a plain-HTTP server, so the TLS
    handshake fails after connecting to it)."""
    port = int(server.rsplit(":", 1)[1])
    looked_up: list[str] = []

    def checked(host, _port):
        looked_up.append(host)
        return ["127.0.0.1"]

    monkeypatch.setattr(fetch, "public_addresses", checked)
    with pytest.raises(ValueError, match="Could not fetch"):
        fetch.fetch_url(f"https://pinned.invalid:{port}/img", 10_000, public_only=True)
    assert looked_up == ["pinned.invalid", "pinned.invalid"]


# ---- redirects and timeouts ----------------------------------------------
class _Hostile(http.server.BaseHTTPRequestHandler):
    """/ftp redirects to an ftp:// URL; /status and /headers trickle the status line and headers;
    others send a body a byte at a time."""

    def do_GET(self):  # pylint: disable=invalid-name  # http.server's naming
        """Answer as the path says."""
        if self.path == "/ftp":
            self.send_response(302)
            self.send_header("Location", "ftp://127.0.0.1:1/secret")
            self.end_headers()
            return
        if self.path == "/headers":  # a header line every 0.1 s, for 3 s
            self.send_response(200)
            self.flush_headers()
            for _ in range(30):
                self.wfile.write(b"X-Slow: 1\r\n")
                self.wfile.flush()
                time.sleep(0.1)
            return
        if self.path == "/status":  # the status line a byte every 0.1 s
            for c in b"HTTP/1.1 200 OK\r\n":
                self.wfile.write(bytes([c]))
                self.wfile.flush()
                time.sleep(0.1)
            return
        self.send_response(200)
        self.send_header("Content-Length", "20")
        self.end_headers()
        for _ in range(20):
            self.wfile.write(b"x")
            self.wfile.flush()
            time.sleep(0.1)

    def log_message(self, *_args):
        pass


@pytest.fixture(name="hostile")
def fixture_hostile():
    """The URL of a local server running ``_Hostile``."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hostile)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_redirects_only_to_http(hostile, monkeypatch):
    """A redirect to ftp:// (which no address check covers) is refused, with or without public_only."""
    with pytest.raises(ValueError, match="isn't http"):
        fetch.fetch_url(hostile + "/ftp", 10_000)
    monkeypatch.setattr(fetch, "public_addresses", lambda host, port: ["127.0.0.1"])
    with pytest.raises(ValueError, match="isn't http"):
        fetch.fetch_url(hostile + "/ftp", 10_000, public_only=True)


def test_timeout_bounds_the_whole_download(hostile):
    """A body sent a byte at a time fails once the timeout has passed, though no single read is slow."""
    t0 = time.monotonic()
    with pytest.raises(ValueError, match=r"took longer than 0\.5 s"):
        fetch.fetch_url(hostile + "/drip", 10_000, timeout=0.5)
    assert time.monotonic() - t0 < 1.5


@pytest.mark.parametrize("path", ["/status", "/headers"])
def test_timeout_bounds_the_headers_too(hostile, path):
    """A status line or headers sent a bit at a time can't hold the caller past the timeout either."""
    t0 = time.monotonic()
    with pytest.raises(ValueError, match=r"took longer than 0\.5 s"):
        fetch.fetch_url(hostile + path, 10_000, timeout=0.5)
    assert time.monotonic() - t0 < 1.5


def test_deadline_shuts_late_sockets_at_once():
    """A connection made after the deadline (a redirect, say) is shut down as soon as it is watched."""
    a, b = socket.socketpair()
    with a, b, fetch._Deadline(0) as deadline:  # pylint: disable=protected-access
        while not deadline.expired:
            time.sleep(0.01)
        deadline.watch(a)
        assert a.recv(1) == b""
