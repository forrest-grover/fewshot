"""Fetching images from URLs (fewshot.fetch): size limits, public addresses only, pinned
connections, redirects and timeouts, against local HTTP servers."""

import http.server
import itertools
import socket
import threading
import time

import pytest

from fewshot import fetch
from tests.conftest import RED, png, serve_http


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
    with serve_http(_Handler) as url:
        yield url


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
    monkeypatch.setattr(fetch, "check_public", lambda url, timeout=None: None if url.endswith("/redirect") else real_check(url, timeout))
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


def _resolves_to(monkeypatch, ip):
    """Make every host name resolve to ``ip`` alone."""
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    sockaddr = (ip, 80, 0, 0) if family == socket.AF_INET6 else (ip, 80)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_k: [(family, socket.SOCK_STREAM, 6, "", sockaddr)])


@pytest.mark.parametrize("ip", [
    "10.0.0.1", "::1",  # private, loopback
    "100.64.0.1",  # carrier-grade NAT (shared address space): neither private nor loopback, but not public
    "169.254.169.254",  # link-local: cloud metadata services
    "0.0.0.0", "::",  # "this host": connecting to it reaches the local machine
    "192.0.0.1",  # IETF protocol assignments
    "198.18.0.1",  # benchmarking
    "240.0.0.1",  # reserved
    "fc00::1",  # unique local
    "fe80::1",  # link-local
    "2001:db8::1",  # documentation
    "::ffff:127.0.0.1",  # IPv4-mapped loopback
    "::ffff:10.0.0.1",  # IPv4-mapped private
    "64:ff9b::7f00:1",  # NAT64 of 127.0.0.1
    "64:ff9b::a9fe:a9fe",  # NAT64 of 169.254.169.254
    "::7f00:1",  # IPv4-compatible 127.0.0.1 (deprecated, but a host may still deliver it)
    "::ffff:0:7f00:1",  # SIIT (IPv4-translated) 127.0.0.1
    "::ffff:0:a9fe:a9fe",  # SIIT 169.254.169.254
    "fec0::1",  # site-local (deprecated, but routed within a site)
])
def test_every_kind_of_non_public_address_is_refused(monkeypatch, ip):
    """Anything but a globally routable address is refused, not just private and loopback ones."""
    _resolves_to(monkeypatch, ip)
    with pytest.raises(ValueError, match="not a public address"):
        fetch.public_addresses("inside.test", 80)


@pytest.mark.parametrize("ip", ["8.8.8.8", "2606:4700:4700::1111", "::ffff:8.8.8.8", "64:ff9b::808:808",
                                "::ffff:0:808:808"])
def test_public_addresses_pass(monkeypatch, ip):
    """Globally routable addresses, also IPv4-mapped or through NAT64, pass."""
    _resolves_to(monkeypatch, ip)
    assert fetch.public_addresses("outside.test", 80) == [ip]


# ---- pinned connections --------------------------------------------------
def test_fetch_connects_to_the_checked_address(server, monkeypatch):
    """Each connection goes to the address that passed the check, looked up once: the host name
    itself is never resolved again (here it can't be resolved at all), and redirects are pinned too."""
    port = int(server.rsplit(":", 1)[1])
    looked_up = []

    def checked(host, _port, _timeout=None):
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

    def checked(host, _port, _timeout=None):
        looked_up.append(host)
        return ["127.0.0.1"]

    monkeypatch.setattr(fetch, "public_addresses", checked)
    with pytest.raises(ValueError, match="Could not fetch"):
        fetch.fetch_url(f"https://pinned.invalid:{port}/img", 10_000, public_only=True)
    assert looked_up == ["pinned.invalid", "pinned.invalid"]


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_public_only_fetches_ignore_proxy_settings(server, monkeypatch, scheme):
    """With public_only no proxy is used, even when HTTP(S)_PROXY name one: a proxy would make the
    connection, to wherever the checked host name resolves for it. Here the proxy is a dead port,
    so a fetch through it would fail."""
    port = int(server.rsplit(":", 1)[1])
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:1")
    connected_for: list[str] = []

    def checked(host, _port, _timeout=None):
        connected_for.append(host)
        return ["127.0.0.1"]

    monkeypatch.setattr(fetch, "public_addresses", checked)
    if scheme == "http":
        assert fetch.fetch_url(f"http://pinned.invalid:{port}/img", 10_000, public_only=True) == png(RED)
    else:  # a plain-HTTP server: the TLS handshake fails, after connecting straight to it
        with pytest.raises(ValueError, match="Could not fetch"):
            fetch.fetch_url(f"https://pinned.invalid:{port}/img", 10_000, public_only=True)
    assert connected_for == ["pinned.invalid", "pinned.invalid"]  # the check, then the connection: no proxy


# ---- redirects and timeouts ----------------------------------------------
# The slow responses send a little every DRIP seconds, so no single read is slow, until the test
# ends or TRICKLE_FOR passes: only the client's deadline can end the request sooner. TIMEOUT is far
# below TRICKLE_FOR, so a deadline that didn't hold shows as a request lasting much longer than it.
DRIP, TRICKLE_FOR, TIMEOUT = 0.05, 30.0, 0.5


class _Hostile(http.server.BaseHTTPRequestHandler):
    """/ftp redirects to an ftp:// URL; /status trickles the status line, /headers header lines,
    and others the body, a byte or a line at a time."""

    done = threading.Event()  # set when the test ends: stop trickling

    def _trickle(self, pieces):
        """Send ``pieces`` one per DRIP seconds, until done or TRICKLE_FOR has passed."""
        for piece in itertools.islice(pieces, int(TRICKLE_FOR / DRIP)):
            if self.done.wait(DRIP):
                return
            self.wfile.write(piece)
            self.wfile.flush()

    def do_GET(self):  # pylint: disable=invalid-name  # http.server's naming
        """Answer as the path says."""
        if self.path == "/ftp":
            self.send_response(302)
            self.send_header("Location", "ftp://127.0.0.1:1/secret")
            self.end_headers()
        elif self.path == "/headers":
            self.send_response(200)
            self.flush_headers()
            self._trickle(itertools.repeat(b"X-Slow: 1\r\n"))
        elif self.path == "/status":
            self._trickle(itertools.chain((bytes([c]) for c in b"HTTP/1.1 200 OK\r\n"), itertools.repeat(b"X: 1\r\n")))
        else:  # no Content-Length: the body lasts until the connection closes
            self.send_response(200)
            self.end_headers()
            self._trickle(itertools.repeat(b"x"))

    def log_message(self, *_args):
        pass


@pytest.fixture(name="hostile")
def fixture_hostile():
    """The URL of a local server running ``_Hostile``."""
    _Hostile.done.clear()
    with serve_http(_Hostile) as url:
        yield url
        _Hostile.done.set()


def test_redirects_only_to_http(hostile, monkeypatch):
    """A redirect to ftp:// (which no address check covers) is refused, with or without public_only."""
    with pytest.raises(ValueError, match="isn't http"):
        fetch.fetch_url(hostile + "/ftp", 10_000)
    monkeypatch.setattr(fetch, "public_addresses", lambda host, port, timeout=None: ["127.0.0.1"])
    with pytest.raises(ValueError, match="isn't http"):
        fetch.fetch_url(hostile + "/ftp", 10_000, public_only=True)


def test_timeout_bounds_the_whole_download(hostile):
    """A body sent a byte at a time fails once the timeout has passed, though no single read is slow."""
    t0 = time.monotonic()
    with pytest.raises(ValueError, match=r"took longer than 0\.5 s"):
        fetch.fetch_url(hostile + "/drip", 10_000, timeout=TIMEOUT)
    assert TIMEOUT <= time.monotonic() - t0 < TRICKLE_FOR / 3


@pytest.mark.parametrize("path", ["/status", "/headers"])
def test_timeout_bounds_the_headers_too(hostile, path):
    """A status line or headers sent a bit at a time can't hold the caller past the timeout either."""
    t0 = time.monotonic()
    with pytest.raises(ValueError, match=r"took longer than 0\.5 s"):
        fetch.fetch_url(hostile + path, 10_000, timeout=TIMEOUT)
    assert TIMEOUT <= time.monotonic() - t0 < TRICKLE_FOR / 3


@pytest.mark.parametrize("pinned", [False, True])
def test_timeout_bounds_host_name_lookups(monkeypatch, pinned):
    """A resolver that doesn't answer can't hold the caller past the timeout either: neither in
    the check before connecting nor in the lookup the connection is pinned to."""
    release = threading.Event()
    calls = []

    def slow(host, port, *_args, **_kw):
        calls.append(host)
        if pinned and len(calls) == 1:  # the check passes; the connection's own lookup hangs
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
        release.wait(10)
        raise socket.gaierror(-2, "too late")

    monkeypatch.setattr(socket, "getaddrinfo", slow)
    t0 = time.monotonic()
    try:
        with pytest.raises(ValueError, match=r"Can't resolve slow\.test: no answer within"):
            fetch.fetch_url("http://slow.test/x.png", 10_000, timeout=TIMEOUT, public_only=True)
        assert time.monotonic() - t0 < TIMEOUT + 1
    finally:
        release.set()
    assert len(calls) == (2 if pinned else 1)


def test_lookup_errors_reach_the_caller(monkeypatch):
    """A lookup that fails (with a timeout set) is reported as for one without."""
    def fail(*_a, **_k):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    with pytest.raises(ValueError, match="Can't resolve nowhere.test: .*not known"):
        fetch.public_addresses("nowhere.test", 80, timeout=5)


def test_deadline_shuts_late_sockets_at_once():
    """A connection made after the deadline (a redirect, say) is shut down as soon as it is watched."""
    a, b = socket.socketpair()
    with a, b, fetch._Deadline(0) as deadline:  # pylint: disable=protected-access
        while not deadline.expired:
            time.sleep(0.01)
        deadline.watch(a)
        assert a.recv(1) == b""
