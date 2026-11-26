"""Download images given by URL, with a size cap and (for the server) a public-address check."""

from __future__ import annotations

import contextlib
import functools
import http.client
import ipaddress
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "fewshot/0.1"


def is_url(s: str) -> bool:
    """Whether ``s`` looks like an http(s) URL rather than a file path."""
    return s.lower().startswith(("http://", "https://"))


def public_addresses(host: str, port: int) -> list[str]:
    """The addresses ``host`` resolves to; ``ValueError`` unless every one is public, so a server
    can't be made to fetch from its own network (SSRF)."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError) as e:
        raise ValueError(f"Can't resolve {host}: {e}") from None
    addrs = []
    for info in infos:
        addr = ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])
        if not addr.is_global or not _embedded_ipv4_public(addr):
            raise ValueError(f"{host} is not a public address ({addr})")
        addrs.append(str(addr))
    return addrs


NAT64 = ipaddress.ip_network("64:ff9b::/96")  # IPv6 addresses that a NAT64 gateway maps to IPv4


def _embedded_ipv4_public(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """For an IPv6 address carrying an IPv4 one (NAT64, IPv4-mapped), whether that IPv4 address
    is public too: 64:ff9b::7f00:1 reaches 127.0.0.1 through a NAT64 gateway."""
    if isinstance(addr, ipaddress.IPv4Address):
        return True
    inner = addr.ipv4_mapped or (ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF) if addr in NAT64 else None)
    return inner is None or inner.is_global


def check_public(url: str) -> None:
    """Raise ``ValueError`` unless ``url`` is http(s) and its host resolves only to public addresses."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"Not an http(s) URL: {url}")
    public_addresses(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))


def _pin(conn: http.client.HTTPConnection) -> None:
    """Make ``conn`` connect to an address checked by ``public_addresses``, looked up once here:
    checking one lookup and connecting after another would let DNS rebinding point the request
    inward. The Host header and the TLS server name stay the URL's host name."""
    addr = public_addresses(conn.host, conn.port)[0]

    def connect(_address: Any, timeout: Any = None, source_address: Any = None) -> socket.socket:
        return socket.create_connection((addr, conn.port), timeout, source_address)

    conn._create_connection = connect  # type: ignore[attr-defined]  # pylint: disable=protected-access


class _Deadline:
    """Shuts down every socket of a download once its time is up, so a server trickling headers
    or body can't hold the caller: blocked reads return at once, whatever phase they are in."""

    def __init__(self, seconds: float) -> None:
        self.expired = False
        self._socks: list[socket.socket] = []
        self._lock = threading.Lock()
        self._timer = threading.Timer(seconds, self._expire)
        self._timer.daemon = True

    def __enter__(self) -> _Deadline:
        self._timer.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._timer.cancel()

    def watch(self, sock: socket.socket) -> None:
        """Shut ``sock`` down at the deadline (at once if it has passed)."""
        with self._lock:
            self._socks.append(sock)
            if self.expired:
                self._shut(sock)

    def _expire(self) -> None:
        with self._lock:
            self.expired = True
            for sock in self._socks:
                self._shut(sock)

    @staticmethod
    def _shut(sock: socket.socket) -> None:
        # socket.socket's own shutdown: ssl.SSLSocket's drops its TLS state under a concurrent read.
        with contextlib.suppress(OSError):
            socket.socket.shutdown(sock, socket.SHUT_RDWR)


class _TimedConnection(http.client.HTTPConnection):
    """A connection whose socket the download's ``_Deadline`` shuts down; with ``public`` it goes
    to the checked public address of its host."""

    def __init__(self, *args: Any, deadline: _Deadline, public: bool, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._deadline, self._public = deadline, public

    def connect(self) -> None:
        if self._public:
            _pin(self)
        super().connect()
        self._deadline.watch(self.sock)


class _TimedHTTPS(_TimedConnection, http.client.HTTPSConnection):
    """The HTTPS kind of ``_TimedConnection``."""


class _TimedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, deadline: _Deadline, public: bool) -> None:
        super().__init__()
        self._conn = functools.partial(_TimedConnection, deadline=deadline, public=public)

    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(self._conn, req)


class _TimedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, deadline: _Deadline, public: bool) -> None:
        super().__init__()
        self._conn = functools.partial(_TimedHTTPS, deadline=deadline, public=public)

    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(self._conn, req, context=self._context)  # type: ignore[attr-defined]


class _HTTPRedirectOnly(urllib.request.HTTPRedirectHandler):
    """Follow redirects only to http(s) URLs: urllib also follows ``ftp://``, whose connection
    no address check covers."""

    def redirect_request(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # urllib's signature
            self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str,
    ) -> urllib.request.Request | None:
        if not is_url(newurl):
            raise urllib.error.URLError(f"redirected to a URL that isn't http(s): {newurl[:200]}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


CHUNK = 64 * 1024


def fetch_url(url: str, max_bytes: int, timeout: float = 20.0, public_only: bool = False) -> bytes:
    """The body at ``url``; raises ``ValueError`` for a bad URL, an HTTP or network error, or a
    body over ``max_bytes`` (read no further). ``timeout`` bounds the whole download (headers too), not just
    each read, so a server sending a byte at a time can't hold the caller. Redirects are followed
    to http(s) URLs only. With ``public_only``, every connection (redirects too) goes to a public
    address, the very one that was checked, and no proxy is used."""
    if not is_url(url):
        raise ValueError(f"Not an http(s) URL: {url}")
    handlers: list[urllib.request.BaseHandler] = [_HTTPRedirectOnly()]
    if public_only:
        check_public(url)
        handlers.append(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _Deadline(timeout) as deadline:
        handlers += [_TimedHTTPHandler(deadline, public_only), _TimedHTTPSHandler(deadline, public_only)]
        try:
            with urllib.request.build_opener(*handlers).open(req, timeout=timeout) as resp:
                data = bytearray()
                while len(data) <= max_bytes and (chunk := resp.read1(CHUNK)):
                    data += chunk
        except urllib.error.HTTPError as e:
            raise ValueError(f"Could not fetch {url}: HTTP {e.code} {e.reason}") from None
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            if deadline.expired:
                raise ValueError(f"Could not fetch {url}: took longer than {timeout:g} s") from None
            reason = getattr(e, "reason", e)
            raise ValueError(f"Could not fetch {url}: {reason}") from None
    if deadline.expired:  # the shut-down socket ended the body early
        raise ValueError(f"Could not fetch {url}: took longer than {timeout:g} s")
    if len(data) > max_bytes:
        raise ValueError(f"{url} is larger than the {max_bytes / 1024 / 1024:g} MB limit")
    return bytes(data)
