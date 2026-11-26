"""Download images given by URL, with a size cap and (for the server) a public-address check."""

from __future__ import annotations

import contextlib
import functools
import http.client
import ipaddress
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "fewshot/0.1"


def is_url(s: str) -> bool:
    """Whether ``s`` looks like an http(s) URL rather than a file path."""
    return s.lower().startswith(("http://", "https://"))


def _resolve(host: str, port: int, timeout: float | None) -> list[Any]:
    """``socket.getaddrinfo`` for TCP, giving up (``ValueError``) after ``timeout`` seconds: the
    system resolver has no timeout of its own to set, and may wait far longer. The lookup runs in
    a thread of its own, which is left to finish by itself if it takes too long."""
    if timeout is None:
        return socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    found: list[Any] = []
    failed: list[BaseException] = []
    done = threading.Event()

    def look_up() -> None:
        try:
            found.extend(socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP))
        except BaseException as e:  # pylint: disable=broad-exception-caught  # raised in the caller's thread
            failed.append(e)
        finally:
            done.set()

    threading.Thread(target=look_up, name="fewshot-dns", daemon=True).start()
    if not done.wait(timeout):
        raise ValueError(f"Can't resolve {host}: no answer within {timeout:g} s")
    if failed:
        raise failed[0]
    return found


def public_addresses(host: str, port: int, timeout: float | None = None) -> list[str]:
    """The addresses ``host`` resolves to (looked up for at most ``timeout`` seconds);
    ``ValueError`` unless every one is public, so a server can't be made to fetch from its own
    network (SSRF)."""
    try:
        infos = _resolve(host, port, timeout)
    except (socket.gaierror, UnicodeError) as e:
        raise ValueError(f"Can't resolve {host}: {e}") from None
    addrs = []
    for info in infos:
        addr = ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])
        if not addr.is_global or not _embedded_ipv4_public(addr):
            raise ValueError(f"{host} is not a public address ({addr})")
        addrs.append(str(addr))
    return addrs


# IPv6 addresses whose last 32 bits are an IPv4 address that a gateway, or the host's own stack,
# may deliver to: NAT64, SIIT (IPv4-translated, ::ffff:0:a.b.c.d) and the deprecated
# IPv4-compatible ::a.b.c.d. IPv4-mapped ones (::ffff:a.b.c.d) are ``ipv4_mapped``.
CARRY_IPV4 = tuple(ipaddress.ip_network(n) for n in ("64:ff9b::/96", "::ffff:0:0:0/96", "::/96"))
SITE_LOCAL = ipaddress.ip_network("fec0::/10")  # deprecated, but still routed within a site; is_global passes it


def _embedded_ipv4_public(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """For an IPv6 address, false if it is site-local or carries an IPv4 address (see
    ``CARRY_IPV4``) that isn't public: 64:ff9b::7f00:1 reaches 127.0.0.1 through a NAT64 gateway,
    and ::7f00:1 may reach it on the host itself."""
    if isinstance(addr, ipaddress.IPv4Address):
        return True
    if addr in SITE_LOCAL:
        return False
    carried = any(addr in n for n in CARRY_IPV4)
    inner = addr.ipv4_mapped or (ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF) if carried else None)
    return inner is None or inner.is_global


def check_public(url: str, timeout: float | None = None) -> None:
    """Raise ``ValueError`` unless ``url`` is http(s) and its host resolves (within ``timeout``
    seconds) only to public addresses."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"Not an http(s) URL: {url}")
    public_addresses(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80), timeout)


def _pin(conn: http.client.HTTPConnection, timeout: float | None) -> None:
    """Make ``conn`` connect to an address checked by ``public_addresses`` (looked up for at most
    ``timeout`` seconds), looked up once here: checking one lookup and connecting after another
    would let DNS rebinding point the request inward. The Host header and the TLS server name
    stay the URL's host name."""
    addr = public_addresses(conn.host, conn.port, timeout)[0]

    def connect(_address: Any, timeout: Any = None, source_address: Any = None) -> socket.socket:
        return socket.create_connection((addr, conn.port), timeout, source_address)

    conn._create_connection = connect  # type: ignore[attr-defined]  # pylint: disable=protected-access


class _Deadline:
    """Shuts down every socket of a download once its time is up, so a server trickling headers
    or body can't hold the caller: blocked reads return at once, whatever phase they are in."""

    def __init__(self, seconds: float) -> None:
        self.expired = False
        self._end = time.monotonic() + seconds
        self._socks: list[socket.socket] = []
        self._lock = threading.Lock()
        self._timer = threading.Timer(seconds, self._expire)
        self._timer.daemon = True

    def __enter__(self) -> _Deadline:
        self._timer.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._timer.cancel()

    def remaining(self) -> float:
        """Seconds left until the deadline (0 once it has passed)."""
        return max(0.0, self._end - time.monotonic())

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
            _pin(self, self._deadline.remaining())
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
    body over ``max_bytes`` (read no further). ``timeout`` bounds the whole download (host name
    lookups and headers too), not just each read, so a server sending a byte at a time, or a slow
    DNS server, can't hold the caller. Redirects are followed
    to http(s) URLs only. With ``public_only``, every connection (redirects too) goes to a public
    address, the very one that was checked, and no proxy is used."""
    if not is_url(url):
        raise ValueError(f"Not an http(s) URL: {url}")
    handlers: list[urllib.request.BaseHandler] = [_HTTPRedirectOnly()]
    if public_only:
        handlers.append(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _Deadline(timeout) as deadline:
        if public_only:  # host name lookups count towards the timeout too
            check_public(url, deadline.remaining())
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
