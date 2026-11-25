"""Download images given by URL, with a size cap and (for the server) a public-address check."""

from __future__ import annotations

import http.client
import ipaddress
import socket
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


class _PinnedHTTP(http.client.HTTPConnection):
    """An HTTP connection to the checked public address of its host."""

    def connect(self) -> None:
        _pin(self)
        super().connect()


class _PinnedHTTPS(http.client.HTTPSConnection):
    """An HTTPS connection to the checked public address of its host."""

    def connect(self) -> None:
        _pin(self)
        super().connect()


class _PublicHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_PinnedHTTP, req)


class _PublicHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_PinnedHTTPS, req)


def fetch_url(url: str, max_bytes: int, timeout: float = 20.0, public_only: bool = False) -> bytes:
    """The body at ``url``; raises ``ValueError`` for a bad URL, an HTTP or network error, or a
    body over ``max_bytes`` (read no further). With ``public_only``, every connection (redirects
    too) goes to a public address, the very one that was checked, and no proxy is used."""
    if not is_url(url):
        raise ValueError(f"Not an http(s) URL: {url}")
    handlers: list[urllib.request.BaseHandler] = []
    if public_only:
        check_public(url)
        handlers += [urllib.request.ProxyHandler({}), _PublicHTTPHandler(), _PublicHTTPSHandler()]
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(req, timeout=timeout) as resp:
            data: bytes = resp.read(max_bytes + 1)
    except urllib.error.HTTPError as e:
        raise ValueError(f"Could not fetch {url}: HTTP {e.code} {e.reason}") from None
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise ValueError(f"Could not fetch {url}: {reason}") from None
    if len(data) > max_bytes:
        raise ValueError(f"{url} is larger than the {max_bytes / 1024 / 1024:g} MB limit")
    return data
