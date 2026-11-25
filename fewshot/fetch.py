"""Download images given by URL, with a size cap and (for the server) a public-address check."""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import IO, Any

USER_AGENT = "fewshot/0.1"


def is_url(s: str) -> bool:
    """Whether ``s`` looks like an http(s) URL rather than a file path."""
    return s.lower().startswith(("http://", "https://"))


def check_public(url: str) -> None:
    """Raise ``ValueError`` unless ``url`` is http(s) and its host resolves only to public
    addresses, so a server can't be made to fetch from its own network (SSRF)."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"Not an http(s) URL: {url}")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 80, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError) as e:
        raise ValueError(f"Can't resolve {parts.hostname}: {e}") from None
    for info in infos:
        addr = ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])
        if not addr.is_global:
            raise ValueError(f"{parts.hostname} is not a public address ({addr})")


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """Re-checks every redirect target, so a public URL can't bounce the request inward."""

    def redirect_request(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # urllib's signature
            self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: Any, newurl: str,
    ) -> urllib.request.Request | None:
        check_public(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_url(url: str, max_bytes: int, timeout: float = 20.0, public_only: bool = False) -> bytes:
    """The body at ``url``; raises ``ValueError`` for a bad URL, an HTTP or network error, or a
    body over ``max_bytes`` (read no further). With ``public_only``, the host and every redirect
    must resolve to public addresses (see ``check_public``)."""
    if not is_url(url):
        raise ValueError(f"Not an http(s) URL: {url}")
    handlers: list[urllib.request.BaseHandler] = []
    if public_only:
        check_public(url)
        handlers.append(_CheckedRedirects())
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
