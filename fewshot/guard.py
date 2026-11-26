"""ASGI pieces of the server's protection: body size limit, Host and cross-site checks."""

from __future__ import annotations

import ipaddress
from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def host_name(host: str) -> str:
    """The name part of a Host header or URL authority, lower-cased, without port or brackets."""
    host = host.strip().lower()
    if host.startswith("["):  # [::1]:8000
        return host[1 : host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def host_allowed(host: str, allowed: set[str]) -> bool:
    """Whether a request's Host header names this server: one of ``allowed`` (``*``: any), or an
    IP address. DNS rebinding needs a host name the attacker controls, so an address is safe."""
    name = host_name(host)
    if "*" in allowed or name in allowed:
        return True
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def cross_site(request: Request) -> bool:
    """Whether a request comes from a page on another site: its Origin isn't this server, or the
    browser says so in Sec-Fetch-Site. Requests without either (curl, the CLI) are not."""
    origin = request.headers.get("origin")
    if origin is not None:
        return origin == "null" or origin.split("://", 1)[-1].lower() != request.headers.get("host", "").lower()
    return request.headers.get("sec-fetch-site", "same-origin") not in {"same-origin", "none"}


class LimitBody:  # pylint: disable=too-few-public-methods  # an ASGI middleware is just __call__
    """Refuse (413) request bodies over ``limit`` bytes, by Content-Length or while streaming, so
    an oversized upload isn't spooled to disk or parsed first. ``limit`` is a number, or a
    function of the request's ASGI scope giving its limit (as ``create_app`` does per route).

    A streamed body that passes the limit gets the 413 at once (if no response has started), and
    the app is told the client went away; whatever it then sends is dropped. Raising from the
    body stream instead would reach FastAPI as a body-parsing error, which it reports as 400.
    """

    def __init__(self, inner: ASGIApp, limit: int | Callable[[Scope], int]) -> None:
        self.inner = inner
        self.limit_for: Callable[[Scope], int] = limit if callable(limit) else lambda _: limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.inner(scope, receive, send)
            return
        headers = dict(scope["headers"])
        limit = self.limit_for(scope)
        length = headers.get(b"content-length")
        if length is not None and length.isdigit() and int(length) > limit:
            await self._refuse(limit, scope, receive, send)
            return
        seen = 0
        started = refused = False

        async def counted() -> Message:
            nonlocal seen, started, refused
            if refused:
                return {"type": "http.disconnect"}
            msg = await receive()
            seen += len(msg.get("body", b""))
            if seen <= limit:
                return msg
            refused = True
            if not started:  # too late for a 413 once a response has begun: just stop
                started = True
                await self._refuse(limit, scope, receive, send)
            return {"type": "http.disconnect"}

        async def tracked(msg: Message) -> None:
            nonlocal started
            if refused:
                return
            started = started or msg["type"] == "http.response.start"
            await send(msg)

        try:
            await self.inner(scope, counted, tracked)
        except Exception:  # pylint: disable=broad-exception-caught  # the app failing on the cut-off body
            if not refused:
                raise

    @staticmethod
    async def _refuse(limit: int, scope: Scope, receive: Receive, send: Send) -> None:
        mb = limit / 1024 / 1024
        await JSONResponse({"detail": f"Request is larger than the {mb:.3g} MB limit"}, 413)(scope, receive, send)
