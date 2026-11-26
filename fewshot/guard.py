"""ASGI pieces of the server's protection: body size limits, Host and cross-site checks."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_JSON_BYTES = 1024 * 1024  # request bodies other than uploads and lists of ids
MAX_BULK_JSON_BYTES = 16 * 1024 * 1024  # JSON bodies listing ids (bulk edit, delete, accept, dismiss)


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


def cross_site(request: Request) -> str | None:
    """Why a request comes from a page on another site, or None if it doesn't: its Origin isn't
    this server, or the browser says so in Sec-Fetch-Site. Requests without either (curl, the
    CLI) are not.

    The Origin's host may match the Host header or, behind a reverse proxy that rewrites Host
    (nginx's default ``proxy_pass``), one in X-Forwarded-Host. Which proxy sent that needn't be
    checked: a page on another site can't add X-Forwarded-Host to its request without a CORS
    preflight, which this server never grants, and a client that isn't a browser can send any
    Origin it likes anyway."""
    origin = request.headers.get("origin")
    if origin is not None:
        host = request.headers.get("host", "")
        forwarded = [h.strip() for h in request.headers.get("x-forwarded-host", "").split(",") if h.strip()]
        if origin != "null" and origin.split("://", 1)[-1].lower() in {h.lower() for h in [host, *forwarded]}:
            return None
        names = f"Host {host!r}" + "".join(f" or X-Forwarded-Host {h!r}" for h in forwarded)
        return (f"Cross-site request refused: its Origin {origin!r} doesn't match the {names}. Behind a reverse "
                "proxy, pass the browser's Host header on (nginx: proxy_set_header Host $host).")
    site = request.headers.get("sec-fetch-site", "same-origin")
    if site in {"same-origin", "none"}:
        return None
    return f"Cross-site request refused: the browser sent it from another site (Sec-Fetch-Site: {site})"


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


# Routes whose bodies may be large: uploads (multipart only) and JSON lists of ids. The limit
# comes from the route, never from what the client says the body is, so a body labelled as an
# upload gets the small limit anywhere else (FastAPI would read it whole to parse JSON).
UPLOAD_ROUTES = re.compile(r"/api/tasks/(?:import|[^/]+/(?:examples|inbox|predict))")  # POST
BULK_ROUTES = {"PATCH": re.compile(r"/api/tasks/[^/]+/examples"),
               "POST": re.compile(r"/api/tasks/[^/]+/(?:examples/delete|inbox/accept|inbox/accept-confident"
                                  r"|inbox/dismiss)")}


def body_limit(max_request_bytes: int, scope: Scope) -> int:
    """The largest body the request in ``scope`` may send: ``max_request_bytes`` for a multipart
    body to an upload route, ``MAX_BULK_JSON_BYTES`` for a route taking a list of ids, else
    ``MAX_JSON_BYTES``."""
    path, method = scope["path"], scope["method"]
    root = scope.get("root_path", "")
    if root and path.startswith(root):
        path = path[len(root):]
    if method == "POST" and UPLOAD_ROUTES.fullmatch(path):
        ctype = dict(scope["headers"]).get(b"content-type", b"").lower()
        if ctype.startswith(b"multipart/form-data"):  # any other body there is no upload: small
            return max_request_bytes
    bulk = BULK_ROUTES.get(method)
    if bulk is not None and bulk.fullmatch(path):
        return MAX_BULK_JSON_BYTES
    return MAX_JSON_BYTES


async def without_input(_: Request, exc: Exception) -> JSONResponse:
    """422 for a request that doesn't validate, like FastAPI's own but without each error's
    ``input``, so a large bad body isn't sent back."""
    assert isinstance(exc, RequestValidationError)
    errors = [{k: v for k, v in e.items() if k != "input"} for e in exc.errors()]
    return JSONResponse({"detail": jsonable_encoder(errors)}, 422)
