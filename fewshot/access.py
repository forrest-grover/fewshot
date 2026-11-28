"""The server's access control, run before every request: the Host and cross-site checks, and
the token or session. The limits and sessions themselves are in ``auth``."""

from __future__ import annotations

import ipaddress
import logging
import secrets
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.responses import Response

from . import __version__
from .auth import LOGIN_WINDOW
from .errors import (
    CROSS_SITE,
    NO_SERVER_TOKEN,
    NOT_SIGNED_IN,
    TOO_MANY_WRONG_TOKENS,
    UNKNOWN_HOST,
    WRONG_TOKEN,
    problem_response,
)
from .guard import cross_site, host_allowed, route_path

if TYPE_CHECKING:
    from .server import Server

log = logging.getLogger(__name__)

SESSION_COOKIE = "fewshot_session"
OPEN_PATHS = {"/api/login", "/api/health"}  # reachable without the token
TOO_MANY_TOKENS = "Too many wrong tokens from this address; try again in a minute"
RETRY_TOKENS = {"Retry-After": str(int(LOGIN_WINDOW))}
NOT_SIGNED_IN_TEXT = ("Not signed in: send Authorization: Bearer <token> (fewshot.client: token=, or $FEWSHOT_TOKEN), "
                      "or sign in with POST /api/login")
WRONG_BEARER = "Wrong token: the Authorization header's Bearer token isn't this server's"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


ALLOW_NO_TOKEN_ENV = "FEWSHOT_ALLOW_NO_TOKEN"
_NO_TOKEN_FIX = ("Set FEWSHOT_TOKEN (e.g. to the output of `openssl rand -hex 16`), or set "
                 "FEWSHOT_ALLOW_NO_TOKEN=1 to serve without one.")
NO_TOKEN = ("This server has no token, so anyone who can reach it could read and change every task, and it "
            "{where} {addr}, which isn't this machine's loopback address. " + _NO_TOKEN_FIX)
NO_TOKEN_PROXIED = ("This server has no token, so anyone who can reach it could read and change every task, and "
                    "this request came through a proxy (it carries {header}), which serves it to other machines. "
                    + _NO_TOKEN_FIX)
# Headers a reverse proxy adds for the client it forwards: a request on loopback that carries one
# came from wherever the proxy serves, not from this machine.
FORWARDING_HEADERS = ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-real-ip")


def is_loopback(host: str) -> bool | None:
    """Whether ``host`` (an address, or ``localhost``) is this machine's loopback; None if it
    isn't an address at all (a Unix socket's path, a test client's made-up name)."""
    if host.strip().lower() == "localhost":
        return True
    try:
        addr = ipaddress.ip_address(host.strip().strip("[]").split("%", 1)[0])
    except ValueError:
        return None
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr.is_loopback


def exposure_problem(host: str, token: str | None, allow_no_token: bool) -> str | None:
    """Why serving on ``host`` (a bind address) with ``token`` is unsafe, or None: a server
    without a token may listen on loopback only, unless ``allow_no_token``. For ``fewshot serve``
    to refuse before starting; ``guard`` refuses the requests of a server started otherwise. A
    host name other than localhost counts as exposed."""
    if token is not None or allow_no_token or is_loopback(host):
        return None
    return NO_TOKEN.format(where="would listen on", addr=host)


def client_address(request: Request) -> str:
    """The client's address: behind a trusted proxy (``forwarded_allow_ips``), the one it reports."""
    return request.client.host if request.client else "?"


def check_token(srv: Server, client: str, given: str) -> int | None:
    """Check a token from ``client`` (at login, or as a Bearer header): None if right (always,
    compared in constant time), else 401, or 429 once the client gave ``LOGIN_FAILURES`` wrong
    tokens within ``LOGIN_WINDOW`` seconds. Only wrong tokens count, and they are what gets
    refused: a stranger guessing from a shared address (behind a proxy that isn't trusted) can't
    lock out the clients there that know the token. That a right guess would still get in is
    safe because the token can't be guessed (``ServerSettings`` refuses weak ones); the limit
    just keeps a guesser slow and noisy."""
    token = srv.settings.token
    if token is None or secrets.compare_digest(given.encode(), token.encode()):
        return None
    blocked = srv.wrong_tokens.blocked(client)
    srv.wrong_tokens.wrong(client)
    return 429 if blocked else 401


def _note_untrusted_proxy(srv: Server, request: Request) -> None:
    """Warn (once) if requests come through a proxy whose X-Forwarded-For isn't trusted: every
    client behind it then has the proxy's address, so wrong tokens are counted for all at once."""
    forwarded = request.headers.get("x-forwarded-for")
    if srv.warned_proxy or forwarded is None:
        return
    client = client_address(request)
    if client not in {a.strip() for a in forwarded.split(",")}:
        srv.warned_proxy = True
        log.warning("Requests come through a proxy at %s whose X-Forwarded-For isn't trusted, so every client "
                    "behind it has its address: one client's wrong tokens get the others' wrong tokens refused "
                    "too (right tokens always work). Set FEWSHOT_FORWARDED_ALLOW_IPS (--forwarded-allow-ips) "
                    "to the proxy's address.", client)


def _refused_cross_site(request: Request) -> str | None:
    """Why an /api request from another site is refused, or None: any write (CSRF), and reads
    too, as a page could make a browser send those by ``<img src>`` or a link, and some run the
    model (inbox suggestions, duplicates) on a server without a token. Reads are refused only by
    Sec-Fetch-Site, which browsers send; example images may be shown anywhere."""
    if request.method not in SAFE_METHODS:
        return cross_site(request)
    site = request.headers.get("sec-fetch-site", "same-origin")
    if site in {"same-origin", "none"} or route_path(request.scope).endswith("/image"):
        return None
    return f"Cross-site request refused: the browser sent it from another site (Sec-Fetch-Site: {site})"


def _reached_from_elsewhere(request: Request) -> str | None:
    """Why a request to a server without a token may come from another machine, or None: it
    arrived on an address other than loopback, on a Unix socket (which only a proxy serving others
    would use; uvicorn gives it no port), or through a proxy, as forwarding headers show (a proxy on
    this machine connects on loopback for whoever it serves)."""
    local = request.scope.get("server")
    if local is None or local[1] is None:
        return NO_TOKEN.format(where="was reached on", addr="a Unix socket")
    if is_loopback(str(local[0])) is False:
        return NO_TOKEN.format(where="was reached at", addr=local[0])
    header = next((h for h in FORWARDING_HEADERS if h in request.headers), None)
    return None if header is None else NO_TOKEN_PROXIED.format(header=header)


async def guard(request: Request, call_next: Any) -> Response:
    """``checked``, with the server's version in X-Fewshot-Version on every answer."""
    response = await checked(request, call_next)
    response.headers["X-Fewshot-Version"] = __version__
    return response


async def checked(request: Request, call_next: Any) -> Response:
    """Refuse every request to a server without a token that arrived on an address other than
    loopback (unless allowed), requests for another host name (DNS rebinding) and writes from
    other sites (CSRF), then check the token or session."""
    srv: Server = request.app.state.server
    if srv.sessions is None and not srv.allow_no_token and (why := _reached_from_elsewhere(request)):
        # Reached from beyond this machine without a token: a container serving on 0.0.0.0 (or
        # ``uvicorn --host 0.0.0.0``), a Unix socket or a proxy on this machine with FEWSHOT_TOKEN
        # unset. Only the liveness check answers.
        if not srv.warned_exposed:
            srv.warned_exposed = True
            log.error("%s", why)
        if route_path(request.scope) != "/healthz":
            return problem_response(403, NO_SERVER_TOKEN, why, request.url.path)
    if not host_allowed(request.headers.get("host", ""), set(srv.settings.allowed_hosts)):
        return problem_response(400, UNKNOWN_HOST, "Unknown host; see FEWSHOT_ALLOWED_HOSTS", request.url.path)
    path = route_path(request.scope)
    if path.startswith("/api/") and (why := _refused_cross_site(request)):
        return problem_response(403, CROSS_SITE, why, request.url.path)
    if srv.sessions is not None:
        _note_untrusted_proxy(srv, request)
    if srv.sessions is not None and path.startswith("/api/") and path not in OPEN_PATHS:
        auth = request.headers.get("authorization", "")
        bearer = auth.lower().startswith("bearer ")
        if bearer:
            refused = check_token(srv, client_address(request), auth[7:])
        else:
            refused = None if srv.sessions.valid(request.cookies.get(SESSION_COOKIE)) else 401
        if refused == 429:
            return problem_response(429, TOO_MANY_WRONG_TOKENS, TOO_MANY_TOKENS, request.url.path, RETRY_TOKENS)
        if refused:
            code, text = (WRONG_TOKEN, WRONG_BEARER) if bearer else (NOT_SIGNED_IN, NOT_SIGNED_IN_TEXT)
            return problem_response(401, code, text, request.url.path, {"WWW-Authenticate": "Bearer"})
    response: Response = await call_next(request)
    if path.startswith("/api/"):  # answers for one user (with the token): no cache may keep them
        response.headers.setdefault("Cache-Control", "no-store")
    return response
