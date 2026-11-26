"""The server's access control, run before every request: the Host and cross-site checks, and
the token or session. The limits and sessions themselves are in ``auth``."""

from __future__ import annotations

import ipaddress
import logging
import os
import secrets
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from .guard import cross_site, host_allowed

if TYPE_CHECKING:
    from .api import Server

log = logging.getLogger(__name__)

SESSION_COOKIE = "fewshot_session"
OPEN_PATHS = {"/api/login", "/api/health"}  # reachable without the token
TOO_MANY_TOKENS = "Too many wrong tokens from this address; try again in a minute"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


ALLOW_NO_TOKEN_ENV = "FEWSHOT_ALLOW_NO_TOKEN"
NO_TOKEN = ("This server has no token, so anyone who can reach it could read and change every task, and it "
            "was reached at {addr}, which isn't this machine's loopback address. Set FEWSHOT_TOKEN (e.g. to "
            "the output of `openssl rand -hex 16`), or set FEWSHOT_ALLOW_NO_TOKEN=1 to serve without one.")


def allow_no_token_from_env() -> bool:
    """Whether ``FEWSHOT_ALLOW_NO_TOKEN`` says to serve other machines without a token."""
    return os.environ.get(ALLOW_NO_TOKEN_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


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
    return NO_TOKEN.format(addr=host)


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
    if site in {"same-origin", "none"} or request.url.path.endswith("/image"):
        return None
    return f"Cross-site request refused: the browser sent it from another site (Sec-Fetch-Site: {site})"


async def guard(request: Request, call_next: Any) -> Response:
    """Refuse every request to a server without a token that arrived on an address other than
    loopback (unless allowed), requests for another host name (DNS rebinding) and writes from
    other sites (CSRF), then check the token or session."""
    srv: Server = request.app.state.server
    local = request.scope.get("server") or ("", None)
    if srv.sessions is None and not srv.allow_no_token and is_loopback(str(local[0])) is False:
        # Reached on another interface without a token: a container serving on 0.0.0.0 (or
        # ``uvicorn --host 0.0.0.0``) with FEWSHOT_TOKEN unset. Only the liveness check answers.
        if not srv.warned_exposed:
            srv.warned_exposed = True
            log.error("%s", NO_TOKEN.format(addr=local[0]))
        if request.url.path != "/healthz":
            return JSONResponse({"detail": NO_TOKEN.format(addr=local[0])}, 403)
    if not host_allowed(request.headers.get("host", ""), set(srv.settings.allowed_hosts)):
        return JSONResponse({"detail": "Unknown host; see FEWSHOT_ALLOWED_HOSTS"}, 400)
    path = request.url.path
    if path.startswith("/api/") and (why := _refused_cross_site(request)):
        return JSONResponse({"detail": why}, 403)
    if srv.sessions is not None:
        _note_untrusted_proxy(srv, request)
    if srv.sessions is not None and path.startswith("/api/") and path not in OPEN_PATHS:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            refused = check_token(srv, client_address(request), auth[7:])
        else:
            refused = None if srv.sessions.valid(request.cookies.get(SESSION_COOKIE)) else 401
        if refused == 429:
            return JSONResponse({"detail": TOO_MANY_TOKENS}, 429)
        if refused:
            return JSONResponse({"detail": "Not signed in"}, 401, headers={"WWW-Authenticate": "Bearer"})
    response: Response = await call_next(request)
    if path.startswith("/api/"):  # answers for one user (with the token): no cache may keep them
        response.headers.setdefault("Cache-Control", "no-store")
    return response
