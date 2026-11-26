"""The server's access control, run before every request: the Host and cross-site checks, and
the token or session. The limits and sessions themselves are in ``auth``."""

from __future__ import annotations

import logging
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


async def guard(request: Request, call_next: Any) -> Response:
    """Refuse requests for another host name (DNS rebinding) and writes from other sites (CSRF),
    then check the token or session."""
    srv: Server = request.app.state.server
    if not host_allowed(request.headers.get("host", ""), set(srv.settings.allowed_hosts)):
        return JSONResponse({"detail": "Unknown host; see FEWSHOT_ALLOWED_HOSTS"}, 400)
    path = request.url.path
    if request.method not in SAFE_METHODS and path.startswith("/api/") and (why := cross_site(request)):
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
    return response
