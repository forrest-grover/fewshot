"""The server's settings, read and checked in one place.

``uvicorn fewshot.api:app`` reads them from the environment with ``from_env``; ``fewshot serve``
does too, then applies the options given on its command line on top, so both read each variable
by the same rules. Either way a bad value is refused with a message naming it, before the server
starts. An empty variable counts as unset.

``FEWSHOT_HOME``
    The workspace (default ``./fewshot_data``).
``FEWSHOT_TOKEN``
    If set, every /api request except /api/login and /api/health needs this token, as a
    ``Bearer`` header or the session cookie /api/login sets. It must be one that can't be guessed
    online (see ``token_problem``): at least 16 characters, not just a few different ones
    repeated. ``openssl rand -hex 16`` makes a good one.
``FEWSHOT_ALLOW_ANY_MODEL=1``
    Let API callers choose any embedder spec. By default only presets, registered embedders and
    specs existing tasks already use are accepted, so a caller can't make the server download
    arbitrary models.
``FEWSHOT_WARM=0``
    Don't load the models existing tasks use at startup (by default they are, in the background).
``FEWSHOT_ALLOWED_HOSTS``
    Comma-separated host names the server answers to, besides IP addresses (``*``: any). A
    request with another Host header gets 400, which stops web pages from reaching the server
    through DNS rebinding. Default: localhost, plus this machine's names when the server listens
    on an address other than a local one (``fewshot serve --host``, or uvicorn's ``--host`` or
    ``UVICORN_HOST``, see ``served_host``).
``FEWSHOT_ALLOW_URLS=1``
    Accept image URLs (``.../predict/urls``, ``.../examples/urls``): the server downloads them,
    from public addresses only.
``FEWSHOT_MAX_UPLOAD_MB`` / ``FEWSHOT_MAX_REQUEST_MB``
    Largest image per upload (default 50) and largest upload request (default 1024): a multipart
    body sent to an upload route (adding examples or inbox images, predicting, importing a
    task). Bodies listing ids (bulk edits and deletes, accepting and dismissing inbox images) are
    limited to ``MAX_BULK_JSON_BYTES``, and every other request body, wherever it is sent and
    whatever its Content-Type says, to ``MAX_JSON_BYTES``.
``FEWSHOT_FORWARDED_ALLOW_IPS``
    Comma-separated addresses of reverse proxies whose ``X-Forwarded-For`` and
    ``X-Forwarded-Proto`` headers are trusted (``*``: any; default 127.0.0.1). Behind a proxy,
    the client address then comes from the proxy's header, so limits on wrong tokens apply per
    client rather than to the proxy as a whole.
``FEWSHOT_WORKERS``
    Threads for heavy work: predictions, evaluations, uploads, imports and thumbnails (default 4).
    Other requests have their own threads, so heavy work can't keep the server from answering.

What still differs under ``uvicorn fewshot.api:app``: the proxy headers are handled by uvicorn
before the app sees the request, so ``FEWSHOT_FORWARDED_ALLOW_IPS`` is passed on only by
``fewshot serve``; with uvicorn use its own ``--forwarded-allow-ips`` (or ``FORWARDED_ALLOW_IPS``).
The address it listens on is read from uvicorn's command line or ``UVICORN_HOST``; a server
started another way (gunicorn, a script) counts as local unless ``FEWSHOT_ALLOWED_HOSTS`` is set.

One process serves the app: logged-out sessions, wrong-token counts and background jobs are kept
in its memory, so running several (``uvicorn --workers N``) isn't supported; a server started
that way warns.
"""

from __future__ import annotations

import os
import socket
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .guard import MAX_BULK_JSON_BYTES, MAX_JSON_BYTES, host_name

__all__ = ["MAX_BULK_JSON_BYTES", "MAX_JSON_BYTES", "ServerSettings", "token_problem"]

MB = 1024 * 1024
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
_TRUE, _FALSE = {"1", "true", "yes", "y", "on"}, {"0", "false", "no", "n", "off"}
MIN_TOKEN_LENGTH = 16
MIN_TOKEN_CHARACTERS = 5  # different characters in it
MAX_WORKERS = 256


def token_problem(token: str) -> str | None:
    """Why ``token`` is too weak to protect a server, or None if it isn't.

    A correct token is always accepted, and wrong ones from an address are refused after a few a
    minute, so what keeps the server safe is a token nobody can guess, even from many addresses.
    Strength can't be measured from the string itself; this refuses the tokens that are weak
    on their face: shorter than ``MIN_TOKEN_LENGTH`` characters (16 random ones, even only hex
    digits, are 2**64 possibilities), or made of fewer than ``MIN_TOKEN_CHARACTERS`` different
    characters ("aaaa...", "abab...", "1234" repeated). A random token of 16 hex digits has
    fewer than 5 different ones about once in two million; one of 32 (``openssl rand -hex 16``)
    practically never.
    """
    if len(token) < MIN_TOKEN_LENGTH:
        return (f"it has {len(token)} characters; use at least {MIN_TOKEN_LENGTH} random ones, "
                "e.g. from `openssl rand -hex 16`")
    if len(set(token)) < MIN_TOKEN_CHARACTERS:
        return (f"it repeats {len(set(token))} different characters; use a random one, "
                "e.g. from `openssl rand -hex 16`")
    return None


def requested_processes(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    """How many processes the server was started with, as far as can be told: uvicorn's (or
    gunicorn's) ``--workers N`` / ``-w N`` in ``argv`` (default ``sys.argv``, which a worker
    process inherits), else ``WEB_CONCURRENCY``, which both read. ``fewshot serve --workers`` sets
    threads, not processes, and isn't counted."""
    argv = sys.argv if argv is None else argv
    env = os.environ if env is None else env
    raw = env.get("WEB_CONCURRENCY", "")
    if _uvicorn_args(argv):
        for i, arg in enumerate(argv):
            if arg in {"--workers", "-w"} and i + 1 < len(argv):
                raw = argv[i + 1]
            elif arg.startswith("--workers="):
                raw = arg.partition("=")[2]
    return int(raw) if raw.strip().isdigit() else 1


def _uvicorn_args(argv: list[str]) -> bool:
    """Whether ``argv`` starts a uvicorn or gunicorn server (``uvicorn ...``, ``python -m uvicorn ...``)."""
    return bool(argv) and any(server in Path(arg).parts for arg in argv[:2] for server in ("uvicorn", "gunicorn"))


def served_host(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> str:
    """The address the server listens on, as far as can be told: uvicorn's ``--host`` in ``argv``
    (default ``sys.argv``), else ``UVICORN_HOST``, else uvicorn's default, 127.0.0.1."""
    argv = sys.argv if argv is None else argv
    env = os.environ if env is None else env
    host = env.get("UVICORN_HOST", "").strip() or "127.0.0.1"
    if _uvicorn_args(argv):
        for i, arg in enumerate(argv):
            if arg == "--host" and i + 1 < len(argv):
                host = argv[i + 1]
            elif arg.startswith("--host="):
                host = arg.partition("=")[2]
    return host


def default_hosts(host: str) -> frozenset[str]:
    """The host names a server listening on ``host`` answers to unless told which: localhost, and
    when ``host`` isn't a local address also this machine's own names, by which other machines
    on the network are likely to reach it."""
    if host in LOCAL_HOSTS:
        return frozenset(LOCAL_HOSTS)
    return frozenset(LOCAL_HOSTS) | ({socket.gethostname().lower(), socket.getfqdn().lower()} - {""})


@dataclass(frozen=True)
class ServerSettings:  # pylint: disable=too-many-instance-attributes  # one field per setting
    """Everything that configures a server; see the module docstring for each setting."""

    home: Path | None = None  # None: FEWSHOT_HOME or ./fewshot_data, as for the library
    token: str | None = None
    allow_any_model: bool = False
    warm: bool = True
    allow_urls: bool = False
    allowed_hosts: frozenset[str] = field(default_factory=lambda: frozenset(LOCAL_HOSTS))
    max_upload_mb: float = 50.0
    max_request_mb: float = 1024.0
    forwarded_allow_ips: str = "127.0.0.1"
    workers: int = 4

    def __post_init__(self) -> None:
        for name in ("max_upload_mb", "max_request_mb"):
            v = getattr(self, name)
            if not 0 < v < float("inf"):
                raise ValueError(f"{name} must be a positive number of megabytes, got {v}")
        if not 1 <= self.workers <= MAX_WORKERS:
            raise ValueError(f"workers must be between 1 and {MAX_WORKERS}, got {self.workers}")
        if self.token is not None and not self.token.strip():
            object.__setattr__(self, "token", None)
        if self.token is not None and (why := token_problem(self.token)):
            raise ValueError(f"The token is too weak to be safe: {why}")
        object.__setattr__(self, "allowed_hosts", frozenset(host_name(h) for h in self.allowed_hosts if h.strip()))

    @property
    def max_upload_bytes(self) -> int:
        """Largest image accepted per upload."""
        return int(self.max_upload_mb * MB)

    @property
    def max_request_bytes(self) -> int:
        """Largest multipart request body accepted."""
        return int(self.max_request_mb * MB)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, host: str | None = None) -> ServerSettings:
        """Settings from the ``FEWSHOT_*`` environment variables (or ``env``), defaults for those
        unset or empty; the default allowed hosts depend on the address the server listens on,
        ``host`` (default: ``served_host()``). Raises ``ValueError`` naming the variable whose
        value is unusable."""
        env = os.environ if env is None else env

        def flag(name: str, default: bool) -> bool:
            raw = env.get(name, "").strip().lower()
            if not raw:
                return default
            if raw not in _TRUE | _FALSE:
                raise ValueError(f"{name} must be 1 or 0, got {env[name]!r}")
            return raw in _TRUE

        def number(name: str, default: float) -> float:
            raw = env.get(name, "").strip()
            try:
                v = float(raw) if raw else default
            except ValueError:
                raise ValueError(f"{name} must be a number, got {raw!r}") from None
            if not 0 < v < float("inf"):
                raise ValueError(f"{name} must be a positive number, got {raw!r}")
            return v

        def whole(name: str, default: int, top: int) -> int:
            raw = env.get(name, "").strip()
            if not raw:
                return default
            if not raw.isdigit() or not 1 <= int(raw) <= top:
                raise ValueError(f"{name} must be a whole number from 1 to {top}, got {raw!r}")
            return int(raw)

        token = env.get("FEWSHOT_TOKEN", "").strip() and env["FEWSHOT_TOKEN"]
        if token and (why := token_problem(token)):
            raise ValueError(f"FEWSHOT_TOKEN is too weak to be safe: {why}")
        hosts = [h for h in env.get("FEWSHOT_ALLOWED_HOSTS", "").split(",") if h.strip()]
        home = env.get("FEWSHOT_HOME", "").strip()
        return cls(home=Path(home) if home else None, token=token or None,
                   allow_any_model=flag("FEWSHOT_ALLOW_ANY_MODEL", False), warm=flag("FEWSHOT_WARM", True),
                   allow_urls=flag("FEWSHOT_ALLOW_URLS", False),
                   allowed_hosts=frozenset(hosts) or default_hosts(served_host(env=env) if host is None else host),
                   max_upload_mb=number("FEWSHOT_MAX_UPLOAD_MB", 50),
                   max_request_mb=number("FEWSHOT_MAX_REQUEST_MB", 1024),
                   forwarded_allow_ips=env.get("FEWSHOT_FORWARDED_ALLOW_IPS", "").strip() or "127.0.0.1",
                   workers=whole("FEWSHOT_WORKERS", 4, MAX_WORKERS))
