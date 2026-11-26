"""The server's settings, read and checked in one place.

``fewshot serve`` builds them from its options (which fall back to the same environment
variables); ``uvicorn fewshot.api:app`` reads them from the environment with ``from_env``. Either
way a bad value is refused with a message naming it, before the server starts.

``FEWSHOT_HOME``
    The workspace (default ``./fewshot_data``).
``FEWSHOT_TOKEN``
    If set, every /api request except /api/login and /api/health needs this token, as a
    ``Bearer`` header or the session cookie /api/login sets.
``FEWSHOT_ALLOW_ANY_MODEL=1``
    Let API callers choose any embedder spec. By default only presets, registered embedders and
    specs existing tasks already use are accepted, so a caller can't make the server download
    arbitrary models.
``FEWSHOT_WARM=0``
    Don't load the models existing tasks use at startup (by default they are, in the background).
``FEWSHOT_ALLOWED_HOSTS``
    Comma-separated host names the server answers to, besides IP addresses (``*``: any; default:
    localhost). A request with another Host header gets 400, which stops web pages from reaching
    the server through DNS rebinding.
``FEWSHOT_ALLOW_URLS=1``
    Accept image URLs (``.../predict/urls``, ``.../examples/urls``): the server downloads them,
    from public addresses only.
``FEWSHOT_MAX_UPLOAD_MB`` / ``FEWSHOT_MAX_REQUEST_MB``
    Largest image per upload (default 50) and largest multipart request body (default 1024).
    Other request bodies (JSON) are limited to ``MAX_JSON_BYTES``.
``FEWSHOT_FORWARDED_ALLOW_IPS``
    Comma-separated addresses of reverse proxies whose ``X-Forwarded-For`` and
    ``X-Forwarded-Proto`` headers are trusted (``*``: any; default 127.0.0.1). Behind a proxy,
    the client address then comes from the proxy's header, so limits on wrong tokens apply per
    client rather than to the proxy as a whole. ``fewshot serve`` passes this to uvicorn; with
    ``uvicorn fewshot.api:app`` use uvicorn's own ``--forwarded-allow-ips``.
``FEWSHOT_WORKERS``
    Threads for heavy work: predictions, evaluations, uploads, imports and thumbnails (default 4).
    Other requests have their own threads, so heavy work can't keep the server from answering.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .guard import host_name

MB = 1024 * 1024
MAX_JSON_BYTES = 1 * MB  # request bodies other than multipart uploads
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"", "0", "false", "no", "off"}


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
        if not 1 <= self.workers <= 256:
            raise ValueError(f"workers must be between 1 and 256, got {self.workers}")
        if self.token is not None and not self.token.strip():
            object.__setattr__(self, "token", None)
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
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServerSettings:
        """Settings from the ``FEWSHOT_*`` environment variables (or ``env``), defaults for those
        unset. Raises ``ValueError`` naming the variable whose value is unusable."""
        env = os.environ if env is None else env

        def flag(name: str, default: bool) -> bool:
            raw = env.get(name)
            if raw is None:
                return default
            if raw.strip().lower() not in _TRUE | _FALSE:
                raise ValueError(f"{name} must be 1 or 0, got {raw!r}")
            return raw.strip().lower() in _TRUE

        def number(name: str, default: float) -> float:
            raw = env.get(name, "").strip()
            try:
                v = float(raw) if raw else default
            except ValueError:
                raise ValueError(f"{name} must be a number, got {raw!r}") from None
            if not 0 < v < float("inf"):
                raise ValueError(f"{name} must be a positive number, got {raw!r}")
            return v

        hosts = [h for h in env.get("FEWSHOT_ALLOWED_HOSTS", "").split(",") if h.strip()]
        home = env.get("FEWSHOT_HOME")
        return cls(home=Path(home) if home else None, token=env.get("FEWSHOT_TOKEN") or None,
                   allow_any_model=flag("FEWSHOT_ALLOW_ANY_MODEL", False), warm=flag("FEWSHOT_WARM", True),
                   allow_urls=flag("FEWSHOT_ALLOW_URLS", False),
                   allowed_hosts=frozenset(hosts or LOCAL_HOSTS),
                   max_upload_mb=number("FEWSHOT_MAX_UPLOAD_MB", 50),
                   max_request_mb=number("FEWSHOT_MAX_REQUEST_MB", 1024),
                   forwarded_allow_ips=env.get("FEWSHOT_FORWARDED_ALLOW_IPS", "").strip() or "127.0.0.1",
                   workers=int(number("FEWSHOT_WORKERS", 4)))
