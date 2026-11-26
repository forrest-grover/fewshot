"""Access control of the server: the limit on wrong tokens, and login sessions.

Module-level limits are read at call time, so a test can change them.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import OrderedDict, deque

LOGIN_FAILURES = 10  # wrong tokens per client within LOGIN_WINDOW before token checks answer 429
LOGIN_WINDOW = 60.0
LOGIN_CLIENTS = 10_000  # clients with recent wrong tokens remembered at once (the oldest are forgotten)
SESSION_DAYS = 30  # how long a login lasts


class WrongTokens:
    """Recent wrong tokens per client address, to answer 429 to a client guessing tokens.

    Only wrong tokens make an entry; entries expire after ``LOGIN_WINDOW`` and at most
    ``LOGIN_CLIENTS`` are kept (the least recently wrong go first), so the map stays small however
    many addresses guess."""

    def __init__(self) -> None:
        self._seen: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def _recent(self, client: str, now: float) -> deque[float]:
        recent = self._seen.get(client, deque())
        while recent and now - recent[0] > LOGIN_WINDOW:
            recent.popleft()
        return recent

    def blocked(self, client: str) -> bool:
        """Whether ``client`` gave ``LOGIN_FAILURES`` wrong tokens within the last ``LOGIN_WINDOW``."""
        with self._lock:
            recent = self._recent(client, time.monotonic())
            if not recent:
                self._seen.pop(client, None)
            return len(recent) >= LOGIN_FAILURES

    def wrong(self, client: str) -> None:
        """Note a wrong token from ``client``."""
        now = time.monotonic()
        with self._lock:
            recent = self._recent(client, now)
            recent.append(now)
            self._seen[client] = recent
            self._seen.move_to_end(client)
            while len(self._seen) > LOGIN_CLIENTS:
                self._seen.popitem(last=False)


class Sessions:
    """Login sessions as signed cookie values: ``v1.<issued>.<expires>.<nonce>.<signature>``.

    The signature is an HMAC with a key derived from the server token, so the cookie never holds
    the token itself, sessions expire, changing the token ends them all, and logging out revokes
    that one session (for as long as this server runs)."""

    def __init__(self, token: str) -> None:
        self._key = hmac.new(token.encode(), b"fewshot session key v1", hashlib.sha256).digest()
        self._revoked: dict[str, float] = {}  # nonce -> its session's expiry
        self._lock = threading.Lock()

    def _sign(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()

    def issue(self, seconds: float = SESSION_DAYS * 86400) -> str:
        """A new session value, valid for ``seconds``."""
        now = int(time.time())
        payload = f"v1.{now}.{now + int(seconds)}.{secrets.token_hex(16)}"
        return f"{payload}.{self._sign(payload)}"

    def _parse(self, value: str) -> tuple[str, float] | None:
        """``(nonce, expiry)`` of a session value this server signed, else None."""
        payload, _, sig = value.rpartition(".")
        parts = payload.split(".")
        if len(parts) != 4 or parts[0] != "v1" or not hmac.compare_digest(sig.encode(), self._sign(payload).encode()):
            return None
        try:
            return parts[3], float(parts[2])
        except ValueError:  # pragma: no cover  # we signed it, so it parses
            return None

    def valid(self, value: str | None) -> bool:
        """Whether ``value`` is a session this server issued, not expired or revoked."""
        parsed = None if value is None else self._parse(value)
        if parsed is None or parsed[1] < time.time():
            return False
        with self._lock:
            return parsed[0] not in self._revoked

    def revoke(self, value: str | None) -> None:
        """End the session ``value`` (if it is one), forgetting revoked sessions once expired."""
        parsed = None if value is None else self._parse(value)
        now = time.time()
        with self._lock:
            self._revoked = {n: exp for n, exp in self._revoked.items() if exp >= now}
            if parsed is not None:
                self._revoked[parsed[0]] = parsed[1]
