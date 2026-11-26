"""Who may use the server: tokens and sessions, wrong-token limits, host names (DNS rebinding) and
cross-site writes."""

import time

import pytest
from fastapi.testclient import TestClient

import fewshot.api
import fewshot.auth
import fewshot.guard
from fewshot.auth import Sessions, WrongTokens
from fewshot.guard import host_allowed
from tests.conftest import RED, api_client, host_client, png


# ---- tokens and sessions -------------------------------------------------
def test_token_auth(tmp_path):
    """With a token set, /api needs it as a Bearer header or the login cookie."""
    client = api_client(tmp_path, token="s3cret")
    assert client.get("/api/tasks").status_code == 401
    assert client.get("/api/tasks").headers["www-authenticate"] == "Bearer"
    assert client.get("/api/tasks", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/tasks", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["auth_required"] is True
    assert client.get("/docs").status_code == 200  # only /api is protected

    assert client.post("/api/login", json={"token": "nope"}).status_code == 401
    r = client.post("/api/login", json={"token": "s3cret"})
    assert r.status_code == 204
    assert "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/api/tasks").status_code == 200  # the cookie now signs requests in
    assert client.post("/api/logout").status_code == 204
    assert client.get("/api/tasks").status_code == 401


def test_login_without_token_configured(client):
    """With no token configured, login succeeds and sets no cookie."""
    r = client.post("/api/login", json={"token": "anything"})
    assert r.status_code == 204
    assert "set-cookie" not in r.headers


@pytest.mark.usefixtures("task_t")
def test_logout_without_a_token_configured(client):
    """Logging out of a server without a token just clears the cookie."""
    assert client.post("/api/logout").status_code == 204


def _cookie(value):
    return {"Cookie": f"{fewshot.api.SESSION_COOKIE}={value}"}


def test_the_session_cookie_is_signed_not_the_token(tmp_path):
    """The cookie holds a signed session id that expires and that logout ends; tampering, the
    raw token, or a cookie from a server with another token is refused."""
    c = api_client(tmp_path, token="s3cret")
    r = c.post("/api/login", json={"token": "s3cret"})
    cookie = r.cookies[fewshot.api.SESSION_COOKIE]
    assert "s3cret" not in r.headers["set-cookie"]
    assert "samesite=strict" in r.headers["set-cookie"].lower()
    fresh = TestClient(c.app)
    assert fresh.get("/api/tasks", headers=_cookie(cookie)).status_code == 200
    for bad in ("s3cret", cookie[:-1] + ("0" if cookie[-1] != "0" else "1"), "v1.1.2.3", "junk"):
        assert fresh.get("/api/tasks", headers=_cookie(bad)).status_code == 401
    other = api_client(tmp_path / "other", token="changed")
    assert other.get("/api/tasks", headers=_cookie(cookie)).status_code == 401
    assert c.post("/api/logout").status_code == 204
    assert fresh.get("/api/tasks", headers=_cookie(cookie)).status_code == 401  # revoked
    sessions = Sessions("s3cret")
    expired = sessions.issue(seconds=-1)
    assert not sessions.valid(expired)
    sessions.revoke(expired)  # forgotten at once: it has expired anyway
    sessions.revoke(None)
    assert not sessions._revoked  # pylint: disable=protected-access


# ---- wrong tokens --------------------------------------------------------
def test_login_is_rate_limited_and_secure_over_https(tmp_path, monkeypatch):
    """Too many wrong tokens get 429 for a while; over HTTPS the cookie is Secure."""
    monkeypatch.setattr(fewshot.auth, "LOGIN_FAILURES", 2)
    c = api_client(tmp_path, token="s3cret")
    assert c.post("/api/login", json={"token": "a"}).status_code == 401
    assert c.post("/api/login", json={"token": "b"}).status_code == 401
    r = c.post("/api/login", json={"token": "s3cret"})
    assert (r.status_code, r.json()["detail"]) == (429, "Too many wrong tokens; try again in a minute")
    monkeypatch.setattr(fewshot.auth, "LOGIN_WINDOW", 0.0)  # the window has passed
    r = c.post("/api/login", json={"token": "s3cret"})
    assert r.status_code == 204
    assert "secure" not in r.headers["set-cookie"].lower()
    https = TestClient(c.app, base_url="https://testserver")
    assert "secure" in https.post("/api/login", json={"token": "s3cret"}).headers["set-cookie"].lower()


def test_wrong_tokens_on_any_request_count_towards_the_limit(tmp_path, monkeypatch):
    """Guessing with Bearer tokens is limited like /api/login (no token at all isn't a guess, and
    a session cookie can't be guessed), and the limit then covers login too."""
    api = fewshot.api
    monkeypatch.setattr(fewshot.auth, "LOGIN_FAILURES", 2)
    c = api_client(tmp_path, token="s3cret")
    for _ in range(3):
        assert c.get("/api/tasks").status_code == 401
        assert c.get("/api/tasks", headers={"Cookie": f"{api.SESSION_COOKIE}=b"}).status_code == 401
    assert c.get("/api/tasks", headers={"Authorization": "Bearer a"}).status_code == 401
    assert c.get("/api/tasks", headers={"Authorization": "Bearer b"}).status_code == 401
    r = c.get("/api/tasks", headers={"Authorization": "Bearer s3cret"})
    assert (r.status_code, r.json()["detail"]) == (429, api.TOO_MANY_TOKENS)
    assert c.post("/api/login", json={"token": "s3cret"}).status_code == 429
    monkeypatch.setattr(fewshot.auth, "LOGIN_WINDOW", 0.0)
    assert c.get("/api/tasks", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_wrong_tokens_are_bounded_and_expire(monkeypatch):
    """Only wrong tokens make entries; old ones expire and the map has a size limit."""
    monkeypatch.setattr(fewshot.auth, "LOGIN_CLIENTS", 3)
    monkeypatch.setattr(fewshot.auth, "LOGIN_FAILURES", 2)
    w = WrongTokens()
    assert not w.blocked("a")
    assert not w._seen  # pylint: disable=protected-access  # asking made no entry
    for client in "abcd":
        w.wrong(client)
    assert list(w._seen) == ["b", "c", "d"]  # pylint: disable=protected-access
    w.wrong("d")
    assert w.blocked("d")
    monkeypatch.setattr(fewshot.auth, "LOGIN_WINDOW", 0.0)
    time.sleep(0.01)
    assert not w.blocked("d")
    assert "d" not in w._seen  # pylint: disable=protected-access  # expired entries go


def test_a_signed_in_browser_keeps_working_while_its_address_is_blocked(tmp_path, monkeypatch):
    """Someone guessing tokens from the same address (say, behind the same proxy) gets 429 for
    token checks, but a session cookie, which can't be guessed, still works."""
    monkeypatch.setattr(fewshot.auth, "LOGIN_FAILURES", 2)
    c = api_client(tmp_path, token="s3cret")
    assert c.post("/api/login", json={"token": "s3cret"}).status_code == 204
    other = TestClient(c.app)
    for guess in ("a", "b"):
        assert other.get("/api/tasks", headers={"Authorization": f"Bearer {guess}"}).status_code == 401
    assert other.post("/api/login", json={"token": "s3cret"}).status_code == 429
    assert c.get("/api/tasks").status_code == 200


# ---- host names (DNS rebinding) ------------------------------------------
def test_host_allowed():
    """Listed names (any port), '*', and IP addresses pass; other names don't."""
    allowed = {"box.lan"}
    for host in ("box.lan", "BOX.LAN:8000", "10.0.0.5:8000", "[fe80::1]:80", "203.0.113.9"):
        assert host_allowed(host, allowed), host
    for host in ("evil.example", "box.lan.evil.example", "", "10.0.0.5.nip.io"):
        assert not host_allowed(host, allowed), host
    assert host_allowed("anything", {"*"})


def test_allowed_host_entries_may_have_ports(tmp_path):
    """An entry written with a port matches its name; the health check by IP always works."""
    c = api_client(tmp_path, allowed_hosts=frozenset({"box.lan:8443"}))
    assert c.get("/api/health", headers={"Host": "box.lan:8443"}).status_code == 200
    assert c.get("/healthz", headers={"Host": "127.0.0.1:8000"}).status_code == 200
    assert c.get("/api/health", headers={"Host": "other.lan"}).status_code == 400


def test_only_local_host_names_by_default(tmp_path, monkeypatch):
    """By default only localhost names are answered; anything else is 400, the UI included."""
    c = host_client(tmp_path, monkeypatch, base_url="http://localhost:8000")
    assert c.get("/api/tasks").status_code == 200
    for host in ("127.0.0.1:8000", "[::1]:8000", "LOCALHOST"):
        assert c.get("/api/tasks", headers={"Host": host}).status_code == 200
    for host in ("evil.example", "evil.example:8000", "localhost.evil.example", ""):
        r = c.get("/api/health", headers={"Host": host})
        assert (r.status_code, r.json()["detail"]) == (400, "Unknown host; see FEWSHOT_ALLOWED_HOSTS")


def test_configured_host_names(tmp_path, monkeypatch):
    """FEWSHOT_ALLOWED_HOSTS lists the names; '*' allows any."""
    c = host_client(tmp_path, monkeypatch, hosts="box.lan, [fe80::1]")
    assert c.get("/api/tasks", headers={"Host": "box.lan:8000"}).status_code == 200
    assert c.get("/api/tasks", headers={"Host": "[fe80::1]:8000"}).status_code == 200
    assert c.get("/api/tasks", headers={"Host": "localhost"}).status_code == 400
    c = host_client(tmp_path, monkeypatch, hosts="*")
    assert c.get("/api/tasks", headers={"Host": "anything"}).status_code == 200


# ---- cross-site writes (CSRF) --------------------------------------------
def test_cross_site_writes_are_refused(tmp_path, monkeypatch):
    """A write whose Origin is another site, or that the browser marks cross-site, is 403;
    same-origin writes, reads, and clients that send neither header go through."""
    c = host_client(tmp_path, monkeypatch, hosts="*", base_url="http://localhost:8000")
    body = {"name": "t", "embedder": "fake:color"}
    for headers in ({"Origin": "https://evil.example"}, {"Origin": "null"}, {"Origin": "http://localhost:9999"},
                    {"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"}):
        r = c.post("/api/tasks", json=body, headers=headers)
        assert (r.status_code, r.json()["detail"]) == (403, "Cross-site requests are not allowed"), headers
    files = [("files", ("a.png", png(RED), "image/png"))]
    assert c.post("/api/tasks/t/examples", data={"label": "red"}, files=files,
                  headers={"Origin": "https://evil.example"}).status_code == 403
    assert c.post("/api/tasks", json=body, headers={"Origin": "http://localhost:8000"}).status_code == 201
    same = {"Sec-Fetch-Site": "same-origin"}
    assert c.post("/api/tasks", json={**body, "name": "u"}, headers=same).status_code == 201
    assert c.post("/api/tasks", json={**body, "name": "v"}).status_code == 201  # curl, the CLI
    assert c.get("/api/tasks", headers={"Origin": "https://evil.example"}).status_code == 200
