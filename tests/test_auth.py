"""Who may use the server: tokens and sessions, wrong-token limits, host names (DNS rebinding) and
cross-site writes."""

import time

import pytest
from fastapi.testclient import TestClient

import fewshot.access
import fewshot.api
import fewshot.auth
import fewshot.guard
from fewshot.auth import Sessions, WrongTokens
from fewshot.access import NO_TOKEN, exposure_problem
from fewshot.guard import host_allowed
from tests.conftest import RED, api_client, api_task, host_client, png


# ---- tokens and sessions -------------------------------------------------
def test_token_auth(tmp_path):
    """With a token set, /api needs it as a Bearer header or the login cookie."""
    client = api_client(tmp_path, token="s3cret-s3cret-s3cret")
    assert client.get("/api/tasks").status_code == 401
    assert client.get("/api/tasks").headers["www-authenticate"] == "Bearer"
    assert client.get("/api/tasks", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/tasks", headers={"Authorization": "Bearer s3cret-s3cret-s3cret"}).status_code == 200
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["auth_required"] is True
    assert client.get("/docs").status_code == 200  # only /api is protected

    assert client.post("/api/login", json={"token": "nope"}).status_code == 401
    r = client.post("/api/login", json={"token": "s3cret-s3cret-s3cret"})
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
    return {"Cookie": f"{fewshot.access.SESSION_COOKIE}={value}"}


def test_the_session_cookie_is_signed_not_the_token(tmp_path):
    """The cookie holds a signed session id that expires and that logout ends; tampering, the
    raw token, or a cookie from a server with another token is refused."""
    c = api_client(tmp_path, token="s3cret-s3cret-s3cret")
    r = c.post("/api/login", json={"token": "s3cret-s3cret-s3cret"})
    cookie = r.cookies[fewshot.access.SESSION_COOKIE]
    assert "s3cret-s3cret-s3cret" not in r.headers["set-cookie"]
    assert "samesite=strict" in r.headers["set-cookie"].lower()
    fresh = TestClient(c.app)
    assert fresh.get("/api/tasks", headers=_cookie(cookie)).status_code == 200
    for bad in ("s3cret-s3cret-s3cret", cookie[:-1] + ("0" if cookie[-1] != "0" else "1"), "v1.1.2.3", "junk"):
        assert fresh.get("/api/tasks", headers=_cookie(bad)).status_code == 401
    other = api_client(tmp_path / "other", token="changed-0123456789")
    assert other.get("/api/tasks", headers=_cookie(cookie)).status_code == 401
    assert c.post("/api/logout").status_code == 204
    assert fresh.get("/api/tasks", headers=_cookie(cookie)).status_code == 401  # revoked
    sessions = Sessions("s3cret-s3cret-s3cret")
    expired = sessions.issue(seconds=-1)
    assert not sessions.valid(expired)
    sessions.revoke(expired)  # forgotten at once: it has expired anyway
    sessions.revoke(None)
    assert not sessions._revoked  # pylint: disable=protected-access


def test_revoked_sessions_are_bounded(monkeypatch):
    """Logged-out sessions are remembered until they expire, at most REVOKED_KEPT at once (those
    expiring soonest are forgotten first), so logging out stays cheap however many do."""
    monkeypatch.setattr(fewshot.auth, "REVOKED_KEPT", 2)
    sessions = Sessions("s3cret-s3cret-s3cret")
    values = [sessions.issue(seconds=100 + i) for i in range(3)]
    for v in values:
        sessions.revoke(v)
    sessions.revoke(values[2])  # again: no second entry
    assert [sessions.valid(v) for v in values] == [True, False, False]  # the soonest to expire went
    assert len(sessions._expiring) == len(sessions._revoked) == 2  # pylint: disable=protected-access
    monkeypatch.setattr(time, "time", lambda: 10**12)  # long after they all expired
    sessions.revoke(None)
    assert not sessions._revoked  # pylint: disable=protected-access


# ---- wrong tokens --------------------------------------------------------
# These changed on purpose: wrong tokens past the limit get 429, but the right token is always
# accepted (it used to get 429 too, which let one stranger behind a shared address lock out
# everyone there). Guessing it is ruled out by refusing weak tokens instead (test_server.py).
TOKEN = "s3cret-s3cret-s3cret"


def test_wrong_logins_are_limited_and_the_cookie_is_secure_over_https(tmp_path, monkeypatch):
    """Past the limit wrong tokens get 429 while the right one still signs in; the limit lasts
    until wrong tokens have stopped for a window. Over HTTPS the cookie is Secure."""
    monkeypatch.setattr(fewshot.auth, "LOGIN_FAILURES", 2)
    c = api_client(tmp_path, token=TOKEN)
    assert c.post("/api/login", json={"token": "a"}).status_code == 401
    assert c.post("/api/login", json={"token": "b"}).status_code == 401
    r = c.post("/api/login", json={"token": "c"})
    assert (r.status_code, r.json()["detail"]) == (429, "Too many wrong tokens from this address; try again in a "
                                                        "minute")
    r = c.post("/api/login", json={"token": TOKEN})
    assert r.status_code == 204
    assert "secure" not in r.headers["set-cookie"].lower()
    monkeypatch.setattr(fewshot.auth, "LOGIN_WINDOW", 0.0)  # the window has passed
    assert c.post("/api/login", json={"token": "d"}).status_code == 401
    https = TestClient(c.app, base_url="https://testserver")
    assert "secure" in https.post("/api/login", json={"token": TOKEN}).headers["set-cookie"].lower()


def test_wrong_tokens_on_any_request_count_towards_the_limit(tmp_path, monkeypatch):
    """Guessing with Bearer tokens is limited like /api/login (no token at all isn't a guess, and
    a session cookie can't be guessed), and the limit then covers login too; the right token
    works throughout."""
    monkeypatch.setattr(fewshot.auth, "LOGIN_FAILURES", 2)
    c = api_client(tmp_path, token=TOKEN)
    for _ in range(3):
        assert c.get("/api/tasks").status_code == 401
        assert c.get("/api/tasks", headers={"Cookie": f"{fewshot.access.SESSION_COOKIE}=b"}).status_code == 401
    assert c.get("/api/tasks", headers={"Authorization": "Bearer a"}).status_code == 401
    assert c.get("/api/tasks", headers={"Authorization": "Bearer b"}).status_code == 401
    r = c.get("/api/tasks", headers={"Authorization": "Bearer c"})
    assert (r.status_code, r.json()["detail"]) == (429, fewshot.access.TOO_MANY_TOKENS)
    assert c.post("/api/login", json={"token": "d"}).status_code == 429
    assert c.get("/api/tasks", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert c.post("/api/login", json={"token": TOKEN}).status_code == 204


def test_wrong_tokens_are_bounded_and_expire(monkeypatch):
    """Only wrong tokens make entries, each keeping its last few; old ones expire and the map
    has a size limit."""
    monkeypatch.setattr(fewshot.auth, "LOGIN_CLIENTS", 3)
    monkeypatch.setattr(fewshot.auth, "LOGIN_FAILURES", 2)
    w = WrongTokens()
    assert not w.blocked("a")
    assert not w._seen  # pylint: disable=protected-access  # asking made no entry
    for client in "abcd":
        w.wrong(client)
    assert list(w._seen) == ["b", "c", "d"]  # pylint: disable=protected-access
    for _ in range(5):
        w.wrong("d")
    assert w.blocked("d")
    assert len(w._seen["d"]) == 2  # pylint: disable=protected-access  # however many came
    monkeypatch.setattr(fewshot.auth, "LOGIN_WINDOW", 0.0)
    time.sleep(0.01)
    assert not w.blocked("d")
    assert "d" not in w._seen  # pylint: disable=protected-access  # expired entries go


def test_a_stranger_guessing_from_a_shared_address_locks_nobody_out(tmp_path, monkeypatch):
    """Someone guessing tokens from the same address (say, behind a proxy that isn't trusted)
    gets 429, but clients there that know the token still sign in and use it, as does a browser
    signed in already."""
    monkeypatch.setattr(fewshot.auth, "LOGIN_FAILURES", 2)
    c = api_client(tmp_path, token=TOKEN)
    assert c.post("/api/login", json={"token": TOKEN}).status_code == 204
    other = TestClient(c.app)
    for guess in ("a", "b", "c", "d"):
        assert other.get("/api/tasks", headers={"Authorization": f"Bearer {guess}"}).status_code in {401, 429}
    assert other.get("/api/tasks", headers={"Authorization": "Bearer e"}).status_code == 429
    assert other.get("/api/tasks", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert other.post("/api/login", json={"token": TOKEN}).status_code == 204
    assert c.get("/api/tasks").status_code == 200


def test_a_proxy_that_isnt_trusted_is_reported_once(tmp_path, caplog):
    """Requests forwarded by a proxy whose X-Forwarded-For isn't trusted log a warning, once;
    a trusted proxy (the client address is the forwarded one) or no token logs none."""
    c = api_client(tmp_path, token=TOKEN)
    with caplog.at_level("WARNING", logger="fewshot.api"):
        assert c.get("/api/health").status_code == 200
        assert not caplog.records
        for _ in range(2):
            c.get("/api/health", headers={"X-Forwarded-For": "203.0.113.5, 10.0.0.1"})
        c.get("/api/health", headers={"X-Forwarded-For": "testclient"})  # what uvicorn makes of a trusted one
        open_server = api_client(tmp_path / "open")
        open_server.get("/api/health", headers={"X-Forwarded-For": "203.0.113.5"})
    assert [r.getMessage()[:56] for r in caplog.records] == [
        "Requests come through a proxy at testclient whose X-Forw"]
    trusted = api_client(tmp_path / "trusted", token=TOKEN)
    with caplog.at_level("WARNING", logger="fewshot.api"):
        caplog.clear()
        trusted.get("/api/health", headers={"X-Forwarded-For": "1.2.3.4, testclient"})
    assert not caplog.records


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
        assert r.status_code == 403, headers
        assert r.json()["detail"].startswith("Cross-site request refused"), headers
    files = [("files", ("a.png", png(RED), "image/png"))]
    assert c.post("/api/tasks/t/examples", data={"label": "red"}, files=files,
                  headers={"Origin": "https://evil.example"}).status_code == 403
    assert c.post("/api/tasks", json=body, headers={"Origin": "http://localhost:8000"}).status_code == 201
    same = {"Sec-Fetch-Site": "same-origin"}
    assert c.post("/api/tasks", json={**body, "name": "u"}, headers=same).status_code == 201
    assert c.post("/api/tasks", json={**body, "name": "v"}).status_code == 201  # curl, the CLI
    assert c.get("/api/tasks", headers={"Origin": "https://evil.example"}).status_code == 200


def test_cross_site_check_behind_a_proxy_that_rewrites_host(tmp_path, monkeypatch):
    """Behind a proxy sending its own Host (nginx's default), the browser's Origin is compared
    with X-Forwarded-Host too; a refusal says what was compared and how to fix the proxy."""
    c = host_client(tmp_path, monkeypatch, hosts="*")
    body = {"name": "t", "embedder": "fake:color"}
    page = {"Host": "127.0.0.1:8000", "Origin": "https://fewshot.example.com", "Sec-Fetch-Site": "same-origin"}
    r = c.post("/api/tasks", json=body, headers=page)
    assert r.status_code == 403
    assert r.json()["detail"] == ("Cross-site request refused: its Origin 'https://fewshot.example.com' doesn't "
                                  "match the Host '127.0.0.1:8000'. Behind a reverse proxy, pass the browser's "
                                  "Host header on (nginx: proxy_set_header Host $host).")
    forwarded = {**page, "X-Forwarded-Host": "fewshot.example.com"}
    assert c.post("/api/tasks", json=body, headers=forwarded).status_code == 201
    evil = {**forwarded, "Origin": "https://evil.example"}
    r = c.post("/api/tasks", json={**body, "name": "u"}, headers=evil)
    assert r.status_code == 403
    assert "Host '127.0.0.1:8000' or X-Forwarded-Host 'fewshot.example.com'" in r.json()["detail"]
    r = c.post("/api/tasks", json={**body, "name": "u"}, headers={"Sec-Fetch-Site": "same-site"})
    assert r.json()["detail"] == ("Cross-site request refused: the browser sent it from another site "
                                  "(Sec-Fetch-Site: same-site)")


# ---- no token on another interface ---------------------------------------
def test_a_server_without_token_refuses_other_interfaces(tmp_path, monkeypatch, caplog):
    """A server without a token reached on an address other than loopback (a container serving on
    0.0.0.0 with FEWSHOT_TOKEN unset) refuses everything but /healthz, and logs why once; on
    loopback, with a token, or with FEWSHOT_ALLOW_NO_TOKEN=1 it serves."""
    app = api_client(tmp_path).app
    outside = TestClient(app, base_url="http://172.17.0.2:8000")  # the address the request arrived at
    refused = NO_TOKEN.format(addr="172.17.0.2")
    for path in ("/api/tasks", "/api/health", "/"):
        r = outside.get(path)
        assert (r.status_code, r.json()["detail"]) == (403, refused), path
    assert outside.post("/api/tasks", json={"name": "t", "embedder": "fake:color"}).status_code == 403
    assert outside.get("/healthz").status_code == 200
    assert [r.getMessage() for r in caplog.records if r.levelname == "ERROR"] == [refused]
    for local in ("http://127.0.0.1:8000", "http://[::1]:8000", "http://[::ffff:127.0.0.1]:8000"):
        assert TestClient(app, base_url=local).get("/api/tasks").status_code == 200, local
    tokened = api_client(tmp_path / "b", base_url="http://172.17.0.2", token="0123456789abcdef0123")
    assert tokened.get("/api/tasks").status_code == 401  # the token protects it
    allowed = TestClient(fewshot.api.create_app(app.state.server.settings, allow_no_token=True),
                         base_url="http://172.17.0.2")
    assert allowed.get("/api/tasks").status_code == 200
    monkeypatch.setenv("FEWSHOT_ALLOW_NO_TOKEN", "1")
    from_env = TestClient(fewshot.api.create_app(app.state.server.settings), base_url="http://172.17.0.2")
    assert from_env.get("/api/tasks").status_code == 200


def test_exposure_problem():
    """For ``fewshot serve`` to refuse before starting: no token on a non-loopback bind address."""
    assert exposure_problem("0.0.0.0", None, False) == NO_TOKEN.format(addr="0.0.0.0")
    assert exposure_problem("::", None, False) is not None
    assert exposure_problem("box.lan", None, False) is not None  # a name: whichever interface it is
    for host in ("127.0.0.1", "127.0.0.2", "::1", "localhost"):
        assert exposure_problem(host, None, False) is None, host
    assert exposure_problem("0.0.0.0", "0123456789abcdef0123", False) is None
    assert exposure_problem("0.0.0.0", None, True) is None


# ---- reads from other sites, caching, and the shape of answers --------------
def test_cross_site_reads_are_refused_but_images(client):
    """A page on another site can't make the browser read /api (say, inbox suggestions running the
    model, by <img src>), except example images; same-origin reads and clients without the
    header go through."""
    task = api_task(client, n=1)
    eid = client.get(f"/api/tasks/{task}").json()["examples"][0]["id"]
    for site in ("cross-site", "same-site"):
        r = client.get(f"/api/tasks/{task}/inbox", headers={"Sec-Fetch-Site": site})
        assert r.status_code == 403, site
        assert r.json()["detail"] == f"Cross-site request refused: the browser sent it from another site (Sec-Fetch-Site: {site})"
    assert client.get(f"/api/tasks/{task}/examples/{eid}/image", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200
    assert client.get(f"/api/tasks/{task}/inbox", headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200
    assert client.get(f"/api/tasks/{task}/inbox", headers={"Sec-Fetch-Site": "none"}).status_code == 200
    assert client.get(f"/api/tasks/{task}/inbox").status_code == 200


def test_api_answers_are_not_cached_but_images_are(client):
    """/api answers are "no-store" (they may need the token); images, whose ids are content
    hashes, are kept by the browser only ("private")."""
    task = api_task(client, n=1)
    eid = client.get(f"/api/tasks/{task}").json()["examples"][0]["id"]
    for path in (f"/api/tasks/{task}", "/api/tasks/nope", f"/api/tasks/{task}/export"):
        assert client.get(path).headers["cache-control"] == "no-store", path
    for size in ("", "?size=32"):
        img = client.get(f"/api/tasks/{task}/examples/{eid}/image{size}")
        assert img.headers["cache-control"] == "private, max-age=31536000, immutable"


def test_no_content_answers_have_no_content_type(tmp_path):
    """204s (login, logout, deleting a task) have no body, so no Content-Type either; login still
    sets its cookie."""
    c = api_client(tmp_path, token=TOKEN)
    r = c.post("/api/login", json={"token": TOKEN})
    assert r.status_code == 204 and "content-type" not in r.headers and fewshot.access.SESSION_COOKIE in r.cookies
    api_task(c, n=1)
    r = c.delete("/api/tasks/c")
    assert r.status_code == 204 and "content-type" not in r.headers
    r = c.post("/api/logout")
    assert r.status_code == 204 and "content-type" not in r.headers


def test_schema_declares_the_access_errors(client):
    """The OpenAPI schema lists what the access checks and size limits can answer anywhere."""
    paths = client.get("/openapi.json").json()["paths"]
    for op in (paths["/api/tasks"]["post"], paths["/api/tasks/{name}/examples"]["post"]):
        assert {"401", "403", "413", "429"} <= set(op["responses"]), op["operationId"]
        assert op["responses"]["403"]["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/Problem"}
