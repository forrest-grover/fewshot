"""The API's contract: one problem envelope with stable codes, the status policy, validation,
merge-patch settings, near-miss class names, deduplicated uploads, ETag/If-Match, jobs,
versioning and an OpenAPI schema fit for generated clients."""

import contextvars
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import fewshot
import fewshot.api
import fewshot.routing
import fewshot.services
from fewshot import preconditions
from fewshot.api import create_app
from fewshot.errors import CODES, violation
from fewshot.exceptions import NoExamples, RevisionChanged, error_code, http_status
from fewshot.jobs import Job, Jobs
from fewshot.server_settings import ServerSettings
from tests.conftest import BLUE, GREEN, RED, api_client, api_task, png, query_files, wait_for_job

PROBLEM_KEYS = {"type", "title", "status", "detail", "instance", "code"}


def _problem(r, status, code):
    """The problem body of response ``r``, checked to have ``status`` and ``code``."""
    assert r.status_code == status, r.text
    assert r.headers["content-type"] == "application/problem+json"
    body = r.json()
    assert set(body) >= PROBLEM_KEYS
    assert (body["status"], body["code"]) == (status, code)
    assert "\n" not in body["detail"]
    return body


def _files(label, seeds, rgb=RED):
    return [("files", (f"{label}{i}.png", png(rgb, noise=15, seed=i), "image/png")) for i in seeds]


@pytest.fixture(name="task")
def task_fixture(client):
    """Task 'c' with three red and three blue examples."""
    return api_task(client)


# ---- the envelope -----------------------------------------------------------------------
def test_every_error_is_a_problem(client, task):
    """Route, method, domain, validation and access errors all answer the same envelope."""
    r = client.get("/api/tasks/pets")
    body = _problem(r, 404, "TASK_NOT_FOUND")
    assert body == {"type": "urn:fewshot:problem:task-not-found", "title": "No such task", "status": 404,
                    "code": "TASK_NOT_FOUND", "instance": "/api/tasks/pets", "task": "pets", "similar_to": [],
                    "detail": "No task 'pets'; create it with POST /api/tasks (fewshot.client: create_task)"}
    body = _problem(client.get("/api/tasks/C"), 404, "TASK_NOT_FOUND")
    assert body["similar_to"] == ["c"]
    assert body["detail"].startswith("No task 'C'; did you mean 'c'? Or create it with POST /api/tasks")
    assert _problem(client.get("/api/nope"), 404, "ROUTE_NOT_FOUND")["type"] == "about:blank"
    body = _problem(client.put("/api/tasks"), 405, "METHOD_NOT_ALLOWED")
    assert (body["type"], body["detail"]) == ("about:blank", "PUT isn't allowed on /api/tasks")
    body = _problem(client.post("/api/tasks", json={"name": 3, "colour": "red"}), 422, "VALIDATION_FAILED")
    assert body["detail"] == ("2 problems with the request: name must be a valid string; got 3; "
                              "unknown field 'colour'")
    assert [e.get("pointer") for e in body["errors"]] == ["/name", "/colour"]
    assert client.post(f"/api/tasks/{task}/examples/{'x' * 10}", files=_files("x", [1])).status_code == 405


def test_a_crash_is_a_problem_without_its_details(tmp_path, monkeypatch, caplog):
    """An unexpected error answers 500 INTERNAL, logged, with nothing of the error in the body."""
    app = create_app(ServerSettings(home=tmp_path / "ws", allowed_hosts=frozenset({"*"}), warm=False))
    client = TestClient(app, raise_server_exceptions=False)

    def broken(*_a, **_k):
        raise RuntimeError("/secret/path exploded")

    monkeypatch.setattr(fewshot.api, "loaded_embedders", broken)
    body = _problem(client.get("/api/health"), 500, "INTERNAL")
    assert (body["type"], body["detail"]) == ("about:blank", "Internal error; see the server log")
    assert "secret" not in str(body)
    assert "Internal error answering GET /api/health: RuntimeError" in caplog.text


def test_validation_reports_every_problem_with_bounds(client, task):
    """Every violation at once, each naming its field, the rule and the value received."""
    r = client.patch(f"/api/tasks/{task}", json={"text_weight": -5, "reject_below": 2, "temperature": 0,
                                                 "classifier": "nope"})
    body = _problem(r, 422, "VALIDATION_FAILED")
    details = {e["pointer"]: e["detail"] for e in body["errors"]}
    assert details.pop("/classifier").endswith("; got 'nope'")
    assert details == {"/text_weight": "text_weight must be greater than or equal to 0; got -5",
                       "/reject_below": "reject_below must be less than or equal to 1; got 2",
                       "/temperature": "temperature must be greater than or equal to 0.01; got 0"}
    assert body["detail"].startswith("4 problems with the request: ")
    assert body["detail"].endswith("; and 1 more")
    assert {"loc", "msg", "code"} <= set(body["errors"][0])  # pydantic's loc and msg, kept for one version
    r = client.get(f"/api/tasks/{task}/inbox", params={"sort": "best"})
    [e] = _problem(r, 422, "VALIDATION_FAILED")["errors"]
    assert (e["parameter"], e["detail"]) == ("sort", "sort must be 'uncertain', 'confident', 'novel' or 'added'; "
                                                     "got best")


def test_violation_messages():
    """Each kind of pydantic error reads as a sentence about the field."""
    assert violation({"type": "missing", "loc": ("body", "label"), "msg": "Field required"})["detail"] == (
        "label is required")
    assert violation({"type": "missing", "loc": ("body",), "msg": "Field required"})["detail"] == (
        "the body is required")
    assert violation({"type": "value_error", "loc": ("body", "x"), "msg": "Value error, x is bad"})["detail"] == (
        "x is bad")
    assert violation({"type": "value_error", "loc": ("body", "x"), "msg": "odd"})["detail"] == "odd"
    big = violation({"type": "string_too_long", "loc": ("body", "note"), "msg": "String should have at most 3",
                     "input": "y" * 500})
    assert big["detail"] == "note string should have at most 3"  # a long value isn't sent back
    assert violation({"loc": ("header", "x-a"), "input": True})["parameter"] == "x-a"
    assert violation({"loc": (), "input": None})["detail"] == "the body is invalid; got null"
    assert violation({"loc": ("query", "on"), "msg": "Bad", "input": False})["detail"] == "on bad; got false"
    assert violation({"loc": ("query", "on"), "msg": "JSON is bad"})["detail"] == "on JSON is bad; got null"  # not jSON
    assert violation({"type": "json_invalid", "loc": ("body",), "msg": "JSON decode error"})["detail"] == (
        "the body isn't valid JSON")
    assert set(CODES) >= {"TASK_NOT_FOUND", "CLASS_NAME_NEAR_MISS", "VALIDATION_FAILED", "PRECONDITION_FAILED"}


def test_a_body_that_isnt_json_says_where_it_stopped(client):
    """Malformed JSON is a 422 saying where parsing stopped, pointing at the whole body, not at a
    'field' named by the character offset."""
    r = client.post("/api/tasks", content=b'{"name":', headers={"Content-Type": "application/json"})
    body = _problem(r, 422, "VALIDATION_FAILED")
    assert body["detail"] == "1 problem with the request: the body isn't valid JSON (at character 8): Expecting value"
    assert (body["errors"][0]["pointer"], body["errors"][0]["code"]) == ("", "json_invalid")


def test_access_errors_are_problems(tmp_path):
    """No token, a wrong token, too many wrong ones, a cross-site write and an unknown host each have their code."""
    client = api_client(tmp_path, token="secret-0123456789")
    body = _problem(client.get("/api/tasks"), 401, "NOT_SIGNED_IN")
    assert body["detail"].startswith("Not signed in: send Authorization: Bearer <token>")
    r = client.get("/api/tasks", headers={"Authorization": "Bearer nope"})
    assert _problem(r, 401, "WRONG_TOKEN")["detail"].startswith("Wrong token")
    assert r.headers["www-authenticate"] == "Bearer"
    _problem(client.post("/api/login", json={"token": "nope"}), 401, "WRONG_TOKEN")
    for _ in range(10):
        client.get("/api/tasks", headers={"Authorization": "Bearer nope"})
    r = client.get("/api/tasks", headers={"Authorization": "Bearer nope"})
    assert (_problem(r, 429, "TOO_MANY_WRONG_TOKENS")["status"], r.headers["retry-after"]) == (429, "60")
    r = client.post("/api/login", json={"token": "nope"})
    assert (_problem(r, 429, "TOO_MANY_WRONG_TOKENS")["status"], r.headers["retry-after"]) == (429, "60")
    auth = {"Authorization": "Bearer secret-0123456789"}
    _problem(client.post("/api/tasks", json={"name": "x"}, headers={**auth, "Origin": "http://evil"}), 403,
             "CROSS_SITE")
    hosts = api_client(tmp_path, allowed_hosts=frozenset({"fewshot.local"}))
    _problem(hosts.get("/api/tasks"), 400, "UNKNOWN_HOST")


def test_payload_too_large_and_unsupported_media_type(client, task):
    """A body over its limit is 413 with the limit; an upload route sent JSON is 415."""
    r = client.post("/api/tasks", content=b" " * (1024 * 1024 + 1), headers={"Content-Type": "application/json"})
    body = _problem(r, 413, "PAYLOAD_TOO_LARGE")
    assert (body["limit_bytes"], body["instance"]) == (1024 * 1024, "/api/tasks")
    for path in ("examples", "inbox", "predict"):
        r = client.post(f"/api/tasks/{task}/{path}", json={"label": "red", "files": []})
        assert "expects multipart/form-data" in _problem(r, 415, "UNSUPPORTED_MEDIA_TYPE")["detail"]


def test_a_json_route_sent_another_kind_of_body_is_415(client, task):
    """A form, or a body without a content type (curl -d without -H), sent to a JSON route is 415
    saying what to send; JSON and +json types are read, and a route without a body ignores it."""
    body = _problem(client.post("/api/tasks", data={"name": "x"}), 415, "UNSUPPORTED_MEDIA_TYPE")
    assert body["detail"] == ("/api/tasks expects a JSON body sent as application/json, not "
                              "application/x-www-form-urlencoded (with curl: -H 'content-type: application/json')")
    r = client.patch(f"/api/tasks/{task}", content=b'{"flip": true}')
    assert "not one without a content type" in _problem(r, 415, "UNSUPPORTED_MEDIA_TYPE")["detail"]
    _problem(client.post("/api/tasks", files=_files("x", [1])), 415, "UNSUPPORTED_MEDIA_TYPE")
    r = client.patch(f"/api/tasks/{task}", content=b'{"flip": true}',
                     headers={"Content-Type": "application/merge-patch+json; charset=utf-8"})
    assert (r.status_code, r.json()["flip"]) == (200, True)
    assert client.post(f"/api/tasks/{task}/undo", data={"x": "y"}).status_code == 200
    _problem(client.post("/api/tasks"), 422, "VALIDATION_FAILED")  # no body at all: the body is required


# ---- status codes -------------------------------------------------------------------------
def test_conflicts_are_409(client, task):
    """Taken names and nothing to undo or redo conflict with the task as it is."""
    client.post("/api/tasks", json={"name": "d", "embedder": "fake:color"})
    _problem(client.post(f"/api/tasks/{task}/rename", json={"name": "d"}), 409, "TASK_EXISTS")
    _problem(client.post(f"/api/tasks/{task}/duplicate", json={"name": "d"}), 409, "TASK_EXISTS")
    _problem(client.patch(f"/api/tasks/{task}/classes/red", json={"name": "blue"}), 409, "CLASS_EXISTS")
    _problem(client.post("/api/tasks/d/undo"), 409, "NOTHING_TO_UNDO")
    _problem(client.post("/api/tasks/d/redo"), 409, "NOTHING_TO_REDO")


def test_adding_a_class_is_201_and_describing_one_200(client, task):
    """POST .../classes creates (201) or, for a class that exists, only sets its description (200)."""
    assert client.post(f"/api/tasks/{task}/classes", json={"name": "green"}).status_code == 201
    r = client.post(f"/api/tasks/{task}/classes", json={"name": "green", "description": "a green thing"})
    assert r.status_code == 200
    assert [c["description"] for c in r.json()["classes"] if c["name"] == "green"] == ["a green thing"]
    rev = r.json()["revision"]
    for body in ({"name": "green"}, {"name": " green ", "description": None}):  # no description: nothing changes
        r = client.post(f"/api/tasks/{task}/classes", json=body)
        assert (r.status_code, r.json()["revision"]) == (200, rev)
        assert [c["description"] for c in r.json()["classes"] if c["name"] == "green"] == ["a green thing"]


def test_a_class_name_close_to_another_is_refused_unless_new_class(client, task):
    """POST .../classes refuses a likely typo of an existing class, as the labelling routes do."""
    body = _problem(client.post(f"/api/tasks/{task}/classes", json={"name": "Red"}), 422, "CLASS_NAME_NEAR_MISS")
    assert (body["similar_to"], body["pointer"]) == (["red"], "/name")
    r = client.post(f"/api/tasks/{task}/classes", json={"name": "Red", "new_class": True})
    assert (r.status_code, [c["name"] for c in r.json()["classes"]]) == (201, ["red", "blue", "Red"])


def test_unknown_query_parameters_are_refused(client, task):
    """A misspelt query parameter is a 422 naming the closest one, not silently ignored."""
    r = client.post(f"/api/tasks/{task}/predict", params={"topk": 1}, files=query_files(1))
    [e] = _problem(r, 422, "VALIDATION_FAILED")["errors"]
    assert (e["parameter"], e["detail"], e["code"]) == ("topk", "unknown parameter 'topk'; did you mean 'top_k'?",
                                                         "unknown_parameter")
    r = client.get(f"/api/tasks/{task}", params={"zzz": 1})
    assert _problem(r, 422, "VALIDATION_FAILED")["errors"][0]["detail"] == (
        "unknown parameter 'zzz'; this route takes none")
    assert client.get(f"/api/tasks/{task}/examples", params={"limit": 1, "offset": 1}).status_code == 200


def test_a_filter_on_a_class_that_doesnt_exist_is_404(client, task):
    """GET .../examples?label= with no such class is a 404 naming the closest, not an empty page."""
    body = _problem(client.get(f"/api/tasks/{task}/examples", params={"label": "Red"}), 404, "CLASS_NOT_FOUND")
    assert (body["label"], body["similar_to"], body["detail"]) == ("Red", ["red"],
                                                                   "No class 'Red' in c; did you mean 'red'?")
    body = _problem(client.get(f"/api/tasks/{task}/examples", params={"label": "zebra"}), 404, "CLASS_NOT_FOUND")
    assert (body["similar_to"], body["detail"]) == ([], "No class 'zebra' in c")
    assert client.get(f"/api/tasks/{task}/examples", params={"label": "red"}).json()["total"] == 3


# ---- near-miss class names -----------------------------------------------------------------
def test_labels_close_to_a_class_are_refused_on_every_route(client, task):
    """A new label one or two edits (or only case) from a class is taken for a typo everywhere a
    label is given, unless new_class; an exactly new name is created and reported."""
    ex = client.get(f"/api/tasks/{task}").json()["examples"]
    r = client.post(f"/api/tasks/{task}/examples", data={"label": "Red"}, files=_files("x", [7]))
    body = _problem(r, 422, "CLASS_NAME_NEAR_MISS")
    assert (body["label"], body["similar_to"], body["pointer"]) == ("Red", ["red"], "/label")
    assert body["detail"] == "No class 'Red' in c; did you mean 'red'? Send new_class=true to create it"
    _problem(client.patch(f"/api/tasks/{task}/examples/{ex[0]['id']}", json={"label": "redd"}), 422,
             "CLASS_NAME_NEAR_MISS")
    _problem(client.patch(f"/api/tasks/{task}/examples", json={"ids": [ex[0]["id"]], "label": "blu"}), 422,
             "CLASS_NAME_NEAR_MISS")
    client.post(f"/api/tasks/{task}/inbox", files=_files("u", [40, 41], BLUE))
    inbox = [i["id"] for i in client.get(f"/api/tasks/{task}/inbox").json()["items"]]
    r = client.post(f"/api/tasks/{task}/inbox/accept", json={"labels": {inbox[0]: "blue", inbox[1]: "Blue"}})
    assert _problem(r, 422, "CLASS_NAME_NEAR_MISS")["pointer"] == f"/labels/{inbox[1]}"
    r = client.post(f"/api/tasks/{task}/examples/urls", json={"label": "reds", "urls": ["http://x/a.png"]})
    _problem(r, 422, "CLASS_NAME_NEAR_MISS")
    assert [c["name"] for c in client.get(f"/api/tasks/{task}").json()["classes"]] == ["red", "blue"]  # nothing made

    r = client.post(f"/api/tasks/{task}/examples", data={"label": "Red", "new_class": "true"}, files=_files("x", [7]))
    assert (r.status_code, r.json()["created_classes"]) == (201, ["Red"])
    r = client.post(f"/api/tasks/{task}/examples", data={"label": "green"}, files=_files("g", [8]))
    assert (r.status_code, r.json()["created_classes"]) == (201, ["green"])
    t = client.post(f"/api/tasks/{task}/inbox/accept", json={"labels": {inbox[1]: "Blue"}, "new_class": True}).json()
    assert ("Blue" in [c["name"] for c in t["classes"]], t["created_classes"]) == (True, ["Blue"])
    t = client.patch(f"/api/tasks/{task}/examples/{ex[0]['id']}", json={"label": "redd", "new_class": True}).json()
    assert ("redd" in [c["name"] for c in t["classes"]], t["created_classes"]) == (True, ["redd"])
    t = client.patch(f"/api/tasks/{task}/examples", json={"ids": [ex[1]["id"]], "label": "Sphynx"}).json()
    assert t["created_classes"] == ["Sphynx"]
    t = client.patch(f"/api/tasks/{task}/examples", json={"ids": [ex[1]["id"]], "label": "red"}).json()
    assert t["created_classes"] == []
    assert client.post(f"/api/tasks/{task}/inbox/accept", json={"labels": {inbox[0]: "blue"}}).json()[
        "created_classes"] == []
    _problem(client.patch(f"/api/tasks/{task}/examples/{ex[1]['id']}", json={"label": " "}), 400, "INVALID_REQUEST")


# ---- deduplicated uploads -----------------------------------------------------------------
def test_uploads_say_what_was_new_unchanged_or_moved(client, task):
    """Sending an upload again stores nothing: 200, with each image in unchanged."""
    r = client.post(f"/api/tasks/{task}/examples", data={"label": "red"}, files=_files("red", range(3)))
    assert r.status_code == 200
    assert (r.json()["added"], len(r.json()["unchanged"]), r.json()["moved"]) == ([], 3, [])
    assert r.headers["etag"] == f'"{r.json()["task"]["revision"]}"'
    r = client.post(f"/api/tasks/{task}/examples", data={"label": "red"},
                    files=[*_files("red", [0]), ("files", ("bad.png", b"junk", "image/png"))])
    assert (r.status_code, len(r.json()["unchanged"]), r.json()["errors"][0]["file"]) == (200, 1, "bad.png")
    r = client.post(f"/api/tasks/{task}/inbox", files=_files("u", [60, 61, 60]))
    assert (r.status_code, len(r.json()["added"]), len(r.json()["unchanged"])) == (201, 2, 1), r.json()["errors"]
    r = client.post(f"/api/tasks/{task}/inbox", files=_files("u", [60]))
    assert (r.status_code, r.json()["added"], len(r.json()["unchanged"])) == (200, [], 1)
    body = _problem(client.post(f"/api/tasks/{task}/inbox", files=_files("red", [0])), 400, "NO_IMAGES_ADDED")
    assert body["detail"].startswith("Nothing added to the inbox: red0.png: ")
    assert [e["file"] for e in body["errors"]] == ["red0.png"]  # every readable image is an example already
    r = client.post(f"/api/tasks/{task}/examples", data={"label": "red"},
                    files=[("files", ("bad.png", b"junk", "image/png")), ("files", ("worse.png", b"", "image/png"))])
    body = _problem(r, 400, "NO_IMAGES_ADDED")
    assert [e["file"] for e in body["errors"]] == ["bad.png", "worse.png"]
    assert all(e["error"] for e in body["errors"])


# ---- settings as a merge patch -------------------------------------------------------------
def test_null_clears_a_setting(client, task):
    """PATCH with null clears reject_below, reject_below_zero_shot or temperature; an omitted
    field is unchanged; null on another field is a 422; the clear_* flags still work."""
    t = client.patch(f"/api/tasks/{task}", json={"reject_below": 0.5, "temperature": 2.0,
                                                 "reject_below_zero_shot": 0.2}).json()
    assert (t["reject_below"], t["temperature"], t["reject_below_zero_shot"]) == (0.5, 2.0, 0.2)
    t = client.patch(f"/api/tasks/{task}", json={"reject_below": None}).json()
    assert (t["reject_below"], t["temperature"]) == (None, 2.0)
    t = client.patch(f"/api/tasks/{task}", json={"temperature": None, "reject_below_zero_shot": None}).json()
    assert (t["temperature"], t["reject_below_zero_shot"]) == (None, None)
    r = client.patch(f"/api/tasks/{task}", json={"classifier": None, "text_weight": None})
    body = _problem(r, 422, "VALIDATION_FAILED")
    assert [(e["pointer"], e["detail"]) for e in body["errors"]] == [
        ("/classifier", "classifier can't be null: omit it to leave it unchanged"),
        ("/text_weight", "text_weight can't be null: omit it to leave it unchanged")]
    client.patch(f"/api/tasks/{task}", json={"reject_below": 0.4})
    r = client.patch(f"/api/tasks/{task}", json={"clear_reject": True})
    assert (r.json()["reject_below"], r.headers["deprecation"]) == (None, "true")
    assert "deprecation" not in client.patch(f"/api/tasks/{task}", json={"reject_below": None}).headers
    schema = client.get("/openapi.json").json()["components"]["schemas"]["TaskSettings"]["properties"]
    assert schema["clear_reject"]["deprecated"] is True


# ---- ETag and If-Match --------------------------------------------------------------------
def test_if_match_refuses_changes_to_a_task_changed_since(client, task):
    """Changes with If-Match naming another revision are refused (412) and change nothing."""
    r = client.get(f"/api/tasks/{task}")
    rev = r.json()["revision"]
    assert r.headers["etag"] == f'"{rev}"'
    t = client.post(f"/api/tasks/{task}/classes", json={"name": "green"}, headers={"If-Match": f'"{rev}"'})
    assert (t.status_code, t.headers["etag"]) == (201, f'"{rev + 1}"')
    r = client.patch(f"/api/tasks/{task}", json={"reject_below": 0.3}, headers={"If-Match": f'"{rev}"'})
    body = _problem(r, 412, "PRECONDITION_FAILED")
    assert (body["task"], body["revision"]) == (task, rev + 1)
    assert client.get(f"/api/tasks/{task}").json()["reject_below"] is None
    r = client.post(f"/api/tasks/{task}/examples", data={"label": "red"}, files=_files("x", [9]),
                    headers={"If-Match": f'"{rev}"'})
    _problem(r, 412, "PRECONDITION_FAILED")
    for tag in ('W/"{}"', "*", '"1", "{}"'):
        now = client.get(f"/api/tasks/{task}").json()["revision"]
        r = client.post(f"/api/tasks/{task}/describe", json={"overwrite": True}, headers={"If-Match": tag.format(now)})
        assert r.status_code == 200
    _problem(client.delete("/api/tasks/nope", headers={"If-Match": '"1"'}), 404, "TASK_NOT_FOUND")


def test_if_match_is_checked_with_the_change_under_the_task_lock(client, task, monkeypatch):
    """A change that passed the check on arrival is still refused if the task moved on before it
    took the task's lock, on Starlette's threads and the heavy-work ones alike."""
    rev = client.get(f"/api/tasks/{task}").json()["revision"]
    client.post(f"/api/tasks/{task}/classes", json={"name": "green"})  # lands between the two checks

    async def as_on_arrival(*_a):  # the task as the check on arrival saw it
        return SimpleNamespace(config=SimpleNamespace(revision=rev))

    monkeypatch.setattr(fewshot.routing, "run_in_threadpool", as_on_arrival)
    r = client.post(f"/api/tasks/{task}/classes", json={"name": "amber"}, headers={"If-Match": f'"{rev}"'})
    body = _problem(r, 412, "PRECONDITION_FAILED")
    assert (body["task"], body["revision"]) == (task, rev + 1)
    r = client.post(f"/api/tasks/{task}/examples", data={"label": "red"}, files=_files("x", [9]),
                    headers={"If-Match": f'"{rev}"'})
    _problem(r, 412, "PRECONDITION_FAILED")
    after = client.get(f"/api/tasks/{task}").json()
    assert (after["revision"], len(after["classes"]), len(after["examples"])) == (rev + 1, 3, 6)
    monkeypatch.undo()
    r = client.post(f"/api/tasks/{task}/classes", json={"name": "amber"}, headers={"If-Match": f'"{rev + 1}"'})
    assert r.status_code == 201  # only the first change of a request is checked: it moved the revision on


def test_concurrent_changes_with_one_if_match_make_one_change(client, task):
    """Twenty changes sent at once naming the same revision: exactly one is made."""
    rev = client.get(f"/api/tasks/{task}").json()["revision"]

    def add(i):
        return client.post(f"/api/tasks/{task}/classes", json={"name": f"class {i}", "new_class": True},
                           headers={"If-Match": f'"{rev}"'}).status_code

    with ThreadPoolExecutor(8) as pool:
        codes = sorted(pool.map(add, range(20)))
    assert codes == [201] + [412] * 19
    assert client.get(f"/api/tasks/{task}").json()["revision"] == rev + 1


def test_an_if_match_is_checked_until_a_change_to_its_task_saves():
    """Outside a request nothing is checked; inside one, each change to the task named, until one
    of them has saved."""

    def changes():
        preconditions.check("a", 1)  # nothing expected
        preconditions.expect("a", '"1"')
        preconditions.check("b", 5)  # another task
        preconditions.check("a", 1)
        with pytest.raises(RevisionChanged):  # nothing saved yet: still checked
            preconditions.check("a", 7)
        preconditions.settle("b")
        preconditions.settle("a")
        preconditions.check("a", 7)  # a change saved: it moved the revision on
        preconditions.expect("a", 'W/"1"')
        with pytest.raises(RevisionChanged, match="at revision 2, not W/\"1\"") as e:
            preconditions.check("a", 2)
        assert (e.value.fields, http_status(e.value), error_code(e.value)) == (
            {"task": "a", "revision": 2}, 412, "PRECONDITION_FAILED")

    contextvars.copy_context().run(changes)


def test_an_upload_in_chunks_is_checked_until_one_saves(ws):
    """A large upload is stored in chunks, each under the lock in turn: while the chunks so far
    changed nothing (images it has already), each later one is checked against If-Match too, so
    a change made meanwhile by someone else refuses the rest; once one has saved, the rest go on."""
    t = ws.create("t", embedder="fake:color")
    old = png(RED, noise=10, seed=1)
    t.add_example(old, "red")
    rev = t.config.revision

    def items(change):
        yield old, "red", "old.png"  # already there: this chunk saves nothing
        if change:  # between chunks, outside the lock, another client changes the task
            contextvars.Context().run(ws.get("t").upsert_class, "other")
        yield png(GREEN, noise=10, seed=2), "red", "new.png"

    def upload(change, if_match):
        preconditions.expect("t", if_match)
        return t.add_examples(items(change), chunk=1)

    with pytest.raises(RevisionChanged):
        contextvars.copy_context().run(upload, True, f'"{rev}"')
    assert (len(t.config.examples), t.config.revision) == (1, rev + 1)  # only the other client's change
    added = contextvars.copy_context().run(upload, False, f'"{rev + 1}"')
    assert (len(added), len(t.config.examples)) == (2, 2)


# ---- jobs ---------------------------------------------------------------------------------
def test_jobs_say_where_to_poll_and_why_they_failed(client, task, monkeypatch):
    """202 carries Location and Retry-After; a failed job has a problem with its code."""
    r = client.post(f"/api/tasks/{task}/evaluate/jobs")
    assert (r.status_code, r.headers["location"], r.headers["retry-after"]) == (
        202, f"/api/jobs/{r.json()['id']}", "1")
    job = wait_for_job(client, r.json()["id"])
    assert (job["status"], job["problem"]) == ("done", None)
    assert "retry-after" not in client.get(f"/api/jobs/{job['id']}").headers

    def gone(*_a, **_k):
        raise NoExamples("Task has no examples to evaluate", task=task)

    monkeypatch.setattr(fewshot.services, "evaluate", gone)
    job = wait_for_job(client, client.post(f"/api/tasks/{task}/evaluate/jobs", params={"folds": 2}).json()["id"])
    assert (job["status"], job["error"]) == ("failed", "Task has no examples to evaluate")
    assert {k: job["problem"][k] for k in ("status", "code", "job_id", "task", "instance")} == {
        "status": 400, "code": "NO_EXAMPLES", "job_id": job["id"], "task": task, "instance": f"/api/jobs/{job['id']}"}


def test_a_job_failing_on_a_missing_thing_or_a_bug():
    """A KeyError fails a job with NOT_FOUND (404); anything else with INTERNAL (500), its message hidden."""
    jobs = Jobs()

    def missing(_progress):
        raise KeyError("No class 'x'")

    def bug(_progress):
        raise RuntimeError("/secret")

    for work, want in ((missing, ("No class 'x'", "NOT_FOUND", 404)),
                       (bug, ("Internal error; see the server log", "INTERNAL", 500))):
        job = jobs.start(Job(task="t", kind=work.__name__), work)
        for _ in range(500):
            if job.status == "failed":
                break
            __import__("time").sleep(0.01)
        assert (job.error, job.code, job.error_status) == want


# ---- versioning and the schema -------------------------------------------------------------
def test_the_server_says_its_version(client):
    """/api/health and every response's X-Fewshot-Version give the version; OpenAPI does too."""
    r = client.get("/api/health")
    assert (r.json()["version"], r.json()["api_version"]) == (fewshot.__version__, "1")
    assert r.headers["x-fewshot-version"] == fewshot.__version__
    assert client.get("/api/nope").headers["x-fewshot-version"] == fewshot.__version__
    assert client.get("/openapi.json").json()["info"]["version"] == fewshot.__version__


def test_openapi_fits_generated_clients(client):
    """Every operation has an id, a summary, tags and a 4xx answer of the Problem schema; enums
    are enums."""
    spec = client.get("/openapi.json").json()
    ops = [(path, method, op) for path, item in spec["paths"].items() for method, op in item.items()]
    assert len(ops) > 30
    for path, method, op in ops:
        where = f"{method.upper()} {path}"
        assert op.get("operationId"), where
        assert op.get("summary"), where
        assert op.get("tags"), where
        assert not op["summary"].istitle() or len(op["summary"].split()) < 2, where  # not a title-cased name
        problems = [code for code, resp in op["responses"].items() if code.startswith("4")
                    and "#/components/schemas/Problem" in str(resp)]
        assert problems, where
    schemas = spec["components"]["schemas"]
    assert schemas["JobOut"]["properties"]["status"]["enum"] == ["running", "cancelling", "done", "failed",
                                                                 "cancelled"]
    sort = next(p for p in spec["paths"]["/api/tasks/{name}/inbox"]["get"]["parameters"] if p["name"] == "sort")
    assert sort["schema"]["enum"] == ["uncertain", "confident", "novel", "added"]
    assert {t["name"] for t in spec["tags"]} == {"tasks", "classes", "examples", "inbox", "predict", "evaluate",
                                                  "jobs", "auth", "system"}
    assert "examples" in schemas["TaskCreate"]


def test_openapi_declares_problem_json_if_match_and_etag(client):
    """Errors are declared as application/problem+json (as sent), changes take an If-Match header,
    responses carrying a task declare its ETag, and JSON routes their 415."""
    spec = client.get("/openapi.json").json()
    assert client.get("/openapi.json").json() == spec  # amended once
    for item in spec["paths"].values():
        for op in item.values():
            for code, resp in op["responses"].items():
                if code[0] in "45":
                    assert list(resp["content"]) == ["application/problem+json"], (op["operationId"], code)
    op = spec["paths"]["/api/tasks/{name}/classes"]["post"]
    header = next(p for p in op["parameters"] if p["in"] == "header")
    assert (header["name"], header["required"]) == ("If-Match", False)
    assert set(op["responses"]["201"]["headers"]) == {"ETag"}
    assert "415" in op["responses"]
    assert "ETag" in spec["paths"]["/api/tasks/{name}/examples"]["post"]["responses"]["201"]["headers"]
    assert "ETag" in spec["paths"]["/api/tasks/{name}/inbox/accept"]["post"]["responses"]["200"]["headers"]
    assert "headers" not in spec["paths"]["/api/health"]["get"]["responses"]["200"]
    assert not any(p["in"] == "header" for p in spec["paths"]["/api/tasks/{name}"]["get"].get("parameters", []))
    assert "415" not in spec["paths"]["/api/tasks/{name}/undo"]["post"]["responses"]


def test_failed_predictions_have_null_scores(client, task):
    """An unreadable image's prediction has null scores, not zeros that would pass for real ones."""
    r = client.post(f"/api/tasks/{task}/predict", files=[("files", ("bad.png", b"junk", "image/png"))])
    [p] = r.json()
    assert (p["top_label"], p["confidence"], p["similarity"], p["label"], p["scores"], p["rejected"]) == (
        None, None, None, None, {}, False)
    assert p["error"]


def test_reports_name_each_examples_file(client, task):
    """Eval reports give each prediction's source, and duplicate pairs their sources, as the CLI's JSON does."""
    report = client.post(f"/api/tasks/{task}/evaluate").json()
    assert {p["source"] for p in report["predictions"]} == {f"{c}{i}.png" for c in ("red", "blue") for i in range(3)}
    client.post(f"/api/tasks/{task}/examples", data={"label": "blue"}, files=_files("again", [0]))
    pairs = client.get(f"/api/tasks/{task}/duplicates", params={"threshold": 0.5}).json()
    assert all(len(d["sources"]) == 2 for d in pairs)
