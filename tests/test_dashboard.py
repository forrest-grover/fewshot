"""Recorded evaluations, warnings for the dashboard, and duplicating tasks."""

import importlib

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import fewshot.api
import fewshot.store
from fewshot import cli
from fewshot.store import Workspace
from tests.conftest import png

RED, BLUE = (220, 30, 30), (30, 30, 220)


def _task(ws, name="t", n=3):
    t = ws.create(name, embedder="fake:color")
    for i in range(n):
        t.add_example(png(RED, noise=10, seed=i), "red")
        t.add_example(png(BLUE, noise=10, seed=i), "blue")
    return t


def test_warnings(ws):
    """Thin classes, imbalance and empty tasks are flagged; described classes may be thin."""
    t = ws.create("t", embedder="fake:color")
    assert t.summary()["warnings"] == ["Add classes and examples to start"]
    for i in range(12):
        t.add_example(png(RED, noise=10, seed=i), "red")
    t.add_example(png(BLUE), "blue")
    t.upsert_class("green", "a green thing")
    w = t.summary()["warnings"]
    assert w == ["Fewer than 2 examples in blue: evaluation can't score them",
                 "Classes are unbalanced (1 to 12 examples)"]
    for c in "abcd":
        t.upsert_class(c)
    assert "in blue, a, b, ...:" in t.summary()["warnings"][0]


def test_recorded_evaluation(ws):
    """record_evaluation keeps the headline; a missing or broken file means none."""
    t = _task(ws)
    assert t.last_evaluation() is None
    t.record_evaluation({"revision": 6, "method": "leave-one-out", "accuracy": 1.0, "macro_f1": 1.0,
                         "n_scored": 6, "n_examples": 6, "confusion": [[1]]})
    ev = t.summary()["last_evaluation"]
    assert (ev["revision"], ev["accuracy"]) == (6, 1.0)
    assert "confusion" not in ev
    (t.path / "eval.json").write_text("[1, 2]")
    assert t.last_evaluation() is None
    (t.path / "eval.json").write_text("{bad")
    assert t.last_evaluation() is None


def test_duplicate(ws):
    """A copy has the settings and classes, examples (and inbox) if asked, its own uid and history."""
    t = _task(ws)
    t.update_settings(classifier="knn")
    t.add_to_inbox([(png(RED, noise=10, seed=9), "x.png")])
    c = ws.duplicate("t", "c")
    assert (c.config.classifier, len(c.config.examples), c.config.inbox) == ("knn", 6, [])
    assert c.config.uid != t.config.uid
    assert not c.can_undo()
    assert c.image_path(c.config.examples[0].id).exists()
    bare = ws.duplicate("t", "bare", examples=False, inbox=True)
    assert (bare.class_names(), len(bare.config.examples), len(bare.config.inbox)) == (["red", "blue"], 0, 1)
    with pytest.raises(ValueError, match="already exists"):
        ws.duplicate("t", "c")
    with pytest.raises(KeyError):
        ws.duplicate("nope", "x")


def test_duplicate_leaves_out_missing_images(ws):
    """An example or inbox image whose file is gone (a damaged task) is left out of the copy."""
    t = _task(ws, n=2)
    t.add_to_inbox([(png(RED, noise=10, seed=9), "x.png")])
    (t.path / t.config.examples[0].file).unlink()
    (t.path / t.config.inbox[0].file).unlink()
    c = ws.duplicate("t", "c", inbox=True)
    assert [e.id for e in c.config.examples] == [e.id for e in t.config.examples[1:]]
    assert c.config.inbox == []


def test_failed_duplicate_leaves_nothing(ws, monkeypatch):
    """If copying fails part-way, the new folder is removed."""
    _task(ws)

    def broken(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(fewshot.store.Task, "_store_image", broken)
    with pytest.raises(OSError, match="disk full"):
        ws.duplicate("t", "c")
    assert not (ws.root / "tasks" / "c").exists()


def test_api(tmp_path, monkeypatch):
    """Evaluations are recorded (sync and as a job) and duplicate works over the API."""
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path / "ws"))
    api = importlib.reload(fewshot.api)
    client = TestClient(api.app)
    _task(api.ws)
    client.post("/api/tasks/t/evaluate")
    s = client.get("/api/tasks").json()[0]
    assert s["last_evaluation"]["accuracy"] == 1.0
    assert s["last_evaluation"]["revision"] == s["revision"]
    r = client.post("/api/tasks/t/duplicate", json={"name": "u", "examples": False})
    assert (r.status_code, r.json()["examples"]) == (201, [])
    assert client.post("/api/tasks/t/duplicate", json={"name": "u"}).status_code == 400
    assert client.post("/api/tasks/nope/duplicate", json={"name": "v"}).status_code == 404

    def gone(*_a):
        raise KeyError("No task 't'")

    monkeypatch.setattr(api.ws, "duplicate", gone)
    assert client.post("/api/tasks/t/duplicate", json={"name": "w"}).status_code == 404


def test_cli(tmp_path):
    """eval records its result for `list`; `copy` duplicates."""
    root = tmp_path / "ws"
    _task(Workspace(root))
    runner = CliRunner()

    def run(*args):
        return runner.invoke(cli.app, [str(a) for a in args] + ["--root", str(root)])

    assert "      -" in run("list").output
    run("eval", "t")
    assert "acc 1.00 " in run("list").output
    run("class", "add", "t", "green")
    assert "acc 1.00*" in run("list").output  # changed since
    assert run("copy", "t", "u", "--no-examples").output.strip() == "Copied t to u (0 examples)"
    assert run("copy", "t", "u").exit_code == 1
