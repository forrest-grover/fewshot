"""Recorded evaluations, warnings for the dashboard, and duplicating tasks."""

import json

import pytest
from typer.testing import CliRunner

import fewshot.evaluations
import fewshot.store
from fewshot import cli
from fewshot.store import Workspace
from tests.conftest import BLUE, RED, api_client, png


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


def test_evaluation_report_and_history(ws, monkeypatch):
    """record_evaluation keeps the whole report and a short history of headlines with how each
    was made; a task evaluated by an older version (headline only) reads as no report."""
    t = _task(ws)
    assert (t.evaluation_report(), t.evaluation_history(), t.last_comparison()) == (None, [], None)
    t.record_evaluation({"revision": 6, "accuracy": 0.5})
    assert t.evaluation_report()["accuracy"] == 0.5
    (t.path / "eval-report.json").unlink()
    (t.path / "eval-history.json").unlink()
    assert t.evaluation_report() is None  # headline only, as older versions kept
    assert [h["accuracy"] for h in t.evaluation_history()] == [0.5]
    monkeypatch.setattr(fewshot.evaluations, "EVAL_HISTORY", 3)
    for i in range(4):
        t.record_evaluation({"revision": 6, "accuracy": i / 4, "embedder": "fake:color", "folds": 2,
                             "finished_at": f"2026-01-0{i + 1}T00:00:00Z", "confusion": [[1]]})
    h = t.evaluation_history()
    assert [x["accuracy"] for x in h] == [0.25, 0.5, 0.75]
    assert [x["at"][:10] for x in h] == ["2026-01-02", "2026-01-03", "2026-01-04"]
    assert (h[-1]["embedder"], h[-1]["folds"], "confusion" in h[-1]) == ("fake:color", 2, False)
    assert t.evaluation_report()["confusion"] == [[1]]
    (t.path / "eval-history.json").write_text('[1, {"at": "x"}]')
    assert t.evaluation_history() == [{"at": "x"}]
    t.record_comparison({"revision": 6, "best": "fake:color"})
    assert t.last_comparison()["best"] == "fake:color"


def test_thresholds_carry_the_evaluation_report(ws):
    """A change evaluation doesn't measure moves the headline and the report to the new revision."""
    t = _task(ws)
    rev = t.config.revision
    t.record_evaluation({"revision": rev, "accuracy": 1.0})
    t.update_settings(reject_below=0.5)
    assert t.last_evaluation()["revision"] == t.evaluation_report()["revision"] == t.config.revision != rev
    (t.path / "eval-report.json").write_text("[]")
    t.set_class_thresholds({"red": 0.4})
    assert t.last_evaluation()["revision"] == t.config.revision


def test_records_skip_a_deleted_task(ws):
    """An evaluation or comparison finishing after its task was deleted is dropped."""
    t = _task(ws)
    ws.delete("t")
    t.record_evaluation({"revision": 1})
    t.record_comparison({"revision": 1})
    assert not (t.path / "eval-report.json").exists()
    assert not (t.path / "compare.json").exists()
    again = _task(ws)  # a new task of the same name
    t.record_comparison({"revision": 1})
    assert again.last_comparison() is None


def test_records_that_cant_be_written_are_logged(ws, monkeypatch, caplog):
    """A task this process can't write to keeps no evaluation or comparison, and says so in the
    log, rather than failing the evaluation or comparison that has been made."""
    t = _task(ws)

    def refuse(path, _data):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(fewshot.evaluations, "_write_atomic", refuse)
    t.record_evaluation({"revision": 1, "accuracy": 1.0})
    t.record_comparison({"revision": 1, "best": "fake:color"})
    assert (t.evaluation_report(), t.last_comparison()) == (None, None)
    assert ["evaluation couldn't be kept" in m or "comparison couldn't be kept" in m
            for m in caplog.messages] == [True, True]


def test_kept_files_that_dont_fit_are_left_out(tmp_path):
    """GET .../evaluation leaves out a kept report, headline or comparison that doesn't fit this
    version's schema (another version's, or edited by hand), and answers with the rest."""
    client = api_client(tmp_path)
    t = _task(client.srv.ws)
    client.post("/api/tasks/t/evaluate")
    (t.path / "eval-history.json").write_text(json.dumps([{"at": "x"}, t.evaluation_history()[-1]]))
    (t.path / "compare.json").write_text('{"best": 3}')
    (t.path / "eval-report.json").write_text('{"revision": 1, "accuracy": 1.0}')
    ev = client.get("/api/tasks/t/evaluation").json()
    assert (ev["report"], len(ev["history"]), ev["comparison"]) == (None, 1, None)


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

    monkeypatch.setattr(fewshot.store.Task, "_add_file", broken)
    with pytest.raises(OSError, match="disk full"):
        ws.duplicate("t", "c")
    assert not (ws.root / "tasks" / "c").exists()
    assert not [p for p in (ws.root / "tasks").iterdir() if p.name.startswith(".")]


def test_api(tmp_path, monkeypatch):
    """Evaluations are recorded (sync and as a job) and duplicate works over the API."""
    client = api_client(tmp_path)
    api = client.srv
    _task(api.ws)
    client.post("/api/tasks/t/evaluate")
    s = client.get("/api/tasks").json()[0]
    assert s["last_evaluation"]["accuracy"] == 1.0
    assert s["last_evaluation"]["revision"] == s["revision"]
    ev = client.get("/api/tasks/t/evaluation").json()
    assert (ev["revision"], ev["report"]["accuracy"], ev["report"]["embedder"]) == (s["revision"], 1.0, "fake:color")
    assert (len(ev["history"]), ev["history"][0]["method"], ev["comparison"]) == (1, "leave-one-out", None)
    assert client.get("/api/tasks/t/jobs").json() == []
    r = client.post("/api/tasks/t/evaluate/jobs", json={"folds": 2})
    jobs = client.get("/api/tasks/t/jobs").json()
    assert [j["id"] for j in jobs] == [r.json()["id"]]
    assert client.get("/api/tasks/nope/evaluation").status_code == 404
    r = client.post("/api/tasks/t/duplicate", json={"name": "u", "examples": False})
    assert (r.status_code, r.json()["examples"]) == (201, [])
    assert client.post("/api/tasks/t/duplicate", json={"name": "u"}).status_code == 409
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

    assert "        -  " in run("list").stdout
    run("eval", "t")
    assert "   1.00   " in run("list").stdout
    run("class", "add", "t", "green")
    r = run("list")
    assert "   1.00*  " in r.stdout  # changed since
    assert "* evaluated on an older revision; re-run `fewshot eval TASK`" in r.stderr
    assert run("copy", "t", "u", "--no-examples").output.strip() == "Copied t to u (0 examples)"
    assert run("copy", "t", "u").exit_code == 1
