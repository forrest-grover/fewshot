"""Per-class reject thresholds: storing them, predicting with them, suggesting them, and the API and CLI."""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from fewshot import cli
from fewshot import evaluate as ev
from fewshot.classifiers import ClassData, make_classifier
from fewshot.evaluate import evaluate_task
from fewshot.pipeline import Pipeline
from fewshot.store import Workspace
from tests.conftest import BLUE, RED, PlainResult, api_client, png, red_blue_task

PURPLE = (150, 30, 160)  # between the two, nearer blue


def _task(ws, n=6):
    t = ws.create("t", embedder="fake:color")
    for i in range(n):
        t.add_example(png(RED, noise=15, seed=i), "red")
        t.add_example(png(BLUE, noise=15, seed=i), "blue")
    return t


# ---- the store --------------------------------------------------------------
def test_set_and_clear_class_thresholds(ws):
    """Thresholds are set and cleared in one undoable change; bad input changes nothing."""
    t = _task(ws, n=1)
    t.set_class_thresholds({"red": 0.9, "blue": 0.8})
    assert [c.reject_below for c in t.config.classes] == [0.9, 0.8]
    assert t.undo_label() == "set thresholds of 2 classes"
    t.set_class_thresholds({"red": None})
    assert t.undo_label() == "set the threshold of red"
    assert [c.reject_below for c in Workspace(ws.root).get("t").config.classes] == [None, 0.8]  # saved
    t.undo()
    assert t.config.classes[0].reject_below == 0.9
    rev = t.config.revision
    with pytest.raises(KeyError, match="No class 'green'"):
        t.set_class_thresholds({"red": 0.1, "green": 0.5})
    with pytest.raises(ValueError, match="between -1 and 1"):
        t.set_class_thresholds({"red": 1.5})
    assert (t.config.revision, t.config.classes[0].reject_below) == (rev, 0.9)


def test_rename_keeps_the_threshold(ws):
    """A renamed class keeps its own threshold."""
    t = _task(ws, n=1)
    t.set_class_thresholds({"red": 0.9})
    t.rename_class("red", "crimson")
    assert t.config.classes[0].model_dump() == {"name": "crimson", "description": "", "reject_below": 0.9}


# ---- prediction ---------------------------------------------------------------
def test_class_threshold_overrides_the_task_one(ws):
    """An image predicted as blue is judged by blue's own threshold, before the task's."""
    t = _task(ws, n=2)
    pipe = Pipeline(ws)
    sim = pipe.predict(t, [png(PURPLE)])[0].similarity
    t.update_settings(reject_below=sim - 0.01)
    assert pipe.predict(t, [png(PURPLE)])[0].label == "blue"
    t.set_class_thresholds({"blue": sim + 0.001})  # stricter for blue only
    p = pipe.predict(t, [png(PURPLE)])[0]
    assert (p.rejected, p.label, p.top_label) == (True, None, "blue")
    t.update_settings(reject_below=None)
    t.set_class_thresholds({"blue": sim - 0.01, "red": None})  # blue's own, with no task threshold
    assert pipe.predict(t, [png(PURPLE)])[0].label == "blue"


def test_other_labels_use_their_own_thresholds(ws):
    """Multi-label: another class is listed when the image reaches that class's own threshold."""
    t = _task(ws, n=2)
    pipe = Pipeline(ws)
    p = pipe.predict(t, [png(PURPLE)])[0]
    clf = pipe.fitted(t)
    per_class = clf.class_similarity(pipe.embed_images("fake:color", [(None, png(PURPLE))]))[0]
    red_sim = float(per_class[clf.classes.index("red")])
    t.set_class_thresholds({"blue": 0.0, "red": red_sim - 0.001})
    assert pipe.predict(t, [png(PURPLE)])[0].labels == ["blue", "red"]
    t.set_class_thresholds({"red": red_sim + 0.001})
    assert pipe.predict(t, [png(PURPLE)])[0].labels == ["blue"]
    assert p.labels == ["blue"]  # with no thresholds at all, just the label


def test_zero_shot_ignores_class_thresholds(ws):
    """Without example images the zero-shot threshold applies, not the classes' own."""
    from fewshot.pipeline import threshold_for  # pylint: disable=import-outside-toplevel

    text = {0: np.array([1, 0, 0], np.float32), 1: np.array([0, 1, 0], np.float32)}
    zero = make_classifier("prototype").fit(ClassData(["a", "b"], np.zeros((0, 3), np.float32), np.zeros(0, int), text))
    assert zero.zero_shot
    t = ws.create("z", embedder="fake:color")
    t.upsert_class("a")
    t.upsert_class("b")
    t.set_class_thresholds({"a": 0.9})
    assert threshold_for(t.config, zero) is None
    t.update_settings(reject_below_zero_shot=0.25)
    thr = threshold_for(t.config, zero)
    assert thr is not None
    assert thr.tolist() == [0.25, 0.25]


# ---- evaluation -----------------------------------------------------------------
def test_evaluation_suggests_class_thresholds(ws):
    """Classes with 5+ correct held-out predictions get a suggested threshold of their own."""
    t = _task(ws, n=5)
    t.add_example(png((30, 220, 30), seed=1), "green")
    r = evaluate_task(Pipeline(ws), t)
    assert set(r["suggested_class_thresholds"]) == {"red", "blue"}
    assert all(0 < v <= 1 for v in r["suggested_class_thresholds"].values())


def test_suggested_threshold_keeps_95_percent_of_new_members():
    """A suggested threshold is a conformal order statistic: a new member as similar as the
    held-out ones reaches it with probability >= 95% (n / (n + 1) below 19 of them, where only the
    lowest will do), which the sample's 5% quantile missed at small n (about 80% at n = 5)."""
    rng = np.random.default_rng(0)
    for n in (5, 10, 19, 40, 100):
        kept = []
        for _ in range(2000):
            sims = rng.random(n)
            kept.append(rng.random() >= ev.keep_threshold(sims))
        assert np.mean(kept) >= min(0.95, n / (n + 1)) - 0.015
    sims = np.arange(1, 40) / 100  # 0.01 .. 0.39
    assert ev.keep_threshold(sims[:5]) == 0.01  # the lowest
    assert ev.keep_threshold(sims[:19]) == 0.01  # floor(0.05 * 20) = 1st
    assert ev.keep_threshold(sims) == 0.02  # floor(0.05 * 40) = 2nd
    assert ev.keep_threshold(np.array([0.8126, 0.9])) == 0.812  # rounded down, so 0.8126 still passes


# ---- API ------------------------------------------------------------------------
@pytest.fixture(name="client")
def client_fixture(tmp_path):
    """A TestClient on a fresh API, with task 't'."""
    c = api_client(tmp_path)
    _task(c.srv.ws, n=5)
    return c


def test_api_class_thresholds(client):
    """PUT .../class-thresholds sets and clears; unknown classes, bad values and nothing at all are refused."""
    t = client.put("/api/tasks/t/class-thresholds", json={"thresholds": {"red": 0.8}}).json()
    assert [c["reject_below"] for c in t["classes"]] == [0.8, None]
    t = client.put("/api/tasks/t/class-thresholds", json={"thresholds": {"red": None}}).json()
    assert t["classes"][0]["reject_below"] is None
    assert client.put("/api/tasks/t/class-thresholds", json={"thresholds": {"x": 0.5}}).status_code == 404
    assert client.put("/api/tasks/t/class-thresholds", json={"thresholds": {"red": 2}}).status_code == 400
    assert client.put("/api/tasks/t/class-thresholds", json={"thresholds": {}}).status_code == 400
    r = client.post("/api/tasks/t/evaluate").json()
    assert set(r["suggested_class_thresholds"]) == {"red", "blue"}


# ---- CLI ------------------------------------------------------------------------
@pytest.fixture(name="run")
def run_fixture(tmp_path):
    """Invoke the CLI against a workspace with task 't'."""
    root = tmp_path / "ws"
    _task(Workspace(root), n=5)
    runner = CliRunner()
    return lambda *args: PlainResult(runner.invoke(cli.app, [str(a) for a in args] + ["--root", str(root)]))


def test_cli_class_threshold(run):
    """`class threshold` sets or clears; show lists it; mistakes are errors."""
    assert run("class", "threshold", "t", "red", "0.85").exit_code == 0
    assert run("class", "threshold", "t", "red", "-0.5").exit_code == 0  # a negative value isn't an option
    assert "reject_below -0.5" in run("show", "t").output
    assert run("class", "threshold", "t", "red", "-1.5").exit_code == 2  # out of range
    assert run("class", "threshold", "t", "red", "0.85").exit_code == 0
    assert "reject_below 0.85" in run("show", "t").output
    assert json.loads(run("show", "t", "--json").output)["classes"][0]["reject_below"] == 0.85
    assert run("class", "threshold", "t", "red", "--clear").exit_code == 0
    assert "reject_below 0.85" not in run("show", "t").output
    assert "either a threshold or --clear" in run("class", "threshold", "t", "red").output
    assert "either a threshold or --clear" in run("class", "threshold", "t", "red", "0.5", "--clear").output
    r = run("class", "threshold", "t", "nope", "0.5")
    assert (r.exit_code, "No class 'nope'" in r.output) == (1, True)


def test_cli_eval_applies_class_thresholds(run):
    """eval prints each class's suggestion, and --apply-class-thresholds sets them."""
    out = run("eval", "t").output
    assert "suggested threshold" in out
    assert "--apply-class-thresholds" in out
    r = run("eval", "t", "--apply-class-thresholds", "--json")
    assert "Set the thresholds of 2 classes" in r.stderr
    suggested = json.loads(r.stdout)["suggested_class_thresholds"]
    classes = json.loads(run("show", "t", "--json").output)["classes"]
    assert {c["name"]: c["reject_below"] for c in classes} == suggested
    assert json.loads(r.stdout)["revision"] == json.loads(run("show", "t", "--json").output)["revision"]
    assert "*" not in run("list").output  # the evaluation still counts as current


def test_cli_eval_apply_with_a_class_gone(run, monkeypatch):
    """A class renamed while the evaluation ran is an error, not a traceback."""
    def gone(_self, _thresholds):
        raise KeyError("No class 'red'")

    monkeypatch.setattr("fewshot.store.Task.set_class_thresholds", gone)
    r = run("eval", "t", "--apply-class-thresholds")
    assert (r.exit_code, "Thresholds not set: No class 'red'" in r.output) == (1, True)


def test_cli_config_says_what_switching_embedder_cleared(run, monkeypatch):
    """`config --embedder` reports the thresholds it cleared, and how to get them back."""
    monkeypatch.setattr("fewshot.services.check_embedder", lambda spec: None)
    run("class", "threshold", "t", "red", "0.8")
    r = run("config", "t", "--embedder", "fake:other")
    assert "Cleared class thresholds: fitted to the previous settings (`fewshot undo t` restores them)" in r.output


def test_cli_eval_apply_keeps_an_old_report_old(run, monkeypatch):
    """An evaluation of an older revision isn't moved to the new one by applying its thresholds."""
    real = evaluate_task

    def older(*a, **k):
        r = real(*a, **k)
        return {**r, "revision": r["revision"] - 1}

    monkeypatch.setattr("fewshot.evaluate.evaluate_task", older)
    r = run("eval", "t", "--apply-class-thresholds", "--json")
    assert json.loads(r.stdout)["revision"] < json.loads(run("show", "t", "--json").output)["revision"] - 1


def test_empty_threshold_change_does_nothing(ws):
    """set_class_thresholds({}) saves nothing and adds no undo step."""
    t = red_blue_task(ws)
    rev = t.config.revision
    t.set_class_thresholds({})
    assert t.config.revision == rev
