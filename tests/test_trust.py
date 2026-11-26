"""Evidence (nearest labelled examples) and confidence calibration."""

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import fewshot.api
import fewshot.evaluate as ev
from fewshot import cli
from fewshot.classifiers import ClassData, make_classifier, temper
from fewshot.evaluate import evaluate_task
from fewshot.pipeline import Pipeline
from fewshot.store import TaskConfig
from tests.conftest import png

RED, GREEN, BLUE = (220, 30, 30), (30, 220, 30), (30, 30, 220)


def unit(rows):
    """Rows scaled to unit length, as float32."""
    a = np.array(rows, np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


# ---- evidence -----------------------------------------------------------
def test_nearest_skips_text_anchors():
    """nearest returns example rows only, most similar first; text anchors and zero-shot give none."""
    data = ClassData(["a", "b", "c"], unit([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]]), np.array([0, 0, 1]),
                     {2: unit([[0, 0, 1]])[0]})
    clf = make_classifier("knn").fit(data)
    idx, sims = clf.nearest(unit([[0, 0.1, 1]]), 5)  # nearest is the text anchor, which is left out
    assert idx.tolist() == [[2, 1, 0]]
    assert sims[0, 0] > sims[0, 1] > sims[0, 2]
    zero = make_classifier("prototype").fit(ClassData(["a", "b"], np.zeros((0, 3), np.float32), np.zeros(0, int),
                                                      {0: unit([[1, 0, 0]])[0], 1: unit([[0, 1, 0]])[0]}))
    assert zero.nearest(unit([[1, 0, 0]]), 3)[0].shape == (1, 0)


def test_predictions_carry_evidence(ws):
    """With evidence=k each prediction lists the k most similar examples; errors get none."""
    t = ws.create("t", embedder="fake:color")
    ids = {label: t.add_example(png(rgb, noise=10, seed=1), label).id for label, rgb in
           (("red", RED), ("green", GREEN), ("blue", BLUE))}
    t.add_example(png(RED, noise=10, seed=2), "red")
    good, bad = Pipeline(ws).predict(t, [png((200, 50, 40)), b"junk"], evidence=2)
    assert [e["label"] for e in good.evidence] == ["red", "red"]
    assert good.evidence[0]["similarity"] >= good.evidence[1]["similarity"]
    assert ids["red"] in {e["id"] for e in good.evidence}
    assert bad.evidence == []
    assert Pipeline(ws).predict(t, [png(RED)])[0].evidence == []


# ---- calibration --------------------------------------------------------
def test_temper():
    """Tempering keeps the top class, softens with T > 1 and sharpens with T < 1."""
    p = np.array([[0.7, 0.2, 0.1], [0.0, 1.0, 0.0]])
    assert temper(p, 1.0) is p
    soft, sharp = temper(p, 3.0), temper(p, 0.3)
    assert (soft.argmax(1) == p.argmax(1)).all()
    assert soft[0, 0] < 0.7 < sharp[0, 0]
    assert np.allclose(soft.sum(1), 1)
    assert soft[1].tolist() == [0.0, 1.0, 0.0]


def test_task_temperature_changes_confidence(ws):
    """The task's temperature is applied to predictions; out-of-range values are refused."""
    t = ws.create("t", embedder="fake:color")
    for rgb, label in ((RED, "red"), ((200, 120, 30), "orange")):
        t.add_example(png(rgb), label)
    pipe = Pipeline(ws)
    q = png((210, 80, 30))
    before = pipe.predict(t, [q])[0].confidence
    t.update_settings(temperature=5.0)
    after = pipe.predict(t, [q])[0]
    assert 0.5 <= after.confidence < before
    with pytest.raises(ValueError, match="temperature must be between"):
        t.update_settings(temperature=0)


def _noisy_task(ws, scale):
    t = ws.create("t", embedder="fake:color", classifier="prototype", classifier_params={"scale": scale})
    rng = np.random.default_rng(0)
    for label, rgb in (("red", (200, 60, 60)), ("orange", (200, 90, 60)), ("pink", (200, 60, 90))):  # overlapping
        for i in range(30):
            t.add_example(png(tuple(int(np.clip(c + rng.normal(0, 25), 0, 255)) for c in rgb), seed=i), label)
    return t


def test_evaluation_suggests_a_temperature(ws):
    """An overconfident task gets a temperature above 1 that lowers the calibration error; applying
    it and evaluating again suggests about 1 times it."""
    t = _noisy_task(ws, scale=200.0)
    pipe = Pipeline(ws)
    cal = evaluate_task(pipe, t)["calibration"]
    assert cal["temperature"] > 1
    assert cal["ece_calibrated"] < cal["ece"]
    assert sum(b["count"] for b in cal["reliability"]) == 90
    t.update_settings(temperature=cal["temperature"])
    again = evaluate_task(pipe, t)["calibration"]
    assert again["temperature"] == pytest.approx(cal["temperature"], rel=0.2)
    assert again["ece"] < cal["ece"]


def test_no_calibration_with_few_examples(ws):
    """Below MIN_CALIBRATION scored examples there is no suggestion."""
    t = ws.create("t", embedder="fake:color")
    for i in range(3):
        t.add_example(png(RED, noise=10, seed=i), "red")
        t.add_example(png(BLUE, noise=10, seed=i), "blue")
    assert evaluate_task(Pipeline(ws), t)["calibration"] is None


def test_fast_leave_one_out_is_tempered_like_refitting():
    """The closed-form leave-one-out applies the task's temperature like a refit does."""
    rng = np.random.default_rng(3)
    X = unit(rng.normal(size=(12, 5)))
    data = ClassData(["a", "b", "c"], X, np.arange(12) % 3)
    cfg = TaskConfig(name="t", classifier="knn", classifier_params={"k": 3}, temperature=2.5)
    fast = ev._fast_leave_one_out(data, cfg, None)  # pylint: disable=protected-access
    slow = ev.HeldOut.empty(12, 3)
    for test in ev._folds(data.y, 12, 0):  # pylint: disable=protected-access
        ev._fold(data, cfg, test, slow)  # pylint: disable=protected-access
    assert np.allclose(fast.conf, slow.conf, atol=1e-5)


# ---- API and CLI --------------------------------------------------------
@pytest.fixture(name="client")
def client_fixture(tmp_path, monkeypatch):
    """A TestClient on a fresh API, with task 'c' (one red and one blue example)."""
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path / "ws"))
    c = TestClient(importlib.reload(fewshot.api).app)
    c.post("/api/tasks", json={"name": "c", "embedder": "fake:color"})
    for label, rgb in (("red", RED), ("blue", BLUE)):
        c.post("/api/tasks/c/examples", data={"label": label}, files=[("files", ("x.png", png(rgb), "image/png"))])
    return c


def test_api(client):
    """predict ?evidence=, temperature set and cleared through PATCH."""
    p = client.post("/api/tasks/c/predict", params={"evidence": 1},
                    files=[("files", ("q.png", png(RED, seed=3), "image/png"))]).json()[0]
    assert [e["label"] for e in p["evidence"]] == ["red"]
    assert client.post("/api/tasks/c/predict", params={"evidence": 99},
                       files=[("files", ("q.png", png(RED), "image/png"))]).status_code == 422
    assert client.patch("/api/tasks/c", json={"temperature": 2.0}).json()["temperature"] == 2.0
    assert client.patch("/api/tasks/c", json={"clear_temperature": True}).json()["temperature"] is None
    r = client.patch("/api/tasks/c", json={"clear_temperature": True, "temperature": 2})
    assert r.status_code == 400
    assert client.patch("/api/tasks/c", json={"temperature": 1000}).status_code == 400


def test_cli(tmp_path):
    """predict --evidence lists similar examples; config --temperature 1 clears it; eval reports
    calibration once there are enough examples."""
    root = tmp_path / "ws"
    runner = CliRunner()

    def run(*args):
        return runner.invoke(cli.app, [str(a) for a in args] + ["--root", str(root)])

    t = _noisy_task(__import__("fewshot").Workspace(root), scale=200.0)
    q = tmp_path / "q.png"
    q.write_bytes(png(RED, seed=9))
    out = run("predict", t.name, q, "--evidence", "2").output
    assert out.count("    like ") == 2
    assert "temperature: 3" in run("config", t.name, "--temperature", "3").output
    assert "temperature: 1" in run("config", t.name, "--temperature", "1").output
    assert "Calibration error" in run("eval", t.name).output
