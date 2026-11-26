"""Confidence calibration: tempering, the task's temperature, suggesting one from evaluation, and
the API and CLI for it."""


import numpy as np
import pytest
from typer.testing import CliRunner

import fewshot.evaluate as ev
from fewshot import cli
from fewshot.classifiers import temper
from fewshot.evaluate import evaluate_task
from fewshot.pipeline import Pipeline
from tests.conftest import BLUE, RED, api_client, png


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


def _held_out(P, truth):
    h = ev.HeldOut.empty(len(truth), P.shape[1])
    h.record(np.arange(len(truth)), P, np.ones(len(truth)), truth, np.ones(len(truth), bool))
    return h


def test_calibration_when_held_out_predictions_are_all_right():
    """All-correct predictions suggest a sharper but bounded temperature, not the grid's minimum,
    and the suggested absolute temperature stays within the range settings accept."""
    n = 60
    truth = np.arange(n) % 3
    P = np.full((n, 3), 0.14)
    P[np.arange(n), truth] = 0.72
    cal = ev._calibration(_held_out(P, truth), truth, 1.0)  # pylint: disable=protected-access
    assert cal is not None
    assert 0.05 < cal["temperature"] < 1
    assert cal["ece_calibrated"] < cal["ece"]
    cal = ev._calibration(_held_out(P, truth), truth, 0.02)  # pylint: disable=protected-access
    assert cal is not None
    assert cal["temperature"] == 0.01  # 0.02 times the fitted one, clamped


# ---- API and CLI ---------------------------------------------------------
@pytest.fixture(name="client")
def client_fixture(tmp_path):
    """A TestClient on a fresh API, with task 'c' (one red and one blue example)."""
    c = api_client(tmp_path)
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
