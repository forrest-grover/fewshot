"""Comparing embedders on a task's own labelled examples: library, API job and CLI."""

import json
import time

import numpy as np
import pytest
from typer.testing import CliRunner

import fewshot.api
import fewshot.evaluate
from fewshot import cli
from fewshot.embedders import Embedder, register_embedder, unregister_embedder
from fewshot.evaluate import compare_embedders
from fewshot.jobs import Cancelled
from fewshot.pipeline import Pipeline
from fewshot.store import Workspace
from tests.conftest import api_client, png
from tests.test_cli import PlainResult

RED, BLUE = (220, 30, 30), (30, 30, 220)


class NoiseEmbedder(Embedder):  # pylint: disable=abstract-method
    """Embeds an image as the colour of its noise only: a weak model for these tasks."""

    cache_version: str | None = "test"

    @property
    def dim(self):
        return 3

    def _embed_image_batch(self, images):
        v = np.stack([np.asarray(im, np.float32).reshape(-1, 3).std(0) + 1.0 for im in images])
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


class BrokenEmbedder(NoiseEmbedder):  # pylint: disable=abstract-method
    """Fails like a model that can't run."""

    def _embed_image_batch(self, images):
        raise RuntimeError("CUDA out of memory")


@pytest.fixture(autouse=True)
def more_embedders():
    """Register "fake:noise" and "fake:broken" next to "fake:color"."""
    register_embedder("fake:noise", NoiseEmbedder())
    register_embedder("fake:broken", BrokenEmbedder())
    yield
    unregister_embedder("fake:noise")
    unregister_embedder("fake:broken")


def _task(ws, name="t"):
    t = ws.create(name, embedder="fake:color")
    for i in range(4):
        t.add_example(png(RED, noise=5 + 10 * i, seed=i), "red")
        t.add_example(png(BLUE, noise=5 + 10 * i, seed=i + 10), "blue")
    return t


def test_compare_embedders(ws):
    """Each embedder is cross-validated; failures are reported; the most accurate is named."""
    t = _task(ws)
    t.update_settings(temperature=2.0)
    stages = []
    r = compare_embedders(Pipeline(ws), t, ["fake:color", "fake:noise", "fake:broken", "fake:color"],
                          progress=lambda stage, done, total: stages.append(stage))
    assert [x["embedder"] for x in r["results"]] == ["fake:color", "fake:noise", "fake:broken"]
    color, noise, broken = r["results"][0], r["results"][1], r["results"][2]
    assert color["accuracy"] == 1.0
    assert color["method"] == "leave-one-out"
    assert noise["accuracy"] < 1.0
    assert broken == {"embedder": "fake:broken", "error": "CUDA out of memory"}
    assert (r["best"], r["current"], r["n_examples"]) == ("fake:color", "fake:color", 8)
    assert r["revision"] == t.config.revision
    assert {"1/3 embedding", "1/3 evaluating", "2/3 embedding"} <= set(stages)
    assert t.config.embedder == "fake:color"  # unchanged
    assert compare_embedders(Pipeline(ws), t, ["fake:noise"])["results"][0]["n_scored"] == 8  # no progress callback


def test_compare_errors(ws):
    """No embedders, no examples, nothing that works, and cancelling."""
    t = _task(ws)
    with pytest.raises(ValueError, match="at least one embedder"):
        compare_embedders(Pipeline(ws), t, [])
    with pytest.raises(ValueError, match="no examples"):
        compare_embedders(Pipeline(ws), ws.create("empty", embedder="fake:color"), ["fake:color"])
    assert compare_embedders(Pipeline(ws), t, ["fake:broken"])["best"] is None

    def cancel(*_):
        raise Cancelled

    with pytest.raises(Cancelled):
        compare_embedders(Pipeline(ws), t, ["fake:color"], progress=cancel)


# ---- API ------------------------------------------------------------------------
@pytest.fixture(name="client")
def client_fixture(tmp_path):
    """A TestClient on a fresh API, with task 't'."""
    c = api_client(tmp_path)
    _task(c.srv.ws)
    return c


def _wait(client, jid):
    deadline = time.time() + 10
    while (job := client.get(f"/api/jobs/{jid}").json())["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
    return job


def test_api_compare_job(client):
    """POST .../compare/jobs runs the comparison in the background; the job's result is the report."""
    r = client.post("/api/tasks/t/compare/jobs", json={"embedders": ["fake:color", " fake:noise ", "fake:color", ""]})
    assert r.status_code == 202
    assert r.json()["kind"] == "compare"
    job = _wait(client, r.json()["id"])
    assert job["status"] == "done", job
    assert [x["embedder"] for x in job["result"]["results"]] == ["fake:color", "fake:noise"]
    assert job["result"]["best"] == "fake:color"


def test_api_compare_refusals(client, monkeypatch):
    """Too few or many embedders, bad folds, an empty task, models off the allowlist, too many jobs."""
    def post(body, task="t"):
        return client.post(f"/api/tasks/{task}/compare/jobs", json=body)

    assert post({"embedders": []}).status_code == 400
    assert post({"embedders": [f"fake:{i}" for i in range(9)]}).status_code == 400
    assert post({"embedders": ["fake:color"], "folds": 1}).status_code == 400
    client.post("/api/tasks", json={"name": "empty", "embedder": "fake:color"})
    assert post({"embedders": ["fake:color"]}, task="empty").status_code == 400
    r = post({"embedders": ["openclip:nope@nowhere"]})
    assert (r.status_code, "not a preset" in r.json()["detail"]) == (403, True)
    assert post({"embedders": ["fake:color"], "extra": 1}).status_code == 422
    monkeypatch.setattr(fewshot.api, "MAX_JOBS", 0)
    assert post({"embedders": ["fake:color"]}).status_code == 429


# ---- CLI ------------------------------------------------------------------------
@pytest.fixture(name="run")
def run_fixture(tmp_path):
    """Invoke the CLI against a workspace with task 't'."""
    root = tmp_path / "ws"
    _task(Workspace(root))
    runner = CliRunner()
    return lambda *args: PlainResult(runner.invoke(cli.app, [str(a) for a in args] + ["--root", str(root)]))


def test_cli_compare(run, monkeypatch):
    """compare prints a line per embedder, marks the best and current, and says how to switch."""
    out = run("compare", "t", "-e", "fake:noise", "-e", "fake:broken").output
    lines = out.splitlines()
    assert lines[0].startswith("*<")
    assert lines[0].endswith("fake:color")
    assert "fake:noise" in lines[1]
    assert lines[2].endswith("fake:broken: CUDA out of memory")
    assert "To switch" not in out
    data = json.loads(run("compare", "t", "-e", "fake:noise", "--json").output)
    assert data["best"] == "fake:color"
    assert "Give another --embedder" in run("compare", "t").output
    assert "Give another --embedder" in run("compare", "t", "-e", "fake:color").output

    def fake(_pipe, _task, specs, _folds, progress):
        progress("1/2 embedding", 1, 1)
        return {"revision": 1, "current": specs[0], "n_examples": 8, "best": specs[1],
                "results": [{"embedder": specs[0], "accuracy": None}, {"embedder": specs[1], "accuracy": 1.0,
                                                                       "macro_f1": 1.0, "seconds": 0.5}]}

    monkeypatch.setattr("fewshot.evaluate.compare_embedders", fake)
    out = run("compare", "t", "-e", "fake:noise").output
    assert "fake:color: nothing scored" in out
    assert "To switch: fewshot config t --embedder fake:noise" in out


def test_cli_compare_presets_and_errors(run, monkeypatch):
    """--presets adds every preset; a ValueError is reported."""
    seen = []

    def fail(_pipe, _task, specs, *_a, **_k):
        seen.extend(specs)
        raise ValueError("Task has no examples to evaluate")

    monkeypatch.setattr("fewshot.evaluate.compare_embedders", fail)
    r = run("compare", "t", "--presets")
    assert (r.exit_code, "no examples" in r.output) == (1, True)
    assert seen[0] == "fake:color"
    assert len(seen) > 2
