"""CLI commands for undo, rename, export/import, examples, duplicates, JSON/CSV output and URLs."""

import csv
import io
import json
import os
import sys

import pytest
from typer.testing import CliRunner

from fewshot import cli
from fewshot.store import Workspace
from tests.conftest import png
from tests.test_cli import ANSI, PlainResult

RED, BLUE = (220, 30, 30), (30, 30, 220)


@pytest.fixture(name="root")
def root_fixture(tmp_path):
    """Workspace directory for the test."""
    return tmp_path / "ws"


@pytest.fixture(name="run")
def run_fixture(root):
    """Invoke the CLI against the temporary workspace."""
    runner = CliRunner()

    def invoke(*args):
        return PlainResult(runner.invoke(cli.app, [str(a) for a in args] + ["--root", str(root)]))

    return invoke


@pytest.fixture(name="task")
def task_fixture(run, tmp_path):
    """Task 't' with three red and three blue examples."""
    assert run("create", "t", "--embedder", "fake:color").exit_code == 0
    for label, rgb in (("red", RED), ("blue", BLUE)):
        d = tmp_path / label
        d.mkdir()
        for i in range(3):
            (d / f"{i}.png").write_bytes(png(rgb, noise=15, seed=i))
        assert run("add", "t", label, d).exit_code == 0
    return "t"


def test_json_output(run, task):
    """models, list and show print JSON with --json."""
    models = CliRunner().invoke(cli.app, ["models", "--json"]).output  # takes no --root
    assert json.loads(models)[0]["spec"].startswith("openclip:")
    tasks = json.loads(run("list", "--json").output)
    assert [(t["name"], t["n_examples"]) for t in tasks] == [("t", 6)]
    shown = json.loads(run("show", task, "--json").output)
    assert shown["embedder"] == "fake:color"
    assert [(c["name"], c["count"]) for c in shown["classes"]] == [("red", 3), ("blue", 3)]


def test_config_zero_shot_threshold_and_flip(run, task):
    """config sets and clears the zero-shot threshold and switches flip."""
    r = run("config", task, "--reject-below-zero-shot", "0.22", "--flip")
    assert r.exit_code == 0
    assert "reject_below_zero_shot: 0.22   flip: True" in r.output
    r = run("config", task, "--clear-reject-zero-shot", "--no-flip")
    assert "reject_below_zero_shot: None   flip: False" in r.output
    r = run("config", task, "--clear-reject-zero-shot", "--reject-below-zero-shot", "0.2")
    assert r.exit_code == 1
    assert "either --reject-below-zero-shot or --clear-reject-zero-shot" in r.output
    run("class", "describe", task, "a {} thing\nthe {}")
    assert '"a red thing" | "the red"' in run("show", task).output


def test_rename_and_undo(run, task, root):
    """rename moves the task; undo steps back one change at a time."""
    assert run("rename", task, "u").output.strip() == "Renamed task t to u"
    assert run("rename", "nope", "v").exit_code == 1
    assert run("rename", "u", "bad name").exit_code == 1
    run("class", "add", "u", "green")
    r = run("undo", "u")
    assert r.exit_code == 0
    assert "6 examples" in r.output
    assert Workspace(root).get("u").class_names() == ["red", "blue"]
    run("create", "empty")
    r = run("undo", "empty")
    assert (r.exit_code, r.output.strip()) == (1, "Error: Nothing to undo")
    assert run("redo", "u").output.strip() == "u: redid add class green (6 examples)"
    r = run("redo", "u")
    assert (r.exit_code, r.output.strip()) == (1, "Error: Nothing to redo")


def test_export_and_import(run, task, tmp_path):
    """export writes a zip that import-task reads back under the same or a new name."""
    out = tmp_path / "t.zip"
    assert run("export", task, out).output.strip() == f"Exported t (6 examples) to {out}"
    assert run("import-task", out, "--name", "copy").output.strip() == "Imported task copy (6 examples)"
    r = run("import-task", out)
    assert (r.exit_code, "already exists" in r.output) == (1, True)
    r = run("import-task", tmp_path / "missing.zip")
    assert "No such file" in r.output


def test_example_commands(run, task, root):
    """example ls/relabel/tag/note/rm work with short id prefixes."""
    ids = [e.id for e in Workspace(root).get(task).config.examples]
    listing = run("example", "ls", task, "--label", "red").output.splitlines()
    assert len(listing) == 3
    assert listing[0].startswith(ids[0][:12])
    assert run("example", "relabel", task, "blue", ids[0][:8]).output.strip() == "Moved 1 examples to blue"
    assert run("example", "relabel", task, " ", ids[0][:8]).exit_code == 1
    assert run("example", "tag", task, ids[0][:8], ids[1][:8], "--add", "hard").output.strip() == "Tagged 2 examples"
    assert run("example", "tag", task, ids[1][:8], "--remove", "hard", "--add", "x").exit_code == 0
    assert run("example", "tag", task, ids[0][:8]).exit_code == 1
    assert run("example", "tag", task, ids[0][:8], "--add", "a,b").exit_code == 1
    assert run("example", "note", task, ids[0][:8], "too dark").output.strip() == "Note saved"
    assert run("example", "note", task, ids[0][:8], "x" * 3000).exit_code == 1
    line = run("example", "ls", task, "--tag", "hard").output.strip()
    assert line.endswith("[hard]  # too dark")
    data = json.loads(run("example", "ls", task, "--tag", "x", "--json").output)
    assert [e["id"] for e in data] == [ids[1]]
    assert run("example", "rm", task, ids[0][:8], ids[1]).output.strip() == "Deleted 2 examples"
    r = run("example", "rm", task, "zzzz")
    assert (r.exit_code, r.output.strip()) == (1, "Error: No example 'zzzz'")


def test_dupes(run, task):
    """dupes lists near-identical pairs, marking different labels; --json prints them."""
    r = run("dupes", task, "--threshold", "0.99")
    assert r.exit_code == 0
    assert r.output.strip().endswith("pairs at similarity >= 0.99")
    assert "DIFFERENT LABELS" not in r.output
    pairs = json.loads(run("dupes", task, "--threshold", "0.99", "--json").output)
    assert all(p["same_label"] for p in pairs)


def test_predict_csv_json_and_urls(run, task, tmp_path, monkeypatch):
    """predict reads URLs, prints CSV, and limits JSON scores to --top."""

    def fake_fetch(url, max_bytes):
        assert max_bytes == cli.MAX_DOWNLOAD
        if "bad" in url:
            raise ValueError(f"Could not fetch {url}: HTTP 404 Not Found")
        return png(BLUE, seed=7)

    monkeypatch.setattr(cli, "fetch_url", fake_fetch)
    q = tmp_path / "q.png"
    q.write_bytes(png(RED, seed=8))
    r = run("predict", task, q, "http://h/b.png", "--csv")
    rows = list(csv.DictReader(io.StringIO(r.output)))
    assert [(row["label"], row["labels"]) for row in rows] == [("red", "red"), ("blue", "blue")]
    assert rows[0]["file"] == str(q)
    r = run("predict", task, "http://h/bad.png", q)
    assert r.exit_code == 1
    assert "http://h/bad.png: error: Could not fetch http://h/bad.png: HTTP 404" in r.output
    data = json.loads(run("predict", task, "http://h/bad.png", "--json").stdout)
    assert data[0]["error"] == "Could not fetch http://h/bad.png: HTTP 404 Not Found"  # the real reason
    data = json.loads(run("predict", task, q, "--json", "--top", "1").output)
    assert list(data[0]["scores"]) == ["red"]
    assert run("add", task, "blue", "http://h/new.png").exit_code == 0
    r = run("add", task, "blue", "http://h/bad.png")
    assert "skip http://h/bad.png: Could not fetch" in r.output


def test_eval_csv_and_metrics(run, task):
    """eval prints macro-F1, top-k and a threshold suggestion; --csv lists held-out predictions."""
    r = run("eval", task)
    assert "macro-F1 1.000" in r.output
    assert "Suggested --reject-below" in r.output
    rows = list(csv.DictReader(io.StringIO(run("eval", task, "--csv").output)))
    assert len(rows) == 6
    assert set(rows[0]) == {"id", "true", "pred", "correct", "confidence", "similarity"}


def test_eval_top_k_lines(run, task, tmp_path):
    """With more than two classes, top-k accuracies are printed."""
    d = tmp_path / "green"
    d.mkdir()
    for i in range(2):
        (d / f"{i}.png").write_bytes(png((30, 220, 30), noise=15, seed=i))
    run("add", task, "green", d)
    assert "top-2 1.000" in run("eval", task).output


def test_progress_bar_on_a_terminal(monkeypatch, capsys):
    """On a terminal the progress helper draws a bar; elsewhere it is silent."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    with cli._progress("Working") as update:  # pylint: disable=protected-access
        update(1, 2)
    assert "Working" in ANSI.sub("", capsys.readouterr().err)


def test_serve_allow_urls(monkeypatch):
    """serve --allow-urls sets FEWSHOT_ALLOW_URLS for the API."""
    seen: dict[str, str] = {}
    import uvicorn  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: seen.update(os.environ))
    for k in ("FEWSHOT_TOKEN", "FEWSHOT_ALLOW_ANY_MODEL", "FEWSHOT_WARM", "FEWSHOT_ALLOW_URLS"):
        monkeypatch.setenv(k, "")
    CliRunner().invoke(cli.app, ["serve", "--allow-urls"])
    assert seen["FEWSHOT_ALLOW_URLS"] == "1"


def test_cache_prune_removes_unused_images(run, task, root):
    """cache prune also deletes stored images no task links to."""
    t = Workspace(root).get(task)
    (t.path / t.config.examples[0].file).unlink()
    r = run("cache", "prune")
    assert "Deleted 1 stored images no task uses" in r.output
    assert "stored images" not in run("cache", "prune", "--dry-run").output


def test_export_to_a_bad_path(run, task, tmp_path):
    """export to a missing folder or onto a folder fails cleanly, leaving no partial file."""
    r = run("export", task, tmp_path / "no" / "such" / "t.zip")
    assert r.exit_code == 1
    assert "Error: Can't write" in r.output
    r = run("export", task, tmp_path)
    assert r.exit_code == 1
    assert not list(tmp_path.glob("*.part"))
