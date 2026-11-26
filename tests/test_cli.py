"""Tests for the ``fewshot`` command-line interface."""

import json
import os
import re
import socket

import numpy as np
import pytest
from typer.testing import CliRunner

from fewshot import cli, embedders
from fewshot.embedders import PRESETS, register_embedder, unregister_embedder
from fewshot.store import Workspace
from tests.conftest import ColorEmbedder, png

RED, GREEN, BLUE = (220, 30, 30), (30, 220, 30), (30, 30, 220)
COLORS = {"red": RED, "green": GREEN, "blue": BLUE}


@pytest.fixture(name="root")
def root_fixture(tmp_path):
    """Workspace directory for the test."""
    return tmp_path / "ws"


ANSI = re.compile(r"\x1b\[[0-9;]*m")


class PlainResult:  # pylint: disable=too-few-public-methods  # stands in for click's Result
    """A click Result with ANSI styling removed from ``output``.

    Typer forces coloured error output when GITHUB_ACTIONS, FORCE_COLOR or PY_COLORS is set,
    and the escape codes split text such as ``--shots`` that tests look for.
    """

    def __init__(self, result):
        self._result = result
        self.output = ANSI.sub("", result.output)

    def __getattr__(self, name):
        return getattr(self._result, name)


@pytest.fixture(name="run")
def run_fixture(root):
    """Invoke the CLI against the temporary workspace; returns the Result with plain-text output."""
    runner = CliRunner()

    def invoke(*args, stdin=None):
        return PlainResult(runner.invoke(cli.app, [str(a) for a in args] + ["--root", str(root)], input=stdin))

    return invoke


def write_images(folder, rgb, n, start=0):
    """Write ``n`` noisy PNGs of colour ``rgb`` into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + n):
        (folder / f"{i}.png").write_bytes(png(rgb, noise=20, seed=i))


@pytest.fixture(name="photos")
def photos_fixture(tmp_path):
    """A FOLDER/<class>/<image> dataset with 4 images per colour plus a stray non-image file."""
    folder = tmp_path / "photos"
    for i, (label, rgb) in enumerate(COLORS.items()):
        write_images(folder / label, rgb, 4, start=10 * i)
    (folder / "red" / "notes.txt").write_text("not an image")
    (folder / "README").write_text("top-level files are ignored")
    return folder


@pytest.fixture(name="task")
def task_fixture(run, photos):
    """A fake-embedder task with the ``photos`` dataset imported."""
    assert run("create", "colors", "--embedder", "fake:color").exit_code == 0
    assert run("import", "colors", photos).exit_code == 0
    return "colors"


def load(root, name="colors"):
    """Read a task fresh from disk."""
    return Workspace(root).get(name)


def test_models_lists_presets():
    """`models` prints every preset spec and label."""
    r = CliRunner().invoke(cli.app, ["models"])
    assert r.exit_code == 0
    for p in PRESETS:
        assert p.spec in r.output
        assert p.label in r.output


def test_create_list_show(run, root):
    """`create` stores its options; `list` and `show` report them."""
    r = run("create", "pets", "--embedder", "fake:color", "--classifier", "knn")
    assert r.exit_code == 0
    assert "Created task pets" in r.output
    assert load(root, "pets").config.classifier == "knn"

    r = run("create", "plain")
    assert r.exit_code == 0
    assert load(root, "plain").config.embedder == embedders.DEFAULT_SPEC

    r = run("list")
    lines = [line for line in r.output.splitlines() if not line.startswith("    !")]  # without warnings
    assert len(lines) == 2
    assert "    ! Add classes and examples to start" in r.output
    assert lines[0].startswith("pets")
    assert "knn" in lines[0]
    assert "fake:color" in lines[0]

    run("class", "add", "pets", "beagle", "--desc", "a small hound")
    run("class", "add", "pets", "pug")
    r = run("show", "pets")
    assert r.exit_code == 0
    assert "classifier: knn" in r.output
    assert "reject_below: None" in r.output
    assert '"a small hound"' in r.output
    assert "pug" in r.output


@pytest.mark.parametrize(("args", "message"), [
    (["pets"], "already exists"),
    (["bad name!"], "Task names use"),
    (["other", "--classifier", "nope"], "Unknown classifier"),
])
def test_create_errors(run, args, message):
    """`create` rejects duplicates, bad names and unknown classifiers with exit code 1."""
    run("create", "pets", "--embedder", "fake:color")
    r = run("create", *args)
    assert r.exit_code == 1
    assert message in r.output


@pytest.mark.parametrize("command", [
    ["show", "missing"],
    ["config", "missing"],
    ["delete", "missing", "--yes"],
    ["class", "add", "missing", "x"],
    ["add", "missing", "x", "."],
    ["predict", "missing", "."],
    ["eval", "missing"],
])
def test_unknown_task_is_a_clean_error(run, command):
    """An unknown task name gives an error message, not a traceback."""
    r = run(*command)
    assert r.exit_code == 1
    assert "Error: No task 'missing'" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)


def test_config(run, root, task):
    """`config` updates settings, merges params and clears the threshold."""
    r = run("config", task, "--classifier", "knn", "--param", "k=3", "--param", "scale=20",
            "--text-weight", "0.5", "--reject-below", "0.7")
    assert r.exit_code == 0
    assert "classifier: knn" in r.output
    c = load(root).config
    assert (c.classifier, c.classifier_params, c.text_weight, c.reject_below) == (
        "knn", {"k": 3, "scale": 20}, 0.5, 0.7)

    # A --param on its own extends the existing params; a negative --reject-below is a threshold.
    assert run("config", task, "--param", "k=5", "--reject-below", "-0.5").exit_code == 0
    c = load(root).config
    assert c.classifier_params == {"k": 5, "scale": 20}
    assert c.reject_below == -0.5
    assert run("config", task, "--clear-reject").exit_code == 0
    assert load(root).config.reject_below is None

    # Naming the current classifier keeps its params; changing it resets them.
    assert run("config", task, "--classifier", "knn").exit_code == 0
    assert load(root).config.classifier_params == {"k": 5, "scale": 20}
    register_embedder("fake:other", ColorEmbedder())
    try:
        assert run("config", task, "--classifier", "prototype", "--embedder", "fake:other").exit_code == 0
    finally:
        unregister_embedder("fake:other")
    c = load(root).config
    assert (c.classifier, c.classifier_params, c.embedder) == ("prototype", {}, "fake:other")


def test_config_and_create_validate_embedders(run, root, task):
    """A spec that can't be loaded is refused before anything is saved."""
    for args in (["config", task, "--embedder", "bogus:x"], ["create", "new", "--embedder", "bogus:x"],
                 ["config", task, "--embedder", ""]):
        r = run(*args)
        assert r.exit_code == 1
        assert "embedder" in r.output.lower()
    assert load(root).config.embedder == "fake:color"
    assert not (root / "tasks" / "new").exists()
    r = run("config", task, "--reject-below", "0.5", "--clear-reject")
    assert r.exit_code == 2  # a usage error
    assert "not both" in r.output


@pytest.mark.parametrize(("args", "message"), [
    (["--classifier", "nope"], "Unknown classifier"),
    (["--param", "k"], "Bad --param 'k'"),
    (["--param", "=3"], "Bad --param '=3'"),
    (["--param", "k=three"], "Bad --param 'k=three'"),
    (["--param", "bogus=1"], "Unknown parameter 'bogus' for gda"),
    (["--classifier", "knn", "--param", 'k="x"'], "'k' must be an integer"),
    (["--reject-below", "2"], "reject_below must be between -1 and 1"),
])
def test_config_errors(run, root, task, args, message):
    """Invalid settings and malformed --param values fail without changing the task."""
    r = run("config", task, *args)
    assert r.exit_code == (2 if message.startswith("Bad --param") else 1)  # 2: a usage error
    assert message in r.output
    assert load(root).config.classifier == "gda"


def test_delete(run, root, task):
    """`delete` asks for confirmation unless --yes is given."""
    r = run("delete", task, stdin="n\n")
    assert r.exit_code == 1
    assert "Aborted" in r.output
    assert (root / "tasks" / task).exists()

    r = run("delete", task, stdin="y\n")
    assert r.exit_code == 0
    assert "Deleted task colors" in r.output
    assert not (root / "tasks" / task).exists()

    run("create", "again", "--embedder", "fake:color")
    assert run("delete", "again", "-y").exit_code == 0
    assert run("list").output == ""


def test_class_commands(run, root, task):
    """`class add`, `class describe` and `class rm` change classes on disk."""
    assert run("class", "add", task, "dog_breed").exit_code == 0
    assert run("class", "add", task, "red", "--desc", "something crimson").exit_code == 0
    r = run("class", "add", task, "  ")
    assert r.exit_code == 1
    assert "Class name is empty" in r.output

    r = run("class", "describe", task, "a photo of a {}, a colour")
    assert r.exit_code == 0
    assert "Set 3 descriptions" in r.output  # green, blue, dog_breed
    descs = {c.name: c.description for c in load(root).config.classes}
    assert descs["dog_breed"] == "a photo of a dog breed, a colour"
    assert descs["red"] == "something crimson"

    r = run("class", "describe", task, "{}!", "--overwrite")
    assert "Set 4 descriptions" in r.output
    assert load(root).config.classes[0].description.endswith("!")

    r = run("class", "describe", task, "no placeholder")
    assert r.exit_code == 1
    assert "must contain '{}'" in r.output

    r = run("class", "rm", task, "red")
    assert r.exit_code == 0
    assert "Removed class red and 4 examples" in r.output
    t = load(root)
    assert "red" not in t.class_names()
    assert len(t.config.examples) == 8

    r = run("class", "rm", task, "red")
    assert r.exit_code == 1
    assert "No class 'red'" in r.output


def test_add(run, root, tmp_path):
    """`add` expands directories, skips unreadable images and rejects missing paths."""
    run("create", "colors", "--embedder", "fake:color")
    write_images(tmp_path / "more" / "nested", RED, 2)
    single = tmp_path / "one.png"
    single.write_bytes(png(RED, seed=99))
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"junk")

    r = run("add", "colors", "red", single, tmp_path / "more", bad)
    assert r.exit_code == 0
    assert "colors: 3 examples" in r.output
    assert "skip" in r.output
    assert "bad.png" in r.output
    assert "Not a readable image" in r.output
    t = load(root)
    assert sorted(e.source for e in t.config.examples) == ["0.png", "1.png", "one.png"]
    assert {e.label for e in t.config.examples} == {"red"}

    r = run("add", "colors", "red", tmp_path / "nope.png")
    assert r.exit_code == 1
    assert "No such file or directory" in r.output
    assert len(load(root).config.examples) == 3


def test_import(run, root, photos):
    """`import` honours --shots, skips bad images and rejects a missing folder."""
    run("create", "colors", "--embedder", "fake:color")
    (photos / "blue" / "broken.png").write_bytes(b"junk")
    r = run("import", "colors", photos, "--shots", "2")
    assert r.exit_code == 0
    assert "Imported 6 images into colors" in r.output
    assert load(root).summary()["counts"] == {"blue": 2, "green": 2, "red": 2}

    r = run("import", "colors", photos)
    assert "Imported 12 images" in r.output  # re-imported duplicates are counted but not stored twice
    assert "skip" in r.output
    assert "broken.png" in r.output
    assert load(root).summary()["counts"] == {"blue": 4, "green": 4, "red": 4}

    r = run("import", "colors", photos, "--shots", "0")
    assert r.exit_code == 2
    assert "--shots" in r.output

    r = run("import", "colors", photos / "missing")
    assert r.exit_code == 1
    assert "Not a directory" in r.output


def test_predict_text_and_json(run, task, tmp_path):
    """`predict` classifies files in a directory, as text or JSON."""
    queries = tmp_path / "queries"
    queries.mkdir()
    (queries / "r.png").write_bytes(png((210, 40, 40), seed=50))
    (queries / "g.png").write_bytes(png((40, 210, 40), seed=51))

    r = run("predict", task, queries, "--top", "1")
    assert r.exit_code == 0
    lines = r.output.splitlines()
    assert len(lines) == 2
    assert "g.png: green" in lines[0]
    assert "r.png: red" in lines[1]
    assert lines[1].count("=") == 2  # sim= plus a single score with --top 1

    r = run("predict", task, queries / "r.png", "--json")
    out = json.loads(r.output)
    assert out[0]["file"].endswith("r.png")
    assert out[0]["label"] == "red"
    assert set(out[0]["scores"]) == {"red", "green", "blue"}
    assert out[0]["rejected"] is False


def test_predict_unknown(run, task, tmp_path):
    """Predictions under reject_below are shown as unknown."""
    q = tmp_path / "q.png"
    q.write_bytes(png((128, 128, 128), seed=7))
    run("config", task, "--reject-below", "0.999")
    r = run("predict", task, q)
    assert r.exit_code == 0
    assert "unknown (closest:" in r.output

    out = json.loads(run("predict", task, q, "--json").output)
    assert out[0]["label"] is None
    assert out[0]["rejected"] is True


def test_predict_errors(run, task, tmp_path):
    """Unreadable images are reported without stopping the others; missing paths are errors."""
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"junk")
    good = tmp_path / "good.png"
    good.write_bytes(png(RED, seed=5))
    r = run("predict", task, bad, good)
    assert r.exit_code == 1
    assert f"{bad}: error: Not a readable image" in r.output
    assert f"{good}: red" in r.output
    assert "1 of 2 images could not be read" in r.output
    r = run("predict", task, bad, good, "--json")
    assert r.exit_code == 1
    assert '"error": "Not a readable image: unrecognised file format"' in r.output

    run("create", "lonely", "--embedder", "fake:color")
    r = run("predict", "lonely", good)
    assert r.exit_code == 1
    assert "Need at least 1 class" in r.output

    r = run("predict", task, tmp_path / "nope.png")
    assert r.exit_code == 1
    assert "No such file or directory" in r.output


def test_eval(run, task, tmp_path):
    """`eval` reports accuracy, per-class metrics and mistakes."""
    r = run("eval", task)
    assert r.exit_code == 0
    assert "leave-one-out: accuracy 1.000 on 12/12 examples" in r.output
    assert "recall=1.00" in r.output
    assert "mistakes" not in r.output

    r = run("eval", task, "--folds", "2", "--json")
    out = json.loads(r.output)
    assert out["method"] == "2-fold"
    assert out["accuracy"] == 1.0

    # A red image labelled blue is always misclassified.
    wrong = tmp_path / "wrong.png"
    wrong.write_bytes(png(RED, seed=123))
    run("add", task, "blue", wrong)
    r = run("eval", task)
    assert "accuracy 0.923 on 13/13" in r.output
    assert "1 mistakes; most confident:" in r.output
    assert "blue -> red" in r.output


def test_eval_edge_cases(run, root):
    """`eval` fails on an empty task and reports when nothing can be scored."""
    run("create", "empty", "--embedder", "fake:color")
    r = run("eval", "empty")
    assert r.exit_code == 1
    assert "no examples" in r.output

    # With one class no fold has two classes to train on, so nothing can be scored.
    t = load(root, "empty")
    t.add_example(png(RED, seed=1), "red")
    t.add_example(png(RED, seed=2), "red")
    t.upsert_class("blue")
    r = run("eval", "empty")
    assert r.exit_code == 0
    assert "Nothing scored" in r.output
    assert "recall=-" in r.output
    assert "precision=-" in r.output


class TextColorEmbedder(ColorEmbedder):
    """ColorEmbedder that also embeds colour words, standing in for a CLIP model."""

    spec = "fake:text"
    supports_text = True

    def _embed_text_batch(self, texts):
        rows = [[float(name in t) for name in COLORS] for t in texts]
        v = np.array(rows, np.float32) + 0.01
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(name="bench_data")
def bench_data_fixture(tmp_path, monkeypatch):
    """A small folder dataset plus a text-capable fake embedder treated as CLIP."""
    folder = tmp_path / "bench"
    for i, (label, rgb) in enumerate(COLORS.items()):
        write_images(folder / label, rgb, 4, start=10 * i)
    write_images(folder / "tiny", (128, 128, 128), 1)  # too few images, so never sampled
    embedders.register_embedder("fake:text", TextColorEmbedder())
    monkeypatch.setattr(cli, "spec_supports_text", lambda spec: spec == "fake:text")
    yield folder
    embedders.unregister_embedder("fake:text")


BENCH = ["--way", "3", "--query", "2", "--episodes", "3", "--shot", "0", "--shot", "1"]


def test_bench(run, bench_data):
    """`bench` runs every embedder/classifier/shot combination, using text only for text embedders."""
    r = run("bench", bench_data, "--embedder", "fake:color", "--embedder", "fake:text",
            "--classifier", "prototype", "--classifier", "knn", "--text-template", "{}", *BENCH)
    assert r.exit_code == 0, r.output
    results = [line for line in r.output.splitlines() if " acc " in line]
    # fake:color has no text, so its 0-shot runs are skipped and the template is ignored.
    assert len(results) == 2 + 4
    color = [line for line in results if line.startswith("fake:color")]
    assert all("1-shot      " in line for line in color)
    text = [line for line in results if line.startswith("fake:text")]
    assert sum("0-shot +text" in line for line in text) == 2
    assert all("+text" in line and "acc 1.000 ± 0.000" in line for line in text)
    assert "Embedding 12 images with fake:color" in r.output


def test_bench_without_template_skips_zero_shot(run, bench_data):
    """Without --text-template a 0-shot run has nothing to classify with."""
    r = run("bench", bench_data, "--embedder", "fake:text", *BENCH)
    results = [line for line in r.output.splitlines() if " acc " in line]
    assert len(results) == 1
    assert "3-way 1-shot" in results[0]
    assert "+text" not in results[0]


def test_bench_default_embedder_and_errors(run, bench_data, monkeypatch):
    """`bench` defaults to DEFAULT_SPEC and reports too-small datasets."""
    monkeypatch.setattr(cli, "DEFAULT_SPEC", "fake:color")
    r = run("bench", bench_data, "--way", "3", "--query", "2", "--episodes", "2", "--shot", "1")
    assert r.exit_code == 0
    assert r.output.count("fake:color") == 2  # progress line + result

    r = run("bench", bench_data, "--embedder", "fake:color", "--way", "5", "--shot", "1", "--query", "2")
    assert r.exit_code == 1
    assert "Only 3 classes have >= 3 images; need 5" in r.output

    r = run("bench", bench_data / "missing", "--embedder", "fake:color")
    assert r.exit_code == 1
    assert "Not a directory" in r.output


def test_serve(monkeypatch, tmp_path):
    """`serve` starts uvicorn with the given host/port and exports --root as FEWSHOT_HOME."""
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.append((app, kw)))
    monkeypatch.delenv("FEWSHOT_HOME", raising=False)
    runner = CliRunner()

    assert runner.invoke(cli.app, ["serve", "--port", "9001"]).exit_code == 0
    assert calls[-1] == ("fewshot.api:app", {"host": "127.0.0.1", "port": 9001})
    assert "FEWSHOT_HOME" not in os.environ

    monkeypatch.setenv("FEWSHOT_HOME", "")  # let monkeypatch restore the variable after the test
    assert runner.invoke(cli.app, ["serve", "--host", "0.0.0.0", "--root", str(tmp_path)]).exit_code == 0
    assert calls[-1] == ("fewshot.api:app", {"host": "0.0.0.0", "port": 8000})
    assert os.environ["FEWSHOT_HOME"] == str(tmp_path)


def test_serve_settings(monkeypatch):
    """serve passes its options to the API through the environment, and warns when an open
    address has no token."""
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: None)
    for k in ("FEWSHOT_TOKEN", "FEWSHOT_ALLOW_ANY_MODEL", "FEWSHOT_WARM"):
        monkeypatch.setenv(k, "")  # let monkeypatch restore them after the test
    monkeypatch.delenv("FEWSHOT_TOKEN")
    runner = CliRunner()
    r = runner.invoke(cli.app, ["serve"])
    settings = ("FEWSHOT_TOKEN", "FEWSHOT_ALLOW_ANY_MODEL", "FEWSHOT_WARM")
    assert [os.environ[k] for k in settings] == ["", "", "1"]
    assert "Warning" not in r.output
    r = runner.invoke(cli.app, ["serve", "--host", "0.0.0.0", "--no-warm", "--allow-any-model"])
    assert "Warning: serving on 0.0.0.0 without --token" in ANSI.sub("", r.output)
    assert (os.environ["FEWSHOT_ALLOW_ANY_MODEL"], os.environ["FEWSHOT_WARM"]) == ("1", "")
    monkeypatch.delenv("FEWSHOT_ALLOWED_HOSTS", raising=False)
    r = runner.invoke(cli.app, ["serve", "--host", "0.0.0.0", "--token", "t0k"])
    assert "Warning" not in r.output
    assert os.environ["FEWSHOT_TOKEN"] == "t0k"
    names = os.environ["FEWSHOT_ALLOWED_HOSTS"].split(",")  # local names and this machine's, never any
    assert names[:3] == ["localhost", "127.0.0.1", "::1"]
    assert socket.gethostname().lower() in names
    assert "*" not in names
    assert "Answering requests by IP address or for: localhost" in r.output
    runner.invoke(cli.app, ["serve", "--host", "0.0.0.0", "--allowed-host", "*"])
    assert os.environ["FEWSHOT_ALLOWED_HOSTS"] == "*"
    monkeypatch.delenv("FEWSHOT_ALLOWED_HOSTS")
    runner.invoke(cli.app, ["serve"])
    assert os.environ["FEWSHOT_ALLOWED_HOSTS"] == "localhost,127.0.0.1,::1"
    monkeypatch.setenv("FEWSHOT_ALLOWED_HOSTS", " box.lan, ,10.0.0.5")  # as in docker-compose.yml
    runner.invoke(cli.app, ["serve", "--host", "0.0.0.0"])
    assert os.environ["FEWSHOT_ALLOWED_HOSTS"] == "box.lan,10.0.0.5"
    runner.invoke(cli.app, ["serve", "--host", "0.0.0.0", "--allowed-host", "a.lan"])  # the option wins
    assert os.environ["FEWSHOT_ALLOWED_HOSTS"] == "a.lan"


def test_folders_and_broken_links_with_image_names_are_skipped(run, task, photos, tmp_path):
    """A directory or dangling symlink called *.png is not treated as an image."""
    odd = tmp_path / "odd"
    write_images(odd, COLORS["red"], 1)
    (odd / "folder.png").mkdir()
    (odd / "folder.png" / "inner.png").write_bytes(png(COLORS["red"], noise=20, seed=77))
    (odd / "broken.png").symlink_to(tmp_path / "missing.png")
    (photos / "blue" / "sub.png").mkdir()
    (photos / "blue" / "dangling.png").symlink_to(tmp_path / "missing.png")

    r = run("add", task, "red", odd)
    assert r.exit_code == 0, r.output  # 0.png and folder.png/inner.png; the dangling link is skipped
    assert "folder.png:" not in r.output
    r = run("import", task, photos)
    assert r.exit_code == 0, r.output
    r = run("predict", task, odd, "--json")
    assert r.exit_code == 0, r.output
    assert [os.path.basename(p["file"]) for p in json.loads(r.output)] == ["0.png", "inner.png"]


def test_add_and_import_fail_when_nothing_is_added(run, task, tmp_path):
    """Exit 1 when no image could be added, instead of reporting success."""
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"junk")
    r = run("add", task, "blue", junk)
    assert r.exit_code == 1
    assert "No images were added" in r.output
    r = run("add", task, "   ", junk)
    assert r.exit_code == 1
    assert "Class name is empty" in r.output
    r = run("add", task, "..", junk)
    assert r.exit_code == 1
    assert "can't be '.' or '..'" in r.output

    empty = tmp_path / "empty"
    (empty / "red").mkdir(parents=True)
    r = run("import", task, empty)
    assert r.exit_code == 1
    assert "No images were imported" in r.output
    r = run("predict", task, empty)
    assert r.exit_code == 1
    assert "No images found" in r.output


def test_unreadable_file_is_skipped_when_adding(run, task, tmp_path, monkeypatch):
    """An OSError reading one file skips it with a message rather than a traceback."""
    good, bad = tmp_path / "good.png", tmp_path / "bad.png"
    good.write_bytes(png(COLORS["red"], noise=20, seed=55))
    bad.write_bytes(png(COLORS["red"], noise=20, seed=56))
    real = type(bad).read_bytes

    def read_bytes(self):
        if self.name == "bad.png":
            raise PermissionError(13, "Permission denied")
        return real(self)

    monkeypatch.setattr(type(bad), "read_bytes", read_bytes)
    r = run("add", task, "red", good, bad)
    assert r.exit_code == 0
    assert f"skip {bad}: Permission denied" in r.output


@pytest.mark.parametrize(("args", "option"), [
    (["--way", "-1"], "--way"), (["--query", "0"], "--query"), (["--episodes", "0"], "--episodes"),
    (["--shot", "-1"], "--shot"),
])
def test_bench_rejects_out_of_range_options(run, bench_data, args, option):
    """Options that would make the benchmark meaningless are usage errors."""
    r = run("bench", bench_data, "--embedder", "fake:color", *args)
    assert r.exit_code == 2
    assert option in r.output


def test_bench_way_one_is_an_error(run, bench_data):
    """One class per episode is meaningless (0 means every class)."""
    r = run("bench", bench_data, "--embedder", "fake:color", "--way", "1")
    assert (r.exit_code, "way must be at least 2, or 0 for every class" in r.output) == (1, True)


def test_cache_info_and_prune(run, root, task):
    """cache info lists namespaces; prune drops vectors of deleted examples, dry run first."""
    assert run("predict", task, root / "tasks" / task / "images").exit_code == 0  # fills the cache
    t = load(root)
    victim = t.config.examples[0].id
    t.delete_example(victim)
    info = run("cache", "info")
    assert info.exit_code == 0
    assert "fake:color#test: 12 vectors" in info.output
    assert "Would delete 0 cached image vectors" in run("cache", "prune", "--dry-run").output  # undo may need it
    t.upsert_class("x")  # an undoable change, then no undo history at all:
    for f in (t.path / "history").glob("*"):
        f.unlink()
    dry = run("cache", "prune", "--dry-run")
    assert "Would delete 1 cached image vectors" in dry.output
    assert "Deleted 1 cached image vectors" in run("cache", "prune").output
    assert "fake:color#test: 11 vectors" in run("cache", "info").output


def test_cache_prune_refuses_with_unreadable_task(run, root, task):
    """A task that can't be read would lose its vectors, so prune stops."""
    (root / "tasks" / task / "task.json").write_text("{broken")
    r = run("cache", "prune")
    assert r.exit_code == 1
    assert f"Can't read task(s) {task}" in r.output
