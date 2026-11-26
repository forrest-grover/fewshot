"""Predicting at an earlier revision, streaming predict output, and `fewshot watch`."""

import csv
import io
import json
import shutil
import time

import pytest

from fewshot import cli
from fewshot.cli_common import images_in
from fewshot.store import Workspace
from tests.conftest import BLUE, RED, api_client, png, write_images


@pytest.fixture(name="task")
def task_fixture(root):
    """Task 't' with three red and three blue examples."""
    t = Workspace(root).create("t", embedder="fake:color")
    for i in range(3):
        t.add_example(png(RED, noise=15, seed=i), "red")
        t.add_example(png(BLUE, noise=15, seed=i), "blue")
    return "t"


# ---- Task.version -------------------------------------------------------
def test_version_returns_kept_versions(root, task):
    """version gives the current config, a kept earlier one, or KeyError; a damaged one is not kept."""
    t = Workspace(root).get(task)
    before = t.config.revision
    t.upsert_class("green")
    assert t.version(t.config.revision) is t.config
    old = t.version(before)
    assert (old.revision, old.name, [c.name for c in old.classes]) == (before, task, ["red", "blue"])
    with pytest.raises(KeyError, match="is not kept"):
        t.version(9999)
    (t.path / "history" / f"{before:010d}.json").write_text("{bad")
    with pytest.raises(KeyError, match=f"Revision {before} of 't' is not kept"):
        t.version(before)


# ---- CLI predict ----------------------------------------------------------
def test_predict_jsonl_from_file_in_chunks(run, task, tmp_path, monkeypatch):
    """--jsonl prints a line per image as each chunk finishes; --from-file - reads paths from stdin."""
    monkeypatch.setattr(cli, "PREDICT_CHUNK", 1)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(png(RED, seed=9))
    b.write_bytes(png(BLUE, seed=9))
    r = run("predict", task, "--from-file", "-", "--jsonl", "--top", "1", stdin=f"{a}\n\n{b}\n")
    assert r.exit_code == 0, r.output
    rows = [json.loads(line) for line in r.stdout.splitlines()]
    assert [(x["file"], x["label"], list(x["scores"])) for x in rows] == [(str(a), "red", ["red"]),
                                                                          (str(b), "blue", ["blue"])]
    assert {x["revision"] for x in rows} == {Workspace(tmp_path / "ws").get(task).config.revision}
    listing = tmp_path / "list.txt"
    listing.write_text(f"{b}\n")
    r = run("predict", task, a, "--from-file", listing, "--csv")
    assert [row["label"] for row in csv.DictReader(io.StringIO(r.output))] == ["red", "blue"]  # one header


def test_predict_argument_errors(run, task, tmp_path):
    """Several output formats, no images, or a revision that isn't kept are errors."""
    q = tmp_path / "q.png"
    q.write_bytes(png(RED, seed=9))
    assert "at most one of" in run("predict", task, q, "--json", "--csv").output
    assert "Give images to classify" in run("predict", task).output
    r = run("predict", task, q, "--revision", "9999")
    assert (r.exit_code, "is not kept" in r.output) == (1, True)


def test_predict_at_an_earlier_revision(run, task, root, tmp_path):
    """--revision classifies with a kept earlier version: here, before the blue examples went."""
    t = Workspace(root).get(task)
    before = t.config.revision
    t.delete_class("blue")
    q = tmp_path / "q.png"
    q.write_bytes(png(BLUE, seed=9))
    now = json.loads(run("predict", task, q, "--json").output)[0]
    assert now["label"] != "blue"
    then = json.loads(run("predict", task, q, "--json", "--revision", before).output)[0]
    assert (then["label"], then["revision"]) == ("blue", before)


def test_predict_value_error(run, task, tmp_path, monkeypatch):
    """A ValueError from the pipeline is reported as an error."""

    def boom(*_a, **_k):
        raise ValueError("no classes")

    monkeypatch.setattr("fewshot.pipeline.Pipeline.predict", boom)
    q = tmp_path / "q.png"
    q.write_bytes(png(RED, seed=9))
    r = run("predict", task, q)
    assert (r.exit_code, "no classes" in r.output) == (1, True)


# ---- CLI watch -------------------------------------------------------------
def test_watch_once_sorts_and_logs(run, task, tmp_path):
    """--once moves each image into OUT/<class>, OUT/_unknown or OUT/_error and logs a JSON line each."""
    inbox, out = tmp_path / "in", tmp_path / "out"
    inbox.mkdir()
    (inbox / "r.png").write_bytes(png(RED, seed=9))
    (inbox / "b.png").write_bytes(png(BLUE, seed=9))
    (inbox / "broken.png").write_bytes(b"not an image")
    (inbox / "empty.png").write_bytes(b"")  # still being written: waits
    r = run("watch", task, inbox, "--out", out, "--once")
    assert r.exit_code == 0, r.output
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.png")) == [
        "_error/broken.png", "blue/b.png", "red/r.png"]
    assert [p.name for p in inbox.iterdir()] == ["empty.png"]
    log = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert {x["to"] for x in log} == {str(out / "_error/broken.png"), str(out / "blue/b.png"), str(out / "red/r.png")}
    assert all(x["task"] == task and "revision" in x for x in log)


def test_watch_copy_min_confidence_and_free_names(run, task, tmp_path):
    """--copy leaves the originals; below --min-confidence goes to _unknown; names are never overwritten."""
    inbox, out = tmp_path / "in", tmp_path / "out"
    inbox.mkdir()
    (inbox / "r.png").write_bytes(png((125, 30, 125), seed=9))  # purple: between red and blue
    for _ in range(2):
        assert run("watch", task, inbox, "--out", out, "--once", "--copy", "--min-confidence", "0.99").exit_code == 0
    assert sorted(p.name for p in (out / "_unknown").iterdir()) == ["r-1.png", "r.png"]
    assert (inbox / "r.png").exists()


def test_watch_waits_for_stable_size_until_interrupted(run, task, tmp_path, monkeypatch):
    """Without --once a file is handled once its size is unchanged for an interval, and only once
    with --copy; Ctrl+C stops the watch."""
    inbox, out = tmp_path / "in", tmp_path / "out"
    inbox.mkdir()
    (inbox / "r.png").write_bytes(png(RED, seed=9))
    looks = []

    def sleep(_seconds):
        looks.append(sorted(p.name for p in out.rglob("*.png")) if out.exists() else [])
        if len(looks) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr("fewshot.cli.time.sleep", sleep)
    r = run("watch", task, inbox, "--out", out, "--copy", "--interval", "0.1")
    assert r.exit_code == 0, r.output
    assert looks == [[], ["r.png"], ["r.png"]]  # first look only notes the size; copied once
    assert r.output.strip().endswith("Stopped")


def _watch_looks(monkeypatch, n, each=None):
    """Make watch stop (as by Ctrl+C) at its ``n``-th sleep, calling ``each(look)`` at every one."""
    looks: list[int] = []

    def sleep(_seconds):
        looks.append(len(looks))
        if each:
            each(len(looks))
        if len(looks) == n:
            raise KeyboardInterrupt

    monkeypatch.setattr("fewshot.cli.time.sleep", sleep)


def test_watch_waits_for_a_task_that_can_predict(run, root, tmp_path, monkeypatch):
    """With no examples yet, --once fails and a watch waits (and sorts once there are some)."""
    t = Workspace(root).create("empty", embedder="fake:color")
    t.upsert_class("red")
    inbox, out = tmp_path / "in", tmp_path / "out"
    inbox.mkdir()
    (inbox / "r.png").write_bytes(png(RED, seed=9))
    r = run("watch", "empty", inbox, "--out", out, "--once")
    assert (r.exit_code, "Need at least 1 class" in r.output) == (1, True)
    _watch_looks(monkeypatch, 4, lambda look: look == 2 and t.add_example(png(RED), "red"))
    r = run("watch", "empty", inbox, "--out", out)
    assert (r.exit_code, "Waiting: Need at least 1 class" in r.output) == (0, True)
    assert (out / "red" / "r.png").exists()


def test_watch_survives_files_that_vanish(run, task, tmp_path, monkeypatch):
    """A file gone before its size is read, or before it is moved, is skipped with a message."""
    inbox, out = tmp_path / "in", tmp_path / "out"
    inbox.mkdir()
    (inbox / "r.png").write_bytes(png(RED, seed=9))
    (inbox / "b.png").write_bytes(png(BLUE, seed=9))
    real_move = shutil.move
    monkeypatch.setattr("fewshot.cli._images_in",
                        lambda folder, recursive: [folder / "gone.png", *images_in(folder, recursive)])

    def move(src, dest):
        if src.endswith("b.png"):
            raise FileNotFoundError(f"No such file: {src}")
        return real_move(src, dest)

    monkeypatch.setattr("fewshot.cli.shutil.move", move)
    r = run("watch", task, inbox, "--out", out, "--once")
    assert r.exit_code == 0, r.output
    assert "b.png: No such file" in r.output
    assert (out / "red" / "r.png").exists()
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 1


def test_watch_retries_unreadable_files_until_settled(run, task, tmp_path, monkeypatch):
    """An unreadable file that just changed is left in place (it may still be arriving); once it is
    older than WATCH_SETTLE it goes to _error."""
    inbox, out = tmp_path / "in", tmp_path / "out"
    inbox.mkdir()
    (inbox / "half.png").write_bytes(b"not yet an image")
    _watch_looks(monkeypatch, 3)
    assert run("watch", task, inbox, "--out", out).exit_code == 0
    assert (inbox / "half.png").exists()
    monkeypatch.setattr(cli, "WATCH_SETTLE", 0.0)
    _watch_looks(monkeypatch, 3)
    assert run("watch", task, inbox, "--out", out).exit_code == 0
    assert (out / "_error" / "half.png").exists()


def test_watch_stops_when_the_task_goes(run, task, root, tmp_path, monkeypatch):
    """A task deleted while watching stops the watch with a message."""
    inbox, out = tmp_path / "in", tmp_path / "out"
    inbox.mkdir()
    (inbox / "r.png").write_bytes(png(RED, seed=9))
    def delete(look):
        if look == 1:
            Workspace(root).delete(task)

    _watch_looks(monkeypatch, 5, delete)
    r = run("watch", task, inbox, "--out", out)
    assert (r.exit_code, "is gone (renamed or deleted)" in r.output) == (1, True)


def test_watch_needs_a_folder(run, task, tmp_path):
    """FOLDER must be a directory."""
    r = run("watch", task, tmp_path / "nope", "--out", tmp_path / "out")
    assert (r.exit_code, "Not a directory" in r.output) == (1, True)


# ---- API -------------------------------------------------------------------
def test_api_predict_at_a_revision(tmp_path):
    """?revision= predicts with a kept version, 404 for one that isn't kept; predictions carry it."""
    client = api_client(tmp_path)
    client.post("/api/tasks", json={"name": "c", "embedder": "fake:color"})
    for label, rgb in (("red", RED), ("blue", BLUE)):
        files = [("files", (f"{label}{i}.png", png(rgb, noise=15, seed=i), "image/png")) for i in range(2)]
        client.post("/api/tasks/c/examples", data={"label": label}, files=files)
    before = client.get("/api/tasks/c").json()["revision"]
    assert client.delete("/api/tasks/c/classes/blue").status_code == 200
    q = [("files", ("q.png", png(BLUE, seed=9), "image/png"))]
    p = client.post("/api/tasks/c/predict", params={"revision": before}, files=q).json()[0]
    assert (p["label"], p["revision"]) == ("blue", before)
    assert client.post("/api/tasks/c/predict", files=q).json()[0]["revision"] > before
    r = client.post("/api/tasks/c/predict", params={"revision": 9999}, files=q)
    assert r.status_code == 404


# ---- CLI predict --from-file ---------------------------------------------
@pytest.mark.usefixtures("task")
def test_from_file_is_read_as_it_arrives(run, tmp_path, monkeypatch):
    """Each batch is classified as soon as its lines have been read, before the list ends."""
    monkeypatch.setattr(cli, "PREDICT_CHUNK", 1)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(png(RED, seed=9))
    b.write_bytes(png(BLUE, seed=9))
    order = []
    real = cli._named_in  # pylint: disable=protected-access

    def lines(src):
        for x in real(src):
            order.append(("read", str(x).rsplit("/", 1)[-1]))
            yield x

    monkeypatch.setattr(cli, "_named_in", lines)
    monkeypatch.setattr(cli, "_output_predictions",
                        lambda part, *_a, **_k: order.append(("out", part[0].name)))
    assert run("predict", "t", "--from-file", "-", "--jsonl", stdin=f"{a}\n{b}\n").exit_code == 0
    assert order == [("read", "a.png"), ("out", "a.png"), ("read", "b.png"), ("out", "b.png")]


@pytest.mark.usefixtures("task")
def test_from_file_entries_that_cant_be_read(run, tmp_path):
    """A listed path that is missing gets an error row (the rest are still classified); a folder is
    expanded; a missing list file or an empty list is an error."""
    a = tmp_path / "a.png"
    a.write_bytes(png(RED, seed=9))
    folder = tmp_path / "more"
    folder.mkdir()
    (folder / "b.png").write_bytes(png(BLUE, seed=9))
    listing = tmp_path / "list.txt"
    listing.write_text(f"{tmp_path / 'gone.png'}\n{a}\n{folder}\nhttp://127.0.0.1:1/x.png\n")
    r = run("predict", "t", "--from-file", listing, "--jsonl")
    rows = [json.loads(line) for line in r.stdout.splitlines()]
    assert [(x["label"], x["error"] is not None) for x in rows] == [
        (None, True), ("red", False), ("blue", False), (None, True)]
    assert rows[3]["file"] == "http://127.0.0.1:1/x.png"
    assert (r.exit_code, "2 of 4 images could not be read" in r.stderr) == (1, True)
    r = run("predict", "t", "--from-file", tmp_path / "nope.txt")
    assert (r.exit_code, "Can't read" in r.output) == (1, True)
    (tmp_path / "empty.txt").write_text("\n")
    r = run("predict", "t", "--from-file", tmp_path / "empty.txt")
    assert (r.exit_code, "No images to classify" in r.output) == (1, True)
    r = run("predict", "t", "--from-file", listing, "--json")
    assert [x["file"] for x in json.loads(r.stdout)][:3] == [str(tmp_path / "gone.png"), str(a), str(folder / "b.png")]


# ---- CLI watch: failed moves ---------------------------------------------
def test_watch_backs_off_from_a_file_it_cannot_move(tmp_path, monkeypatch):
    """Failures double the wait before the next try, up to a maximum."""
    retry = cli.Backoff(1.0)
    f = tmp_path / "a.png"
    assert not retry.waiting(f)
    assert [retry.failed(f) for _ in range(3)] == [1.0, 2.0, 4.0]
    assert retry.waiting(f)
    for _ in range(20):
        retry.failed(f)
    assert retry.failed(f) == cli.WATCH_MAX_BACKOFF
    retry.keep([])
    assert not retry.waiting(f)
    retry.failed(f)
    retry.succeeded(f)
    assert not retry.waiting(f)
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    retry.failed(f)
    clock[0] = 1.5
    assert not retry.waiting(f)  # due again


def test_watch_retries_a_failed_move_later(run, task, tmp_path, monkeypatch):
    """watch doesn't retry a file it couldn't move at every look."""
    inbox = tmp_path / "in"
    write_images(inbox, RED, 1)
    tries = []

    def place(f, folder, copy):  # pylint: disable=unused-argument
        """Fail to move anything."""
        tries.append(f.name)
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "_place", place)
    sleeps = []

    def sleep(_s):
        """Stop after a few looks."""
        sleeps.append(1)
        if len(sleeps) > 6:
            raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", sleep)
    r = run("watch", task, inbox, "--out", tmp_path / "out", "--interval", "100")
    assert "trying again in 100 s" in r.output
    assert len(tries) == 1  # waits out its back-off instead of retrying every look
