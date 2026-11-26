"""Bounded memory for many images, streamed CLI input, transparent images, the job limit under
concurrency, and quotes in uploaded file names."""

import io
import json
import threading
import time
import uuid

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from fewshot import classifiers, cli, pipeline
from fewshot.classifiers import ClassData, make_classifier
from fewshot.images import load_image, to_rgb
from fewshot.jobs import Job, Jobs, TooManyJobs
from fewshot.store import Workspace
from tests.conftest import api_client, png
from tests.test_cli import PlainResult
from tests.test_consistency import _task

RED, BLUE = (255, 0, 0), (0, 0, 255)  # as in test_consistency


def test_predict_reads_files_a_chunk_at_a_time(ws, tmp_path, monkeypatch):
    """Files are read READ_CHUNK at a time; errors keep their index; progress covers the whole."""
    monkeypatch.setattr(pipeline, "READ_CHUNK", 2)
    t = _task(ws)
    files = []
    for i, rgb in enumerate((RED, BLUE, RED, BLUE)):
        f = tmp_path / f"{i}.png"
        f.write_bytes(png(rgb, seed=40 + i))
        files.append(f)
    files.insert(3, tmp_path / "missing.png")
    reads = []
    real = pipeline._keyed  # pylint: disable=protected-access
    def keyed(part, errs):
        reads.append(len(part))
        return real(part, errs)

    monkeypatch.setattr(pipeline, "_keyed", keyed)
    seen = []
    preds = pipeline.Pipeline(ws).predict(t, files, progress=lambda done, total: seen.append((done, total)))
    assert reads == [2, 2, 1]
    assert [p.label for p in preds] == ["red", "blue", "red", None, "blue"]
    assert "Could not read" in (preds[3].error or "")
    assert seen[-1] == (5, 5)
    assert all(total == 5 for _, total in seen)


@pytest.mark.parametrize("name", ["prototype", "knn", "logreg", "gda"])
def test_queries_are_scored_in_blocks(monkeypatch, name):
    """Scoring PREDICT_ROWS queries at a time gives the same results as all at once."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    clf = make_classifier(name).fit(ClassData(["a", "b", "c"], X, np.arange(12) % 3))
    Q = rng.normal(size=(7, 4)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    whole = clf.predict_full(Q)
    monkeypatch.setattr(classifiers, "PREDICT_ROWS", 3)
    for a, b in zip(whole, clf.predict_full(Q), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6)


def test_knn_compares_with_examples_once(monkeypatch):
    """kNN's vote and per-class similarity share one queries x examples product."""
    X = np.eye(3, dtype=np.float32)
    clf = make_classifier("knn", k=2).fit(ClassData(["a", "b", "c"], X, np.arange(3)))
    calls = []
    real = classifiers.Classifier._support_similarity  # pylint: disable=protected-access

    def counted(self, Q):
        calls.append(len(Q))
        return real(self, Q)

    monkeypatch.setattr(classifiers.Classifier, "_support_similarity", counted)
    clf.predict_full(X)
    assert calls == [3]
    clf._image_logprobs(X)  # pylint: disable=protected-access  # without S, the hook computes it
    assert calls == [3, 3]


def test_transparent_pixels_become_white():
    """RGBA, LA and palette transparency are composited onto white; opaque images are unchanged;
    PIL images passed to predict are turned upright too."""
    for im in (Image.new("RGBA", (4, 4), (0, 0, 0, 0)), Image.new("LA", (4, 4), (0, 0))):
        assert to_rgb(im).getpixel((0, 0)) == (255, 255, 255)
    buf = io.BytesIO()
    Image.new("P", (4, 4), 0).save(buf, "PNG", transparency=0)
    assert load_image(buf.getvalue()).getpixel((0, 0)) == (255, 255, 255)
    assert to_rgb(Image.new("RGB", (4, 4), RED)).getpixel((0, 0)) == RED
    tall = Image.new("RGB", (4, 8), RED)
    exif = tall.getexif()
    exif[0x0112] = 6  # rotated 90 degrees
    buf = io.BytesIO()
    tall.save(buf, "JPEG", exif=exif.tobytes())
    assert pipeline._to_pil(Image.open(io.BytesIO(buf.getvalue()))).size == (8, 4)  # pylint: disable=protected-access


def test_job_limit_is_checked_with_the_start():
    """Jobs.start refuses past ``limit``, even when many starts race."""
    jobs = Jobs()
    gate = threading.Event()
    started, refused = [], []

    def start():
        try:
            job = Job(task="t", kind="k", key=uuid.uuid4().hex)  # different work, not shared
            started.append(jobs.start(job, lambda _p: gate.wait(5), limit=3))
        except TooManyJobs:
            refused.append(1)

    threads = [threading.Thread(target=start) for _ in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert (len(started), len(refused)) == (3, 17)
    gate.set()
    deadline = time.time() + 5
    while jobs.running() and time.time() < deadline:
        time.sleep(0.01)
    assert jobs.start(Job(task="t", kind="k"), lambda _p: None, limit=3).status in {"running", "done"}


def test_quotes_in_uploaded_file_names(tmp_path):
    """A '"' that a browser sent as %22 comes back as a quote."""
    c = api_client(tmp_path)
    _task(c.srv.ws)
    r = c.post("/api/tasks/t/predict", files=[("files", ('say "cheese".png', png(RED, seed=3), "image/png"))])
    assert r.json()[0]["file"] == 'say "cheese".png'


# ---- CLI predict --from-file -------------------------------------------------
@pytest.fixture(name="run")
def run_fixture(tmp_path):
    """Invoke the CLI against a workspace with task 't'."""
    root = tmp_path / "ws"
    _task(Workspace(root))
    runner = CliRunner()

    def invoke(*args, stdin=None):
        return PlainResult(runner.invoke(cli.app, [str(a) for a in args] + ["--root", str(root)], input=stdin))

    return invoke


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


def test_empty_threshold_change_does_nothing(ws):
    """set_class_thresholds({}) saves nothing and adds no undo step."""
    t = _task(ws)
    rev = t.config.revision
    t.set_class_thresholds({})
    assert t.config.revision == rev
