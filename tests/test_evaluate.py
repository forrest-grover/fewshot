"""Tests for fewshot.evaluate: cross-validation of tasks and episodic folder benchmarks."""

from typing import Any

import numpy as np
import pytest

from fewshot import Pipeline, benchmark, evaluate_task
from fewshot.classifiers import ClassData
from fewshot.embedders import cache_namespace, register_embedder, unregister_embedder
from fewshot.evaluate import _folds, _subset, load_folder_dataset  # pylint: disable=protected-access  # unit tests
from tests.conftest import ColorEmbedder, png

COLORS = {"red": (220, 30, 30), "green": (30, 220, 30), "blue": (30, 30, 220), "yellow": (220, 220, 30)}
TEXT_SPEC = "openclip:fake-color"  # "openclip:" prefix makes the pipeline use class descriptions


class ColorTextEmbedder(ColorEmbedder):
    """ColorEmbedder that also embeds text naming a colour, in the same space as its images."""

    spec = TEXT_SPEC
    supports_text = True

    def _embed_text_batch(self, texts):
        rgbs = [next(rgb for name, rgb in COLORS.items() if name in t) for t in texts]
        v = np.array(rgbs, np.float32) + 1.0
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(name="text_embedder")
def fixture_text_embedder():
    """Register the text-capable fake embedder for the duration of a test."""
    register_embedder(TEXT_SPEC, ColorTextEmbedder())
    yield TEXT_SPEC
    unregister_embedder(TEXT_SPEC)


def colour_task(ws, n=3, classes=("red", "green", "blue"), embedder="fake:color", **settings):
    """Task with ``n`` noisy examples of each colour class."""
    t = ws.create("colors", embedder=embedder, **settings)
    for label in classes:
        for i in range(n):
            t.add_example(png(COLORS[label], noise=15, seed=100 * len(label) + i), label, f"{label}{i}.png")
    return t


# ---- _folds / _subset ---------------------------------------------------

def test_folds_leave_one_out_when_k_covers_all():
    """k >= n gives one fold per example."""
    assert [f.tolist() for f in _folds(np.array([0, 1, 1]), 3, 0)] == [[0], [1], [2]]
    assert len(_folds(np.array([0, 1, 1]), 10, 0)) == 3


def test_folds_are_a_stratified_partition():
    """k-fold assignment partitions indices and spreads each class over folds."""
    y = np.repeat([0, 1, 2], 4)
    folds = _folds(y, 4, seed=1)
    assert len(folds) == 4
    assert sorted(np.concatenate(folds).tolist()) == list(range(12))
    for f in folds:
        assert sorted(y[f].tolist()) == [0, 1, 2]  # one of each class per fold
    assert [f.tolist() for f in _folds(y, 4, seed=1)] == [f.tolist() for f in folds]  # deterministic


def test_subset_drops_empty_classes_but_keeps_text_ones():
    """Classes without training rows are dropped unless they have text."""
    X = np.eye(4, dtype=np.float32)
    data = ClassData(["a", "b", "c", "d"], X, np.array([0, 1, 1, 3]), {2: np.ones(4, np.float32)})
    sub, back = _subset(data, np.array([1, 3]))
    assert sub.classes == ["b", "c", "d"]
    assert sub.y.tolist() == [0, 2]
    assert back.tolist() == [1, 2, 3]
    assert list(sub.text) == [1]
    np.testing.assert_array_equal(sub.X, X[[1, 3]])


# ---- evaluate_task -----------------------------------------------------

def test_empty_task_is_an_error(ws):
    """Evaluating a task with no examples raises."""
    t = ws.create("empty", embedder="fake:color")
    with pytest.raises(ValueError, match="no examples"):
        evaluate_task(Pipeline(ws), t)


def test_k_fold_report(ws):
    """An explicit fold count gives a k-fold report with full metrics."""
    t = colour_task(ws, n=4)
    r = evaluate_task(Pipeline(ws), t, folds=2)
    assert r["method"] == "2-fold"
    assert r["n_examples"] == r["n_scored"] == 12
    assert r["accuracy"] == 1.0
    assert r["classes"] == ["red", "green", "blue"]
    assert r["confusion"] == [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
    assert r["per_class"]["green"] == {"support": 4, "recall": 1.0, "precision": 1.0}
    assert r["mistakes"] == []


def test_mislabelled_example_is_reported_as_mistake(ws):
    """A mislabelled example shows up as a mistake and in the metrics."""
    t = colour_task(ws, n=3)
    bad = t.add_example(png(COLORS["red"], seed=7), "blue")
    r = evaluate_task(Pipeline(ws), t)
    assert r["method"] == "leave-one-out"
    assert len(r["mistakes"]) == 1
    m = r["mistakes"][0]
    assert (m["id"], m["true"], m["pred"]) == (bad.id, "blue", "red")
    assert 0.5 < m["confidence"] <= 1.0
    assert r["accuracy"] == pytest.approx(9 / 10)
    assert r["confusion"][2][0] == 1
    assert r["per_class"]["blue"]["recall"] == pytest.approx(3 / 4)
    assert r["per_class"]["red"]["precision"] == pytest.approx(3 / 4)


def test_mistakes_sorted_by_confidence(ws):
    """Mistakes are listed most confident first."""
    t = colour_task(ws, n=3)
    t.add_example(png(COLORS["red"], seed=7), "blue")
    t.add_example(png(COLORS["green"], seed=8), "red")
    mistakes = evaluate_task(Pipeline(ws), t)["mistakes"]
    assert len(mistakes) == 2
    assert mistakes[0]["confidence"] >= mistakes[1]["confidence"]


def test_nothing_scorable_gives_none_metrics(ws):
    """When no fold can train, nothing is scored and metrics are None."""
    # Two singleton classes: every LOO fold leaves only one class to train on.
    t = colour_task(ws, n=1, classes=("red", "green"))
    r = evaluate_task(Pipeline(ws), t)
    assert r["n_scored"] == 0
    assert r["n_examples"] == 2
    assert r["accuracy"] is None
    assert r["confusion"] == [[0, 0], [0, 0]]
    assert r["mistakes"] == []
    assert r["per_class"]["red"] == {"support": 0, "recall": None, "precision": None}


def test_described_class_stays_trainable_in_every_fold(ws, text_embedder):
    """A described class can be predicted from text when its only image is held out."""
    # A singleton class with a description can still be predicted from its text when its image is held out.
    t = colour_task(ws, n=2, classes=("red", "green"), embedder=text_embedder)
    t.add_example(png(COLORS["blue"]), "blue")
    t.upsert_class("blue", "something blue")
    r = evaluate_task(Pipeline(ws), t)
    assert r["n_scored"] == 5
    assert r["accuracy"] == 1.0
    assert r["per_class"]["blue"]["recall"] == 1.0


# ---- load_folder_dataset / benchmark --------------------------------------

def make_folder(root, per_class=6, classes=tuple(COLORS)):
    """Write ``root/<class>/<i>.png`` images of noisy solid colours."""
    for c in classes:
        d = root / c
        d.mkdir(parents=True)
        for i in range(per_class):
            (d / f"{i}.png").write_bytes(png(COLORS[c], noise=15, seed=1000 * len(c) + i))
    return root


def test_load_folder_dataset_filters(tmp_path):
    """Only subfolders with image files (any case extension) become classes."""
    make_folder(tmp_path, per_class=2, classes=("red", "green"))
    (tmp_path / "green" / "notes.txt").write_text("x")
    (tmp_path / "green" / "UPPER.PNG").write_bytes(png(COLORS["green"]))
    (tmp_path / "empty").mkdir()
    (tmp_path / "only_text").mkdir()
    (tmp_path / "only_text" / "a.txt").write_text("x")
    (tmp_path / "stray.png").write_bytes(png(COLORS["red"]))
    ds = load_folder_dataset(tmp_path)
    assert list(ds) == ["green", "red"]
    assert [f.name for f in ds["green"]] == ["0.png", "1.png", "UPPER.PNG"]
    assert [f.name for f in ds["red"]] == ["0.png", "1.png"]


def test_benchmark_separable_colours(ws, tmp_path):
    """Separable colours score 100% and progress is reported once."""
    folder = make_folder(tmp_path / "data")
    messages: list[str] = []
    r = benchmark(Pipeline(ws), folder, "fake:color", way=3, shot=2, query=3, episodes=5, progress=messages.append)
    assert r.pop("flip") is False
    assert r == {
        "embedder": "fake:color", "classifier": "gda", "way": 3, "shot": 2, "query": 3,
        "episodes": 5, "text_template": None, "accuracy": 1.0, "ci95": 0.0,
    }
    assert messages == ["Embedding 24 images with fake:color (cached after first run)"]


def test_benchmark_is_seeded_and_passes_classifier_params(ws, tmp_path):
    """The same seed gives the same result; classifier params reach the classifier."""
    folder = make_folder(tmp_path / "data", per_class=4, classes=("red", "yellow"))
    (folder / "red" / "0.png").write_bytes(png(COLORS["yellow"]))  # one confusing image
    pipe = Pipeline(ws)
    kw: dict[str, Any] = {"way": 2, "shot": 1, "query": 1, "episodes": 6, "classifier": "knn", "k": 1}
    runs = [benchmark(pipe, folder, "fake:color", seed=s, **kw) for s in (0, 0, 1, 2, 3)]
    assert runs[0] == runs[1]
    assert runs[0]["classifier"] == "knn"
    assert len({r["accuracy"] for r in runs}) > 1  # different seeds draw different episodes
    with pytest.raises(ValueError, match="Unknown parameter 'k' for gda"):
        benchmark(pipe, folder, "fake:color", way=2, shot=1, query=1, episodes=1, k=1)  # gda has no k


def test_benchmark_single_episode_has_zero_ci(ws, tmp_path):
    """One episode reports a zero-width confidence interval."""
    folder = make_folder(tmp_path / "data", classes=("red", "blue"))
    r = benchmark(Pipeline(ws), folder, "fake:color", way=2, shot=1, query=1, episodes=1)
    assert r["episodes"] == 1
    assert r["ci95"] == 0.0
    assert r["accuracy"] == 1.0


def test_benchmark_ci_from_varying_episodes(ws, tmp_path):
    """Episodes of differing accuracy give a mean below 1 and a positive CI."""
    folder = make_folder(tmp_path / "data", per_class=4, classes=("red", "yellow"))
    # Swap one image so 'red' contains a yellow one: episodes that query it score 0.5.
    (folder / "red" / "0.png").write_bytes(png(COLORS["yellow"]))
    r = benchmark(Pipeline(ws), folder, "fake:color", way=2, shot=1, query=1, episodes=30)
    assert 0.5 < r["accuracy"] < 1.0
    assert r["ci95"] > 0.0


def test_benchmark_too_few_classes(ws, tmp_path):
    """Too few classes with shot + query images raises with counts."""
    folder = make_folder(tmp_path / "data", per_class=4, classes=("red", "green"))
    with pytest.raises(ValueError, match=r"Only 2 classes have >= 4 images; need 3"):
        benchmark(Pipeline(ws), folder, "fake:color", way=3, shot=1, query=3)
    with pytest.raises(ValueError, match=r"Only 0 classes have >= 5 images; need 2"):
        benchmark(Pipeline(ws), folder, "fake:color", way=2, shot=2, query=3)


def test_benchmark_zero_shot_with_text_template(ws, tmp_path, text_embedder):
    """shot=0 with a text template is zero-shot over class-name prompts."""
    folder = make_folder(tmp_path / "data")
    (folder / "red").rename(folder / "dark_red")  # underscores become spaces in the prompt
    r = benchmark(Pipeline(ws), folder, text_embedder, way=4, shot=0, query=3, episodes=4,
                  text_template="a photo of something {}")
    assert r["shot"] == 0
    assert r["text_template"] == "a photo of something {}"
    assert r["accuracy"] == 1.0
    cached = Pipeline(ws).cache.get_many(cache_namespace(text_embedder), ["text:a photo of something dark red"])
    assert len(cached) == 1


def test_benchmark_zero_shot_without_text_fails(ws, tmp_path):
    """shot=0 without a text template has nothing to classify with: refused before embedding."""
    folder = make_folder(tmp_path / "data", classes=("red", "blue"))
    with pytest.raises(ValueError, match=r"shot=0 .* needs a text_template"):
        benchmark(Pipeline(ws), folder, "fake:color", way=2, shot=0, query=2, episodes=1)


@pytest.mark.parametrize(("arg", "value"), [("way", 1), ("shot", -1), ("query", 0), ("episodes", 0)])
def test_benchmark_rejects_meaningless_arguments(ws, tmp_path, arg, value):
    """Arguments that would give NaN or an empty benchmark are ValueErrors."""
    folder = make_folder(tmp_path / "data", classes=("red", "blue"))
    kw = {"way": 2, "shot": 1, "query": 1, "episodes": 1, arg: value}
    with pytest.raises(ValueError, match=f"{arg} must be at least"):
        benchmark(Pipeline(ws), folder, "fake:color", **kw)


@pytest.mark.parametrize(("classifier", "n", "method"), [
    ("gda", 100, "leave-one-out"), ("gda", 102, "10-fold"), ("prototype", 102, "leave-one-out")])
def test_default_folds_depend_on_the_classifier(ws, classifier, n, method):
    """Refitting classifiers use leave-one-out up to 100 examples, then 10 folds; prototype and kNN
    (closed form) up to 300."""
    t = ws.create("t", embedder="fake:color", classifier=classifier)
    for i in range(n // 2):
        t.add_example(png((200, 30, 30), noise=10, seed=i), "red")
        t.add_example(png((30, 30, 200), noise=10, seed=1000 + i), "blue")
    assert evaluate_task(Pipeline(ws), t)["method"] == method
