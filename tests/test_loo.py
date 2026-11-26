"""The closed-form leave-one-out of fewshot.evaluate: it must match refitting per example."""

import numpy as np
import pytest

import fewshot.evaluate as ev
from fewshot.classifiers import ClassData
from fewshot.evaluate import evaluate_task
from fewshot.models import TaskConfig
from fewshot.pipeline import Pipeline
from tests.conftest import RED, png


def _slow_leave_one_out(data, cfg):
    out = ev.HeldOut.empty(len(data.y), len(data.classes))
    for test in ev._folds(data.y, len(data.y), 0):  # pylint: disable=protected-access
        ev._fold(data, cfg, test, out)  # pylint: disable=protected-access
    return out


@pytest.mark.parametrize(("classifier", "params"), [("prototype", {}), ("prototype", {"scale": 5.0}),
                                                    ("knn", {"k": 3}), ("knn", {"k": 50})])
@pytest.mark.parametrize("labels", [[0, 1, 2, 0, 1, 2, 0, 1, 0], [0, 0, 0, 1, 1, 2, 2, 2, 3], [0] * 5, [0, 1]])
def test_fast_leave_one_out_matches_refitting(classifier, params, labels):
    """The closed-form leave-one-out gives the same results as refitting per example."""
    rng = np.random.default_rng(len(labels))
    X = rng.normal(size=(len(labels), 6)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.array(labels)
    data = ClassData([f"c{i}" for i in range(y.max() + 1)], X, y)
    cfg = TaskConfig(name="t", classifier=classifier, classifier_params=params)
    fast = ev._fast_leave_one_out(data, cfg, None)  # pylint: disable=protected-access
    slow = _slow_leave_one_out(data, cfg)
    assert np.array_equal(fast.pred, slow.pred)
    assert np.array_equal(fast.rank, slow.rank)
    assert np.allclose(fast.conf, slow.conf, atol=1e-5)
    scored = slow.pred >= 0
    assert np.allclose(fast.sim[scored], slow.sim[scored], atol=1e-5)


@pytest.mark.parametrize(("k", "at", "labels"), [(2, [1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0]),
                                                 (1, [0, 2, 2, 0, 0, 0, 0], [1, 1, 1, 0, 2, 2, 1]),
                                                 (3, [2, 2, 2, 0, 2, 2, 0, 0], [0, 2, 2, 1, 1, 2, 0, 2])])
def test_fast_knn_leave_one_out_matches_refitting_on_ties(k, at, labels):
    """With exact ties (repeated vectors), the closed-form kNN leave-one-out still matches refitting:
    both give tied neighbours to the lower column, and taking a column out keeps that order."""
    data = ClassData(["a", "b", "c"][: max(labels) + 1], np.eye(3, dtype=np.float32)[at], np.array(labels))
    cfg = TaskConfig(name="t", classifier="knn", classifier_params={"k": k})
    fast = ev._fast_leave_one_out(data, cfg, None)  # pylint: disable=protected-access
    slow = _slow_leave_one_out(data, cfg)
    assert np.array_equal(fast.pred, slow.pred)
    assert np.allclose(fast.conf, slow.conf, atol=1e-5)


def test_fast_knn_with_a_single_example(ws):
    """One example leaves nothing to vote with: nothing is scored."""
    t = ws.create("t", embedder="fake:color", classifier="knn")
    t.add_example(png(RED), "red")
    assert evaluate_task(Pipeline(ws), t)["n_scored"] == 0


def test_fast_leave_one_out_with_unused_descriptions(monkeypatch):
    """With text_weight 0 and every class keeping an example, descriptions don't matter, so the
    closed form is used; a class that would be left with only its description refits."""
    X = np.eye(4, dtype=np.float32)[[0, 0, 1, 1, 2]] + 0.01
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    text = {0: np.eye(4, dtype=np.float32)[3]}
    two_each = ClassData(["a", "b"], X[:4], np.array([0, 0, 1, 1]), text)
    used: list[int] = []
    real = ev._fast_leave_one_out  # pylint: disable=protected-access
    def counted(*args):
        used.append(1)
        return real(*args)

    monkeypatch.setattr(ev, "_fast_leave_one_out", counted)
    ev._cross_val_predict(two_each, TaskConfig(name="t", classifier="prototype", text_weight=0.0), 4, 0)  # pylint: disable=protected-access
    assert used == [1]
    ev._cross_val_predict(two_each, TaskConfig(name="t", classifier="prototype", text_weight=1.0), 4, 0)  # pylint: disable=protected-access
    single = ClassData(["a", "b", "c"], X, np.array([0, 0, 1, 1, 2]), text)
    ev._cross_val_predict(single, TaskConfig(name="t", classifier="prototype", text_weight=0.0), 5, 0)  # pylint: disable=protected-access
    assert used == [1]


def unit(rows):
    """Rows scaled to unit length, as float32."""
    a = np.array(rows, np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


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


@pytest.mark.parametrize(("classifier", "params"), [("prototype", {}), ("knn", {"k": 3})])
def test_fast_leave_one_out_in_chunks_matches_refitting(monkeypatch, classifier, params):
    """Held out a few rows at a time (LOO_CHUNK), every chunk, not just the first, gives what
    refitting gives: each row is compared with all the others but itself."""
    monkeypatch.setattr(ev, "LOO_CHUNK", 4)
    rng = np.random.default_rng(3)
    y = np.array([0, 1, 2] * 5)
    X = rng.normal(size=(len(y), 6)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    data = ClassData(["a", "b", "c"], X, y)
    cfg = TaskConfig(name="t", classifier=classifier, classifier_params=params)
    seen = []
    fast = ev._fast_leave_one_out(data, cfg, lambda done, n: seen.append(done))  # pylint: disable=protected-access
    slow = _slow_leave_one_out(data, cfg)
    assert seen == [0, 4, 8, 12, 15]
    assert np.array_equal(fast.pred, slow.pred)
    assert np.allclose(fast.conf, slow.conf, atol=1e-5)
    assert np.allclose(fast.sim, slow.sim, atol=1e-5)
