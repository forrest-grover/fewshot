"""Prediction results: multi-label and top-k, thresholds, one-class tasks, evidence (nearest
examples), progress and near-duplicates."""

import numpy as np
import pytest

from fewshot.classifiers import ClassData, make_classifier
from fewshot.evaluate import evaluate_task
from fewshot.pipeline import Pipeline
from tests.conftest import BLUE, GREEN, RED, png

PINK = (200, 0, 60)


# ---- labels, top-k and thresholds ----------------------------------------
def test_multi_label_and_top_k(ws):
    """With a threshold, every class whose own similarity reaches it is listed (best first);
    top_k trims the scores, which come highest first."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    t.add_example(png((220, 120, 30)), "orange")
    t.add_example(png(BLUE), "blue")
    pipe = Pipeline(ws)
    q = png((220, 80, 30))  # between red and orange
    p = pipe.predict(t, [q])[0]
    assert p.labels == [p.label]  # no threshold: just the label
    assert list(p.scores) == sorted(p.scores, key=lambda c: -p.scores[c])
    t.update_settings(reject_below=0.97)
    p = pipe.predict(t, [q], top_k=2)[0]
    assert sorted(p.labels) == ["orange", "red"]
    assert p.labels[0] == p.label
    assert len(p.scores) == 2
    far = pipe.predict(t, [png((128, 128, 128))])[0]
    assert far.rejected
    assert far.labels == []


def test_top_k_must_be_positive(ws):
    """top_k below 1 is a ValueError, not a silent trim from the end."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        Pipeline(ws).predict(t, [png(RED)], top_k=0)


def test_zero_shot_threshold(ws, text_spec):
    """Zero-shot predictions use reject_below_zero_shot, not reject_below."""
    t = ws.create("t", embedder=text_spec)
    t.upsert_class("red", "a red thing")
    t.upsert_class("blue", "a blue thing")
    t.update_settings(reject_below=0.9999)
    pipe = Pipeline(ws)
    q = png((200, 90, 60))
    assert pipe.predict(t, [q])[0].label == "red"  # the image threshold doesn't apply
    t.update_settings(reject_below_zero_shot=0.9999)
    assert pipe.predict(t, [q])[0].rejected


def test_threshold_applies_to_the_predicted_class():
    """When a description tips the vote to a class the image is less similar to, predict reports
    that class's own similarity, so a threshold between the two rejects it instead of accepting
    it below the line."""
    from fewshot.pipeline import _to_pred  # pylint: disable=import-outside-toplevel

    X = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    data = ClassData(["a", "b"], X, np.array([0, 1]), {1: np.array([0.6, 0.8, 0], np.float32)})
    clf = make_classifier("prototype", text_weight=3.0).fit(data)
    q = np.array([[0.8, 0.6, 0]], np.float32)
    probs, sims = clf.predict(q)
    per_class = clf.class_similarity(q)[0]
    assert per_class[0] > per_class[1]  # nearer to a...
    assert probs[0].argmax() == 1  # ...but b's description wins
    assert sims[0] == pytest.approx(per_class[1])
    p = _to_pred(clf, np.array([0.7, 0.7]), (probs[0], float(sims[0]), per_class), None)
    assert (p.rejected, p.label, p.labels) == (True, None, ["a"])  # rejected as b, but a clears its threshold


def test_rejected_top_class_keeps_other_matching_labels(ws):
    """When the top class is under its own threshold, a class that clears its own is still listed."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    t.add_example(png(PINK), "pink")
    t.set_class_thresholds({"red": 0.9999, "pink": 0.5})
    p = Pipeline(ws).predict(t, [png((240, 0, 30))])[0]
    assert (p.label, p.top_label, p.rejected, p.labels) == (None, "red", True, ["pink"])


def test_one_class_task(ws):
    """A single class predicts that class; its threshold separates members from outsiders."""
    t = ws.create("t", embedder="fake:color")
    for i in range(3):
        t.add_example(png(RED, noise=10, seed=i), "red")
    pipe = Pipeline(ws)
    t.update_settings(reject_below=0.99)
    member, outsider = pipe.predict(t, [png(RED, seed=9), png(BLUE)])
    assert member.label == "red"
    assert outsider.rejected
    r = evaluate_task(pipe, t)
    assert r["accuracy"] == 1.0
    assert r["suggested_reject_below"] > 0.99
    assert r["top_k_accuracy"] == {}


def test_predict_reports_progress(ws):
    """predict calls progress with the images embedded."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    seen = []
    Pipeline(ws).predict(t, [png(BLUE), png(GREEN)], progress=lambda d, n: seen.append((d, n)))
    assert seen == [(2, 2)]


# ---- evidence ------------------------------------------------------------
def unit(rows):
    """Rows scaled to unit length, as float32."""
    a = np.array(rows, np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


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


# ---- near-duplicates -----------------------------------------------------
def test_near_duplicates(ws):
    """Near-identical examples are paired, most similar first, marking different labels."""
    t = ws.create("t", embedder="fake:color")
    a = t.add_example(png(RED), "red")
    b = t.add_example(png(RED, noise=2, seed=1), "red")
    c = t.add_example(png(RED, noise=2, seed=2), "blue")
    t.add_example(png(GREEN), "green")
    pairs = Pipeline(ws).near_duplicates(t, threshold=0.999)
    assert {frozenset((d["a"], d["b"])) for d in pairs} == {frozenset(x) for x in ((a.id, b.id), (a.id, c.id),
                                                                                   (b.id, c.id))}
    assert [d["similarity"] for d in pairs] == sorted((d["similarity"] for d in pairs), reverse=True)
    mixed = next(d for d in pairs if {d["a"], d["b"]} == {a.id, c.id})
    assert not mixed["same_label"]
    assert set(mixed["labels"]) == {"red", "blue"}
    assert len(Pipeline(ws).near_duplicates(t, threshold=0.999, limit=1)) == 1


def test_near_duplicates_stays_bounded_and_finds_identical(ws):
    """Only the best ``limit`` pairs are kept while scanning, and threshold 1.0 still finds
    embeddings that are identical up to float rounding."""
    t = ws.create("t", embedder="fake:color")
    for size in (8, 9, 10):  # same colour, different files: identical embeddings
        t.add_example(png(RED, size=size), "red")
    t.add_example(png(GREEN), "green")
    pipe = Pipeline(ws)
    assert len(pipe.near_duplicates(t, threshold=1.0)) == 3
    assert len(pipe.near_duplicates(t, threshold=0.5, limit=2)) == 2
