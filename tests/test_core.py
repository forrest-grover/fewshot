"""End-to-end classification behaviour with the fake colour embedder."""

import subprocess
import sys

import numpy as np
import pytest

import fewshot
from fewshot import ClassData, Pipeline, Workspace, evaluate_task, history, make_classifier, store
from tests.conftest import png

RED, GREEN, BLUE = (220, 30, 30), (30, 220, 30), (30, 30, 220)


def make_task(ws, classifier="prototype", n=4):
    """Task "colors" with ``n`` noisy red, green and blue examples each."""
    t = ws.create("colors", embedder="fake:color", classifier=classifier)
    for label, rgb in [("red", RED), ("green", GREEN), ("blue", BLUE)]:
        for i in range(n):
            t.add_example(png(rgb, noise=20, seed=hash((label, i)) % 1000), label, f"{label}{i}.png")
    return t


@pytest.mark.parametrize("clf", ["prototype", "knn", "logreg"])
def test_predict_and_evaluate(ws, clf):
    """Each classifier predicts clear colours correctly and scores 100% in leave-one-out."""
    t = make_task(ws, clf)
    pipe = Pipeline(ws)
    preds = pipe.predict(t, [png(RED, seed=99), png(BLUE, seed=98)])
    assert [p.label for p in preds] == ["red", "blue"]
    assert abs(sum(preds[0].scores.values()) - 1) < 1e-6
    r = evaluate_task(pipe, t)
    assert r["method"] == "leave-one-out"
    assert r["accuracy"] == 1.0
    assert np.trace(np.array(r["confusion"])) == 12


def test_duplicate_image_relabels_instead_of_duplicating(ws):
    """Re-adding the same bytes relabels the existing example."""
    t = ws.create("d", embedder="fake:color")
    a = t.add_example(png(RED), "red")
    b = t.add_example(png(RED), "blue")
    assert a.id == b.id
    assert len(t.config.examples) == 1
    assert t.config.examples[0].label == "blue"


def test_rejects_non_images(ws):
    """Undecodable bytes are refused before anything is stored."""
    t = ws.create("d", embedder="fake:color")
    with pytest.raises(ValueError, match="Not a readable image"):
        t.add_example(b"not an image", "x")
    assert t.config.examples == []


def test_reject_threshold_marks_unknown(ws):
    """Below ``reject_below`` similarity the label is None but the top label is kept."""
    t = make_task(ws)
    t.update_settings(reject_below=0.999)
    p = Pipeline(ws).predict(t, [png((128, 128, 128))])[0]
    assert p.rejected
    assert p.label is None
    assert p.top_label in {"red", "green", "blue"}


def test_refit_after_change_and_persistence(ws, monkeypatch):
    """Renames and deletions take effect on the next prediction and persist to disk."""
    monkeypatch.setattr(history, "HISTORY_KEPT", 0)  # no undo: deleted images go at once
    t = make_task(ws)
    pipe = Pipeline(ws)
    assert pipe.predict(t, [png(GREEN, seed=5)])[0].label == "green"
    t.rename_class("green", "verde")
    assert pipe.predict(t, [png(GREEN, seed=5)])[0].label == "verde"
    t.delete_class("blue")
    t2 = Workspace(ws.root).get("colors")  # fresh load from disk
    assert t2.class_names() == ["red", "verde"]
    assert len(t2.config.examples) == 8
    assert len(list((t2.path / "images").iterdir())) == 8


def test_singleton_class_is_skipped_not_crashing_in_loo(ws):
    """A class with one example cannot be held out, but is still reported."""
    t = make_task(ws, n=3)
    t.add_example(png((200, 200, 0)), "yellow")
    r = evaluate_task(Pipeline(ws), t)
    assert r["n_examples"] == 10
    assert r["per_class"]["yellow"]["support"] == 1


def test_text_only_class_zero_shot_and_modality_shift():
    """Text-only classes classify across the modality gap, with or without images."""
    rng = np.random.default_rng(0)
    d = 16
    proto = rng.normal(size=(3, d))
    gap = rng.normal(size=d) * 3  # text lives far from images

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    X = unit(np.concatenate([proto[i] + 0.1 * rng.normal(size=(3, d)) for i in range(2)]))
    y = np.repeat([0, 1], 3)
    text = {i: unit(proto[i] + gap) for i in range(3)}  # class 2 has text only
    data = ClassData(["a", "b", "c"], X.astype(np.float32), y, {k: v.astype(np.float32) for k, v in text.items()})
    for name in ["prototype", "knn", "logreg"]:
        clf = make_classifier(name).fit(data)
        q = unit(proto + 0.05 * rng.normal(size=(3, d))).astype(np.float32)
        assert clf.predict(q)[0].argmax(1).tolist() == [0, 1, 2], name

    empty = ClassData(["a", "b", "c"], np.zeros((0, d), np.float32), np.zeros(0, int), data.text)
    zs = make_classifier("prototype").fit(empty)
    assert zs.predict(unit(proto + gap).astype(np.float32))[0].argmax(1).tolist() == [0, 1, 2]


def test_class_without_data_is_an_error():
    """Fitting with a class that has neither images nor text fails clearly."""
    with pytest.raises(ValueError, match="no images and no text"):
        make_classifier("prototype").fit(ClassData(["a", "b"], np.eye(2, dtype=np.float32)[:1], np.array([0])))


def test_package_exports_are_lazy():
    """Top-level names resolve to their submodule objects without importing torch or the pipeline."""
    assert fewshot.Workspace is store.Workspace
    assert set(fewshot.__all__) >= {"Pipeline", "Workspace", "register_embedder"}
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        _ = fewshot.nope
    code = "import sys, fewshot; fewshot.Workspace; print('torch' in sys.modules, 'fewshot.pipeline' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False"]
