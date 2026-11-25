"""Pipeline: embedding cache use, training data, classifier caching and prediction inputs."""

import numpy as np
import pytest
from PIL import Image

from fewshot.classifiers import KNNClassifier
from fewshot.embedders import register_embedder, unregister_embedder
from fewshot.pipeline import Pipeline, Prediction
from fewshot.store import image_id
from tests.conftest import ColorEmbedder, png

RED, GREEN, BLUE = (220, 30, 30), (30, 220, 30), (30, 30, 220)
TEXT_SPEC = "openclip:fake-color"


class CountingEmbedder(ColorEmbedder):
    """Colour embedder that records how many images and texts it was asked to embed."""

    supports_text = True

    def __init__(self):
        super().__init__()
        self.images = 0
        self.texts: list[str] = []

    def _embed_image_batch(self, images):
        self.images += len(images)
        return super()._embed_image_batch(images)

    def _embed_text_batch(self, texts):
        self.texts += texts
        words = {"red": RED, "green": GREEN, "blue": BLUE}
        v = np.array([[float(c) + 1.0 for c in next(rgb for w, rgb in words.items() if w in t)] for t in texts])
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(name="emb")
def fixture_emb():
    """A counting embedder registered both as "fake:color" and as a text-capable CLIP-like spec."""
    e = CountingEmbedder()
    register_embedder("fake:color", e)
    register_embedder(TEXT_SPEC, e)
    yield e
    unregister_embedder(TEXT_SPEC)


def make_task(ws, name="t", embedder="fake:color", **settings):
    """Task with two examples each of red, green and blue."""
    t = ws.create(name, embedder=embedder, **settings)
    for label, rgb in [("red", RED), ("green", GREEN), ("blue", BLUE)]:
        for i in range(2):
            t.add_example(png(rgb, noise=15, seed=i), label)
    return t


def test_prediction_to_dict():
    """to_dict exposes every field."""
    p = Prediction("a", "a", 0.9, 0.8, False, {"a": 0.9, "b": 0.1})
    assert p.to_dict() == {"label": "a", "top_label": "a", "confidence": 0.9, "similarity": 0.8,
                           "rejected": False, "scores": {"a": 0.9, "b": 0.1}}


# ---- embeddings --------------------------------------------------------
def test_embed_images_uses_and_fills_cache(ws, emb):
    """Keyed images are embedded once; unkeyed (PIL) ones every time; order is preserved."""
    pipe = Pipeline(ws)
    red, green = png(RED), png(GREEN)
    blue_im = Image.new("RGB", (4, 4), BLUE)
    out = pipe.embed_images("fake:color", [("r", red), (None, blue_im)])
    assert out.shape == (2, 3)
    assert out.dtype == np.float32
    assert emb.images == 2
    assert set(pipe.cache.get_many("fake:color", ["r", "g"])) == {"r"}
    out2 = pipe.embed_images("fake:color", [("g", green), ("r", red), (None, blue_im)])
    assert emb.images == 4  # only "g" and the PIL image
    np.testing.assert_allclose(out2[1], out[0], atol=1e-6)
    np.testing.assert_allclose(out2[2], out[1], atol=1e-6)
    assert out2[0].argmax() == 1
    assert out2[1].argmax() == 0
    assert out2[2].argmax() == 2
    assert pipe.embed_images("fake:color", [("g", green)]).shape == (1, 3)
    assert emb.images == 4


def test_embed_images_empty(ws, emb):
    """No items gives an empty matrix without touching the model."""
    assert Pipeline(ws).embed_images("fake:color", []).shape == (0, 0)
    assert emb.images == 0


def test_embed_texts_caches_and_dedupes(ws, emb):
    """Each distinct text is embedded once, ever; rows follow input order."""
    pipe = Pipeline(ws)
    out = pipe.embed_texts(TEXT_SPEC, ["a red thing", "blue", "a red thing"])
    assert emb.texts == ["a red thing", "blue"]
    assert out.shape == (3, 3)
    assert [int(r.argmax()) for r in out] == [0, 2, 0]
    out2 = pipe.embed_texts(TEXT_SPEC, ["blue", "green"])
    assert emb.texts == ["a red thing", "blue", "green"]
    np.testing.assert_allclose(out2[0], out[1])
    assert set(pipe.cache.get_many(TEXT_SPEC, ["text:blue", "text:green"])) == {"text:blue", "text:green"}
    assert pipe.embed_texts(TEXT_SPEC, []).shape == (0, 0)


# ---- training data -----------------------------------------------------
def test_class_data_and_exclude(ws, emb):
    """Rows follow the examples; excluded indices are dropped; image-only specs ignore text."""
    t = make_task(ws)
    t.upsert_class("red", "a red thing")
    pipe = Pipeline(ws)
    d = pipe.class_data(t)
    assert d.classes == ["red", "green", "blue"]
    assert d.y.tolist() == [0, 0, 1, 1, 2, 2]
    assert d.X.shape == (6, 3)
    assert not d.text
    assert emb.texts == []
    d2 = pipe.class_data(t, exclude={0, 3})
    assert d2.y.tolist() == [0, 1, 2, 2]
    assert d2.X.shape == (4, 3)
    np.testing.assert_allclose(d2.X[0], d.X[1])


def test_class_data_text_for_described_classes(ws, emb):
    """With a text-capable spec only classes with a non-blank description get text vectors."""
    t = make_task(ws, embedder=TEXT_SPEC)
    pipe = Pipeline(ws)
    assert not pipe.class_data(t).text
    t.upsert_class("blue", "something blue")
    t.upsert_class("red", "   ")
    t.upsert_class("purple", "a green-ish purple")
    d = pipe.class_data(t)
    assert sorted(d.text) == [2, 3]
    assert emb.texts == ["something blue", "a green-ish purple"]
    assert d.text[2].argmax() == 2
    assert d.text[3].argmax() == 1


def test_text_only_class_is_predicted(ws, emb):
    """A class known only from its description can win a prediction."""
    t = ws.create("t", embedder=TEXT_SPEC)
    t.add_example(png(RED), "red")
    t.add_example(png(GREEN), "green")
    t.upsert_class("blue", "pure blue")
    preds = Pipeline(ws).predict(t, [png(BLUE, seed=3), png(RED, seed=3)])
    assert [p.label for p in preds] == ["blue", "red"]
    assert emb.texts == ["pure blue"]


# ---- classifier caching ------------------------------------------------
def test_fitted_is_reused_until_task_changes(ws, emb):
    """The classifier is fitted once per revision, with the task's settings."""
    t = make_task(ws, classifier="knn", classifier_params={"k": 3}, text_weight=0.25)
    pipe = Pipeline(ws)
    clf = pipe.fitted(t)
    assert isinstance(clf, KNNClassifier)
    assert clf.k == 3
    assert clf.text_weight == 0.25
    n = emb.images
    assert pipe.fitted(t) is clf
    assert emb.images == n
    t.update_settings(classifier_params={"k": 1})
    clf2 = pipe.fitted(t)
    assert isinstance(clf2, KNNClassifier)
    assert clf2 is not clf
    assert clf2.k == 1
    assert emb.images == n  # refit reused cached example embeddings


def test_recreated_task_with_same_revision_is_refitted(ws, emb):
    """A deleted and recreated task is never served the old task's classifier."""
    pipe = Pipeline(ws)
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "a")
    t.add_example(png(GREEN), "b")
    assert pipe.predict("t", [png(RED, seed=1)])[0].label == "a"
    old_rev = t.config.revision
    ws.delete("t")
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "b")
    t.add_example(png(GREEN), "a")
    assert t.config.revision == old_rev
    assert pipe.predict("t", [png(RED, seed=1)])[0].label == "b"
    assert emb.images == 2  # the query equals the red example, and the refit is served by the cache


# ---- prediction --------------------------------------------------------
def test_predict_accepts_bytes_paths_and_pil(ws, emb, tmp_path):
    """bytes, str and Path inputs are cached by content id; PIL images are not."""
    t = make_task(ws)
    pipe = Pipeline(ws)
    red, green, blue = png(RED, seed=7), png(GREEN, seed=7), png(BLUE, seed=7)
    (tmp_path / "g.png").write_bytes(green)
    (tmp_path / "b.png").write_bytes(blue)
    pil = Image.new("L", (4, 4), 200)  # grey, converted to RGB
    preds = pipe.predict("t", [red, str(tmp_path / "g.png"), tmp_path / "b.png", pil])
    assert [p.label for p in preds] == ["red", "green", "blue", preds[3].top_label]
    assert all(not p.rejected and abs(sum(p.scores.values()) - 1) < 1e-6 for p in preds)
    assert preds[0].confidence == max(preds[0].scores.values())
    assert 0 < preds[0].similarity <= 1
    cached = pipe.cache.get_many("fake:color", [image_id(b) for b in (red, green, blue)])
    assert len(cached) == 3
    n = emb.images
    pipe.predict(t, [red, pil])
    assert emb.images == n + 1  # only the PIL image is re-embedded
    assert pipe.predict(t, []) == []


def test_predict_threshold_keeps_confident_labels(ws, emb):
    """With reject_below set, similar images keep their label and dissimilar ones are rejected."""
    t = make_task(ws, reject_below=0.95)
    preds = Pipeline(ws).predict(t, [png(RED, seed=4), png((128, 128, 128))])
    assert preds[0].label == "red"
    assert not preds[0].rejected
    assert preds[0].similarity >= 0.95
    assert preds[1].label is None
    assert preds[1].rejected
    assert preds[1].similarity < 0.95
    assert emb.texts == []


def test_predict_rejects_unreadable_input(ws, emb):
    """Undecodable query bytes raise ValueError; a missing path raises FileNotFoundError."""
    t = make_task(ws)
    pipe = Pipeline(ws)
    with pytest.raises(ValueError, match="Not a readable image"):
        pipe.predict(t, [b"junk"])
    with pytest.raises(FileNotFoundError):
        pipe.predict(t, ["/nonexistent/x.png"])
    with pytest.raises(KeyError, match="No task 'nope'"):
        pipe.predict("nope", [png(RED)])
    assert emb.texts == []
