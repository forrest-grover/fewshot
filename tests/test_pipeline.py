"""Pipeline: embedding cache use, training data, classifier caching and prediction inputs."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import fewshot.pipeline as pipeline_mod
from fewshot.classifiers import KNNClassifier
from fewshot.embedders import Embedder, cache_namespace, get_embedder, register_embedder, unregister_embedder
from fewshot.images import image_id
from fewshot.pipeline import Pipeline, Prediction, binomial_tail, names_match
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


class ImageOnly(Embedder):
    """Image-only view of another embedder, so both specs share its counters."""

    cache_version = "test"

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    @property
    def dim(self):
        return self.inner.dim

    def _embed_image_batch(self, images):
        return self.inner._embed_image_batch(images)  # pylint: disable=protected-access


@pytest.fixture(name="emb")
def fixture_emb():
    """A counting embedder registered as a text-capable CLIP-like spec and, image-only, as "fake:color"."""
    e = CountingEmbedder()
    register_embedder("fake:color", ImageOnly(e))
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
    p = Prediction("a", "a", 0.9, 0.8, False, {"a": 0.9, "b": 0.1}, ["a"])
    assert p.to_dict() == {"label": "a", "top_label": "a", "confidence": 0.9, "similarity": 0.8,
                           "rejected": False, "scores": {"a": 0.9, "b": 0.1}, "labels": ["a"], "evidence": [],
                           "error": None, "revision": None}
    assert Prediction.failed("bad").to_dict() == {"label": None, "top_label": "", "confidence": 0.0,
                                                  "similarity": 0.0, "rejected": False, "scores": {},
                                                  "labels": [], "evidence": [], "error": "bad", "revision": None}


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
    assert set(pipe.cache.get_many(cache_namespace("fake:color"), ["r", "g"])) == {"r"}
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
    cached = pipe.cache.get_many(cache_namespace(TEXT_SPEC), ["text:blue", "text:green"])
    assert set(cached) == {"text:blue", "text:green"}
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
    """With a text-capable spec (and names_as_text off) only classes with a non-blank description get
    text vectors."""
    t = make_task(ws, embedder=TEXT_SPEC, names_as_text=False)
    pipe = Pipeline(ws)
    assert not pipe.class_data(t).text
    t.upsert_class("blue", "something blue")
    t.upsert_class("red", "   ")
    t.upsert_class("purple", "a green-ish purple")
    d = pipe.class_data(t)
    assert sorted(d.text) == [2, 3]
    # descriptions first, then the undescribed classes' names (for the modality gap only)
    assert emb.texts == ["something blue", "a green-ish purple", "red", "green"]
    assert sorted(d.name_text) == [0, 1]
    assert d.text[2].argmax() == 2
    assert d.text[3].argmax() == 1


def test_text_only_class_is_predicted(ws, emb):
    """A class known only from its description can win a prediction."""
    t = ws.create("t", embedder=TEXT_SPEC, names_as_text=False)
    t.add_example(png(RED), "red")
    t.add_example(png(GREEN), "green")
    t.upsert_class("blue", "pure blue")
    preds = Pipeline(ws).predict(t, [png(BLUE, seed=3), png(RED, seed=3)])
    assert [p.label for p in preds] == ["blue", "red"]
    assert emb.texts == ["pure blue", "red", "green"]


def test_names_are_not_embedded_without_descriptions(ws, emb):
    """With names_as_text off, class names are only embedded when some class has a description."""
    t = make_task(ws, embedder=TEXT_SPEC, names_as_text=False)
    assert not Pipeline(ws).class_data(t).name_text
    assert emb.texts == []


def test_names_as_text_when_they_match(ws, emb):
    """Classes with examples but no description use their names as text when the names classify
    the examples better than chance; names that don't are left out, as are classes without examples."""
    t = make_task(ws, embedder=TEXT_SPEC)
    t.upsert_class("yellow")  # no examples: not described by its name
    d = Pipeline(ws).class_data(t)
    assert sorted(d.text) == [0, 1, 2]
    assert emb.texts == ["a photo of a red.", "a photo of a green.", "a photo of a blue."]
    assert [int(d.text[i].argmax()) for i in range(3)] == [0, 1, 2]
    wrong = ws.create("wrong", embedder=TEXT_SPEC)  # every name points at another class's colour
    for label, rgb in [("green-a", RED), ("blue-b", GREEN), ("red-c", BLUE)]:
        for i in range(2):
            wrong.add_example(png(rgb, noise=15, seed=i), label)
    assert not Pipeline(ws).class_data(wrong).text


def test_binomial_tail():
    """P(X >= k) for a binomial, exact at the edges and stable for large n."""
    assert binomial_tail(0, 5, 0.5) == 1.0
    assert binomial_tail(6, 5, 0.5) == 0.0
    assert binomial_tail(2, 2, 0.5) == pytest.approx(0.25)
    assert binomial_tail(3, 3, 1.0) == 1.0
    assert binomial_tail(2000, 2000, 0.01) < 1e-300  # no overflow
    assert binomial_tail(500, 1000, 0.5) == pytest.approx(0.5126, abs=1e-3)


def test_names_match_allows_for_unbalanced_classes():
    """Names that send every image to the largest class match no better than chance, however many
    examples that gets right; names that tell the classes apart do, even with one example each."""
    truth = np.repeat([0, 1], [30, 5])
    assert not names_match(np.zeros(35, int), truth)  # 30/35 right, but so would any constant guess
    assert names_match(truth.copy(), truth)
    one_each = np.arange(5)
    assert names_match(one_each.copy(), one_each)
    assert names_match(np.array([0, 1, 2, 3, 9]), one_each)  # 9: a described class, never right
    assert not names_match(np.array([1, 2, 3, 4, 0]), one_each)


def test_same_lazy_pil_image_decoded_more_than_once():
    """A lazily loaded PIL image given several times decodes the same each time (its file is
    read once at a time, not by several threads at once)."""
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    Image.fromarray((rng.random((600, 600, 3)) * 255).astype(np.uint8)).save(buf, "JPEG", quality=95)
    ref = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    for _ in range(3):
        im = Image.open(io.BytesIO(buf.getvalue()))
        errors: dict[int, str] = {}
        out = pipeline_mod._decode([(i, None, im) for i in range(8)], errors)  # pylint: disable=protected-access
        assert not errors
        assert all(np.array_equal(np.asarray(r), ref) for _, _, r in out)


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
    cached = pipe.cache.get_many(cache_namespace("fake:color"), [image_id(b) for b in (red, green, blue)])
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
    """Unreadable inputs get an error result each while the rest are classified."""
    t = make_task(ws)
    pipe = Pipeline(ws)
    bad, good = pipe.predict(t, [b"junk", png(RED, seed=9)])
    assert bad.error == "Not a readable image: unrecognised file format"
    assert (bad.label, bad.scores) == (None, {})
    assert (good.error, good.label) == (None, "red")
    with pytest.raises(ValueError, match="Not a readable image"):
        pipe.embed_images("fake:color", [(None, b"junk")])
    truncated = Image.open(io.BytesIO(png(RED, size=64, noise=50)[:-40]))  # fails only when loaded
    missing, folder, cut, later = pipe.predict(t, ["/nonexistent/x.png", ws.root, truncated, Path(ws.root / "r.png")])
    assert missing.error == "Could not read /nonexistent/x.png: No such file or directory"
    assert folder.error == f"Could not read {ws.root}: Is a directory"
    assert cut.error is not None
    assert cut.error.startswith("Not a readable image: image file is truncated")
    assert later.error == f"Could not read {ws.root / 'r.png'}: No such file or directory"
    (ws.root / "r.png").write_bytes(png(RED, seed=9))
    assert pipe.predict(t, [ws.root / "r.png"])[0].label == "red"
    with pytest.raises(KeyError, match="No task 'nope'"):
        pipe.predict("nope", [png(RED)])
    assert emb.texts == []


def test_empty_class_is_left_out(ws):
    """A class with no examples and no description doesn't stop predict, and isn't scored."""
    t = make_task(ws)
    t.upsert_class("purple")
    pipe = Pipeline(ws)
    assert pipe.class_data(t).classes == ["red", "green", "blue"]
    (p,) = pipe.predict(t, [png(BLUE, seed=9)])
    assert p.label == "blue"
    assert set(p.scores) == {"red", "green", "blue"}


def test_registered_embedder_decides_text_support(ws, emb):
    """Descriptions are used for any spec whose registered embedder supports text, whatever its name."""
    register_embedder("mine:v1", emb)
    try:
        t = make_task(ws, embedder="mine:v1", names_as_text=False)
        t.upsert_class("red", "a red thing")
        assert sorted(Pipeline(ws).class_data(t).text) == [0]
    finally:
        unregister_embedder("mine:v1")


def test_reregistered_embedder_gets_fresh_vectors_and_fit(ws):
    """Replacing the model under a spec never reuses the old model's cached vectors or classifier."""

    class Inverted(ColorEmbedder):  # pylint: disable=abstract-method
        """A different model: every vector negated."""

        cache_version = None  # fingerprinted

        def _embed_image_batch(self, images):
            return -super()._embed_image_batch(images)

    register_embedder("mine:v1", ColorEmbedder())
    try:
        t = make_task(ws, embedder="mine:v1")
        pipe = Pipeline(ws)
        first = pipe.example_embeddings(t)
        clf = pipe.fitted(t)
        register_embedder("mine:v1", Inverted())
        np.testing.assert_allclose(pipe.example_embeddings(t), -first)
        assert pipe.fitted(t) is not clf
        ns = cache_namespace("mine:v1")
        assert ns.startswith("mine:v1#")
        assert ns != "mine:v1#test"
        assert cache_namespace("mine:v1") == ns  # computed once
        register_embedder("mine:v1", Inverted())  # same weights, same fingerprint: cache reused
        assert cache_namespace("mine:v1") == ns
    finally:
        unregister_embedder("mine:v1")
    assert cache_namespace("mine:v1") == "mine:v1"
    assert cache_namespace("openclip:ViT-B-32@laion2b_s34b_b79k") == "openclip:ViT-B-32@laion2b_s34b_b79k"


def test_namespace_of_a_replaced_embedder_is_not_kept():
    """If the embedder is replaced while its fingerprint is computed, that result isn't stored."""

    class Replaced(ColorEmbedder):  # pylint: disable=abstract-method
        """Re-registers the spec with another model during its own fingerprint."""

        cache_version = None

        def _embed_image_batch(self, images):
            register_embedder("mine:v2", ColorEmbedder())
            return super()._embed_image_batch(images)

    register_embedder("mine:v2", Replaced())
    try:
        assert cache_namespace("mine:v2").startswith("mine:v2#")
        assert cache_namespace("mine:v2") == "mine:v2#test"  # the replacement's own namespace
    finally:
        unregister_embedder("mine:v2")


# ---- chunked embedding, progress and cache checks -------------------------
def test_embedding_is_chunked_with_progress(ws, emb):
    """Uncached images are embedded and cached one batch at a time, with progress after each."""
    get_embedder("fake:color").batch_size = 2  # the registered wrapper is what the pipeline batches with
    pipe = Pipeline(ws)
    calls = []
    items = [(f"k{i}", png(RED, noise=10, seed=i)) for i in range(5)]
    out = pipe.embed_images("fake:color", items, progress=lambda d, t: calls.append((d, t)))
    assert out.shape == (5, 3)
    assert calls == [(2, 5), (4, 5), (5, 5)]
    assert emb.images == 5
    pipe.embed_images("fake:color", items, progress=lambda d, t: calls.append((d, t)))
    assert emb.images == 5  # all cached: no model work and no progress
    assert len(calls) == 3


def test_cancelled_embedding_keeps_finished_batches(ws, emb):  # pylint: disable=unused-argument  # registers the spec
    """If progress raises (a cancel), the batches already embedded stay cached."""
    get_embedder("fake:color").batch_size = 2  # the registered wrapper is what the pipeline batches with
    pipe = Pipeline(ws)
    items = [(f"k{i}", png(RED, noise=10, seed=i)) for i in range(5)]

    def stop(done, _total):
        raise KeyboardInterrupt(done)

    with pytest.raises(KeyboardInterrupt):
        pipe.embed_images("fake:color", items, progress=stop)
    assert set(pipe.cache.get_many(cache_namespace("fake:color"), [k for k, _ in items])) == {"k0", "k1"}


def test_cached_vectors_of_another_length_are_reembedded(ws, emb):
    """Vectors cached under the spec by a model with another dimension are replaced."""
    pipe = Pipeline(ws)
    ns = cache_namespace("fake:color")
    pipe.cache.put_many(ns, {"a": np.ones(5, np.float32)})
    pipe.embed_images("fake:color", [("b", png(BLUE))])
    out = pipe.embed_images("fake:color", [("a", png(RED)), ("b", png(BLUE))])  # mixed lengths in the cache
    assert out.shape == (2, 3)
    assert len(pipe.cache.get_many(ns, ["a"])["a"]) == 3
    assert emb.images == 2

    tns = cache_namespace(TEXT_SPEC)
    pipe.cache.put_many(tns, {"text:a red thing": np.ones(5, np.float32), "text:a blue thing": np.ones(3, np.float32)})
    t = pipe.embed_texts(TEXT_SPEC, ["a red thing", "a blue thing"])
    assert t.shape == (2, 3)
    assert emb.texts == ["a red thing"]


def test_embed_files_reads_each_file_once(ws, emb, tmp_path, monkeypatch):
    """embed_files hashes and decodes from one read per file, in chunks, with progress."""
    files = []
    for i in range(5):
        f = tmp_path / f"{i}.png"
        f.write_bytes(png(GREEN, noise=10, seed=i))
        files.append(f)
    reads = []
    real = Path.read_bytes
    def counted(p):
        reads.append(p.name)
        return real(p)

    monkeypatch.setattr(Path, "read_bytes", counted)
    calls = []
    pipe = Pipeline(ws)
    out = pipe.embed_files("fake:color", files, progress=lambda d, t: calls.append((d, t)), chunk=2)
    assert out.shape == (5, 3)
    assert sorted(reads) == sorted(f.name for f in files)
    assert calls == [(2, 5), (4, 5), (5, 5)]
    np.testing.assert_allclose(out, pipe.embed_images("fake:color", [(None, f) for f in files]))
    assert pipe.embed_files("fake:color", []).shape == (0, 0)
    assert emb.images == 10
