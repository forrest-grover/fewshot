"""Unit tests for fewshot.classifiers on small hand-built embeddings."""

import numpy as np
import pytest

from fewshot import classifiers
from fewshot.classifiers import (
    CLASSIFIERS,
    ClassData,
    Classifier,
    GDAClassifier,
    KNNClassifier,
    LogRegClassifier,
    PrototypeClassifier,
    make_classifier,
)

ALL = ["prototype", "knn", "logreg", "gda"]


def unit(v):
    """L2-normalise rows as float32."""
    v = np.asarray(v, np.float32)
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def two_clusters():
    """Two tight clusters around the x and y axes in 3-d."""
    X = unit([[1, 0.1, 0], [1, -0.1, 0], [0.1, 1, 0], [-0.1, 1, 0]])
    return ClassData(["x", "y"], X, np.array([0, 0, 1, 1]))


def test_registry_and_factory():
    """make_classifier builds the registered class with the given params."""
    assert sorted(CLASSIFIERS) == sorted(ALL)
    assert CLASSIFIERS["knn"] is KNNClassifier
    clf = make_classifier("knn", k=2, text_weight=0.5)
    assert isinstance(clf, KNNClassifier)
    assert clf.k == 2
    assert clf.text_weight == 0.5
    assert isinstance(make_classifier("prototype"), PrototypeClassifier)
    logreg = make_classifier("logreg", C=1.0)
    assert isinstance(logreg, LogRegClassifier)
    assert logreg.C == 1.0


def test_unknown_classifier_lists_choices():
    """An unknown name raises and lists the valid choices."""
    names = r"\['gda', 'knn', 'logreg', 'prototype'\]"
    with pytest.raises(ValueError, match=r"Unknown classifier 'svm'; choose from " + names):
        make_classifier("svm")


def test_dim_from_images_or_text():
    """dim comes from X, or from the text vectors when there are no images."""
    assert two_clusters().dim == 3
    text = {0: np.ones(7, np.float32), 1: np.ones(7, np.float32)}
    zero_shot = ClassData(["a", "b"], np.zeros((0, 0), np.float32), np.zeros(0, int), text)
    assert zero_shot.dim == 7


@pytest.mark.parametrize("name", ALL)
def test_one_class(name):
    """A single class is allowed: every query gets it, with its similarity to the class, which is
    what tells members from outsiders. No class at all is an error."""
    data = ClassData(["only"], unit([[1, 0, 0], [0.9, 0.1, 0]]), np.array([0, 0]))
    clf = make_classifier(name).fit(data)
    probs, sims = clf.predict(unit([[1, 0, 0], [0, 0, 1]]))
    assert probs.tolist() == [[1.0], [1.0]]
    assert sims[0] > 0.9 > 0.1 > sims[1]
    assert clf.class_similarity(unit([[0, 0, 1]])).shape == (1, 1)
    with pytest.raises(ValueError, match="at least 1 class"):
        make_classifier(name).fit(ClassData([], np.zeros((0, 3), np.float32), np.zeros(0, int)))


@pytest.mark.parametrize("name", ALL)
def test_separable_clusters(name):
    """Each classifier separates two clusters with normalised probabilities."""
    clf = make_classifier(name).fit(two_clusters())
    assert clf.classes == ["x", "y"]
    assert not clf.zero_shot
    assert clf.Pt is None
    probs, sims = clf.predict(unit([[1, 0.05, 0], [0, 1, 0.05]]))
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(1), 1, rtol=1e-6)
    assert probs.argmax(1).tolist() == [0, 1]
    assert np.all(sims > 0.98)
    assert np.all(sims <= 1 + 1e-6)


def test_fit_returns_self_and_copies_class_list():
    """fit is chainable and does not alias the caller's class list."""
    data = two_clusters()
    clf = PrototypeClassifier()
    assert clf.fit(data) is clf
    data.classes.append("z")
    assert clf.classes == ["x", "y"]


def test_prototype_similarity_is_to_centroid_not_examples():
    """Prototype similarity is cosine to the class centroid."""
    clf = make_classifier("prototype").fit(two_clusters())
    _, sims = clf.predict(unit([[1, 0.1, 0]]))  # identical to one example, off the centroid
    assert sims[0] == pytest.approx(float(unit([1, 0.1, 0]) @ unit([1, 0, 0])), abs=1e-6)
    assert sims[0] < 0.999


@pytest.mark.parametrize("name", ["knn", "logreg"])
def test_example_based_similarity_is_to_nearest_example(name):
    """kNN and logreg report cosine to the nearest support example."""
    clf = make_classifier(name).fit(two_clusters())
    _, sims = clf.predict(unit([[1, 0.1, 0]]))
    assert sims[0] == pytest.approx(1.0, abs=1e-6)


def test_prototype_scale_sharpens_probabilities():
    """A larger scale gives more confident prototype predictions."""
    q = unit([[1, 0.6, 0]])
    soft = make_classifier("prototype", scale=1.0).fit(two_clusters()).predict(q)[0][0, 0]
    sharp = make_classifier("prototype", scale=50.0).fit(two_clusters()).predict(q)[0][0, 0]
    assert 0.5 < soft < sharp


def test_knn_k_larger_than_support_and_k1_vote():
    """k=1 follows the nearest neighbour; k beyond the support size is clipped."""
    # One "x" example right next to the query, three "y" examples further away.
    X = unit([[1, 0, 0], [0.6, 1, 0], [0.5, 1, 0], [0.4, 1, 0]])
    data = ClassData(["x", "y"], X, np.array([0, 1, 1, 1]))
    q = unit([[0.9, 0.4, 0]])
    assert make_classifier("knn", k=1).fit(data).predict(q)[0].argmax(1).tolist() == [0]
    probs, _ = make_classifier("knn", k=50, scale=0.0).fit(data).predict(q)  # unweighted vote of all 4
    np.testing.assert_allclose(probs[0], [0.25, 0.75], atol=1e-5)


@pytest.mark.parametrize("name", ALL)
def test_predict_before_fit_is_an_error(name):
    """Predicting with an unfitted classifier raises a clear error instead of failing on None."""
    with pytest.raises(RuntimeError, match=r"not fitted; call fit\(\) first"):
        make_classifier(name).predict(unit([[1, 0, 0]]))


def test_base_classifier_requires_subclass_hooks():
    """The base class leaves image fitting and scoring to subclasses."""
    with pytest.raises(NotImplementedError):
        Classifier().fit(two_clusters())
    with pytest.raises(NotImplementedError):
        Classifier()._image_logprobs(unit([[1, 0, 0]]))  # pylint: disable=protected-access  # testing the hook


def text_task():
    """Three classes on the axes; text lives across a modality gap. Class 'b' is image-only."""
    gap = np.array([0, 0, 0, 3.0])
    X = unit([[1, 0.1, 0, 0], [1, -0.1, 0, 0], [0.1, 1, 0, 0], [-0.1, 1, 0, 0]])
    text = {0: unit(np.array([1, 0, 0, 0]) + gap), 2: unit(np.array([0, 0, 1, 0]) + gap)}
    return ClassData(["a", "b", "c"], X, np.array([0, 0, 1, 1]), text), gap


@pytest.mark.parametrize("name", ALL)
def test_image_only_class_gets_shifted_text_vector(name):
    """Image-only classes get a text-side vector moved across the modality gap."""
    data, _ = text_task()
    clf = make_classifier(name).fit(data)
    assert clf.Pt is not None
    assert clf.Pt.shape == (3, 4)
    np.testing.assert_allclose(clf.Pt[0], data.text[0], atol=1e-6)
    # 'b' centroid (y axis) moved from image space into text space: mostly y plus the gap direction.
    assert clf.Pt[1].argmax() in (1, 3)
    assert clf.Pt[1][1] > 0.25
    assert clf.Pt[1][3] > 0.5
    np.testing.assert_allclose(np.linalg.norm(clf.Pt, axis=1), 1, atol=1e-6)
    # Text-only 'c' is reachable from an image near the z axis; a/b still work.
    q = unit([[1, 0, 0.05, 0], [0, 1, 0.05, 0], [0.1, 0.1, 1, 0]])
    assert clf.predict(q)[0].argmax(1).tolist() == [0, 1, 2]


def test_text_weight_zero_ignores_text_but_keeps_text_only_anchor():
    """text_weight=0 drops text scoring but text-only classes stay predictable."""
    data, _ = text_task()
    clf = make_classifier("prototype", text_weight=0.0).fit(data)
    assert clf.Pt is None
    q = unit([[0.1, 0.1, 1, 0]])
    assert clf.predict(q)[0].argmax(1).tolist() == [2]


def test_text_term_shifts_ambiguous_image():
    """The text term can flip an image that is ambiguous on image evidence alone."""
    X = unit([[1, 0, 0], [0, 1, 0]])
    data = ClassData(["a", "b"], X, np.array([0, 1]), {1: unit([0.6, 1, 0])})
    q = unit([[1, 0.9, 0]])  # slightly closer to 'a' in image space
    assert make_classifier("prototype", text_weight=0.0).fit(data).predict(q)[0].argmax(1).tolist() == [0]
    assert make_classifier("prototype", text_weight=1.0).fit(data).predict(q)[0].argmax(1).tolist() == [1]


def test_zero_shot_uses_clip_scale_and_text_similarity():
    """With no images, scoring is a softmax of 100x image-text cosine."""
    t = {0: unit([1, 0, 0]), 1: unit([0, 1, 0])}
    data = ClassData(["a", "b"], np.zeros((0, 3), np.float32), np.zeros(0, int), t)
    clf = make_classifier("knn", text_weight=0.0).fit(data)  # zero-shot uses text even with weight 0
    assert clf.zero_shot
    q = unit([[1, 0.98, 0]])
    probs, sims = clf.predict(q)
    cos = q @ np.stack([t[0], t[1]]).T
    expected = np.exp(100 * (cos[0] - cos[0].max()))
    expected /= expected.sum()
    assert 0.6 < probs[0, 0] < 0.95  # 100x logit scale: a 0.007 cosine gap is decisive but not certain
    np.testing.assert_allclose(probs[0], expected, rtol=1e-5)
    assert sims[0] == pytest.approx(cos[0].max())


def test_logreg_respects_regularisation():
    """Weaker regularisation (larger C) gives more confident logreg predictions."""
    data = two_clusters()
    q = unit([[1, 0.2, 0]])
    weak = make_classifier("logreg", C=0.01).fit(data).predict(q)[0][0, 0]
    strong = make_classifier("logreg", C=100.0).fit(data).predict(q)[0][0, 0]
    assert 0.5 < weak < strong
    assert isinstance(make_classifier("logreg"), LogRegClassifier)


def test_knn_low_similarities_still_decide():
    """With every similarity far below 1, the nearer class still wins clearly (no flattening by the floor)."""
    X = unit([[1, 0, 0], [0, 1, 0]]).astype(np.float32)
    data = ClassData(["a", "b"], X, np.array([0, 1]))
    q = unit([[0.3, 0.1, 1.0]]).astype(np.float32)  # sim 0.29 to a, 0.10 to b
    probs, _ = make_classifier("knn", k=2).fit(data).predict(q)
    expected, _ = make_classifier("prototype").fit(data).predict(q)
    np.testing.assert_allclose(probs, expected, atol=1e-5)
    assert probs[0, 0] > 0.99


@pytest.mark.parametrize(("name", "params", "message"), [
    ("nope", {}, "Unknown classifier 'nope'"),
    ("knn", {"bogus": 1}, "Unknown parameter 'bogus' for knn"),
    ("knn", {"k": "abc"}, "'k' must be an integer"),
    ("knn", {"k": 2.5}, "'k' must be an integer"),
    ("knn", {"k": True}, "'k' must be an integer"),
    ("knn", {"k": 0}, "'k' must be > 0"),
    ("logreg", {"C": 0}, "'C' must be > 0"),
    ("prototype", {"scale": -1}, "'scale' must be >= 0"),
    ("prototype", {"scale": float("nan")}, "'scale' must be >= 0"),
    ("prototype", {"text_weight": "x"}, "'text_weight' must be a number"),
    ("prototype", {"scale": 1e308}, "'scale' must be >= 0.0 and <= 1000"),
    ("prototype", {"text_weight": 101}, "'text_weight' must be >= 0.0 and <= 100"),
    ("knn", {"k": 10**400}, "'k' must be > 0 and <= 100000"),
    ("knn", {"k": np.float64(3)}, "'k' must be an integer"),
    ("knn", {"k": np.bool_(True)}, "'k' must be an integer"),
    ("logreg", {"C": float("inf")}, "'C' must be > 0"),
])
def test_make_classifier_rejects_bad_params(name, params, message):
    """Bad classifier names and parameters raise ValueError up front, not at predict time."""
    with pytest.raises(ValueError, match=message):
        make_classifier(name, **params)
    make_classifier("prototype", scale=0, text_weight=0)  # zero is allowed for these
    make_classifier("knn", k=np.int64(3), scale=np.float32(20))  # NumPy numbers are fine


def test_single_described_class_uses_names_for_the_gap():
    """One image-only and one text-only class: the gap comes from the image class's name.

    Without names the text mean is the one description, the shift lands the text-only class's
    anchor exactly on the other class's centroid, and every prediction is 50/50.
    """
    gap = np.array([0, 0, 0, 1.0])
    cat_img, dog_img = unit([1, 0.2, 0, 0]), unit([0.2, 1, 0, 0])
    cat_txt, dog_txt = unit(cat_img + gap), unit(dog_img + gap)
    X = np.stack([cat_img, cat_img]).astype(np.float32)
    q = np.stack([cat_img, dog_img]).astype(np.float32)
    without = ClassData(["cat", "dog"], X, np.array([0, 0]), {1: dog_txt.astype(np.float32)})
    probs, _ = make_classifier("prototype").fit(without).predict(q)
    np.testing.assert_allclose(probs, 0.5, atol=1e-6)  # the degenerate fallback

    named = ClassData(["cat", "dog"], X, np.array([0, 0]), {1: dog_txt.astype(np.float32)},
                      {0: cat_txt.astype(np.float32)})
    for name in ("prototype", "knn", "logreg", "gda"):
        probs, _ = make_classifier(name).fit(named).predict(q)
        assert probs.argmax(1).tolist() == [0, 1], name


def test_gda_with_identical_examples():
    """Examples without any spread within their classes still fit (the spread has a floor)."""
    X = unit([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
    clf = make_classifier("gda").fit(ClassData(["a", "b"], X, np.array([0, 0, 1, 1])))
    probs, _ = clf.predict(unit([[0.9, 0.1, 0]]))
    assert probs[0].argmax() == 0


def test_gda_with_text_and_a_single_image():
    """One image plus a text-only class: the image term gets the prior weight, the scale stays 1."""
    X = unit([[1, 0.1, 0]])
    text = {0: unit([0.9, 0, 0.3]), 1: unit([0, 1, 0])}
    clf = make_classifier("gda").fit(ClassData(["a", "b"], X, np.array([0]), text))
    assert clf.logit_scale == 1.0
    probs, _ = clf.predict(unit([[0.1, 1, 0], [1, 0, 0.1]]))
    assert list(probs.argmax(1)) == [1, 0]


def noise(rng, n, d=64):
    """``n`` random unit vectors in ``d`` dimensions."""
    return unit(rng.normal(size=(n, d)))


def test_gda_confidence_when_only_one_class_can_be_left_out():
    """Image-only GDA on pure noise with 3 examples of one class and 1 of the other: leaving out
    the lone example leaves nothing to score it against, and the other class's examples, all
    "right", can't show how often its class is mistaken for the other. The scale must not then
    make every prediction certain."""
    rng = np.random.default_rng(9)
    confs = []
    for _ in range(10):
        clf = make_classifier("gda").fit(ClassData(["a", "b"], noise(rng, 4), np.array([0, 0, 0, 1])))
        confs.append(clf.predict(noise(rng, 200))[0].max(1).mean())
    assert np.mean(confs) < 0.9


@pytest.mark.parametrize("k", [2, 4])
def test_gda_image_only_confidence_matches_accuracy(k):
    """Image-only GDA on Gaussian classes with k images each: the mean confidence is about the
    accuracy. Leaving one of 2 images out leaves a mean of one, a harder problem than the fit's,
    so a scale fit to it without allowing for that made 2-image confidence 0.06 at 0.57 accuracy."""
    accs, confs = [], []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        centres = rng.normal(size=(20, 64)) / 8
        y, yq = np.repeat(np.arange(20), k), np.repeat(np.arange(20), 40)
        X, Q = (unit(centres[t] + 0.25 * rng.normal(size=(len(t), 64))) for t in (y, yq))
        probs = GDAClassifier().fit(ClassData([str(i) for i in range(20)], X, y)).predict(Q)[0]
        accs.append((probs.argmax(1) == yq).mean())
        confs.append(probs.max(1).mean())
    assert 0.4 < np.mean(accs) < 0.8
    assert np.mean(confs) == pytest.approx(np.mean(accs), abs=0.06)


def test_gda_confidence_with_too_few_images_for_a_spread():
    """With 3 images in 2 classes (one degree of freedom) the covariance is nu * I, and the scale
    is still fit from the class with two images; a nearer query gets the higher confidence."""
    X = unit([[1, 0.2, 0], [1, -0.2, 0], [0, 1, 0]])
    clf = GDAClassifier().fit(ClassData(["a", "b"], X, np.array([0, 0, 1])))
    probs, _ = clf.predict(unit([[1, 0, 0], [1, 0.6, 0]]))
    assert probs.argmax(1).tolist() == [0, 0]
    assert probs[0, 0] > probs[1, 0]


def test_gda_leaves_an_image_only_examples_own_text_side_out(monkeypatch):
    """With some classes described, an image-only class's text side is its centroid moved across
    the gap. Choosing the fusion weight, an example's own class is scored by the centroid without
    it, and an example alone in its class (with nothing left to stand for it) is left out."""
    seen: dict[str, np.ndarray] = {}
    fusion = classifiers._fusion  # pylint: disable=protected-access

    def spy(zt, loo, exact, y, k, text_weight, keep):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        seen.update(zt=zt, y=y[keep])
        return fusion(zt, loo, exact, y, k, text_weight, keep)

    monkeypatch.setattr(classifiers, "_fusion", spy)
    rng = np.random.default_rng(3)
    y = np.array([0, 0, 1, 1, 2, 2, 3])  # 0, 1 described; 2 image-only with 2 examples; 3 alone
    X = noise(rng, len(y))
    clf = make_classifier("gda").fit(ClassData(list("abcd"), X, y, {0: noise(rng, 1)[0], 1: noise(rng, 1)[0]}))
    assert seen["y"].tolist() == [0, 0, 1, 1, 2, 2]
    for row, other in ((4, 5), (5, 4)):
        side = unit(X[other] + clf.gap)
        assert seen["zt"][row, 2] == pytest.approx(100 * float(X[row] @ side), abs=1e-3)


def test_gda_text_weight_scales_the_tuned_balance():
    """The image term's weight is tuned for text at weight 1; ``text_weight`` then multiplies the
    text's share (in the scores and in the means' pull towards it), so a small one leaves the
    image term to decide and a large one the text."""
    rng = np.random.default_rng(2)
    centres = unit(rng.normal(size=(5, 32)))
    text = dict(enumerate(unit(centres + 0.5 * rng.normal(size=(5, 32)))))
    y = np.repeat(np.arange(5), 3)
    X = unit(centres[y] + 0.25 * rng.normal(size=(len(y), 32)))
    Q = unit(centres[np.repeat(np.arange(5), 20)] + 0.3 * rng.normal(size=(100, 32)))
    data = ClassData(list("abcde"), X, y, text)
    fits = {w: GDAClassifier(text_weight=w) for w in (0.001, 1.0, 100.0)}
    for clf in fits.values():
        clf.fit(data)
    image_only = fits[0.001]._image_logprobs(Q).argmax(1)  # pylint: disable=protected-access
    text_only = (Q @ np.stack(list(text.values())).T).argmax(1)
    assert (fits[0.001].predict(Q)[0].argmax(1) == image_only).all()
    assert (fits[100.0].predict(Q)[0].argmax(1) == text_only).mean() > 0.9
    assert not (image_only == text_only).all()


def test_gda_pulls_class_means_towards_their_text(monkeypatch):
    """With a few noisy images per class and text that knows where each class lies (across a
    modality gap), pulling the means towards the text beats leaving them as the images' means."""
    rng = np.random.default_rng(4)
    centres = unit(rng.normal(size=(10, 32)))
    gap = 0.5 * unit(rng.normal(size=32))
    text = dict(enumerate(unit(centres + gap)))
    y = np.repeat(np.arange(10), 2)
    yq = np.repeat(np.arange(10), 30)
    X = unit(centres[y] + 0.45 * rng.normal(size=(len(y), 32)))
    Q = unit(centres[yq] + 0.45 * rng.normal(size=(len(yq), 32)))
    data = ClassData([str(i) for i in range(10)], X, y, text)
    clf = GDAClassifier()
    clf.fit(data)
    pulled = (clf.predict(Q)[0].argmax(1) == yq).mean()
    monkeypatch.setattr(classifiers, "TEXT_PRIOR", 0.0)
    plain = (GDAClassifier().fit(data).predict(Q)[0].argmax(1) == yq).mean()
    assert pulled > plain + 0.02


def _prior_strengths(monkeypatch):
    """The lam values each GDA fit pulls its means with, recorded as the fits run."""
    seen = []
    real = classifiers._GDALeaveOneOut.pull_towards  # pylint: disable=protected-access

    def spy(self, anchors, lam):
        seen.append(lam)
        real(self, anchors, lam)
    monkeypatch.setattr(classifiers._GDALeaveOneOut, "pull_towards", spy)  # pylint: disable=protected-access
    return seen


def _noisy_task(text_noise, seed=5, C=20, k=3):
    rng = np.random.default_rng(seed)
    centres = unit(rng.normal(size=(C, 32)))
    text = dict(enumerate(unit(centres + text_noise * rng.normal(size=(C, 32)))))
    y = np.repeat(np.arange(C), k)
    return ClassData([str(i) for i in range(C)], unit(centres[y] + 0.6 * rng.normal(size=(len(y), 32))), y, text)


def test_gda_prior_needs_text_that_matches_the_images(monkeypatch):
    """Text whose anchors don't pick out their own class means (here shuffled across classes)
    doesn't pull the means at all; text that does, does."""
    seen = _prior_strengths(monkeypatch)
    data = _noisy_task(0.3)
    GDAClassifier().fit(data)
    assert (seen[-1] > 0).all()
    order = np.roll(np.arange(len(data.classes)), 1)
    GDAClassifier().fit(ClassData(data.classes, data.X, data.y, {i: data.text[int(j)] for i, j in enumerate(order)}))
    assert (seen[-1] == 0).all()
    # A single described class can't show it matches.
    GDAClassifier().fit(ClassData(data.classes, data.X, data.y, {0: data.text[0]}))
    assert (seen[-1] == 0).all()


def test_gda_text_weight_scales_the_prior(monkeypatch):
    """The means' pull towards the text scales with ``text_weight``, so a tiny one fits almost
    as with no text at all."""
    seen = _prior_strengths(monkeypatch)
    data = _noisy_task(0.3)
    GDAClassifier(text_weight=1.0).fit(data)
    GDAClassifier(text_weight=0.5).fit(data)
    assert seen[-1] == pytest.approx(seen[-2] * 0.5)
    tiny, none = GDAClassifier(text_weight=1e-6).fit(data), GDAClassifier(text_weight=0.0).fit(data)
    Q = data.X + 0.1
    assert (tiny.predict(Q)[0].argmax(1) == none.predict(Q)[0].argmax(1)).all()


def test_grid_searches_log_when_they_hit_an_end(caplog):
    """A grid search that chooses an end of its grid (where the best value may lie beyond it) says
    so at debug level; one that chooses inside it doesn't. So do evaluation's temperatures."""
    import fewshot.evaluate as ev  # pylint: disable=import-outside-toplevel

    caplog.set_level("DEBUG", logger="fewshot")
    y = np.array([0, 1])
    sure = np.array([[1.0, -1.0], [-1.0, 1.0]])
    grid = np.array([0.5, 1.0, 2.0])
    assert classifiers._least_nll([g * sure for g in grid], grid, y, what="probe") == 2.0  # pylint: disable=protected-access
    assert "probe chose 2, the high end of its grid" in caplog.text
    caplog.clear()
    assert classifiers._least_nll([g * sure for g in (0.5, 5, 10)], grid, y, smooth=True) == 0.5  # pylint: disable=protected-access
    assert "value chose 0.5, the low end of its grid" in caplog.text
    caplog.clear()
    assert classifiers._least_nll([sure * 0, sure, sure * 0], grid, y) == 1.0  # pylint: disable=protected-access
    assert not caplog.text
    ev._fit_temperature(np.array([[0.501, 0.499], [0.499, 0.501]] * 50), np.array([0, 1] * 50))  # pylint: disable=protected-access
    assert "temperature chose" in caplog.text


@pytest.mark.parametrize("name", ALL)
def test_refitting_forgets_the_previous_fit(name):
    """Fitting a classifier again (here without text, after a fit with it) gives what a fresh one
    would: no fitted scale, text vectors or gap carry over."""
    rng = np.random.default_rng(0)
    X, y = noise(rng, 12, 8), np.arange(12) % 3
    text = dict(enumerate(noise(rng, 3, 8)))
    plain = ClassData(list("abc"), X, y)
    Q = noise(rng, 5, 8)
    clf = make_classifier(name).fit(ClassData(list("abc"), X, y, text))
    assert clf.Pt is not None
    clf.fit(plain)
    fresh = make_classifier(name).fit(plain)
    assert clf.Pt is None
    assert clf.gap is None
    assert clf.logit_scale == fresh.logit_scale
    np.testing.assert_allclose(clf.predict(Q)[0], fresh.predict(Q)[0], rtol=1e-6)


def test_gda_pulls_means_with_the_classifiers_one_gap(monkeypatch):
    """GDA moves the descriptions across the same modality gap as the rest of the fit (one
    estimate, from ``_modality_gap``), not a second estimate of its own."""
    rng = np.random.default_rng(1)
    centres = noise(rng, 4, 16)
    y = np.repeat(np.arange(4), 3)
    X = unit(centres[y] + 0.3 * rng.normal(size=(12, 16)))
    data = ClassData(list("abcd"), X, y, dict(enumerate(unit(centres + 0.2))))
    real = GDAClassifier()
    real.fit(data)
    assert real.gap is not None
    monkeypatch.setattr(classifiers, "_modality_gap", lambda d: np.zeros(16, np.float32))
    zero = GDAClassifier()
    zero.fit(data)
    assert zero.W is not None
    assert real.W is not None
    assert not np.allclose(zero.W, real.W)


def test_gda_prior_with_two_classes(monkeypatch):
    """With 2 classes the class-mean test can't pass (at best p = 1/4), so the images are counted:
    matching text pulls a binary task's means, swapped text doesn't, and 2 images can't show it."""
    seen = _prior_strengths(monkeypatch)
    data = _noisy_task(0.3, seed=8, C=2, k=4)
    GDAClassifier().fit(data)
    assert (seen[-1] > 0).all()
    GDAClassifier().fit(ClassData(data.classes, data.X, data.y, {0: data.text[1], 1: data.text[0]}))
    assert (seen[-1] == 0).all()
    two = _noisy_task(0.3, seed=8, C=2, k=1)
    GDAClassifier().fit(two)
    assert (seen[-1] == 0).all()


def test_gda_stand_in_anchors_are_shortened_without_the_pull(monkeypatch):
    """A class whose only image is left out is represented by its anchor, shortened to the class
    means' length, also when the text doesn't pull the means (here shuffled descriptions)."""
    seen, lengths = _prior_strengths(monkeypatch), []
    real = classifiers._GDALeaveOneOut.scores  # pylint: disable=protected-access

    def spy(self, W, b, anchors=None, exact=True):
        lengths.append((np.linalg.norm(anchors, axis=1), np.linalg.norm(self.mu, axis=1).mean()))
        return real(self, W, b, anchors, exact)

    monkeypatch.setattr(classifiers._GDALeaveOneOut, "scores", spy)  # pylint: disable=protected-access
    data = _noisy_task(0.3, k=3)
    order = np.roll(np.arange(len(data.classes)), 1)
    shuffled = {i: data.text[int(j)] for i, j in enumerate(order)}
    GDAClassifier().fit(ClassData(data.classes, data.X[2:], data.y[2:], shuffled))  # class 0: one image
    assert (seen[-1] == 0).all()
    norms, mean = lengths[-1]
    assert mean < 0.9
    np.testing.assert_allclose(norms, mean, rtol=1e-6)
