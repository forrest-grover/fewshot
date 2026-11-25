"""Unit tests for fewshot.classifiers on small hand-built embeddings."""

import numpy as np
import pytest

from fewshot.classifiers import (
    CLASSIFIERS,
    ClassData,
    Classifier,
    KNNClassifier,
    LogRegClassifier,
    PrototypeClassifier,
    make_classifier,
)

ALL = ["prototype", "knn", "logreg"]


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
    assert isinstance(clf, KNNClassifier) and clf.k == 2 and clf.text_weight == 0.5
    assert isinstance(make_classifier("prototype"), PrototypeClassifier)
    assert make_classifier("logreg", C=1.0).C == 1.0


def test_unknown_classifier_lists_choices():
    """An unknown name raises and lists the valid choices."""
    with pytest.raises(ValueError, match=r"Unknown classifier 'svm'; choose from \['knn', 'logreg', 'prototype'\]"):
        make_classifier("svm")


def test_dim_from_images_or_text():
    """dim comes from X, or from the text vectors when there are no images."""
    assert two_clusters().dim == 3
    zero_shot = ClassData(["a", "b"], np.zeros((0, 0), np.float32), np.zeros(0, int), {0: np.ones(7), 1: np.ones(7)})
    assert zero_shot.dim == 7


@pytest.mark.parametrize("name", ALL)
def test_needs_two_classes(name):
    """A single-class problem is rejected."""
    data = ClassData(["only"], unit([[1, 0, 0]]), np.array([0]))
    with pytest.raises(ValueError, match="at least 2 classes"):
        make_classifier(name).fit(data)


@pytest.mark.parametrize("name", ALL)
def test_separable_clusters(name):
    """Each classifier separates two clusters with normalised probabilities."""
    clf = make_classifier(name).fit(two_clusters())
    assert clf.classes == ["x", "y"] and not clf.zero_shot and clf.Pt is None
    probs, sims = clf.predict(unit([[1, 0.05, 0], [0, 1, 0.05]]))
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(1), 1, rtol=1e-6)
    assert probs.argmax(1).tolist() == [0, 1]
    assert np.all(sims > 0.98) and np.all(sims <= 1 + 1e-6)


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
    assert clf.Pt.shape == (3, 4)
    np.testing.assert_allclose(clf.Pt[0], data.text[0], atol=1e-6)
    # 'b' centroid (y axis) moved from image space into text space: mostly y plus the gap direction.
    assert clf.Pt[1].argmax() in (1, 3) and clf.Pt[1][1] > 0.25 and clf.Pt[1][3] > 0.5
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
