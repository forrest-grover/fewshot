"""Few-shot classifiers over L2-normalised embeddings.

All classifiers share one input contract (``ClassData``): image embeddings with
labels, plus optional text embeddings per class (CLIP only).

Scoring is ``log p_image(c | x) + text_weight * 100 * cos(x, text_c)``. The
image term comes from the specific classifier (prototype, kNN, linear probe);
the text term is CLIP zero-shot scoring at CLIP's own logit scale. Fusing at
the logit level beats averaging text and image vectors: on Oxford Pets 5-way
1-shot it scores 97.3% vs 95.3%. See ``fewshot bench``.

CLIP text and image vectors occupy different regions (the "modality gap").
Where a class lacks one side we move the other across by a gap vector: a
text-only class gets its text vector minus the gap as its image-side anchor; an
image-only class (in a task where others have text) gets its image centroid plus
the gap as its text-side vector. A task with no images at all is pure CLIP
zero-shot.

A single class is allowed: every image gets that class, and its similarity score is what tells
members from outsiders (one-class or anomaly tasks, with a reject threshold).

The gap is the mean of (text - image centroid) over the classes with images,
taking each class's description, or else the embedding of its name
(``ClassData.name_text``). Without names it falls back to the difference of the
modality means, which degenerates when only one class has text: that text
vector *is* the text mean, so the shift erases every class difference.
"""

from __future__ import annotations

import inspect
import numbers
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from .embedders import Vectors

CLIP_LOGIT_SCALE = 100.0

Labels = npt.NDArray[np.int_]  # class indices
Scores = npt.NDArray[np.floating[Any]]  # log-probabilities, probabilities or similarities

_T = TypeVar("_T")


@dataclass
class ClassData:
    """Training data for a classifier: labelled image embeddings plus optional text embeddings."""

    classes: list[str]
    X: Vectors  # (n, d) image embeddings
    y: Labels  # (n,) int class indices into ``classes``
    text: dict[int, Vectors] = field(default_factory=dict)  # class index -> (d,) text embedding
    # Class index -> (d,) embedding of the class name, for classes without ``text``. Only used to
    # estimate the modality gap, never to score.
    name_text: dict[int, Vectors] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        """Embedding dimension, taken from the images or, for zero-shot data, the text."""
        if len(self.X):
            return int(self.X.shape[1])
        return int(next(iter(self.text.values())).shape[0])


def _unit(v: npt.NDArray[np.floating[Any]]) -> Vectors:
    out: Vectors = (v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)).astype(np.float32)
    return out


def _softmax(z: Scores) -> Scores:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fitted(value: _T | None) -> _T:
    """``value`` if ``fit`` has set it; raises RuntimeError on an unfitted classifier."""
    if value is None:
        raise RuntimeError("Classifier is not fitted; call fit() first")
    return value


def _modality_gap(data: ClassData) -> Vectors:
    """Estimated (text - image) offset; ``data`` has images and at least one text embedding."""
    with_images = sorted(set(data.y.tolist()))
    sides = [(i, t) for i in with_images if (t := data.text.get(i, data.name_text.get(i))) is not None]
    if len(sides) == len(with_images):
        gap: Vectors = np.mean([t - data.X[data.y == i].mean(0) for i, t in sides], axis=0).astype(np.float32)
        return gap
    return (np.mean(list(data.text.values()), axis=0) - data.X.mean(0)).astype(np.float32)


class Classifier:
    """Base class: handles text fusion and modality-gap anchors; subclasses score images."""

    name = "base"

    def __init__(self, text_weight: float = 1.0) -> None:
        self.text_weight = text_weight
        self.classes: list[str] = []
        self.zero_shot = False
        self.Pt: Vectors | None = None  # (C, d) text-side vector per class, if any text is used
        self._X: Vectors | None = None  # image-space support points (examples + text anchors)
        self._y: Labels | None = None  # class of each support point

    # Subclasses implement these on image-space data where every class has >= 1 point.
    def _fit_image(self, X: Vectors, y: Labels, w: Vectors) -> None:
        raise NotImplementedError

    def _image_logprobs(self, Q: Vectors) -> Scores:
        raise NotImplementedError

    def _class_similarity_image(self, Q: Vectors) -> Scores:
        """(m, C) cosine similarity of each query to each class's nearest support point."""
        S = Q @ _fitted(self._X).T
        y = _fitted(self._y)
        return np.stack([S[:, y == c].max(axis=1) for c in range(len(self.classes))], axis=1)

    def class_similarity(self, Q: Vectors) -> Scores:
        """(m, C) similarity of each query to each class: to its nearest example (its prototype, for
        the prototype classifier), or to its text for zero-shot tasks. Unlike the probabilities,
        each class's score doesn't depend on the others, so several classes can pass a threshold
        (multi-label prediction)."""
        if self.zero_shot:
            return Q @ _fitted(self.Pt).T
        return self._class_similarity_image(Q)

    def fit(self, data: ClassData) -> Classifier:
        """Fit on ``data``; every class needs at least one image or a text embedding."""
        C = len(data.classes)
        if C < 1:
            raise ValueError("Need at least 1 class with examples or a description")
        with_images = set(data.y.tolist())
        missing = [c for i, c in enumerate(data.classes) if i not in with_images and i not in data.text]
        if missing:
            raise ValueError(f"Classes with no images and no text description: {missing}")
        self.classes = list(data.classes)
        self.zero_shot = len(data.X) == 0
        self.Pt = None

        if data.text and (self.zero_shot or self.text_weight > 0):
            gap = None if self.zero_shot else _modality_gap(data)
            Pt = np.zeros((C, data.dim), np.float32)
            for i in range(C):
                if i in data.text:
                    Pt[i] = data.text[i]
                else:  # image-only class: centroid moved into text space
                    Pt[i] = _unit(data.X[data.y == i].mean(0) + _fitted(gap))
            self.Pt = _unit(Pt)
        if self.zero_shot:
            return self

        X, y, w = data.X, data.y, np.ones(len(data.X), np.float32)
        text_only = [i for i in data.text if i not in with_images]
        if text_only:
            gap = _modality_gap(data)
            T = _unit(np.stack([data.text[i] - gap for i in text_only]))
            X = np.concatenate([X, T])
            y = np.concatenate([y, text_only])
            w = np.concatenate([w, np.ones(len(T), np.float32)])
        self._X, self._y = X, y
        self._fit_image(X, y, w)
        return self

    def predict(self, Q: Vectors) -> tuple[Scores, Scores]:
        """Return (probs (m, C), similarity of each query to its best match (m,)).

        Similarity is image-to-image cosine, except for zero-shot tasks where it
        is image-to-text cosine (typically 0.2-0.35 for CLIP), so reject
        thresholds are not interchangeable between the two. Raises RuntimeError
        if called before ``fit``.
        """
        sims = self.class_similarity(Q).max(axis=1)
        if self.zero_shot:
            return _softmax(CLIP_LOGIT_SCALE * (Q @ _fitted(self.Pt).T)), sims
        if len(self.classes) == 1:
            return np.ones((len(Q), 1)), sims
        logits = self._image_logprobs(Q)
        if self.Pt is not None and self.text_weight > 0:
            logits = logits + self.text_weight * CLIP_LOGIT_SCALE * (Q @ self.Pt.T)
        return _softmax(logits), sims


class PrototypeClassifier(Classifier):
    """Nearest class centroid (Prototypical Networks / SimpleShot style)."""

    name = "prototype"

    def __init__(self, text_weight: float = 1.0, scale: float = 30.0) -> None:
        super().__init__(text_weight)
        self.scale = scale
        self.P: Vectors | None = None

    def _fit_image(self, X: Vectors, y: Labels, w: Vectors) -> None:
        P = np.zeros((len(self.classes), X.shape[1]), np.float32)
        np.add.at(P, y, X * w[:, None])
        self.P = _unit(P)

    def _image_logprobs(self, Q: Vectors) -> Scores:
        z = self.scale * (Q @ _fitted(self.P).T)
        z = z - z.max(1, keepdims=True)
        return z - np.log(np.exp(z).sum(1, keepdims=True))

    def _class_similarity_image(self, Q: Vectors) -> Scores:
        return Q @ _fitted(self.P).T


class KNNClassifier(Classifier):
    """Similarity-weighted k-nearest-neighbour vote over individual examples."""

    name = "knn"

    def __init__(self, text_weight: float = 1.0, k: int = 5, scale: float = 30.0) -> None:
        super().__init__(text_weight)
        self.k, self.scale = k, scale
        self.y: Labels | None = None
        self.w: Vectors | None = None

    def _fit_image(self, X: Vectors, y: Labels, w: Vectors) -> None:
        self.y, self.w = y, w

    def _image_logprobs(self, Q: Vectors) -> Scores:
        S = Q @ _fitted(self._X).T
        y, w = _fitted(self.y), _fitted(self.w)
        k = min(self.k, S.shape[1])
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        scores = np.full((len(Q), len(self.classes)), 1e-6, np.float64)
        top_s = np.take_along_axis(S, top, axis=1)
        # Votes are relative to each query's nearest neighbour, so the 1e-6 floor for classes with no
        # neighbour in the top k stays negligible however low the absolute similarities are.
        votes = w[top] * np.exp(self.scale * (top_s - top_s.max(axis=1, keepdims=True)))
        rows = np.repeat(np.arange(len(Q)), k)
        np.add.at(scores, (rows, y[top.ravel()]), votes.ravel())
        return np.log(scores / scores.sum(axis=1, keepdims=True))


class LogRegClassifier(Classifier):
    """Linear probe (multinomial logistic regression) on embeddings."""

    name = "logreg"

    def __init__(self, text_weight: float = 1.0, C: float = 10.0) -> None:
        super().__init__(text_weight)
        self.C = C
        self.model: Any = None  # sklearn LogisticRegression once fitted (sklearn is untyped)

    def _fit_image(self, X: Vectors, y: Labels, w: Vectors) -> None:
        # Lazy import: sklearn is slow to import and only this classifier needs it.
        from sklearn.linear_model import LogisticRegression  # pylint: disable=import-outside-toplevel

        if len(self.classes) == 1:  # nothing to separate; predict() doesn't use the model
            return
        self.model = LogisticRegression(C=self.C, max_iter=2000).fit(X, y, sample_weight=w)

    def _image_logprobs(self, Q: Vectors) -> Scores:
        model = _fitted(self.model)
        lp = np.full((len(Q), len(self.classes)), -1e9, np.float64)
        lp[:, model.classes_] = model.predict_log_proba(Q)
        return lp


CLASSIFIERS: dict[str, type[Classifier]] = {
    c.name: c for c in (PrototypeClassifier, KNNClassifier, LogRegClassifier)
}


# Upper bounds keep the scores finite: logits are at most about scale + text_weight * 100.
PARAM_MAX = {"k": 100_000, "scale": 1000.0, "C": 1e6, "text_weight": 100.0}
POSITIVE_PARAMS = {"k", "C"}


def check_params(name: str, params: dict[str, Any]) -> None:
    """Raise ``ValueError`` unless ``params`` are valid keyword arguments for classifier ``name``.

    Every parameter is a number between 0 and ``PARAM_MAX``; ``k`` must be a positive integer and
    ``C`` positive. Python and NumPy numbers are accepted; bools are not.
    """
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier {name!r}; choose from {sorted(CLASSIFIERS)}")
    defaults = {
        p.name: p.default for p in inspect.signature(CLASSIFIERS[name].__init__).parameters.values()
        if p.default is not inspect.Parameter.empty
    }
    for key, value in params.items():
        if key not in defaults:
            raise ValueError(f"Unknown parameter {key!r} for {name}; choose from {sorted(defaults)}")
        want_int = isinstance(defaults[key], int)
        kind = numbers.Integral if want_int else numbers.Real
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, kind):
            raise ValueError(f"{name} parameter {key!r} must be {'an integer' if want_int else 'a number'}, "
                             f"got {value!r}")
        low = 1 if want_int else 0.0
        high = PARAM_MAX.get(key, 1e6)
        try:
            v = float(value)
            ok = low <= v <= high and (key not in POSITIVE_PARAMS or v > 0)
        except OverflowError:  # an int too large for a float
            ok = False
        if not ok:  # also catches NaN, which fails every comparison
            bound = "> 0" if key in POSITIVE_PARAMS else f">= {low}"
            raise ValueError(f"{name} parameter {key!r} must be {bound} and <= {high:g}, got {value!r}")


def make_classifier(name: str, **params: Any) -> Classifier:
    """Instantiate the classifier registered as ``name`` with keyword ``params``.

    Raises ``ValueError`` for an unknown classifier or invalid parameters.
    """
    check_params(name, params)
    return CLASSIFIERS[name](**params)
