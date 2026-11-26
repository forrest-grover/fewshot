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
(``ClassData.names``). Without names it falls back to the difference of the
modality means, which degenerates when only one class has text: that text
vector *is* the text mean, so the shift erases every class difference.

With ``ClassData.use_names``, the names of the classes with images but no description are
also their text, if they classify the images better than chance would (``names_match``). That
is decided in ``fit``, from the images it is given, so cross-validation decides it per fold
without the held-out images.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import math
import numbers
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from .embedders import Vectors

CLIP_LOGIT_SCALE = 100.0
log = logging.getLogger(__name__)

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
    # Class index -> (d,) embedding of the class's name in a template ("a photo of a {}."), for
    # classes without ``text``. They estimate the modality gap, and with ``use_names`` the names
    # of the classes with images are their text if they pass ``names_match`` (see ``with_names``).
    names: dict[int, Vectors] = field(default_factory=dict)
    use_names: bool = False

    @property
    def dim(self) -> int:
        """Embedding dimension, taken from the images or, for zero-shot data, the text."""
        if len(self.X):
            return int(self.X.shape[1])
        return int(next(iter(self.text.values())).shape[0])


def _unit(v: npt.NDArray[np.floating[Any]]) -> Vectors:
    out: Vectors = (v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)).astype(np.float32)
    return out


def temper(probs: Scores, temperature: float) -> Scores:
    """Probabilities softened (``temperature`` > 1) or sharpened (< 1): ``p ** (1/T)``, renormalised.
    For softmax outputs this is the same as dividing the logits by T, so the most probable class
    never changes; it is how a fitted temperature calibrates confidence."""
    if temperature == 1.0:
        return probs
    with np.errstate(divide="ignore"):
        z = np.log(probs) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: Scores = e / e.sum(axis=1, keepdims=True)
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


NAME_TEST_LEVEL = 0.05  # names are used as text when matching the examples this well is unlikely by chance


def names_match(guess: Labels, truth: Labels, named: npt.NDArray[np.bool_], texts: int) -> bool:
    """Whether text guesses ``guess`` of every example's class ``truth`` hit it, among the examples
    of the classes described by their names (``named``), more often than chance would (one-sided
    binomial test at ``NAME_TEST_LEVEL``). Chance is the hit rate of guesses unrelated to the
    images: one is class c as often as the guesses for all the examples are, so texts that send
    every image to the largest class score no better than chance however unbalanced the classes
    are. With every example in one class that rate can't show whether the text tells the class
    apart, so chance is then one in the number of ``texts`` (classes with text)."""
    rate = np.bincount(guess, minlength=int(max(guess.max(), truth.max())) + 1) / len(guess)
    chance = 1.0 / texts if len(set(truth.tolist())) == 1 else float(rate[truth[named]].mean())
    return binomial_tail(int((guess == truth)[named].sum()), int(named.sum()), chance) < NAME_TEST_LEVEL


def with_names(data: ClassData) -> ClassData:
    """``data`` with the names of its classes that have images but no text as their text, if
    ``use_names`` is set and those names classify the images better than chance would (see
    ``names_match``, guessing among every class with text or such a name): meaningless names
    (codes, numbers) would only pull predictions off. Otherwise ``data`` itself."""
    named = {i: v for i, v in data.names.items() if i not in data.text and (data.y == i).any()}
    if not data.use_names or not named:
        return data
    candidates = sorted(set(data.text) | set(named))
    T = np.stack([data.text[i] if i in data.text else named[i] for i in candidates])
    guess = np.array(candidates)[(data.X @ T.T).argmax(1)]
    if not names_match(guess, data.y, np.isin(data.y, list(named)), len(candidates)):
        return data
    return dataclasses.replace(data, text={**data.text, **named})


def _modality_gap(data: ClassData) -> Vectors:
    """Estimated (text - image) offset; ``data`` has images and at least one text embedding."""
    with_images = sorted(set(data.y.tolist()))
    sides = [(i, t) for i in with_images if (t := data.text.get(i, data.names.get(i))) is not None]
    if len(sides) == len(with_images):
        gap: Vectors = np.mean([t - data.X[data.y == i].mean(0) for i, t in sides], axis=0).astype(np.float32)
        return gap
    return (np.mean(list(data.text.values()), axis=0) - data.X.mean(0)).astype(np.float32)


# Queries are scored in blocks of about this many (query x support point) similarities, or (query x
# dimension) values where that is more: about 16 MB per float32 array a block makes.
PREDICT_CELLS = 4_000_000


class Classifier:  # pylint: disable=too-many-instance-attributes  # fitted state plus settings
    """Base class: handles text fusion and modality-gap anchors; subclasses score images."""

    name = "base"

    def __init__(self, text_weight: float = 1.0) -> None:
        self.text_weight = text_weight
        self.logit_scale = 1.0  # multiplies the fused logits (a classifier may fit it, for calibration)
        self.temperature = 1.0  # applied to the probabilities (see ``temper``); a task setting
        self.n_images = 0  # support points that are example images (the rest are text anchors)
        self.classes: list[str] = []
        self.zero_shot = False
        self.Pt: Vectors | None = None  # (C, d) text-side vector per class, if any text is used
        # Whether each class has its own text (the others' ``Pt`` rows are image centroids moved
        # across the gap ``gap``), when ``Pt`` is set.
        self.described: npt.NDArray[np.bool_] | None = None
        self.gap: Vectors | None = None
        self._X: Vectors | None = None  # image-space support points (examples + text anchors)
        self._y: Labels | None = None  # class of each support point
        self.names_used = False  # whether the last fit used class names as text (see ``with_names``)

    def _reset(self) -> None:
        """Forget a previous fit's state, so refitting the same object matches a fresh one."""
        self.logit_scale, self.n_images, self.names_used = 1.0, 0, False
        self.Pt = self.described = self.gap = self._X = self._y = None

    # Subclasses implement these on image-space data where every class has >= 1 point.
    def _fit_image(self, X: Vectors, y: Labels, w: Vectors) -> None:
        raise NotImplementedError

    # ``S`` is the queries' similarity to every support point (``Q @ X.T``) when the caller has it,
    # so a classifier that needs it in both hooks computes it once.
    needs_support = True  # whether the hooks use ``S``; if not, it isn't computed for them

    def _image_logprobs(self, Q: Vectors, S: Scores | None = None) -> Scores:
        raise NotImplementedError

    def _support_similarity(self, Q: Vectors) -> Scores:
        return Q @ _fitted(self._X).T

    def _class_similarity_image(self, Q: Vectors, S: Scores | None = None) -> Scores:
        """(m, C) cosine similarity of each query to each class's nearest support point."""
        S = self._support_similarity(Q) if S is None else S
        y = _fitted(self._y)
        return np.stack([S[:, y == c].max(axis=1) for c in range(len(self.classes))], axis=1)

    def class_similarity(self, Q: Vectors, S: Scores | None = None) -> Scores:
        """(m, C) similarity of each query to each class: to its nearest example (its prototype, for
        the prototype classifier), or to its text for zero-shot tasks. Unlike the probabilities,
        each class's score doesn't depend on the others, so several classes can pass a threshold
        (multi-label prediction)."""
        if self.zero_shot:
            return Q @ _fitted(self.Pt).T
        return self._class_similarity_image(Q, S)

    def fit(self, data: ClassData) -> Classifier:
        """Fit on ``data``; every class needs at least one image or a text embedding."""
        C = len(data.classes)
        if C < 1:
            raise ValueError("Need at least 1 class with examples or a description")
        with_images = set(data.y.tolist())
        missing = [c for i, c in enumerate(data.classes) if i not in with_images and i not in data.text]
        if missing:
            raise ValueError(f"Classes with no images and no text description: {missing}")
        self._reset()
        described = len(data.text)
        data = with_names(data)
        self.names_used = len(data.text) > described
        self.classes = list(data.classes)
        self.zero_shot = len(data.X) == 0
        gap = None if self.zero_shot or not data.text else _modality_gap(data)

        if data.text and (self.zero_shot or self.text_weight > 0):
            Pt = np.zeros((C, data.dim), np.float32)
            for i in range(C):
                if i in data.text:
                    Pt[i] = data.text[i]
                else:  # image-only class: centroid moved into text space
                    Pt[i] = _unit(data.X[data.y == i].mean(0) + _fitted(gap))
            self.Pt = _unit(Pt)
            self.described = np.array([i in data.text for i in range(C)])
            self.gap = gap
        if self.zero_shot:
            return self

        X, y, w = data.X, data.y, np.ones(len(data.X), np.float32)
        text_only = [i for i in data.text if i not in with_images]
        if text_only:
            T = _unit(np.stack([data.text[i] - _fitted(gap) for i in text_only]))
            X = np.concatenate([X, T])
            y = np.concatenate([y, text_only])
            w = np.concatenate([w, np.ones(len(T), np.float32)])
        self._X, self._y = X, y
        self.n_images = len(data.X)
        self._fit_image(X, y, w)
        return self

    def nearest(self, Q: Vectors, k: int) -> tuple[npt.NDArray[np.int_], Scores]:
        """The ``k`` example images most similar to each query, as (row indices into the fitted
        ``ClassData.X``, cosine similarities), most similar first. Evidence to show beside a
        prediction; for the prototype classifier the decision itself uses class means."""
        if self.zero_shot or self.n_images == 0:
            return np.zeros((len(Q), 0), int), np.zeros((len(Q), 0))
        S = Q @ _fitted(self._X)[: self.n_images].T
        k = min(k, self.n_images)
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, top, axis=1), axis=1, kind="stable")
        idx = np.take_along_axis(top, order, axis=1)
        return idx, np.take_along_axis(S, idx, axis=1)

    def predict(self, Q: Vectors) -> tuple[Scores, Scores]:
        """``predict_full`` without the per-class similarities."""
        probs, sims, _ = self.predict_full(Q)
        return probs, sims

    def predict_full(self, Q: Vectors) -> tuple[Scores, Scores, Scores]:
        """Return (probs (m, C), similarity of each query to its predicted class (m,), and its
        similarity to every class (m, C), see ``class_similarity``).

        The similarity is that of the most probable class (see ``class_similarity``), not of the
        nearest one, which can differ when descriptions tip the scores: a reject threshold on it
        then never accepts a class the image is less similar to than the threshold. It is
        image-to-image cosine, except for zero-shot tasks where it is image-to-text cosine
        (typically 0.2-0.35 for CLIP), so reject thresholds are not interchangeable between the
        two. Raises RuntimeError if called before ``fit``.

        Queries are scored in blocks of ``PREDICT_CELLS`` / (support points, classes or dimensions,
        whichever is most) at a time, so the (queries x examples) similarity matrix and the other
        per-block arrays stay small however many of both there are.
        """
        width = max(len(self._X) if self._X is not None else 0, len(self.classes), Q.shape[1])
        rows = max(1, PREDICT_CELLS // width)
        if len(Q) > rows:
            parts = [self._predict_rows(Q[i : i + rows]) for i in range(0, len(Q), rows)]
            return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                    np.concatenate([p[2] for p in parts]))
        return self._predict_rows(Q)

    def _predict_rows(self, Q: Vectors) -> tuple[Scores, Scores, Scores]:
        S = self._support_similarity(Q) if self.needs_support and not self.zero_shot else None
        per_class = self.class_similarity(Q, S)
        if self.zero_shot:
            probs = _softmax(CLIP_LOGIT_SCALE * per_class)
        elif len(self.classes) == 1:
            probs = np.ones((len(Q), 1))
        else:
            logits = self._image_logprobs(Q, S)
            if self.Pt is not None and self.text_weight > 0:
                logits = logits + self.text_weight * CLIP_LOGIT_SCALE * (Q @ self.Pt.T)
            probs = _softmax(self.logit_scale * logits)
        probs = temper(probs, self.temperature)
        return probs, per_class[np.arange(len(Q)), probs.argmax(axis=1)], per_class


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

    needs_support = False  # it compares with class means instead

    def _image_logprobs(self, Q: Vectors, S: Scores | None = None) -> Scores:
        z = self.scale * (Q @ _fitted(self.P).T)
        z = z - z.max(1, keepdims=True)
        return z - np.log(np.exp(z).sum(1, keepdims=True))

    def _class_similarity_image(self, Q: Vectors, S: Scores | None = None) -> Scores:
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

    def _image_logprobs(self, Q: Vectors, S: Scores | None = None) -> Scores:
        S = self._support_similarity(Q) if S is None else S
        y, w = _fitted(self.y), _fitted(self.w)
        k = min(self.k, S.shape[1])
        top = top_k(S, k)
        scores = np.full((len(Q), len(self.classes)), 1e-6, np.float64)
        top_s = np.take_along_axis(S, top, axis=1)
        # Votes are relative to each query's nearest neighbour, so the 1e-6 floor for classes with no
        # neighbour in the top k stays negligible however low the absolute similarities are.
        votes = w[top] * np.exp(self.scale * (top_s - top_s.max(axis=1, keepdims=True)))
        rows = np.repeat(np.arange(len(Q)), k)
        np.add.at(scores, (rows, y[top.ravel()]), votes.ravel())
        return np.log(scores / scores.sum(axis=1, keepdims=True))


def top_k(S: Scores, k: int) -> npt.NDArray[np.int_]:
    """(m, k) columns of the ``k`` largest entries of each row of ``S`` (1 <= k <= columns), in
    column order. Ties go to the lower column, so the choice among tied columns doesn't change
    when another column is taken out: leave-one-out in closed form then matches refitting."""
    kth = np.partition(S, S.shape[1] - k, axis=1)[:, S.shape[1] - k, None]
    above, tied = kth < S, kth == S
    need = k - above.sum(1)
    crowded = np.flatnonzero(tied.sum(1) > need)  # rows where more columns tie than fit
    tied[crowded] &= np.cumsum(tied[crowded], axis=1) <= need[crowded, None]
    out: npt.NDArray[np.int_] = np.nonzero(above | tied)[1].reshape(len(S), k)
    return out


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

    def _image_logprobs(self, Q: Vectors, S: Scores | None = None) -> Scores:
        model = _fitted(self.model)
        lp = np.full((len(Q), len(self.classes)), -1e9, np.float64)
        lp[:, model.classes_] = model.predict_log_proba(Q)
        return lp


GDA_ALPHAS = np.geomspace(0.01, 100, 41)  # image-term weights tried, relative to the text term
TEXT_PRIOR = 1.5  # multiplies the empirical-Bayes weight of a class's text in its mean
PRIOR_TEST_LEVEL = 0.05  # the text pulls the means only if its anchors match them this unlikely by chance
ONE_SHOT_SPREAD = 0.1  # with 1 image per class: share of the image-to-text distance put down to noise


class GDAClassifier(Classifier):
    """Gaussian discriminant analysis with a shared, shrunk covariance (LDA), on features centred
    on the mean of the class means and re-normalised; with text, fused with zero-shot scores by a
    weight chosen per task.

    The covariance is shrunk towards ``nu * I`` by ``D / (D + dof)`` (dof = images minus
    classes): with one image per class it is a nearest-class-mean classifier, and with many it
    uses how the classes actually spread.

    With text, each described class's mean is also pulled towards its text moved across the
    modality gap (its anchor), as if the anchor were lam more images: a Gaussian prior on the
    mean, with lam = ``TEXT_PRIOR`` * sigma^2 / tau^2 by empirical Bayes (sigma^2: the images'
    spread around their class mean; tau^2: the true means' spread around their anchors, i.e. the
    class means' distance from them less the share their own noise explains). A few images are
    a noisy estimate of where a class lies and the text a biased but steady one, so this helps
    most at 2-8 images per class (about +1 point), on top of fusing the text scores below.

    The image term's weight against the text term is chosen
    by leave-one-out cross-entropy on the examples, pulled (in log space) towards
    ``0.2 * k**0.6`` for k images per class, the more so the fewer images there are; a class
    whose only image is left out is represented by its text moved across the modality gap.
    Without text, the same leave-one-out picks the scores' scale (so confidences are sensible).
    """

    name = "gda"

    def __init__(self, text_weight: float = 1.0) -> None:
        super().__init__(text_weight)
        self.center: Vectors | None = None
        self.W: Vectors | None = None
        self.b: Vectors | None = None
        self.weight = 1.0  # multiplies the discriminant scores

    def _centred(self, Q: Vectors) -> Vectors:
        return _unit(Q - _fitted(self.center))

    def _image_logprobs(self, Q: Vectors, S: Scores | None = None) -> Scores:
        out: Scores = self.weight * (self._centred(Q) @ _fitted(self.W).T + _fitted(self.b))
        return out

    def _fit_image(self, X: Vectors, y: Labels, w: Vectors) -> None:
        C = len(self.classes)
        n = np.bincount(y, minlength=C).astype(np.float64)
        raw_mu = np.zeros((C, X.shape[1]))
        np.add.at(raw_mu, y, X.astype(np.float64))
        raw_mu /= n[:, None]
        self.center = raw_mu.mean(0).astype(np.float32)
        Z = self._centred(X).astype(np.float64)
        mu = np.zeros_like(raw_mu)
        np.add.at(mu, y, Z)
        mu /= n[:, None]
        m = self.n_images  # the rows after these are text anchors of classes without images
        prec, nu, downdate = _shared_precision(Z, y, mu, n, m)
        loo = _GDALeaveOneOut(Z[:m], y[:m], mu, n, prec, downdate)
        if self.Pt is None or self.text_weight <= 0:
            self._set_means(loo.mu, prec)
            self.weight = loo.scale(_fitted(self.W), _fitted(self.b), default=15.0 * nu)
        else:
            self._fit_with_text(X[:m], raw_mu, loo, nu)

    def _fit_with_text(self, X: Vectors, raw_mu: Scores, loo: _GDALeaveOneOut, nu: float) -> None:
        """The class means pulled towards their text, and the text fusion weight and scale."""
        y = loo.y
        imgs = np.bincount(y, minlength=len(self.classes))
        # Each class's text moved across the gap into this space: where the text says its images are.
        gap = (_fitted(self.Pt)[imgs > 0] - raw_mu[imgs > 0]).mean(0)
        anchors = self._centred((_fitted(self.Pt) - gap).astype(np.float32)).astype(np.float64)
        # text_weight scales the text's pull on the means as it does its scores, so 0 is no text.
        lam = self.text_weight * self._prior_strength(loo.mu, anchors, imgs, nu, loo.downdate is not None)
        loo.pull_towards(anchors, lam)
        self._set_means(loo.mu, loo.prec)
        stand_in = anchors.copy()
        if (lam > 0).any():  # a class whose only image is left out is represented by its anchor
            # alone, shortened to the class means' length: a longer mean would be penalised for it
            stand_in[lam > 0] *= float(np.linalg.norm(loo.mu[lam > 0], axis=1).mean())
        zt, keep = self._text_scores(X, y, raw_mu, imgs)
        W, b = _fitted(self.W), _fitted(self.b)
        self.weight, self.logit_scale = _fusion(
            zt, loo.scores(W, b, stand_in, exact=False), lambda: loo.scores(W, b, stand_in),
            y, float(np.median(imgs[imgs > 0])), self.text_weight, keep)

    def _set_means(self, mu: Scores, prec: Scores) -> None:
        """The discriminant's weights and offsets for class means ``mu``."""
        W = mu @ prec
        self.W, self.b = W.astype(np.float32), (-0.5 * (W * mu).sum(1)).astype(np.float32)

    def _prior_strength(self, mu: Scores, anchors: Scores, imgs: Labels, nu: float,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                        spread_known: bool) -> Scores:
        """How many images' worth of weight each class's text anchor gets in its mean (see the
        class doc): ``TEXT_PRIOR`` times sigma^2 / tau^2, the images' spread around their class
        mean over the class means' spread around their anchors (per dimension, estimated from
        all the classes), or 0 for a class without both images and its own text.

        Or 0 for every class unless the anchors pick out their own class means (each mean's
        nearest anchor is its own) more often than chance would, by a one-sided binomial test at
        ``PRIOR_TEST_LEVEL``. The estimate can't tell text unrelated to the images from steady
        text when the images scatter widely: shuffling the descriptions gave lam about 1 and cut
        accuracy at 2 images per class from 45% (no pull) to 30%."""
        lam = np.zeros(len(imgs))
        on = _fitted(self.described) & (imgs > 0)
        if not on.any():
            return lam
        dist = (mu[on] ** 2).sum(1)[:, None] + (anchors[on] ** 2).sum(1)[None, :] - 2 * mu[on] @ anchors[on].T
        hits = int((dist.argmin(1) == np.arange(len(dist))).sum())
        if binomial_tail(hits, len(dist), 1.0 / len(dist)) >= PRIOR_TEST_LEVEL:
            return lam
        d2 = float(np.diag(dist).mean()) / mu.shape[1]
        s2 = nu if spread_known else ONE_SHOT_SPREAD * d2  # no spread to measure with 1 image each
        k = float(np.median(imgs[imgs > 0]))
        t2 = max(d2 - s2 / k, 1e-3 * s2, 1e-12)  # a mean of k images is off by s2 / k itself
        lam[on] = TEXT_PRIOR * s2 / t2
        return lam

    def _text_scores(self, X: Vectors, y: Labels, raw_mu: Scores, imgs: Labels
                     ) -> tuple[Scores, npt.NDArray[np.bool_]]:
        """Each example's text scores, and whether it can help choose the fusion weight. An
        image-only class's text side is its own centroid moved across the gap, so an example of one
        is scored against the centroid without it; one alone in its class has nothing left to
        stand for it (on either side) and is left out."""
        zt = CLIP_LOGIT_SCALE * (X @ _fitted(self.Pt).T).astype(np.float64)
        described = _fitted(self.described)[y]
        keep = described | (imgs[y] > 1)
        own = np.flatnonzero(~described & keep)
        if len(own):
            c, n = y[own], imgs[y[own], None]
            centroid = (n * raw_mu[c] - X[own]) / (n - 1)
            zt[own, c] = CLIP_LOGIT_SCALE * (X[own] * _unit(centroid + _fitted(self.gap))).sum(1)
        return zt, keep


def _fusion(zt: Scores, loo: Scores, exact: Callable[[], Scores], y: Labels, k: float,  # pylint: disable=too-many-arguments,too-many-positional-arguments
            text_weight: float, keep: npt.NDArray[np.bool_]) -> tuple[float, float]:
    """(weight of the image scores ``loo`` against the text scores ``zt``, scale of the fused
    logits): see ``GDAClassifier``. ``exact()`` gives the image scores with each example also left
    out of the covariance, for the scale. The weight is chosen for text at weight 1; the fused
    logits (and so the scale) have the text scores times ``text_weight``, which therefore
    multiplies the text's share against the chosen balance. Only the rows in ``keep`` are used to
    choose; all set the two kinds of scores' relative spread."""
    rel = float(zt.std()) / max(float(loo.std()), 1e-12)
    prior = 0.2 * k**0.6
    zt, loo, y = zt[keep], loo[keep], y[keep]
    if len(y) < 2:
        return prior * rel, 1.0
    chosen = _least_nll([zt + a * rel * loo for a in GDA_ALPHAS], GDA_ALPHAS, y, what="image-term weight")
    pull = 1.0 / (1.0 + (k - 1) / 2)  # all prior at 1 image per class, less with more
    weight = float(np.exp((1 - pull) * np.log(chosen) + pull * np.log(prior))) * rel
    fused = text_weight * zt + weight * exact()[keep]
    scales = np.geomspace(0.1, 10, 41)
    return weight, _least_nll([s * fused for s in scales], scales, y, what="fused logit scale")


def _least_nll(logits: list[Scores], values: Scores, y: Labels, smooth: bool = False,
               what: str = "value") -> float:
    """The value whose logits give the true classes ``y`` the least mean cross-entropy. With
    ``smooth``, against targets smoothed by 1/(n+2) (as for a task's temperature): examples that
    all came out right don't prove the scores are never wrong, and fitting them as certain would
    push the scale, and every confidence, to the limit. A choice at either end of the grid may be
    clamped; it is logged at debug level as ``what``."""
    rows = np.arange(len(y))
    if smooth:
        eps = 1.0 / (len(y) + 2)
        C = logits[0].shape[1]
        Y = np.full(logits[0].shape, eps / max(C - 1, 1))  # (one class: nothing to spread it over)
        Y[rows, y] = 1.0 - eps
        best = int(np.argmin([-(Y * _log_softmax(z)).sum(1).mean() for z in logits]))
    else:
        best = int(np.argmin([-_log_softmax(z)[rows, y].mean() for z in logits]))
    log_clamped(what, values, best)
    return float(values[best])


def log_clamped(what: str, grid: Scores, best: int) -> None:
    """Log at debug level when a grid search chose an end of its grid ``grid``: the best value may
    lie beyond it."""
    if best in (0, len(grid) - 1):
        log.debug("%s chose %g, the %s end of its grid", what, float(grid[best]), "low" if best == 0 else "high")


def _shared_precision(Z: Scores, y: Labels, mu: Scores, n: Scores, m: int  # pylint: disable=too-many-arguments,too-many-positional-arguments
                      ) -> tuple[Scores, float, tuple[Scores, Scores] | None]:
    """The inverse of the shrunk pooled covariance of ``Z`` around class means ``mu``, the
    shrinkage target's scale nu, and the leave-one-out downdate (see ``_GDALeaveOneOut``) for the
    first ``m`` rows, or None when there is no spread to estimate (then the covariance is nu*I)."""
    D, dof = Z.shape[1], len(Z) - len(mu)
    if dof <= 1:  # isotropic, i.e. nearest class mean
        return np.eye(D) * D, 1.0 / D, None
    scatter = (Z - mu[y]).T @ (Z - mu[y])
    nu = max(float(np.trace(scatter)) / dof / D, 1e-8)  # identical examples: no spread
    g = D / (D + dof)
    prec = np.linalg.inv((1 - g) / dof * scatter + g * nu * np.eye(D))
    # Leaving one example out also takes it out of the covariance: with one degree of freedom
    # fewer, the covariance is B - u u^T for this B and each example's u.
    b_inv = np.linalg.inv((1 - g) / (dof - 1) * scatter + g * nu * np.eye(D))
    ny = n[y[:m], None]
    u = np.sqrt((1 - g) / (dof - 1) * ny / np.maximum(ny - 1, 1)) * (Z[:m] - mu[y[:m]])
    return prec, nu, (b_inv, u)


class _GDALeaveOneOut:  # pylint: disable=too-many-instance-attributes  # a helper holding fit-time arrays
    """Leave-one-out discriminant scores of a GDA fit's examples: each with itself taken out of its
    class mean and (given ``downdate``) of the shared covariance, by a rank-one update.

    ``downdate`` is (B^-1, U): example i's left-out covariance is B - u_i u_i^T, whose inverse is
    B^-1 + B^-1 u u^T B^-1 / (1 - u^T B^-1 u) (Sherman-Morrison). Without it the covariance's
    inverse is ``prec`` for every example. The shrinkage amount and target are kept.
    """

    def __init__(self, Z: Scores, y: Labels, mu: Scores, n: Scores, prec: Scores,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                 downdate: tuple[Scores, Scores] | None = None) -> None:
        """``mu`` are the class means of ``Z`` (over ``n`` rows each): the fit's means ``self.mu``
        unless ``pull_towards`` moves them."""
        self.Z, self.y, self.n, self.prec, self.downdate = Z, y, n, prec, downdate
        self.sample_mu = self.mu = mu
        self.lam: Scores = np.zeros(len(n))
        self.pull = np.zeros_like(mu)

    def pull_towards(self, anchors: Scores, lam: Scores) -> None:
        """Pull class c's mean to ``anchors[c]`` as if that were ``lam[c]`` more examples: the
        fit's means become (n mu + lam anchor) / (n + lam), and so do the left-out ones."""
        self.lam, self.pull = lam, lam[:, None] * anchors
        self.mu = (self.n[:, None] * self.sample_mu + self.pull) / (self.n + lam)[:, None]

    def scores(self, W: Vectors, b: Vectors, anchors: Scores | None = None, exact: bool = True) -> Scores:  # pylint: disable=too-many-locals
        """(n, C) scores; an example alone in its class gets ``anchors[class]`` as the left-out
        mean, or NaN in its row without anchors. ``W`` and ``b`` are the full fit's, used when
        there is no downdate. ``exact=False`` keeps the full covariance (only the mean is left
        out), which overrates the image term a little; choosing the text fusion weight with it
        measured best (the prior on the weight was tuned together with it).

        ``exact=True`` scores are for fitting the scale, so they should be as hard to tell apart
        as a new example's. But a mean of N - 1 examples (N counting a text prior's weight) is
        further from a new member of its class than one of N: by (1 + 1/(N-1)) vs (1 + 1/N)
        times the spread, in expected squared Mahalanobis distance. So an example's distance to
        its own left-out mean is scaled by the ratio, (N+1)(N-1)/N^2 (3/4 at 2 images per
        class). Without it, the scale fit to 2 images per class came out a third to a half of the
        right one, and image-only confidence far too low (14% at 56% accuracy on DTD)."""
        Z, y, n, mu = self.Z, self.y, self.n, self.mu
        rows = np.arange(len(y))
        alone = n[y] <= 1
        left = n[y] - 1 + self.lam[y]  # own class mean without it (an anchor if nothing is left)
        M = (n[y, None] * self.sample_mu[y] - Z + self.pull[y]) / np.where(left > 0, left, 1)[:, None]
        if anchors is not None:
            M[alone] = anchors[y[alone]]
        if self.downdate is None or not exact:
            L: Scores = Z @ W.T.astype(np.float64) + b
            Wm = M @ self.prec
            L[rows, y] = (Z * Wm).sum(1) - 0.5 * (M * Wm).sum(1)
            if not exact:
                return self._mark_alone(L, anchors)
            zpz = ((Z @ self.prec) * Z).sum(1)
        else:
            P, U = self.downdate
            ZP, UP, mu_p = Z @ P, U @ P, mu @ P
            den = 1.0 - (U * UP).sum(1)
            zu = (ZP * U).sum(1)
            up_mu = UP @ mu.T  # (n, C): u_i P mu_c
            L = ZP @ mu.T + zu[:, None] * up_mu / den[:, None]
            L -= 0.5 * ((mu_p * mu).sum(1)[None, :] + up_mu**2 / den[:, None])
            MP = M @ P
            up_m = (UP * M).sum(1)
            L[rows, y] = ((ZP * M).sum(1) + zu * up_m / den) - 0.5 * ((MP * M).sum(1) + up_m**2 / den)
            zpz = (ZP * Z).sum(1) + zu**2 / den
        # A score is (zPz - distance) / 2, so scaling the distance by f makes it f L + (1 - f) zPz / 2.
        N = n[y] + self.lam[y]
        f = np.where(alone, 1.0, (N + 1) * (N - 1) / N**2)
        L[rows, y] = f * L[rows, y] + 0.5 * (1 - f) * zpz
        return self._mark_alone(L, anchors)

    def _mark_alone(self, L: Scores, anchors: Scores | None) -> Scores:
        """``L`` with NaN rows for the examples alone in their class, unless anchors stand in."""
        if anchors is None:
            L[self.n[self.y] <= 1] = np.nan
        return L

    def scale(self, W: Vectors, b: Vectors, default: float) -> float:
        """The factor on the scores with the least (smoothed) leave-one-out cross-entropy: the
        default if no example can be left out, at most the default if they all have one class."""
        L = self.scores(W, b)
        ok = ~np.isnan(L).any(1)
        if not ok.any():
            return default
        L, t = L[ok], self.y[ok]
        base = 10.0 / max(float(L.std()), 1e-12)
        fitted = _least_nll([a * base * L for a in GDA_ALPHAS], GDA_ALPHAS, t, smooth=True, what="score scale") * base
        # Examples of a single class can't show how often the others are mistaken for it, so then
        # the fit only caps the default.
        return fitted if len(set(t.tolist())) > 1 else min(fitted, default)


def binomial_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), computed in log space so large n doesn't overflow."""
    if k <= 0 or p >= 1:
        return 1.0
    if k > n:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * math.log(p)
            + (n - j) * math.log1p(-p) for j in range(k, n + 1)]
    top = max(logs)
    return math.exp(top) * sum(math.exp(v - top) for v in logs)


def _log_softmax(z: Scores) -> Scores:
    z = z - z.max(1, keepdims=True)
    out: Scores = z - np.log(np.exp(z).sum(1, keepdims=True))
    return out


CLASSIFIERS: dict[str, type[Classifier]] = {
    c.name: c for c in (PrototypeClassifier, KNNClassifier, LogRegClassifier, GDAClassifier)
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
