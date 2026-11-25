"""Few-shot classifiers over L2-normalised embeddings.

All classifiers share one input contract (``ClassData``): image embeddings with
labels, plus optional text embeddings per class (CLIP only).

Scoring is ``log p_image(c | x) + text_weight * 100 * cos(x, text_c)``. The
image term comes from the specific classifier (prototype, kNN, linear probe);
the text term is CLIP zero-shot scoring at CLIP's own logit scale. Fusing at
the logit level beats averaging text and image vectors: on Oxford Pets 5-way
1-shot it scores 97.3% vs 95.3%. See ``fewshot bench``.

CLIP text and image vectors occupy different regions (the "modality gap").
Where a class lacks one side we move the other across by the difference of
modality means: a text-only class gets a shifted text vector as its image-side
anchor; an image-only class (in a task where others have text) gets its shifted
image centroid as its text-side vector. A task with no images at all is pure
CLIP zero-shot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLIP_LOGIT_SCALE = 100.0


@dataclass
class ClassData:
    classes: list[str]
    X: np.ndarray  # (n, d) image embeddings
    y: np.ndarray  # (n,) int class indices into ``classes``
    text: dict[int, np.ndarray] = field(default_factory=dict)  # class index -> (d,) text embedding

    @property
    def dim(self) -> int:
        if len(self.X):
            return self.X.shape[1]
        return next(iter(self.text.values())).shape[0]


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)).astype(np.float32)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Classifier:
    name = "base"

    def __init__(self, text_weight: float = 1.0) -> None:
        self.text_weight = text_weight
        self.classes: list[str] = []

    # Subclasses implement these on image-space data where every class has >= 1 point.
    def _fit_image(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        raise NotImplementedError

    def _image_logprobs(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _image_similarity(self, Q: np.ndarray) -> np.ndarray:
        """Cosine similarity of each query to its nearest support point/prototype."""
        return (Q @ self._X.T).max(axis=1)

    def fit(self, data: ClassData) -> "Classifier":
        C = len(data.classes)
        if C < 2:
            raise ValueError("Need at least 2 classes")
        with_images = set(data.y.tolist())
        missing = [c for i, c in enumerate(data.classes) if i not in with_images and i not in data.text]
        if missing:
            raise ValueError(f"Classes with no images and no text description: {missing}")
        self.classes = list(data.classes)
        self.zero_shot = len(data.X) == 0
        self.Pt = None

        if data.text and (self.zero_shot or self.text_weight > 0):
            mean_t = np.mean(list(data.text.values()), axis=0)
            mean_x = data.X.mean(0) if len(data.X) else None
            Pt = np.zeros((C, data.dim), np.float32)
            for i in range(C):
                if i in data.text:
                    Pt[i] = data.text[i]
                else:  # image-only class: centroid moved into text space
                    Pt[i] = _unit(data.X[data.y == i].mean(0) - mean_x + mean_t)
            self.Pt = _unit(Pt)
        if self.zero_shot:
            return self

        X, y, w = data.X, data.y, np.ones(len(data.X), np.float32)
        text_only = [i for i in data.text if i not in with_images]
        if text_only:
            mean_x, mean_t = data.X.mean(0), np.mean(list(data.text.values()), axis=0)
            T = _unit(np.stack([data.text[i] - mean_t + mean_x for i in text_only]))
            X = np.concatenate([X, T])
            y = np.concatenate([y, text_only])
            w = np.concatenate([w, np.ones(len(T), np.float32)])
        self._X = X
        self._fit_image(X, y, w)
        return self

    def predict(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (probs (m, C), similarity of each query to its best match (m,)).

        Similarity is image-to-image cosine, except for zero-shot tasks where it
        is image-to-text cosine (typically 0.2-0.35 for CLIP), so reject
        thresholds are not interchangeable between the two.
        """
        if self.zero_shot:
            S = Q @ self.Pt.T
            return _softmax(CLIP_LOGIT_SCALE * S), S.max(axis=1)
        logits = self._image_logprobs(Q)
        if self.Pt is not None and self.text_weight > 0:
            logits = logits + self.text_weight * CLIP_LOGIT_SCALE * (Q @ self.Pt.T)
        return _softmax(logits), self._image_similarity(Q)


class PrototypeClassifier(Classifier):
    """Nearest class centroid (Prototypical Networks / SimpleShot style)."""

    name = "prototype"

    def __init__(self, text_weight: float = 1.0, scale: float = 30.0) -> None:
        super().__init__(text_weight)
        self.scale = scale

    def _fit_image(self, X, y, w):
        P = np.zeros((len(self.classes), X.shape[1]), np.float32)
        np.add.at(P, y, X * w[:, None])
        self.P = _unit(P)

    def _image_logprobs(self, Q):
        z = self.scale * (Q @ self.P.T)
        return z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)

    def _image_similarity(self, Q):
        return (Q @ self.P.T).max(axis=1)


class KNNClassifier(Classifier):
    """Similarity-weighted k-nearest-neighbour vote over individual examples."""

    name = "knn"

    def __init__(self, text_weight: float = 1.0, k: int = 5, scale: float = 30.0) -> None:
        super().__init__(text_weight)
        self.k, self.scale = k, scale

    def _fit_image(self, X, y, w):
        self.y, self.w = y, w

    def _image_logprobs(self, Q):
        S = Q @ self._X.T
        k = min(self.k, S.shape[1])
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        scores = np.full((len(Q), len(self.classes)), 1e-6)
        rows = np.repeat(np.arange(len(Q)), k)
        cols = top.ravel()
        np.add.at(scores, (rows, self.y[cols]), self.w[cols] * np.exp(self.scale * (S[rows, cols] - 1.0)))
        return np.log(scores / scores.sum(axis=1, keepdims=True))


class LogRegClassifier(Classifier):
    """Linear probe (multinomial logistic regression) on embeddings."""

    name = "logreg"

    def __init__(self, text_weight: float = 1.0, C: float = 10.0) -> None:
        super().__init__(text_weight)
        self.C = C

    def _fit_image(self, X, y, w):
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(C=self.C, max_iter=2000).fit(X, y, sample_weight=w)

    def _image_logprobs(self, Q):
        lp = np.full((len(Q), len(self.classes)), -1e9)
        lp[:, self.model.classes_] = self.model.predict_log_proba(Q)
        return lp


CLASSIFIERS: dict[str, type[Classifier]] = {
    c.name: c for c in (PrototypeClassifier, KNNClassifier, LogRegClassifier)
}


def make_classifier(name: str, **params) -> Classifier:
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier {name!r}; choose from {sorted(CLASSIFIERS)}")
    return CLASSIFIERS[name](**params)
