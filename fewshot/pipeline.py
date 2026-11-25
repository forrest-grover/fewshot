"""Ties a task's stored examples to an embedder, the cache and a classifier."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from PIL import Image

from .cache import EmbeddingCache
from .classifiers import Classifier, ClassData, make_classifier
from .embedders import get_embedder, spec_supports_text
from .store import Task, Workspace, image_id, load_image

ImageInput = Union[bytes, str, Path, Image.Image]


@dataclass
class Prediction:
    """Classification result for one image."""

    label: str | None  # None when rejected as unknown
    top_label: str
    confidence: float
    similarity: float
    rejected: bool
    scores: dict[str, float]

    def to_dict(self) -> dict:
        """JSON-ready dict of the fields."""
        return asdict(self)


class Pipeline:
    """Embeds and classifies images for tasks in one workspace.

    Fitted classifiers are cached per task revision, so repeated predictions
    only re-fit after the task changes.
    """

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace
        self.cache = EmbeddingCache(workspace.cache_path)
        # task name -> (task object, revision, classifier). The object identity check stops a
        # deleted-and-recreated task that reaches the same revision from reusing a stale fit.
        self._fitted: dict[str, tuple[Task, int, Classifier]] = {}
        self._lock = threading.Lock()

    # ---- embeddings --------------------------------------------------
    def embed_images(self, spec: str, items: Sequence[tuple[str | None, Image.Image | bytes | Path]]) -> np.ndarray:
        """Embed ``(cache_key or None, image)`` pairs, using and filling the cache."""
        keys = [k for k, _ in items if k]
        hit = self.cache.get_many(spec, keys) if keys else {}
        todo = [(i, k, im) for i, (k, im) in enumerate(items) if not (k and k in hit)]
        out: list[np.ndarray | None] = [hit.get(k) if k else None for k, _ in items]
        if todo:
            emb = get_embedder(spec)
            ims = [_to_pil(im) for _, _, im in todo]
            vecs = emb.embed_images(ims)
            new = {}
            for (i, k, _), v in zip(todo, vecs):
                out[i] = v
                if k:
                    new[k] = v
            self.cache.put_many(spec, new)
        if not items:
            return np.zeros((0, 0), np.float32)
        return np.stack(out).astype(np.float32)

    def embed_texts(self, spec: str, texts: Sequence[str]) -> np.ndarray:
        """Embed strings (rows in input order), using and filling the cache."""
        keys = ["text:" + t for t in texts]
        hit = self.cache.get_many(spec, keys)
        missing = [t for t in dict.fromkeys(texts) if "text:" + t not in hit]
        if missing:
            vecs = get_embedder(spec).embed_texts(missing)
            new = {"text:" + t: v for t, v in zip(missing, vecs)}
            self.cache.put_many(spec, new)
            hit.update(new)
        return np.stack([hit[k] for k in keys]) if texts else np.zeros((0, 0), np.float32)

    def example_embeddings(self, task: Task) -> np.ndarray:
        """Embeddings of the task's examples, in ``task.config.examples`` order."""
        exs = task.config.examples
        return self.embed_images(task.config.embedder, [(e.id, task.path / e.file) for e in exs])

    # ---- training data -----------------------------------------------
    def class_data(self, task: Task, exclude: set[int] | None = None) -> ClassData:
        """Training data for ``task``, leaving out example indices in ``exclude``."""
        cfg = task.config
        classes = task.class_names()
        idx = {c: i for i, c in enumerate(classes)}
        X = self.example_embeddings(task)
        y = np.array([idx[e.label] for e in cfg.examples], int)
        if exclude:
            keep = np.array([i not in exclude for i in range(len(y))], bool)
            X, y = X[keep], y[keep]
        text: dict[int, np.ndarray] = {}
        if spec_supports_text(cfg.embedder):
            described = [(i, c.description) for i, c in enumerate(cfg.classes) if c.description.strip()]
            if described:
                T = self.embed_texts(cfg.embedder, [d for _, d in described])
                text = {i: T[j] for j, (i, _) in enumerate(described)}
        return ClassData(classes, X, y, text)

    def _new_classifier(self, task: Task) -> Classifier:
        cfg = task.config
        return make_classifier(cfg.classifier, text_weight=cfg.text_weight, **cfg.classifier_params)

    def fitted(self, task: Task) -> Classifier:
        """The task's classifier, re-fitted only when the task has changed."""
        rev = task.config.revision
        with self._lock:
            hit = self._fitted.get(task.name)
            if hit and hit[0] is task and hit[1] == rev:
                return hit[2]
        clf = self._new_classifier(task).fit(self.class_data(task))
        with self._lock:
            self._fitted[task.name] = (task, rev, clf)
        return clf

    # ---- prediction --------------------------------------------------
    def predict(self, task: Task | str, images: Sequence[ImageInput]) -> list[Prediction]:
        """Classify images given as bytes, file paths or PIL images."""
        task = self.ws.get(task) if isinstance(task, str) else task
        clf = self.fitted(task)
        if not images:
            return []
        items = [(_key(im), im) for im in images]
        Q = self.embed_images(task.config.embedder, items)
        probs, sims = clf.predict(Q)
        return [self._to_pred(task, clf, p, s) for p, s in zip(probs, sims)]

    def _to_pred(self, task: Task, clf: Classifier, p: np.ndarray, s: float) -> Prediction:
        j = int(p.argmax())
        thr = task.config.reject_below
        rejected = thr is not None and float(s) < thr
        return Prediction(
            label=None if rejected else clf.classes[j],
            top_label=clf.classes[j],
            confidence=float(p[j]),
            similarity=float(s),
            rejected=rejected,
            scores={c: float(v) for c, v in zip(clf.classes, p)},
        )


def _key(im: ImageInput) -> str | None:
    if isinstance(im, bytes):
        return image_id(im)
    if isinstance(im, (str, Path)):
        return image_id(Path(im).read_bytes())
    return None  # PIL images are not cached


def _to_pil(im: ImageInput) -> Image.Image:
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    if isinstance(im, (str, Path)):
        im = Path(im).read_bytes()
    return load_image(im)
