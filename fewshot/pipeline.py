"""Ties a task's stored examples to an embedder, the cache and a classifier."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .cache import EmbeddingCache
from .classifiers import ClassData, Classifier, make_classifier
from .embedders import Vectors, get_embedder, spec_supports_text
from .store import Task, TaskConfig, Workspace, image_id, load_image

ImageInput = bytes | str | Path | Image.Image


@dataclass
class Prediction:
    """Classification result for one image."""

    label: str | None  # None when rejected as unknown
    top_label: str
    confidence: float
    similarity: float
    rejected: bool
    scores: dict[str, float]
    error: str | None = None  # set (with every other field empty) when the image could not be read

    @classmethod
    def failed(cls, error: str) -> Prediction:
        """Result for an image that could not be decoded."""
        return cls(label=None, top_label="", confidence=0.0, similarity=0.0, rejected=False, scores={}, error=error)

    def to_dict(self) -> dict[str, Any]:
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
        # task name -> (task uid, revision, classifier). The uid stops a deleted-and-recreated
        # task that reaches the same revision from reusing a stale fit.
        self._fitted: dict[str, tuple[str, int, Classifier]] = {}
        self._lock = threading.Lock()

    # ---- embeddings --------------------------------------------------
    def embed_images(self, spec: str, items: Sequence[tuple[str | None, ImageInput]],
                     errors: dict[int, str] | None = None) -> Vectors:
        """Embed ``(cache_key or None, image)`` pairs, using and filling the cache.

        An undecodable image raises ``ValueError``, unless ``errors`` is given: then the message is
        stored there under the item's index and that row is all zeros.
        """
        keys = [k for k, _ in items if k]
        hit = self.cache.get_many(spec, keys) if keys else {}
        todo = [(i, k, im) for i, (k, im) in enumerate(items) if not (k and k in hit)]
        out = {i: hit[k] for i, (k, _) in enumerate(items) if k and k in hit}  # row index -> vector
        if todo:
            emb = get_embedder(spec)
            decoded = _decode(todo, errors)
            out.update({i: np.zeros(emb.dim, np.float32) for i in errors or {}})
            vecs = emb.embed_images([im for _, _, im in decoded])
            new: dict[str, Vectors] = {}
            for (i, k, _), v in zip(decoded, vecs, strict=True):
                out[i] = v
                if k:
                    new[k] = v
            self.cache.put_many(spec, new)
        if not items:
            return np.zeros((0, 0), np.float32)
        return np.stack([out[i] for i in range(len(items))]).astype(np.float32)

    def embed_texts(self, spec: str, texts: Sequence[str]) -> Vectors:
        """Embed strings (rows in input order), using and filling the cache."""
        keys = ["text:" + t for t in texts]
        hit = self.cache.get_many(spec, keys)
        missing = [t for t in dict.fromkeys(texts) if "text:" + t not in hit]
        if missing:
            vecs = get_embedder(spec).embed_texts(missing)
            new = {"text:" + t: v for t, v in zip(missing, vecs, strict=True)}
            self.cache.put_many(spec, new)
            hit.update(new)
        return np.stack([hit[k] for k in keys]) if texts else np.zeros((0, 0), np.float32)

    def example_embeddings(self, task: Task, cfg: TaskConfig | None = None) -> Vectors:
        """Embeddings of the task's examples, in ``task.config.examples`` order (or ``cfg``'s)."""
        cfg = cfg or task.config
        return self.embed_images(cfg.embedder, [(e.id, task.path / e.file) for e in cfg.examples])

    # ---- training data -----------------------------------------------
    def class_data(self, task: Task, exclude: set[int] | None = None, cfg: TaskConfig | None = None) -> ClassData:
        """Training data for ``task`` (as of ``cfg``, default its current config), leaving out
        example indices in ``exclude``."""
        cfg = cfg or task.config  # one snapshot: another thread may swap in a reloaded config
        classes = [c.name for c in cfg.classes]
        idx = {c: i for i, c in enumerate(classes)}
        X = self.example_embeddings(task, cfg)
        y = np.array([idx[e.label] for e in cfg.examples], int)
        if exclude:
            keep = np.array([i not in exclude for i in range(len(y))], bool)
            X, y = X[keep], y[keep]
        text: dict[int, Vectors] = {}
        if spec_supports_text(cfg.embedder):
            described = [(i, c.description) for i, c in enumerate(cfg.classes) if c.description.strip()]
            if described:
                T = self.embed_texts(cfg.embedder, [d for _, d in described])
                text = {i: T[j] for j, (i, _) in enumerate(described)}
        # A class with neither examples nor text (e.g. just created) has nothing to fit; leave it out.
        labelled = set(y.tolist())
        used = [i for i in range(len(classes)) if i in text or i in labelled]
        if len(used) < len(classes):
            remap = {old: new for new, old in enumerate(used)}
            classes = [classes[i] for i in used]
            y = np.array([remap[i] for i in y], int)
            text = {remap[i]: v for i, v in text.items()}
        return ClassData(classes, X, y, text)

    @staticmethod
    def _new_classifier(cfg: TaskConfig) -> Classifier:
        return make_classifier(cfg.classifier, text_weight=cfg.text_weight, **cfg.classifier_params)

    def fitted(self, task: Task, cfg: TaskConfig | None = None) -> Classifier:
        """The task's classifier (as of ``cfg``, default its current config), re-fitted only when
        the task has changed."""
        cfg = cfg or task.config  # one snapshot: another thread may swap in a reloaded config
        with self._lock:
            hit = self._fitted.get(cfg.name)
            if hit and hit[:2] == (cfg.uid, cfg.revision):
                return hit[2]
        clf = self._new_classifier(cfg).fit(self.class_data(task, cfg=cfg))
        with self._lock:
            self._fitted[cfg.name] = (cfg.uid, cfg.revision, clf)
        return clf

    # ---- prediction --------------------------------------------------
    def predict(self, task: Task | str, images: Sequence[ImageInput]) -> list[Prediction]:
        """Classify images given as bytes, file paths or PIL images.

        An image that cannot be read or decoded (including a missing file) gets a ``Prediction``
        with ``error`` set; the others are still classified. Classes with neither examples nor a
        description are left out.
        """
        task = self.ws.get(task) if isinstance(task, str) else task
        cfg = task.config
        clf = self.fitted(task, cfg)
        if not images:
            return []
        errors: dict[int, str] = {}
        items: list[tuple[str | None, ImageInput]] = []
        for i, im in enumerate(images):
            if isinstance(im, (str, Path)):  # read once here: for the cache key and for decoding
                try:
                    data = Path(im).read_bytes()
                except OSError as e:
                    errors[i] = f"Could not read {im}: {e.strerror or e}"
                    items.append((None, b""))
                    continue
                items.append((image_id(data), data))
            else:  # PIL images are not cached
                items.append((image_id(im) if isinstance(im, bytes) else None, im))
        Q = self.embed_images(cfg.embedder, items, errors)
        probs, sims = clf.predict(Q)
        return [Prediction.failed(errors[i]) if i in errors else self._to_pred(cfg, clf, p, s)
                for i, (p, s) in enumerate(zip(probs, sims, strict=True))]

    @staticmethod
    def _to_pred(cfg: TaskConfig, clf: Classifier, p: np.ndarray, s: float) -> Prediction:
        j = int(p.argmax())
        thr = cfg.reject_below
        rejected = thr is not None and float(s) < thr
        return Prediction(
            label=None if rejected else clf.classes[j],
            top_label=clf.classes[j],
            confidence=float(p[j]),
            similarity=float(s),
            rejected=rejected,
            scores={c: float(v) for c, v in zip(clf.classes, p, strict=True)},
        )


def _decode(todo: list[tuple[int, str | None, ImageInput]], errors: dict[int, str] | None
            ) -> list[tuple[int, str | None, Image.Image]]:
    """Decode each ``(index, key, image)``; failures raise, or go to ``errors`` when it is given.

    Items already in ``errors`` are skipped.
    """
    decoded = []
    for i, k, im in todo:
        if errors is not None and i in errors:
            continue
        try:
            decoded.append((i, k, _to_pil(im)))
        except (ValueError, OSError) as e:
            if errors is None:
                raise
            errors[i] = str(e) if isinstance(e, ValueError) else f"Not a readable image: {e}"
    return decoded


def _to_pil(im: ImageInput) -> Image.Image:
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    if isinstance(im, (str, Path)):
        im = Path(im).read_bytes()
    return load_image(im)
