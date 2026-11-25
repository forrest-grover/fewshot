"""Ties a task's stored examples to an embedder, the cache and a classifier."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .cache import EmbeddingCache
from .classifiers import ClassData, Classifier, make_classifier
from .embedders import Embedder, Vectors, cache_namespace, get_embedder, spec_supports_text
from .store import Task, TaskConfig, Workspace, image_id, load_image

ImageInput = bytes | str | Path | Image.Image
Progress = Callable[[int, int], None]  # (done, total); may raise to cancel the work


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
        # task name -> ((task uid, revision, cache namespace), classifier). The uid stops a
        # deleted-and-recreated task that reaches the same revision from reusing a stale fit.
        self._fitted: dict[str, tuple[tuple[str, int, str], Classifier]] = {}
        self._lock = threading.Lock()

    # ---- embeddings --------------------------------------------------
    def embed_images(self, spec: str, items: Sequence[tuple[str | None, ImageInput]],
                     errors: dict[int, str] | None = None, progress: Progress | None = None) -> Vectors:
        """Embed ``(cache_key or None, image)`` pairs, using and filling the cache.

        Uncached images are decoded, embedded and cached one batch at a time, so memory stays
        bounded however many there are, and an interrupted run keeps what it finished. An
        undecodable image raises ``ValueError``, unless ``errors`` is given: then the message is
        stored there under the item's index and that row is all zeros. ``progress(done, total)`` is
        called after each batch with counts of images needing the model; it may raise to cancel.
        """
        ns = cache_namespace(spec)
        hit = self.cache.get_many(ns, [k for k, _ in items if k])
        emb = None
        if any(not (k and k in hit) for k, _ in items) or len({len(v) for v in hit.values()}) > 1:
            # Some images need the model, or the cache holds vectors of different lengths (written
            # by another model under this name): load it and re-embed any that don't fit.
            emb = get_embedder(spec)
            hit = {k: v for k, v in hit.items() if len(v) == emb.dim}
        todo = [(i, k, im) for i, (k, im) in enumerate(items) if not (k and k in hit)]
        out = {i: hit[k] for i, (k, _) in enumerate(items) if k and k in hit}  # row index -> vector
        if emb is not None and todo:
            out.update(self._embed_uncached(emb, ns, todo, errors, progress))
        if not items:
            return np.zeros((0, 0), np.float32)
        return np.stack([out[i] for i in range(len(items))]).astype(np.float32)

    def _embed_uncached(self, emb: Embedder, ns: str, todo: list[tuple[int, str | None, ImageInput]],
                        errors: dict[int, str] | None, progress: Progress | None) -> dict[int, Vectors]:
        """Decode, embed and cache ``todo`` one batch at a time; returns row index -> vector."""
        out: dict[int, Vectors] = {}
        step = emb.batch_size
        for start in range(0, len(todo), step):
            chunk = todo[start : start + step]
            decoded = _decode(chunk, errors)
            if errors:
                out.update({i: np.zeros(emb.dim, np.float32) for i, _, _ in chunk if i in errors})
            new: dict[str, Vectors] = {}
            for (i, k, _), v in zip(decoded, emb.embed_images([im for _, _, im in decoded]), strict=True):
                out[i] = v
                if k:
                    new[k] = v
            self.cache.put_many(ns, new)
            if progress:
                progress(min(start + step, len(todo)), len(todo))
        return out

    def embed_files(self, spec: str, paths: Sequence[Path], progress: Progress | None = None,
                    chunk: int = 256) -> Vectors:
        """Embed image files, reading each once (for both its cache key and decoding) and holding
        only ``chunk`` files' bytes at a time. ``progress(done, total)`` counts files."""
        parts = []
        for start in range(0, len(paths), chunk):
            blobs = [p.read_bytes() for p in paths[start : start + chunk]]
            parts.append(self.embed_images(spec, [(image_id(b), b) for b in blobs]))
            if progress:
                progress(min(start + chunk, len(paths)), len(paths))
        return np.concatenate(parts) if parts else np.zeros((0, 0), np.float32)

    def embed_texts(self, spec: str, texts: Sequence[str]) -> Vectors:
        """Embed strings (rows in input order), using and filling the cache."""
        ns = cache_namespace(spec)
        keys = ["text:" + t for t in texts]
        hit = self.cache.get_many(ns, keys)
        if len({len(v) for v in hit.values()}) > 1:  # mixed lengths: keep only the model's own
            dim = get_embedder(spec).dim
            hit = {k: v for k, v in hit.items() if len(v) == dim}
        missing = [t for t in dict.fromkeys(texts) if "text:" + t not in hit]
        if missing:
            vecs = get_embedder(spec).embed_texts(missing)
            new = {"text:" + t: v for t, v in zip(missing, vecs, strict=True)}
            self.cache.put_many(ns, new)
            hit.update(new)
        return np.stack([hit[k] for k in keys]) if texts else np.zeros((0, 0), np.float32)

    def example_embeddings(self, task: Task, cfg: TaskConfig | None = None,
                           progress: Progress | None = None) -> Vectors:
        """Embeddings of the task's examples, in ``task.config.examples`` order (or ``cfg``'s)."""
        cfg = cfg or task.config
        return self.embed_images(cfg.embedder, [(e.id, task.path / e.file) for e in cfg.examples],
                                 progress=progress)

    # ---- training data -----------------------------------------------
    def class_data(self, task: Task, exclude: set[int] | None = None, cfg: TaskConfig | None = None,
                   progress: Progress | None = None) -> ClassData:
        """Training data for ``task`` (as of ``cfg``, default its current config), leaving out
        example indices in ``exclude``. ``progress`` reports images embedded (see ``embed_images``)."""
        cfg = cfg or task.config  # one snapshot: another thread may swap in a reloaded config
        classes = [c.name for c in cfg.classes]
        idx = {c: i for i, c in enumerate(classes)}
        X = self.example_embeddings(task, cfg, progress)
        y = np.array([idx[e.label] for e in cfg.examples], int)
        if exclude:
            keep = np.array([i not in exclude for i in range(len(y))], bool)
            X, y = X[keep], y[keep]
        text: dict[int, Vectors] = {}
        if spec_supports_text(cfg.embedder):
            text = self._indexed_texts(cfg.embedder, {i: c.description for i, c in enumerate(cfg.classes)
                                                      if c.description.strip()})
        # A class with neither examples nor text (e.g. just created) has nothing to fit; leave it out.
        labelled = set(y.tolist())
        used = [i for i in range(len(classes)) if i in text or i in labelled]
        # With some descriptions, the other classes' names help estimate the modality gap.
        names = self._indexed_texts(cfg.embedder, {i: classes[i].replace("_", " ") for i in used
                                                   if i not in text}) if text else {}
        if len(used) < len(classes):
            remap = {old: new for new, old in enumerate(used)}
            classes = [classes[i] for i in used]
            y = np.array([remap[i] for i in y], int)
            text = {remap[i]: v for i, v in text.items()}
            names = {remap[i]: v for i, v in names.items()}
        return ClassData(classes, X, y, text, names)

    def _indexed_texts(self, spec: str, texts: dict[int, str]) -> dict[int, Vectors]:
        """Embed ``{class index: text}``, keeping the indices."""
        if not texts:
            return {}
        return dict(zip(texts, self.embed_texts(spec, list(texts.values())), strict=True))

    @staticmethod
    def _new_classifier(cfg: TaskConfig) -> Classifier:
        return make_classifier(cfg.classifier, text_weight=cfg.text_weight, **cfg.classifier_params)

    def fitted(self, task: Task, cfg: TaskConfig | None = None) -> Classifier:
        """The task's classifier (as of ``cfg``, default its current config), re-fitted only when
        the task has changed."""
        cfg = cfg or task.config  # one snapshot: another thread may swap in a reloaded config
        # The model's namespace is part of the key, so re-registering a spec's embedder refits.
        stamp = (cfg.uid, cfg.revision, cache_namespace(cfg.embedder))
        with self._lock:
            hit = self._fitted.get(cfg.name)
            if hit and hit[0] == stamp:
                return hit[1]
        clf = self._new_classifier(cfg).fit(self.class_data(task, cfg=cfg))
        with self._lock:
            self._fitted[cfg.name] = (stamp, clf)
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
