"""Ties a task's stored examples to an embedder, the cache and a classifier."""

from __future__ import annotations

import heapq
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .cache import EmbeddingCache
from .classifiers import ClassData, Classifier, make_classifier
from .embedders import Embedder, Vectors, cache_namespace, get_embedder, spec_supports_text
from .images import image_id, load_image
from .store import Task, TaskConfig, Workspace
from .text import prompts

ImageInput = bytes | str | Path | Image.Image
Progress = Callable[[int, int], None]  # (done, total); may raise to cancel the work
FLIP_KEY = ":flip"  # cache key suffix for flip-averaged embeddings


@dataclass
class Prediction:  # pylint: disable=too-many-instance-attributes  # one field per result value
    """Classification result for one image."""

    label: str | None  # None when rejected as unknown
    top_label: str
    confidence: float
    similarity: float
    rejected: bool
    scores: dict[str, float]  # probability per class, highest first
    # Every class the image matches (multi-label): with a reject threshold, each class whose own
    # similarity reaches it, ``label`` first; without one, just ``label``.
    labels: list[str] = field(default_factory=list)
    # The labelled examples most like the image ({id, label, similarity}), when asked for.
    evidence: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None  # set (with every other field empty) when the image could not be read
    revision: int | None = None  # of the task version that made the prediction

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
                     errors: dict[int, str] | None = None, progress: Progress | None = None,
                     flip: bool = False) -> Vectors:
        """Embed ``(cache_key or None, image)`` pairs, using and filling the cache.

        With ``flip``, each vector is the normalised mean of the image's and its mirror image's
        embeddings (test-time augmentation), cached under the key plus ``FLIP_KEY``.

        Uncached images are decoded, embedded and cached one batch at a time, so memory stays
        bounded however many there are, and an interrupted run keeps what it finished. An
        undecodable image raises ``ValueError``, unless ``errors`` is given: then the message is
        stored there under the item's index and that row is all zeros. ``progress(done, total)`` is
        called after each batch with counts of images needing the model; it may raise to cancel.
        """
        if not items:
            return np.zeros((0, 0), np.float32)
        ns = cache_namespace(spec)
        if flip:
            items = [(k + FLIP_KEY if k else None, im) for k, im in items]
        emb, hit = self._cached(spec, ns, [k for k, _ in items])
        out = {i: hit[k] for i, (k, _) in enumerate(items) if k and k in hit}  # row index -> vector
        if emb is not None:
            first: dict[str, int] = {}  # key -> first row with it: the same image is embedded once
            todo = []
            for i, (k, im) in enumerate(items):
                if i in out or (k and k in first):
                    continue
                if k:
                    first[k] = i
                todo.append((i, k, im))
            out.update(self._embed_uncached(emb, ns, todo, errors, progress, flip))
            for i, (k, _) in enumerate(items):
                if i not in out and k:  # a repeat of an image embedded (or failed) above
                    out[i] = out[first[k]]
                    if errors is not None and first[k] in errors:
                        errors[i] = errors[first[k]]
        return np.stack([out[i] for i in range(len(items))]).astype(np.float32)

    def _cached(self, spec: str, ns: str, keys: list[str | None]) -> tuple[Embedder | None, dict[str, Vectors]]:
        """Cached vectors for ``keys``, plus the loaded embedder if any key misses the cache."""
        hit = self.cache.get_many(ns, [k for k in keys if k])
        emb = None
        if any(not (k and k in hit) for k in keys) or len({len(v) for v in hit.values()}) > 1:
            # Some images need the model, or the cache holds vectors of different lengths (written
            # by another model under this name): load it and re-embed any that don't fit.
            emb = get_embedder(spec)
            hit = {k: v for k, v in hit.items() if len(v) == emb.dim}
        return emb, hit

    def _embed_uncached(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # internal helper
            self, emb: Embedder, ns: str, todo: list[tuple[int, str | None, ImageInput]],
            errors: dict[int, str] | None, progress: Progress | None, flip: bool = False) -> dict[int, Vectors]:
        """Decode, embed and cache ``todo`` one batch at a time; returns row index -> vector."""
        out: dict[int, Vectors] = {}
        step = emb.batch_size
        for start in range(0, len(todo), step):
            chunk = todo[start : start + step]
            decoded = _decode(chunk, errors)
            if errors:  # unreadable rows are zeros
                out.update({i: np.zeros(emb.dim, np.float32) for i, _, _ in chunk if i in errors})
            vecs = _embed_decoded(emb, [im for _, _, im in decoded], flip)
            out.update({i: v for (i, _, _), v in zip(decoded, vecs, strict=True)})
            self.cache.put_many(ns, {k: v for (_, k, _), v in zip(decoded, vecs, strict=True) if k})
            if progress:
                progress(min(start + step, len(todo)), len(todo))
        return out

    def embed_files(self, spec: str, paths: Sequence[Path], progress: Progress | None = None,
                    chunk: int = 256, flip: bool = False) -> Vectors:
        """Embed image files, reading each once (for both its cache key and decoding) and holding
        only ``chunk`` files' bytes at a time. ``progress(done, total)`` counts files."""
        parts = []
        for start in range(0, len(paths), chunk):
            blobs = [p.read_bytes() for p in paths[start : start + chunk]]
            parts.append(self.embed_images(spec, [(image_id(b), b) for b in blobs], flip=flip))
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

    def embed_descriptions(self, spec: str, descriptions: Sequence[str]) -> Vectors:
        """One text vector per description: the normalised mean of its prompts (one per line)."""
        groups = [prompts(d) or [d] for d in descriptions]
        flat = self.embed_texts(spec, [t for g in groups for t in g])
        out, at = [], 0
        for g in groups:
            v = flat[at : at + len(g)].mean(0)
            out.append(v / max(float(np.linalg.norm(v)), 1e-12))
            at += len(g)
        return np.stack(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)

    def example_embeddings(self, task: Task, cfg: TaskConfig | None = None,
                           progress: Progress | None = None) -> Vectors:
        """Embeddings of the task's examples, in ``task.config.examples`` order (or ``cfg``'s)."""
        cfg = cfg or task.config
        return self.embed_images(cfg.embedder, [(e.id, task.path / e.file) for e in cfg.examples],
                                 progress=progress, flip=cfg.flip)

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
                                                      if prompts(c.description)})
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
        return dict(zip(texts, self.embed_descriptions(spec, list(texts.values())), strict=True))


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
        clf = new_classifier(cfg).fit(self.class_data(task, cfg=cfg))
        with self._lock:
            # A slow fit of an older revision mustn't replace a newer one fitted meanwhile.
            cur = self._fitted.get(cfg.name)
            if cur is None or cur[0][0] != stamp[0] or cur[0][1] <= stamp[1]:
                self._fitted[cfg.name] = (stamp, clf)
        return clf

    # ---- prediction --------------------------------------------------
    def predict(self, task: Task | str, images: Sequence[ImageInput], top_k: int | None = None,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                progress: Progress | None = None, evidence: int = 0, cfg: TaskConfig | None = None) -> list[Prediction]:
        """Classify images given as bytes, file paths or PIL images.

        An image that cannot be read or decoded (including a missing file) gets a ``Prediction``
        with ``error`` set; the others are still classified. Classes with neither examples nor a
        description are left out. ``top_k`` keeps only that many of the highest ``scores``;
        ``progress`` counts images embedded (see ``embed_images``); ``evidence`` adds that many of
        the most similar labelled examples to each prediction. ``cfg`` predicts with that version of
        the task (see ``Task.version``) instead of the current one.
        """
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        task = self.ws.get(task) if isinstance(task, str) else task
        cfg = cfg or task.config
        clf = self.fitted(task, cfg)
        if not images:
            return []
        errors: dict[int, str] = {}
        Q = self.embed_images(cfg.embedder, _keyed(images, errors), errors, progress, cfg.flip)
        probs, sims, per_class = clf.predict_full(Q)
        thr = threshold_for(cfg, clf)
        preds = [Prediction.failed(errors[i]) if i in errors else _to_pred(clf, thr, row, top_k)
                 for i, row in enumerate(zip(probs, sims, per_class, strict=True))]
        if evidence > 0:
            _attach_evidence(clf, cfg, Q, preds, evidence)
        return [replace(p, revision=cfg.revision) for p in preds]

    # ---- inbox ---------------------------------------------------------
    def inbox_suggestions(self, task: Task, sort: str = "uncertain") -> list[dict[str, Any]]:
        """The task's inbox images with the model's suggestion for each, sorted for labelling:

        ``uncertain`` (default): smallest margin between the two most probable classes first; the
        images whose labels teach the model most (active learning). ``confident``: most confident
        first, for accepting suggestions in bulk. ``novel``: least like any labelled example first,
        which finds new kinds of image. ``added``: oldest first.

        Each entry is the inbox item plus ``prediction`` (as from ``predict``, top 3 scores) and
        ``margin``; with no classes yet there is no prediction.
        """
        orders = {"uncertain", "confident", "novel", "added"}
        if sort not in orders:
            raise ValueError(f"sort must be one of {sorted(orders)}, got {sort!r}")
        cfg = task.config
        items = [x.model_dump() for x in cfg.inbox]
        if not items or not cfg.classes:
            return items
        preds = self.predict(task, [task.path / x.file for x in cfg.inbox], top_k=3)
        for item, p in zip(items, preds, strict=True):
            top = list(p.scores.values())
            item["prediction"] = p.to_dict()
            item["margin"] = round(top[0] - (top[1] if len(top) > 1 else 0.0), 4) if top else 0.0
        keys = {"uncertain": lambda i: (i["margin"],), "confident": lambda i: (-i["prediction"]["confidence"],),
                "novel": lambda i: (i["prediction"]["similarity"],), "added": lambda i: ()}
        return sorted(items, key=keys[sort])

    # ---- data checks -------------------------------------------------
    def near_duplicates(self, task: Task, threshold: float = 0.95, limit: int = 500) -> list[dict[str, Any]]:
        """Pairs of the task's examples whose embeddings have cosine similarity >= ``threshold``,
        most similar first (at most ``limit``). Pairs with different labels are often labelling
        mistakes; identical files are already stored once, so these are re-encodes, crops and the like.
        """
        cfg = task.config
        X = self.example_embeddings(task, cfg)
        exs = cfg.examples
        pairs: list[tuple[float, int, int]] = []
        floor = threshold - 1e-5  # float32 rounding: identical images can score 0.99999976
        for start in range(0, len(X), 512):
            S = X[start : start + 512] @ X.T
            S[np.arange(len(S))[:, None] >= np.arange(len(X))[None, :] - start] = -np.inf  # each pair once
            flat = S.ravel()
            hits = np.flatnonzero(flat >= floor)
            if len(hits) > limit:  # keep only this chunk's best few, so memory stays bounded
                hits = hits[np.argpartition(-flat[hits], limit - 1)[:limit]]
            pairs += [(float(flat[h]), start + h // len(X), h % len(X)) for h in hits.tolist()]
            pairs = heapq.nlargest(limit, pairs)
        return [{"a": exs[i].id, "b": exs[j].id, "similarity": round(sim, 4),
                 "labels": [exs[i].label, exs[j].label], "same_label": exs[i].label == exs[j].label}
                for sim, i, j in pairs[:limit]]


def _attach_evidence(clf: Classifier, cfg: TaskConfig, Q: Vectors, preds: list[Prediction], k: int) -> None:
    """Give each readable prediction the ``k`` labelled examples most like its image. The fit's
    example rows are ``cfg.examples`` in order (the snapshot it was fitted from)."""
    idx, near = clf.nearest(Q, k)
    for i, p in enumerate(preds):
        if p.error is None:
            p.evidence = [{"id": cfg.examples[j].id, "label": cfg.examples[j].label, "similarity": round(float(sv), 4)}
                          for j, sv in zip(idx[i], near[i], strict=True)]


def new_classifier(cfg: TaskConfig) -> Classifier:
    """An unfitted classifier with ``cfg``'s settings, including its calibration temperature."""
    clf = make_classifier(cfg.classifier, text_weight=cfg.text_weight, **cfg.classifier_params)
    clf.temperature = cfg.temperature or 1.0
    return clf


def threshold_for(cfg: TaskConfig, clf: Classifier) -> float | None:
    """The reject threshold that applies to ``clf``'s predictions: the zero-shot one for a task
    without images, otherwise ``reject_below``."""
    return cfg.reject_below_zero_shot if clf.zero_shot else cfg.reject_below


def _keyed(images: Sequence[ImageInput], errors: dict[int, str]) -> list[tuple[str | None, ImageInput]]:
    """``(cache key, image)`` per image. Files are read here, once, for both the key and decoding;
    one that can't be read gets a message in ``errors``. PIL images aren't cached."""
    items: list[tuple[str | None, ImageInput]] = []
    for i, im in enumerate(images):
        if isinstance(im, (str, Path)):
            try:
                data = Path(im).read_bytes()
            except OSError as e:
                errors[i] = f"Could not read {im}: {e.strerror or e}"
                items.append((None, b""))
                continue
            items.append((image_id(data), data))
        else:
            items.append((image_id(im) if isinstance(im, bytes) else None, im))
    return items


def _to_pred(clf: Classifier, thr: float | None, row: tuple[np.ndarray, float, np.ndarray],
             top_k: int | None) -> Prediction:
    """The prediction from one image's (probabilities, similarity to the predicted class,
    similarity per class)."""
    p, s, per_class = row
    j = int(p.argmax())
    rejected = thr is not None and float(s) < thr
    label = None if rejected else clf.classes[j]
    labels = [] if label is None else [label]
    if thr is not None and label is not None:
        order = np.argsort(-per_class, kind="stable")
        labels += [clf.classes[i] for i in order if i != j and per_class[i] >= thr]
    ranked = sorted(zip(clf.classes, p.tolist(), strict=True), key=lambda cv: -cv[1])
    return Prediction(
        label=label,
        top_label=clf.classes[j],
        confidence=float(p[j]),
        similarity=float(s),
        rejected=rejected,
        scores=dict(ranked[:top_k] if top_k else ranked),
        labels=labels,
    )


def _embed_decoded(emb: Embedder, images: list[Image.Image], flip: bool) -> Vectors:
    """Embed decoded images; with ``flip``, average each with its mirror image."""
    if not flip:
        return emb.embed_images(images)
    v = emb.embed_images(images + [ImageOps.mirror(im) for im in images])
    v = v[: len(images)] + v[len(images) :]
    out: Vectors = (v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)).astype(np.float32)
    return out


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
