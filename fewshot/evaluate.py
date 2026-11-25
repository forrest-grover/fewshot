"""Evaluation: cross-validation on a task, and episodic N-way K-shot benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .classifiers import ClassData, make_classifier
from .pipeline import Pipeline
from .store import IMAGE_EXTS, Task, fill_template, image_id


def _subset(data: ClassData, train: np.ndarray) -> tuple[ClassData, np.ndarray]:
    """Training ClassData restricted to rows ``train``, dropping classes left with
    neither images nor text. Returns it plus a map from subset class index to
    original class index."""
    X, y = data.X[train], data.y[train]
    present = sorted(set(y.tolist()) | set(data.text))
    remap = {c: i for i, c in enumerate(present)}
    sub = ClassData(
        [data.classes[c] for c in present],
        X,
        np.array([remap[c] for c in y], int),
        {remap[c]: v for c, v in data.text.items()},
        {remap[c]: v for c, v in data.name_text.items() if c in remap},
    )
    return sub, np.array(present)


def _folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Stratified fold assignment; k >= len(y) means leave-one-out."""
    n = len(y)
    if k >= n:
        return [np.array([i]) for i in range(n)]
    rng = np.random.default_rng(seed)
    fold_of = np.empty(n, int)
    offset = 0
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold_of[idx] = (np.arange(len(idx)) + offset) % k
        offset += len(idx)
    return [np.flatnonzero(fold_of == f) for f in range(k)]


def _cross_val_predict(data: ClassData, task: Task, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Held-out predicted class index and confidence per example; -1 where a fold
    left fewer than 2 classes to train on."""
    n = len(data.y)
    cfg = task.config
    pred = np.full(n, -1)
    conf = np.zeros(n)
    for test in _folds(data.y, k, seed):
        sub, back = _subset(data, np.setdiff1d(np.arange(n), test))
        if len(sub.classes) < 2:
            continue
        clf = make_classifier(cfg.classifier, text_weight=cfg.text_weight, **cfg.classifier_params).fit(sub)
        p, _ = clf.predict(data.X[test])
        pred[test] = back[p.argmax(1)]
        conf[test] = p.max(1)
    return pred, conf


def _per_class(classes: list[str], cm: np.ndarray) -> dict[str, dict[str, Any]]:
    """Support, recall and precision per class from a confusion matrix (rows = truth)."""
    out = {}
    for i, c in enumerate(classes):
        tp, support, predicted = cm[i, i], cm[i].sum(), cm[:, i].sum()
        out[c] = {
            "support": int(support),
            "recall": float(tp / support) if support else None,
            "precision": float(tp / predicted) if predicted else None,
        }
    return out


def _mistakes(task: Task, data: ClassData, pred: np.ndarray, conf: np.ndarray) -> list[dict[str, Any]]:
    """Scored, misclassified examples, most confident first."""
    exs = task.config.examples
    wrong = np.flatnonzero((pred >= 0) & (pred != data.y))
    mistakes = [
        {"id": exs[i].id, "true": data.classes[data.y[i]], "pred": data.classes[pred[i]],
         "confidence": float(conf[i])}
        for i in wrong
    ]
    mistakes.sort(key=lambda m: -m["confidence"])
    return mistakes


def evaluate_task(pipe: Pipeline, task: Task, folds: int | None = None, seed: int = 0) -> dict[str, Any]:
    """Cross-validate the task's current settings on its own labelled examples.

    Defaults to leave-one-out for up to 300 examples, otherwise 5-fold. Every
    held-out example is predicted by a classifier that never saw it.
    """
    data = pipe.class_data(task)
    n, C = len(data.y), len(data.classes)
    if n == 0:
        raise ValueError("Task has no examples to evaluate")
    if folds is not None and folds < 2:
        raise ValueError(f"folds must be at least 2, got {folds}")
    k = folds or (n if n <= 300 else 5)
    pred, conf = _cross_val_predict(data, task, k, seed)

    scored = pred >= 0
    cm = np.zeros((C, C), int)
    for t, p in zip(data.y[scored], pred[scored], strict=True):
        cm[t, p] += 1
    return {
        "method": "leave-one-out" if k >= n else f"{k}-fold",
        "n_scored": int(scored.sum()),
        "n_examples": n,
        "accuracy": float((pred[scored] == data.y[scored]).mean()) if scored.any() else None,
        "classes": data.classes,
        "confusion": cm.tolist(),
        "per_class": _per_class(data.classes, cm),
        "mistakes": _mistakes(task, data, pred, conf),
    }


def load_folder_dataset(folder: Path) -> dict[str, list[Path]]:
    """Map each subfolder name of ``folder`` to its sorted image files; empty classes are skipped."""
    ds = {}
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if files:
            ds[d.name] = files
    return ds


@dataclass
class _EpisodeSampler:
    """Draws N-way K-shot episodes from a folder dataset's precomputed embeddings."""

    names: list[str]  # sorted class names
    sizes: list[int]  # number of images per class
    E: np.ndarray  # embeddings of every image, grouped by class in ``names`` order
    T: np.ndarray | None  # one text embedding per class, or None

    @classmethod
    def embed(cls, pipe: Pipeline, ds: dict[str, list[Path]], embedder: str, text_template: str | None,
              progress: Callable[[str], None] | None) -> _EpisodeSampler:
        """Embed every image in ``ds`` (and each class's ``text_template`` text, if given)."""
        names = sorted(ds)
        files = [f for c in names for f in ds[c]]
        if progress:
            progress(f"Embedding {len(files)} images with {embedder} (cached after first run)")
        E = pipe.embed_images(embedder, [(image_id(f.read_bytes()), f) for f in files])
        T = None
        if text_template:
            T = pipe.embed_texts(embedder, [fill_template(text_template, c) for c in names])
        return cls(names, [len(ds[c]) for c in names], E, T)

    def sample(self, rng: np.random.Generator, way: int, shot: int, query: int
               ) -> tuple[ClassData, np.ndarray, np.ndarray]:
        """Return support ClassData, query embeddings and query labels for one episode."""
        cls = rng.choice(len(self.names), way, replace=False)
        Xs, Xq, text = [], [], {}
        for j, ci in enumerate(cls):
            pick = rng.choice(self.sizes[ci], shot + query, replace=False) + sum(self.sizes[:ci])
            Xs.append(self.E[pick[:shot]])
            Xq.append(self.E[pick[shot:]])
            if self.T is not None:
                text[j] = self.T[ci]
        data = ClassData([self.names[c] for c in cls], np.concatenate(Xs), np.repeat(np.arange(way), shot), text)
        return data, np.concatenate(Xq), np.repeat(np.arange(way), query)


# The many arguments are benchmark's public API (the CLI passes them positionally); pylint also
# counts those 13 parameters as locals, which alone nearly exhaust the too-many-locals budget.
def benchmark(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    pipe: Pipeline,
    folder: Path,
    embedder: str,
    classifier: str = "prototype",
    way: int = 5,
    shot: int = 5,
    query: int = 15,
    episodes: int = 200,
    text_template: str | None = None,
    text_weight: float = 1.0,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
    **classifier_params: Any,
) -> dict[str, Any]:
    """Mean accuracy over random N-way K-shot episodes from ``folder/<class>/*``.

    ``shot=0`` with a ``text_template`` such as ``"a photo of a {}"`` measures
    CLIP zero-shot accuracy.
    """
    ds = {c: fs for c, fs in load_folder_dataset(folder).items() if len(fs) >= shot + query}
    if len(ds) < way:
        raise ValueError(f"Only {len(ds)} classes have >= {shot + query} images; need {way}")
    sampler = _EpisodeSampler.embed(pipe, ds, embedder, text_template, progress)

    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(episodes):
        data, Q, yq = sampler.sample(rng, way, shot, query)
        p, _ = make_classifier(classifier, text_weight=text_weight, **classifier_params).fit(data).predict(Q)
        accs.append(float((p.argmax(1) == yq).mean()))
    a = np.array(accs)
    return {
        "embedder": embedder, "classifier": classifier, "way": way, "shot": shot, "query": query,
        "episodes": episodes, "text_template": text_template,
        "accuracy": float(a.mean()),
        "ci95": float(1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
    }
