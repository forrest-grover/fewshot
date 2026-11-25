"""Evaluation: cross-validation on a task, and episodic N-way K-shot benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .classifiers import ClassData, make_classifier
from .pipeline import Pipeline
from .store import IMAGE_EXTS, Task, image_id


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


def evaluate_task(pipe: Pipeline, task: Task, folds: int | None = None, seed: int = 0) -> dict[str, Any]:
    """Cross-validate the task's current settings on its own labelled examples.

    Defaults to leave-one-out for up to 300 examples, otherwise 5-fold. Every
    held-out example is predicted by a classifier that never saw it.
    """
    data = pipe.class_data(task)
    n, C = len(data.y), len(data.classes)
    if n == 0:
        raise ValueError("Task has no examples to evaluate")
    k = folds or (n if n <= 300 else 5)
    pred = np.full(n, -1)
    conf = np.zeros(n)
    for test in _folds(data.y, k, seed):
        train = np.setdiff1d(np.arange(n), test)
        sub, back = _subset(data, train)
        if len(sub.classes) < 2:
            continue
        clf = make_classifier(task.config.classifier, text_weight=task.config.text_weight,
                              **task.config.classifier_params).fit(sub)
        p, _ = clf.predict(data.X[test])
        pred[test] = back[p.argmax(1)]
        conf[test] = p.max(1)

    scored = pred >= 0
    cm = np.zeros((C, C), int)
    for t, p in zip(data.y[scored], pred[scored]):
        cm[t, p] += 1
    per_class = {}
    for i, c in enumerate(data.classes):
        tp, support, predicted = cm[i, i], cm[i].sum(), cm[:, i].sum()
        per_class[c] = {
            "support": int(support),
            "recall": float(tp / support) if support else None,
            "precision": float(tp / predicted) if predicted else None,
        }
    exs = task.config.examples
    mistakes = [
        {"id": exs[i].id, "true": data.classes[data.y[i]], "pred": data.classes[pred[i]], "confidence": float(conf[i])}
        for i in np.flatnonzero(scored & (pred != data.y))
    ]
    mistakes.sort(key=lambda m: -m["confidence"])
    return {
        "method": "leave-one-out" if k >= n else f"{k}-fold",
        "n_scored": int(scored.sum()),
        "n_examples": n,
        "accuracy": float((pred[scored] == data.y[scored]).mean()) if scored.any() else None,
        "classes": data.classes,
        "confusion": cm.tolist(),
        "per_class": per_class,
        "mistakes": mistakes,
    }


def load_folder_dataset(folder: Path) -> dict[str, list[Path]]:
    ds = {}
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if files:
            ds[d.name] = files
    return ds


def benchmark(
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
    names = sorted(ds)
    files = [f for c in names for f in ds[c]]
    if progress:
        progress(f"Embedding {len(files)} images with {embedder} (cached after first run)")
    E = pipe.embed_images(embedder, [(image_id(f.read_bytes()), f) for f in files])
    start, i = {}, 0
    for c in names:
        start[c] = i
        i += len(ds[c])
    T = None
    if text_template:
        T = pipe.embed_texts(embedder, [text_template.format(c.replace("_", " ")) for c in names])

    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(episodes):
        cls = rng.choice(len(names), way, replace=False)
        Xs, ys, Xq, yq, text = [], [], [], [], {}
        for j, ci in enumerate(cls):
            c = names[ci]
            pick = rng.choice(len(ds[c]), shot + query, replace=False) + start[c]
            Xs.append(E[pick[:shot]]); ys += [j] * shot
            Xq.append(E[pick[shot:]]); yq += [j] * query
            if T is not None:
                text[j] = T[ci]
        data = ClassData([names[c] for c in cls], np.concatenate(Xs), np.array(ys, int), text)
        clf = make_classifier(classifier, text_weight=text_weight, **classifier_params).fit(data)
        p, _ = clf.predict(np.concatenate(Xq))
        accs.append(float((p.argmax(1) == np.array(yq)).mean()))
    a = np.array(accs)
    return {
        "embedder": embedder, "classifier": classifier, "way": way, "shot": shot, "query": query,
        "episodes": episodes, "text_template": text_template,
        "accuracy": float(a.mean()), "ci95": float(1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
    }
