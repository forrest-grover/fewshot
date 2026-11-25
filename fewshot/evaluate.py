"""Evaluation: cross-validation on a task, and episodic N-way K-shot benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .classifiers import ClassData, KNNClassifier, PrototypeClassifier, make_classifier, temper
from .images import IMAGE_EXTS
from .pipeline import Pipeline, new_classifier
from .store import Task, TaskConfig
from .text import fill_template

EvalProgress = Callable[[str, int, int], None]  # (stage, done, total); may raise to cancel
TOP_K = (2, 3, 5)  # top-k accuracies reported, for k below the number of classes
KEEP_CORRECT = 0.95  # the suggested reject threshold keeps this share of correct predictions


@dataclass
class HeldOut:
    """Cross-validated results per example: predicted class index (-1 if a fold had too few
    classes to train on), its probability, the similarity to the best match, and the rank of the
    true class among the predictions (0 = top; the number of classes if it couldn't be predicted)."""

    pred: np.ndarray
    conf: np.ndarray
    sim: np.ndarray
    rank: np.ndarray
    probs: np.ndarray  # (n, C) probabilities over all classes (0 for classes a fold left out)

    @classmethod
    def empty(cls, n: int, n_classes: int) -> HeldOut:
        """Results with nothing scored yet."""
        return cls(np.full(n, -1), np.zeros(n), np.zeros(n), np.full(n, n_classes), np.zeros((n, n_classes)))

    def record(self, rows: np.ndarray, probs: np.ndarray, sim: np.ndarray, truth: np.ndarray,
               known: np.ndarray) -> None:
        """Store results for examples ``rows`` from their probabilities over all classes; ``known``
        says whether each one's true class was among those trained on."""
        self.pred[rows] = probs.argmax(1)
        self.conf[rows] = probs.max(1)
        self.probs[rows] = probs
        self.sim[rows] = sim
        self.rank[rows] = _ranks(probs, truth, known)


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


def _cross_val_predict(data: ClassData, cfg: TaskConfig, k: int, seed: int,
                       progress: Callable[[int, int], None] | None = None) -> HeldOut:
    """Held-out predictions per example. A fold whose training part has fewer than 2 classes (or,
    for a one-class task, none) leaves its examples unscored."""
    n, C = len(data.y), len(data.classes)
    if k >= n and cfg.classifier in {"prototype", "knn"} and _text_unused(data, cfg):
        return _fast_leave_one_out(data, cfg, progress)
    out = HeldOut.empty(n, C)
    folds = _folds(data.y, k, seed)
    for done, test in enumerate(folds):
        if progress:
            progress(done, len(folds))
        _fold(data, cfg, test, out)
    if progress:
        progress(len(folds), len(folds))
    return out


def _fold(data: ClassData, cfg: TaskConfig, test: np.ndarray, out: HeldOut) -> None:
    """Fit on every example but ``test`` and record the predictions for ``test`` in ``out``."""
    C = len(data.classes)
    sub, back = _subset(data, np.setdiff1d(np.arange(len(data.y)), test))
    if len(sub.classes) < min(2, C):
        return
    p, s = new_classifier(cfg).fit(sub).predict(data.X[test])
    full = np.zeros((len(test), C))  # probabilities over all classes, 0 for those left out
    full[:, back] = p
    out.record(test, full, s, data.y[test], np.isin(data.y[test], back))


def _text_unused(data: ClassData, cfg: TaskConfig) -> bool:
    """Whether descriptions play no part in any leave-one-out fold: there are none, or their
    weight is 0 and every class keeps an example when one is held out (a class left with only
    its description would score through it)."""
    if not data.text:
        return True
    return cfg.text_weight == 0 and int(np.bincount(data.y, minlength=len(data.classes)).min()) >= 2


def _ranks(probs: np.ndarray, truth: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Rank of each row's true class (0 = most probable); the number of classes where the true
    class wasn't among those trained on (``known`` False). Ties count against the true class: kNN
    gives every class without a near neighbour the same tiny score, and counting those ties in the
    true class's favour would credit it with a top-k place it never earned."""
    true_p = probs[np.arange(len(truth)), truth]
    ranks: np.ndarray = np.where(known, (probs >= true_p[:, None]).sum(1) - 1, probs.shape[1])
    return ranks


def _fast_leave_one_out(data: ClassData, cfg: TaskConfig,
                        progress: Callable[[int, int], None] | None) -> HeldOut:
    """Leave-one-out for the prototype and kNN classifiers without refitting per example: each
    example's own contribution is taken out of the fitted model. Gives the same results as
    refitting, for tasks without text descriptions."""
    clf = new_classifier(cfg)
    X, y = data.X.astype(np.float64), data.y
    n, C = len(y), len(data.classes)
    present = np.ones((n, C), bool)  # classes left with examples once example i is held out
    present[np.arange(n), y] = np.bincount(y, minlength=C)[y] > 1
    out = HeldOut.empty(n, C)
    for start in range(0, n, 1024):
        if progress:
            progress(start, n)
        rows = np.arange(start, min(start + 1024, n))
        if isinstance(clf, PrototypeClassifier):
            probs, sim = _loo_prototype(clf, X, y, rows, present[rows])
        else:
            assert isinstance(clf, KNNClassifier)
            probs, sim = _loo_knn(clf, X, y, rows, present[rows])
        probs = temper(probs, clf.temperature)
        ok = present[rows].sum(1) >= min(2, C)  # as many classes as refitting would need
        out.record(rows[ok], probs[ok], sim[ok], y[rows[ok]], present[rows[ok], y[rows[ok]]])
    if progress:
        progress(n, n)
    return out


def _masked_softmax(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask, z, -np.inf)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.where(mask, np.exp(z), 0.0)
    return e / e.sum(1, keepdims=True)


def _loo_prototype(clf: PrototypeClassifier, X: np.ndarray, y: np.ndarray, rows: np.ndarray,
                   present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities and best similarity for ``rows``, each against prototypes built without it."""
    C = present.shape[1]
    sums = np.zeros((C, X.shape[1]))
    np.add.at(sums, y, X)
    norms = np.linalg.norm(sums, axis=1).clip(1e-12)
    xs = X[rows]
    cos = (xs @ sums.T) / norms
    own = sums[y[rows]] - xs  # the example's class sum without it
    cos[np.arange(len(rows)), y[rows]] = (xs * own).sum(1) / np.linalg.norm(own, axis=1).clip(1e-12)
    with np.errstate(invalid="ignore"):  # rows with no class present are dropped by the caller
        probs = _masked_softmax(clf.scale * cos, present)
    return probs, np.where(present, cos, -np.inf).max(1)


def _loo_knn(clf: KNNClassifier, X: np.ndarray, y: np.ndarray, rows: np.ndarray,
             present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities and the similarity to the predicted class (its nearest other example) for
    ``rows``, each voted on by the other examples."""
    C = present.shape[1]
    S = X[rows] @ X.T
    S[np.arange(len(rows)), rows] = -np.inf  # not its own neighbour
    k = min(clf.k, len(X) - 1)
    if k < 1:  # a single example: nothing left to vote
        return np.zeros((len(rows), C)), np.full(len(rows), -np.inf)
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    top_s = np.take_along_axis(S, top, axis=1)
    votes = np.exp(clf.scale * (top_s - top_s.max(axis=1, keepdims=True)))
    scores = np.where(present, 1e-6, 0.0)  # the same floor as KNNClassifier, for classes still present
    np.add.at(scores, (np.repeat(np.arange(len(rows)), k), y[top.ravel()]), votes.ravel())
    pred = scores.argmax(1)
    sim = np.array([S[i, y == c].max() for i, c in enumerate(pred)])
    return scores / scores.sum(1, keepdims=True), sim


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


def _predictions(cfg: TaskConfig, data: ClassData, h: HeldOut) -> list[dict[str, Any]]:
    """The held-out prediction for every scored example, in task order."""
    exs = cfg.examples
    return [{"id": exs[i].id, "true": data.classes[data.y[i]], "pred": data.classes[h.pred[i]],
             "confidence": round(float(h.conf[i]), 4), "similarity": round(float(h.sim[i]), 4),
             "correct": bool(h.pred[i] == data.y[i])}
            for i in np.flatnonzero(h.pred >= 0)]


def _ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    """Expected calibration error (how far confidence is from accuracy, weighted by how many
    predictions have that confidence) and the per-bin reliability it is computed from."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece, table = 0.0, []
    for b in range(bins):
        sel = which == b
        if sel.any():
            ece += abs(float(conf[sel].mean()) - float(correct[sel].mean())) * float(sel.mean())
            table.append({"confidence": round(float(conf[sel].mean()), 4),
                          "accuracy": round(float(correct[sel].mean()), 4), "count": int(sel.sum())})
    return round(ece, 4), table


MIN_CALIBRATION = 50  # scored examples needed before a temperature is suggested (fewer: too noisy)


def _nll(P: np.ndarray, t: np.ndarray, T: float) -> float:
    return float(-np.log(np.clip(temper(P, T)[np.arange(len(t)), t], 1e-12, 1)).sum())


def _calibration(h: HeldOut, truth: np.ndarray, current: float) -> dict[str, Any] | None:
    """A temperature that makes held-out confidence match held-out accuracy (fitted by maximum
    likelihood over a grid), with the calibration error before and after. ``None`` with too few
    scored examples or classes. The temperature is absolute: ``current`` times the one fitted
    on top of the task's present setting."""
    rows = np.flatnonzero((h.pred >= 0) & (h.probs[np.arange(len(truth)), truth] > 0))
    if len(rows) < MIN_CALIBRATION or h.probs.shape[1] < 2:
        return None
    P, t = h.probs[rows], truth[rows]
    best = float(min(np.geomspace(0.01, 100, 81), key=lambda T: _nll(P, t, float(T))))
    correct = (P.argmax(1) == t).astype(float)
    ece, table = _ece(P.max(1), correct)
    ece_after, table_after = _ece(temper(P, best).max(1), correct)
    return {"temperature": round(current * best, 4), "ece": ece, "ece_calibrated": ece_after,
            "reliability": table, "reliability_calibrated": table_after}


def _macro_f1(cm: np.ndarray) -> float | None:
    """Mean F1 over the classes with examples (None if none were scored)."""
    support, predicted, tp = cm.sum(1), cm.sum(0), np.diag(cm)
    has = support > 0
    if not has.any():
        return None
    return float(np.mean(2 * tp[has] / (support[has] + predicted[has])))


def _suggest_threshold(h: HeldOut, truth: np.ndarray) -> float | None:
    """A reject threshold that keeps ``KEEP_CORRECT`` of the correct held-out predictions:
    images less similar to the task's examples than nearly all real members are likely outsiders."""
    ok = (h.pred >= 0) & (h.pred == truth)
    if not ok.any():
        return None
    return round(float(np.quantile(h.sim[ok], 1 - KEEP_CORRECT)), 3)


def _confusion(truth: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Confusion matrix: rows are true classes, columns predicted ones."""
    cm = np.zeros((n_classes, n_classes), int)
    np.add.at(cm, (truth, pred), 1)
    return cm


def _stage(progress: EvalProgress | None, stage: str) -> Callable[[int, int], None] | None:
    """``progress`` with its stage name filled in."""
    if progress is None:
        return None
    return lambda done, total: progress(stage, done, total)


def evaluate_task(pipe: Pipeline, task: Task, folds: int | None = None, seed: int = 0,
                  progress: EvalProgress | None = None) -> dict[str, Any]:
    """Cross-validate the task's current settings on its own labelled examples.

    Defaults to leave-one-out for up to 300 examples, otherwise 5-fold. Every
    held-out example is predicted by a classifier that never saw it.
    ``progress(stage, done, total)`` is called with stage ``"embedding"`` (images) and then
    ``"evaluating"`` (folds); it may raise to cancel.
    """
    cfg = task.config  # one snapshot, so a concurrent edit can't misalign examples and results
    if folds is not None and folds < 2:
        raise ValueError(f"folds must be at least 2, got {folds}")
    data = pipe.class_data(task, cfg=cfg, progress=_stage(progress, "embedding"))
    n, C = len(data.y), len(data.classes)
    if n == 0:
        raise ValueError("Task has no examples to evaluate")
    k = folds or (n if n <= 300 else 5)
    h = _cross_val_predict(data, cfg, k, seed, _stage(progress, "evaluating"))

    scored = h.pred >= 0
    cm = _confusion(data.y[scored], h.pred[scored], C)
    predictions = _predictions(cfg, data, h)
    return {
        "revision": cfg.revision,
        "method": "leave-one-out" if k >= n else f"{k}-fold",
        "n_scored": int(scored.sum()),
        "n_examples": n,
        "accuracy": float((h.pred[scored] == data.y[scored]).mean()) if scored.any() else None,
        "macro_f1": _macro_f1(cm),
        "top_k_accuracy": {str(t): float((h.rank[scored] < t).mean()) for t in TOP_K if t < C and scored.any()},
        "classes": data.classes,
        "confusion": cm.tolist(),
        "per_class": _per_class(data.classes, cm),
        "mistakes": sorted((p for p in predictions if not p["correct"]), key=lambda m: -m["confidence"]),
        "predictions": predictions,
        "suggested_reject_below": _suggest_threshold(h, data.y),
        "calibration": _calibration(h, data.y, cfg.temperature or 1.0),
    }


def load_folder_dataset(folder: Path) -> dict[str, list[Path]]:
    """Map each subfolder name of ``folder`` to its sorted image files; empty classes are skipped."""
    ds = {}
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS and f.is_file())
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
    def embed(cls, pipe: Pipeline, ds: dict[str, list[Path]], embedder: str,  # pylint: disable=too-many-arguments,too-many-positional-arguments
              text_template: str | None, progress: Callable[[str], None] | None, flip: bool = False
              ) -> _EpisodeSampler:
        """Embed every image in ``ds`` (and each class's ``text_template`` text, if given; a
        template of several lines gives several prompts, which are averaged)."""
        names = sorted(ds)
        files = [f for c in names for f in ds[c]]
        if progress:
            progress(f"Embedding {len(files)} images with {embedder} (cached after first run)")
        E = pipe.embed_files(embedder, files, flip=flip)
        T = None
        if text_template:
            T = pipe.embed_descriptions(embedder, [fill_template(text_template, c) for c in names])
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
    flip: bool = False,
    **classifier_params: Any,
) -> dict[str, Any]:
    """Mean accuracy over random N-way K-shot episodes from ``folder/<class>/*``.

    ``shot=0`` with a ``text_template`` such as ``"a photo of a {}"`` measures
    CLIP zero-shot accuracy. ``flip`` averages each image with its mirror image.
    """
    for arg, value, low in (("way", way, 2), ("shot", shot, 0), ("query", query, 1), ("episodes", episodes, 1)):
        if value < low:
            raise ValueError(f"{arg} must be at least {low}, got {value}")
    if shot == 0 and not text_template:  # checked before embedding the whole dataset
        raise ValueError("shot=0 (zero-shot) needs a text_template such as 'a photo of a {}'")
    ds = {c: fs for c, fs in load_folder_dataset(folder).items() if len(fs) >= shot + query}
    if len(ds) < way:
        raise ValueError(f"Only {len(ds)} classes have >= {shot + query} images; need {way}")
    sampler = _EpisodeSampler.embed(pipe, ds, embedder, text_template, progress, flip)

    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(episodes):
        data, Q, yq = sampler.sample(rng, way, shot, query)
        p, _ = make_classifier(classifier, text_weight=text_weight, **classifier_params).fit(data).predict(Q)
        accs.append(float((p.argmax(1) == yq).mean()))
    a = np.array(accs)
    return {
        "embedder": embedder, "classifier": classifier, "way": way, "shot": shot, "query": query,
        "episodes": episodes, "text_template": text_template, "flip": flip,
        "accuracy": float(a.mean()),
        "ci95": float(1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
    }
