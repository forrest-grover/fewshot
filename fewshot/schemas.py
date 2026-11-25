"""Response models of the REST API, so /docs shows (and clients can generate) exact types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Health(BaseModel):
    """GET /api/health."""

    status: str
    auth_required: bool
    models_loaded: list[str]
    jobs_running: int
    url_fetch: bool  # whether the .../urls endpoints are enabled


class EmbedderOut(BaseModel):
    """An embedder preset."""

    spec: str
    label: str
    supports_text: bool
    notes: str


class ModelsOut(BaseModel):
    """GET /api/models."""

    embedders: list[EmbedderOut]
    classifiers: list[str]


class TaskSummary(BaseModel):
    """One entry of GET /api/tasks."""

    name: str
    embedder: str
    classifier: str
    n_classes: int
    n_examples: int
    counts: dict[str, int]  # examples per class
    created: str
    revision: int


class ClassOut(BaseModel):
    """A class of a task."""

    name: str
    description: str  # text prompts, one per line
    count: int  # examples


class ExampleOut(BaseModel):
    """A labelled example image."""

    id: str
    label: str
    file: str
    source: str
    added: str
    tags: list[str]
    note: str


class TaskDetail(BaseModel):
    """A task's settings, classes and examples."""

    name: str
    embedder: str
    supports_text: bool
    classifier: str
    classifier_params: dict[str, Any]
    text_weight: float
    reject_below: float | None
    reject_below_zero_shot: float | None
    flip: bool
    revision: int
    can_undo: bool
    classes: list[ClassOut]
    examples: list[ExampleOut]


class ExamplePage(BaseModel):
    """GET .../examples: ``examples`` is the page starting at ``offset`` of ``total`` matches."""

    total: int
    offset: int
    examples: list[ExampleOut]


class FileError(BaseModel):
    """A file (or URL) that could not be used."""

    file: str | None
    error: str


class AddResult(BaseModel):
    """POST .../examples."""

    added: list[ExampleOut]
    errors: list[FileError]
    task: TaskDetail


class PredictionOut(BaseModel):
    """The result for one image."""

    file: str | None
    label: str | None  # None when rejected as unknown, or on error
    labels: list[str]  # every class the image matches (multi-label), ``label`` first
    top_label: str  # the most probable class, even if rejected
    scores: dict[str, float]  # probability per class, highest first
    confidence: float
    similarity: float
    rejected: bool
    error: str | None


class DuplicatePair(BaseModel):
    """Two examples whose embeddings are nearly the same."""

    a: str
    b: str
    similarity: float
    labels: list[str]
    same_label: bool


class ClassMetrics(BaseModel):
    """Per-class evaluation results."""

    support: int
    recall: float | None
    precision: float | None


class HeldOutPrediction(BaseModel):
    """An example's prediction by a classifier that didn't see it."""

    id: str
    true: str
    pred: str
    confidence: float
    similarity: float
    correct: bool


class EvalReport(BaseModel):
    """Cross-validation report."""

    method: str
    n_scored: int
    n_examples: int
    accuracy: float | None
    macro_f1: float | None
    top_k_accuracy: dict[str, float]  # "2", "3", "5" -> accuracy counting the top k guesses
    classes: list[str]
    confusion: list[list[int]]  # rows: true class, columns: predicted
    per_class: dict[str, ClassMetrics]
    mistakes: list[HeldOutPrediction]  # most confident first
    predictions: list[HeldOutPrediction]  # every scored example, in task order
    suggested_reject_below: float | None  # keeps 95% of correct predictions


class JobOut(BaseModel):
    """A background job."""

    id: str
    task: str
    kind: str
    status: str  # running | done | failed | cancelled
    stage: str
    done: int
    total: int
    result: EvalReport | None
    error: str | None
