"""Request and response models of the REST API, so /docs shows (and clients can generate) exact types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import MAX_NOTE
from .text import MAX_DESCRIPTION

MAX_URLS = 100  # per request


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


class LastEvaluation(BaseModel):
    """The headline of a task's last evaluation."""

    at: str
    revision: int | None
    method: str | None
    accuracy: float | None
    macro_f1: float | None
    n_scored: int | None
    n_examples: int | None


class TaskSummary(BaseModel):
    """One entry of GET /api/tasks."""

    name: str
    embedder: str
    classifier: str
    n_classes: int
    n_examples: int
    n_inbox: int
    counts: dict[str, int]  # examples per class
    created: str
    revision: int
    last_evaluation: LastEvaluation | None  # the last evaluation's headline, if any
    warnings: list[str]  # problems worth fixing


class ClassOut(BaseModel):
    """A class of a task."""

    name: str
    description: str  # text prompts, one per line
    count: int  # examples
    reject_below: float | None  # the class's own reject threshold (None: the task's)


class ExampleOut(BaseModel):
    """A labelled example image."""

    id: str
    label: str
    file: str
    source: str
    added: str
    tags: list[str]
    note: str


class InboxItemOut(BaseModel):
    """An unlabelled image waiting in a task's inbox."""

    id: str
    file: str
    source: str
    added: str


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
    names_as_text: bool
    temperature: float | None
    revision: int
    inbox_count: int  # images waiting in the inbox (listed by GET .../inbox)
    can_undo: bool
    can_redo: bool
    undo_label: str  # what undo would take back, e.g. "delete 3 examples" ("" if unknown)
    redo_label: str
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


class Evidence(BaseModel):
    """A labelled example similar to a predicted image."""

    id: str
    label: str
    similarity: float


class PredictionOut(BaseModel):
    """The result for one image."""

    file: str | None = None  # the uploaded file's name or URL (not given for inbox suggestions)
    label: str | None  # None when rejected as unknown, or on error
    labels: list[str]  # every class the image matches (multi-label), ``label`` first unless rejected
    top_label: str  # the most probable class, even if rejected
    scores: dict[str, float]  # probability per class, highest first
    confidence: float
    similarity: float
    rejected: bool
    evidence: list[Evidence]  # the most similar labelled examples, when asked for (?evidence=N)
    error: str | None
    revision: int | None = None  # the task version that made the prediction


class InboxEntry(InboxItemOut):
    """An inbox image with the model's suggestion (absent while the task has no classes)."""

    prediction: PredictionOut | None = None
    margin: float | None = None  # top probability minus the second: small means unsure


class InboxPage(BaseModel):
    """GET .../inbox: ``items`` is the page starting at ``offset`` of ``total``, in the sort asked for."""

    total: int
    offset: int
    items: list[InboxEntry]


class InboxAdded(BaseModel):
    """POST .../inbox."""

    added: list[InboxItemOut]
    errors: list[FileError]
    task: TaskDetail


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


class ReliabilityBin(BaseModel):
    """Predictions with about this confidence, and how often they were right."""

    confidence: float
    accuracy: float
    count: int


class Calibration(BaseModel):
    """How well confidence matches accuracy, and the temperature that would make it match."""

    temperature: float  # the setting to use (absolute)
    ece: float  # expected calibration error now
    ece_calibrated: float  # ... with ``temperature``
    reliability: list[ReliabilityBin]
    reliability_calibrated: list[ReliabilityBin]


class EvalReport(BaseModel):
    """Cross-validation report."""

    revision: int  # of the task when evaluated
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
    suggested_class_thresholds: dict[str, float]  # the same per class, for classes with 5+ correct
    calibration: Calibration | None  # None with fewer than 50 scored examples


class CompareResult(BaseModel):
    """One embedder's cross-validation on the task (or why it couldn't be used)."""

    embedder: str
    accuracy: float | None = None
    macro_f1: float | None = None
    top_k_accuracy: dict[str, float] = {}
    n_scored: int = 0
    seconds: float | None = None  # embedding (cached images are fast) and evaluation
    method: str | None = None
    error: str | None = None


class CompareReport(BaseModel):
    """Several embedders evaluated on the same labelled examples."""

    revision: int
    current: str  # the task's embedder
    n_examples: int
    results: list[CompareResult]  # in the order asked for
    best: str | None  # the most accurate embedder


class JobOut(BaseModel):
    """A background job."""

    id: str
    task: str
    kind: str
    status: str  # running | done | failed | cancelled
    stage: str
    done: int
    total: int
    result: EvalReport | CompareReport | None
    error: str | None


# ---- request bodies -----------------------------------------------------
class _Body(BaseModel):
    """Request bodies refuse unknown fields, so a misspelt setting is an error, not a no-op."""

    model_config = ConfigDict(extra="forbid")


class Login(_Body):
    """Body of POST /api/login."""

    token: str


class TaskCreate(_Body):
    """Body of POST /api/tasks."""

    name: str
    embedder: str | None = None
    classifier: str = "gda"


class TaskSettings(_Body):
    """Body of PATCH /api/tasks/{name}; omitted fields are left unchanged."""

    embedder: str | None = None
    classifier: str | None = None
    classifier_params: dict[str, Any] | None = None
    text_weight: float | None = None
    reject_below: float | None = None
    clear_reject: bool = False
    reject_below_zero_shot: float | None = None
    clear_reject_zero_shot: bool = False
    flip: bool | None = None
    names_as_text: bool | None = None
    temperature: float | None = None
    clear_temperature: bool = False


class Rename(_Body):
    """Body of POST .../rename."""

    name: str


class Duplicate(_Body):
    """Body of POST .../duplicate."""

    name: str
    examples: bool = True  # copy the examples (else only settings and classes)
    inbox: bool = False


class ClassBody(_Body):
    """Body of POST .../classes. A description holds at most ``MAX_PROMPTS`` prompts (lines)."""

    name: str
    description: str | None = Field(None, max_length=MAX_DESCRIPTION)


class ClassPatch(_Body):
    """Body of PATCH .../classes/{label}: rename and/or set the description."""

    name: str | None = None
    description: str | None = Field(None, max_length=MAX_DESCRIPTION)


class Describe(_Body):
    """Body of POST .../describe."""

    template: str = Field("a photo of a {}", max_length=MAX_DESCRIPTION)
    overwrite: bool = False


class ClassThresholds(_Body):
    """Body of PUT .../class-thresholds."""

    thresholds: dict[str, float | None]  # class -> similarity (-1 to 1), or null to use the task's


class Urls(_Body):
    """Body of POST .../predict/urls."""

    urls: list[str] = Field(min_length=1, max_length=MAX_URLS)


class LabelledUrls(Urls):
    """Body of POST .../examples/urls."""

    label: str


class ExampleEdit(_Body):
    """Body of PATCH .../examples/{eid}: any of a new label, tags (replacing them) and a note."""

    label: str | None = None
    tags: list[str] | None = None
    note: str | None = Field(None, max_length=MAX_NOTE)


class ExampleIds(_Body):
    """Body of POST .../examples/delete."""

    ids: list[str] = Field(min_length=1)


class BulkEdit(ExampleIds):
    """Body of PATCH .../examples: a new label and/or tags to add or remove."""

    label: str | None = None
    add_tags: list[str] = Field(default_factory=list)
    remove_tags: list[str] = Field(default_factory=list)


class InboxLabels(_Body):
    """Body of POST .../inbox/accept: the class for each inbox image to label."""

    labels: dict[str, str] = Field(min_length=1)


class Compare(_Body):
    """Body of POST .../compare/jobs."""

    embedders: list[str]
    folds: int | None = None
