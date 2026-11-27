"""A small client for a fewshot server's REST API, using only the standard library.

    from fewshot.client import Client

    fs = Client()  # $FEWSHOT_URL (default http://127.0.0.1:8000) and $FEWSHOT_TOKEN
    fs.add_examples("pets", "Beagle", ["a.jpg", "b.jpg"])
    for p in fs.predict("pets", ["new.jpg"]):
        print(p["file"], p["label"], p["confidence"])

Images are file paths, bytes, binary file objects, ``(filename, bytes)`` pairs or PIL images.
Every method returns the API's JSON, typed as the ``TypedDict`` of its schema (plain dicts at
runtime: a newer server's extra fields are kept, and code must ignore the ones it doesn't know).

Errors: every exception is a ``FewshotError``. An error response raises an ``ApiError`` subclass
by status (``NotFoundError``, ``ConflictError``, ``ValidationError``...) carrying the server's
stable ``code``, its ``detail`` and the whole problem ``body``; a server that can't be reached
raises ``APIConnectionError``, and one that doesn't answer in time ``APITimeoutError``. Redirects
aren't followed, so the token is never sent to another server and a POST is never repeated as a
GET. Background jobs (``start_evaluate``, ``start_compare``) are ``Job`` handles; one that fails
raises a ``JobError``.

Many images are sent in batches of at most ``BATCH`` images (the server takes at most 1000 files
per request) and at most half the server's request size limit, reading only one batch of files at
a time; the results are combined. The methods cover the common work (tasks, adding images, the
inbox, predicting, evaluating, comparing embedders); ``request`` reaches every other endpoint (see
the server's /docs).
"""

from __future__ import annotations

import http.client
import io
import json
import mimetypes
import os
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, TypedDict, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

type Image = str | Path | bytes | BinaryIO | tuple[str, bytes] | PILImage
BATCH = 256  # images per upload request, at most
BATCH_BYTES = 256 * 1024 * 1024  # bytes per upload request, at most, if the server doesn't say its limit
DEFAULT_URL = "http://127.0.0.1:8000"
URL_ENV, TOKEN_ENV = "FEWSHOT_URL", "FEWSHOT_TOKEN"


# ---- the API's types (they mirror fewshot.schemas; a test keeps them in sync) ----------
# pylint: disable=duplicate-code
class Health(TypedDict):
    """GET /api/health."""

    status: str
    version: str
    api_version: str
    auth_required: bool
    models_loaded: list[str]
    jobs_running: int
    jobs_cancelling: int
    url_fetch: bool
    max_request_bytes: int
    max_upload_bytes: int


class LastEvaluation(TypedDict):
    """The headline of a task's last evaluation."""

    at: str
    revision: int | None
    method: str | None
    accuracy: float | None
    macro_f1: float | None
    n_scored: int | None
    n_examples: int | None


class TaskSummary(TypedDict):
    """One entry of ``tasks()``."""

    name: str
    embedder: str
    classifier: str
    n_classes: int
    n_examples: int
    n_inbox: int
    counts: dict[str, int]
    created: str
    revision: int
    last_evaluation: LastEvaluation | None
    warnings: list[str]


class ClassInfo(TypedDict):
    """A class of a task."""

    name: str
    description: str
    count: int
    reject_below: float | None


class Example(TypedDict):
    """A labelled example image."""

    id: str
    label: str
    file: str
    source: str
    added: str
    tags: list[str]
    note: str


class InboxItem(TypedDict):
    """An unlabelled image waiting in a task's inbox."""

    id: str
    file: str
    source: str
    added: str


class TaskDetail(TypedDict):
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
    inbox_count: int
    can_undo: bool
    can_redo: bool
    undo_label: str
    undo_group: str | None
    redo_label: str
    classes: list[ClassInfo]
    examples: list[Example]


class ExamplePage(TypedDict):
    """One page of ``examples()``."""

    total: int
    offset: int
    examples: list[Example]


class FileError(TypedDict):
    """A file (or URL) that could not be used; ``file`` is None for a whole batch refused."""

    file: str | None
    error: str


class MovedExample(TypedDict):
    """An image that was an example of another class, moved by adding it."""

    id: str
    previous_label: str


class AddResult(TypedDict):
    """``add_examples()``: what was new, already there, moved, and couldn't be used."""

    added: list[Example]
    unchanged: list[Example]
    moved: list[MovedExample]
    errors: list[FileError]
    created_classes: list[str]
    task: TaskDetail


class InboxAdded(TypedDict):
    """``add_to_inbox()``."""

    added: list[InboxItem]
    unchanged: list[InboxItem]
    errors: list[FileError]
    task: TaskDetail


class Evidence(TypedDict):
    """A labelled example similar to a predicted image."""

    id: str
    label: str
    similarity: float


class Prediction(TypedDict):
    """The result for one image; with ``error`` set, the scores are None and the lists empty."""

    file: str | None
    label: str | None
    labels: list[str]
    top_label: str | None
    scores: dict[str, float]
    confidence: float | None
    similarity: float | None
    rejected: bool
    evidence: list[Evidence]
    error: str | None
    revision: int | None


class InboxEntry(InboxItem):
    """An inbox image with the model's suggestion (None while the task has no classes)."""

    prediction: Prediction | None
    margin: float | None


class InboxPage(TypedDict):
    """One page of ``inbox()``."""

    total: int
    offset: int
    items: list[InboxEntry]


class DuplicatePair(TypedDict):
    """Two examples whose embeddings are nearly the same."""

    a: str
    b: str
    similarity: float
    labels: list[str]
    sources: list[str]
    same_label: bool


class ClassMetrics(TypedDict):
    """Per-class evaluation results."""

    support: int
    recall: float | None
    precision: float | None


class HeldOutPrediction(TypedDict):
    """An example's prediction by a classifier that didn't see it."""

    id: str
    source: str
    true: str
    pred: str
    confidence: float
    similarity: float
    correct: bool


class ReliabilityBin(TypedDict):
    """Predictions with about this confidence, and how often they were right."""

    confidence: float
    accuracy: float
    count: int


class Calibration(TypedDict):
    """How well confidence matches accuracy, and the temperature that would make it match."""

    temperature: float
    ece: float
    ece_calibrated: float
    temperature_helps: bool
    reliability: list[ReliabilityBin]
    reliability_calibrated: list[ReliabilityBin]


class EvalReport(TypedDict):
    """A cross-validation report."""

    revision: int
    method: str
    n_scored: int
    n_examples: int
    accuracy: float | None
    macro_f1: float | None
    top_k_accuracy: dict[str, float]
    classes: list[str]
    confusion: list[list[int]]
    per_class: dict[str, ClassMetrics]
    mistakes: list[HeldOutPrediction]
    predictions: list[HeldOutPrediction]
    suggested_reject_below: float | None
    suggested_class_thresholds: dict[str, float]
    calibration: Calibration | None


class CompareResult(TypedDict):
    """One embedder's cross-validation on the task, or why it couldn't be used."""

    embedder: str
    accuracy: float | None
    macro_f1: float | None
    top_k_accuracy: dict[str, float]
    n_scored: int
    seconds: float | None
    method: str | None
    error: str | None


class CompareReport(TypedDict):
    """Several embedders evaluated on the same labelled examples."""

    revision: int
    current: str
    n_examples: int
    results: list[CompareResult]
    best: str | None


class JobInfo(TypedDict):
    """A background job, as the server reports it (``Job.info``)."""

    id: str
    task: str
    kind: str
    status: str  # running | cancelling | done | failed | cancelled
    stage: str
    done: int
    total: int
    result: Any  # EvalReport or CompareReport, once done
    error: str | None
    problem: dict[str, Any] | None
    expired: bool


# pylint: enable=duplicate-code
# ---- errors ----------------------------------------------------------------------------
class FewshotError(Exception):
    """Any error of this client."""


class ApiError(FewshotError):
    """An error response from the server: its ``status`` (also ``status_code``), stable ``code``
    (see the README's code table), one-line ``detail``, the whole problem ``body``, and its other
    members as ``fields`` (``task``, ``label``, ``similar_to``...). ``str()`` is ``code: detail``."""

    def __init__(self, status: int, detail: str, code: str | None = None, body: Any = None) -> None:
        self.status = self.status_code = status
        self.code = code or f"HTTP_{status}"
        self.detail = detail
        self.body = body
        known = {"type", "title", "status", "detail", "instance", "code"}
        self.fields: dict[str, Any] = ({k: v for k, v in body.items() if k not in known}
                                       if isinstance(body, dict) else {})
        super().__init__(f"{self.code}: {detail}")


class BadRequestError(ApiError):
    """400: the request is well-formed but can't be done as asked."""


class AuthenticationError(ApiError):
    """401: not signed in, or a wrong token (403 too: see ``PermissionDeniedError``)."""


class PermissionDeniedError(AuthenticationError):
    """403: refused by the server's settings (a model or URL fetching it doesn't allow, a cross-site request)."""


class NotFoundError(ApiError):
    """404: no such task, class, example, job or route; ``fields["similar_to"]`` has the closest names."""


class ConflictError(ApiError):
    """409: conflicts with the task as it is (a name taken, nothing to undo)."""


class PreconditionFailedError(ApiError):
    """412: ``if_match`` named a revision the task has moved on from; ``fields["revision"]`` is the current one."""


class ValidationError(ApiError):
    """422: the request doesn't validate; ``errors`` lists every problem (``pointer`` or ``parameter``, ``detail``)."""

    @property
    def errors(self) -> list[dict[str, Any]]:
        """Each problem the server found."""
        found = self.body.get("errors") if isinstance(self.body, dict) else None
        return found if isinstance(found, list) else []


class RateLimitError(ApiError):
    """429: too many wrong tokens, or too many jobs running; ``retry_after`` is the seconds to wait, if said."""

    retry_after: float | None = None


class ServerError(ApiError):
    """5xx: the server failed (see its log)."""


_BY_STATUS: dict[int, type[ApiError]] = {400: BadRequestError, 401: AuthenticationError, 403: PermissionDeniedError,
                                        404: NotFoundError, 409: ConflictError, 412: PreconditionFailedError,
                                        422: ValidationError, 429: RateLimitError}


class APIConnectionError(FewshotError):
    """The server couldn't be reached (or dropped the connection); ``url`` is where it was looked for."""

    def __init__(self, message: str, url: str) -> None:
        super().__init__(message)
        self.url = url


class APITimeoutError(FewshotError):
    """No answer in time: from the server (``url``), or from a job still running (``job_id``)."""

    def __init__(self, message: str, url: str, job_id: str | None = None) -> None:
        super().__init__(message)
        self.url, self.job_id = url, job_id


class JobError(FewshotError):
    """A background job that didn't give its result; ``job`` is its last state."""

    def __init__(self, message: str, job: JobInfo) -> None:
        super().__init__(message)
        self.job = job
        self.job_id = job["id"]


class JobFailedError(JobError):
    """The job failed; ``code`` and ``detail`` say why (as an error response would)."""

    def __init__(self, job: JobInfo) -> None:
        problem = job.get("problem") or {}
        self.code: str = problem.get("code") or "JOB_FAILED"
        self.detail: str = job.get("error") or "unknown error"
        super().__init__(f"{self.code}: job {job['id']} failed: {self.detail}", job)


class JobCancelledError(JobError):
    """The job was cancelled."""


class JobResultExpiredError(JobError):
    """The job finished, but the server no longer keeps its result (it keeps the most recent few)."""


def _api_error(status: int, raw: bytes, headers: Any = None) -> ApiError:
    """An error response as the ``ApiError`` subclass of its status, from its problem body (or
    the body as text, from a proxy say)."""
    try:
        body = json.loads(raw)
    except ValueError:
        body = None
    detail: Any = body.get("detail") if isinstance(body, dict) else None
    if detail is None:
        detail = raw.decode(errors="replace")
    elif not isinstance(detail, str):  # an older server's 422 list
        detail = json.dumps(detail)
    code = body.get("code") if isinstance(body, dict) else None
    kind = _BY_STATUS.get(status, ServerError if status >= 500 else ApiError)
    err = kind(status, detail, code if isinstance(code, str) else None, body)
    if isinstance(err, RateLimitError) and headers is not None:
        try:
            err.retry_after = float(headers.get("Retry-After"))
        except (TypeError, ValueError):
            err.retry_after = None
    return err


# ---- images ----------------------------------------------------------------------------
def _is_pil(img: Any) -> bool:
    """Whether ``img`` is a PIL image (PIL is imported only if something looks like one)."""
    if not (hasattr(img, "save") and hasattr(img, "size") and type(img).__module__.startswith("PIL.")):
        return False
    from PIL import Image as PIL  # pylint: disable=import-outside-toplevel  # only for PIL images

    return isinstance(img, PIL.Image)


def _check(images: Sequence[Any], start: int = 0) -> None:
    """``TypeError`` for the first of ``images`` that isn't one of the kinds accepted."""
    for i, img in enumerate(images, start):
        ok = (isinstance(img, str | Path | bytes | bytearray | memoryview) or hasattr(img, "read")
              or (isinstance(img, tuple) and len(img) == 2 and isinstance(img[1], bytes | bytearray)) or _is_pil(img))
        if not ok:
            raise TypeError(f"image {i}: expected a path, bytes, file object, (name, bytes) or PIL.Image, "
                            f"got {type(img).__name__}")


def _size(img: Any) -> int:
    """About how many bytes ``img`` will add to a request (0 for a file that can't be read: it fails when sent)."""
    if isinstance(img, tuple):
        return len(img[1])
    if isinstance(img, bytes | bytearray | memoryview):
        return len(img)
    if isinstance(img, str | Path):
        try:
            return Path(img).stat().st_size
        except OSError:
            return 0
    if _is_pil(img):
        w, h = img.size
        return int(w * h * 3)
    return 0  # a file object: not known without reading it


def _batches(images: Iterable[Any], max_bytes: int) -> Iterator[list[Any]]:
    """``images`` in lists of at most ``BATCH`` images and ``max_bytes`` (an image larger than that
    goes alone, for the server to judge). A list of images is checked whole before the first batch."""
    if isinstance(images, Sequence):
        _check(images)
    batch: list[Any] = []
    size = 0
    for img in images:
        n = _size(img)
        if batch and (len(batch) >= BATCH or size + n > max_bytes):
            yield batch
            batch, size = [], 0
        batch.append(img)
        size += n
    if batch:
        yield batch


def _read(batch: Sequence[Any], start: int = 0) -> tuple[list[tuple[str, bytes]], list[tuple[int, str, str]]]:
    """The images of ``batch`` that can be read, as ``(name, bytes)``, and ``(position, file, reason)``
    for each file that can't be. Images without a file name are named ``image-<i>`` (``i`` counted from ``start``)."""
    _check(batch, start)
    loaded: list[tuple[str, bytes]] = []
    unread: list[tuple[int, str, str]] = []
    for i, img in enumerate(batch):
        n = start + i
        if isinstance(img, tuple):
            loaded.append((img[0], bytes(img[1])))
        elif isinstance(img, bytes | bytearray | memoryview):
            loaded.append((f"image-{n}", bytes(img)))
        elif isinstance(img, str | Path):
            try:
                loaded.append((Path(img).name, Path(img).read_bytes()))
            except OSError as e:
                unread.append((i, _text(str(img)), e.strerror or str(e)))
        elif _is_pil(img):
            buf = io.BytesIO()
            img.save(buf, "PNG")
            loaded.append((f"image-{n}.png", buf.getvalue()))
        else:  # a file object
            name = Path(str(getattr(img, "name", "") or "")).name or f"image-{n}"
            try:
                loaded.append((name, img.read()))
            except OSError as e:
                unread.append((i, name, e.strerror or str(e)))
    return loaded, unread


def _failed(file: str, error: str) -> Prediction:
    """The prediction of an image that couldn't be read here: every key, the scores empty."""
    return {"file": file, "label": None, "labels": [], "top_label": None, "scores": {}, "confidence": None,
            "similarity": None, "rejected": False, "evidence": [], "error": error, "revision": None}


class _NoRedirects(HTTPRedirectHandler):
    """Refuses every redirect: urllib would send the Authorization header on to whatever server
    it names, and turn a POST into a GET without its body."""

    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any,  # pylint: disable=too-many-arguments,too-many-positional-arguments  # urllib's signature
                         newurl: str) -> None:
        fp.close()
        raise ApiError(code, f"The server redirected to {newurl}; give that URL instead (redirects aren't "
                             "followed, so the token isn't sent elsewhere)", "REDIRECTED")


_OPENER = build_opener(_NoRedirects)


# ---- jobs ------------------------------------------------------------------------------
class Job[R]:
    """A background job on the server (an evaluation or a comparison). ``info`` is its state when
    last fetched; ``wait()`` polls until it ends and returns its result."""

    ACTIVE = frozenset({"running", "cancelling"})

    def __init__(self, client: Client, info: JobInfo) -> None:
        self.client = client
        self.info = info

    @property
    def id(self) -> str:
        """The job's id (GET /api/jobs/{id})."""
        return self.info["id"]

    def refresh(self) -> JobInfo:
        """Fetch the job's state now."""
        self.info = self.client.request("GET", f"/jobs/{_seg(self.id)}")
        return self.info

    def status(self) -> str:
        """The job's status now: running, cancelling, done, failed or cancelled."""
        return self.refresh()["status"]

    def cancel(self) -> JobInfo:
        """Ask the job to stop (it does at its next progress step)."""
        self.info = self.client.request("DELETE", f"/jobs/{_seg(self.id)}")
        return self.info

    def wait(self, timeout: float | None = None, on_progress: Callable[[JobInfo], None] | None = None,
             poll: float = 1.0) -> R:
        """Poll every ``poll`` seconds until the job ends, calling ``on_progress(info)`` each time
        (``info["stage"]``, ``["done"]`` and ``["total"]`` say how far it is), and return its result
        (see ``result``). After ``timeout`` seconds raises ``APITimeoutError``, leaving the job running."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.info["status"] in self.ACTIVE:
            if on_progress:
                on_progress(self.info)
            if deadline is not None and time.monotonic() >= deadline:
                raise APITimeoutError(f"Job {self.id} is still running after {timeout:g} s; it goes on on the server: "
                                      f"Client.job({self.id!r}).wait() waits again, .cancel() stops it",
                                      self.client.url, self.id)
            time.sleep(poll if deadline is None else max(0.0, min(poll, deadline - time.monotonic())))
            self.refresh()
        return self.result()

    def result(self) -> R:
        """The job's result, as last fetched: ``JobFailedError`` if it failed, ``JobCancelledError``
        if it was cancelled, ``JobResultExpiredError`` if the server dropped it, ``JobError`` if
        it is still running."""
        status = self.info["status"]
        if status == "done":
            if self.info.get("result") is None:
                raise JobResultExpiredError(f"Job {self.id} finished, but the server no longer keeps its result (it "
                                            "keeps only the most recent ones); start it again", self.info)
            return cast(R, self.info["result"])
        if status == "failed":
            raise JobFailedError(self.info)
        if status == "cancelled":
            raise JobCancelledError(f"Job {self.id} was cancelled", self.info)
        raise JobError(f"Job {self.id} is still {status}; wait() for it", self.info)


# ---- the client ------------------------------------------------------------------------
class Client:  # pylint: disable=too-many-public-methods  # one method per common operation
    """Calls the API of the fewshot server at ``url`` (default ``$FEWSHOT_URL``, else
    http://127.0.0.1:8000), sending ``token`` (default ``$FEWSHOT_TOKEN``) if the server needs one."""

    def __init__(self, url: str | None = None, token: str | None = None, timeout: float = 300) -> None:
        self.url = (url or os.environ.get(URL_ENV) or DEFAULT_URL).rstrip("/")
        self.token = token if token is not None else os.environ.get(TOKEN_ENV) or None
        self.timeout = timeout
        self._batch_bytes: int | None = None  # half the server's request limit, once asked

    # ---- transport -----------------------------------------------------
    def request(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches  # keyword-only body options
            self, method: str, path: str, *, params: dict[str, Any] | None = None, json_body: Any = None,
            fields: dict[str, str] | None = None, files: Sequence[Any] = (),
            headers: dict[str, str] | None = None) -> Any:
        """Send a request to ``path`` (under ``/api``) and return the decoded JSON (None if empty).

        ``params`` become the query string (None values are left out); ``json_body`` is sent as JSON;
        ``fields`` and ``files`` as a multipart form, the files under the name ``files`` (a file
        that can't be read raises ``FewshotError``, and nothing is sent); ``headers`` are added.
        """
        query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
        sent = {"Accept": "application/json, application/problem+json", **(headers or {})}
        if self.token:
            sent["Authorization"] = f"Bearer {self.token}"
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode()
            sent["Content-Type"] = "application/json"
        elif fields is not None or files:
            loaded, unread = _read(files)
            if unread:
                raise FewshotError(f"Can't read {unread[0][1]}: {unread[0][2]}")
            data, sent["Content-Type"] = _multipart(fields or {}, loaded)
        req = Request(f"{self.url}/api{path}{'?' + query if query else ''}", data=data, headers=sent, method=method)
        try:
            with _OPENER.open(req, timeout=self.timeout) as r:
                status, body = r.status, r.read()
        except HTTPError as e:
            raise _api_error(e.code, e.read(), e.headers) from None
        except TimeoutError as e:
            raise APITimeoutError(f"{self.url} didn't answer {method} {path} within {self.timeout:g} s "
                                  "(Client(timeout=...) waits longer)", self.url) from e
        except (URLError, OSError, http.client.HTTPException) as e:
            reason = getattr(e, "reason", None) or e
            if isinstance(reason, TimeoutError):
                raise APITimeoutError(f"{self.url} didn't answer {method} {path} within {self.timeout:g} s "
                                      "(Client(timeout=...) waits longer)", self.url) from e
            hint = (" (the request may be larger than the server accepts)" if data and len(data) > 1024 * 1024 else
                    f"; is the server running? Start one with `fewshot serve`, or set {URL_ENV} to its address")
            raise APIConnectionError(f"Connection to {self.url} failed: {reason}{hint}", self.url) from None
        try:
            return json.loads(body) if body else None
        except ValueError:  # an HTML page from a proxy, say
            start = body[:200].decode(errors="replace")
            raise ApiError(status, f"The response from {self.url} isn't JSON: {start!r}", "NOT_JSON") from None

    def batch_bytes(self) -> int:
        """The most bytes of images sent per request: half the server's request size limit (as its
        health check reports it), or ``BATCH_BYTES`` if it doesn't say."""
        if self._batch_bytes is None:
            limit = self.health().get("max_request_bytes")
            self._batch_bytes = limit // 2 if isinstance(limit, int) and limit > 0 else BATCH_BYTES
        return self._batch_bytes

    # ---- tasks ---------------------------------------------------------
    def health(self) -> Health:
        """Server status and version: loaded models, running jobs, whether a token is required."""
        return cast(Health, self.request("GET", "/health"))

    def tasks(self) -> list[TaskSummary]:
        """A summary of every task."""
        return cast(list[TaskSummary], self.request("GET", "/tasks"))

    def task(self, name: str) -> TaskDetail:
        """One task's details: classes, examples, settings, revision."""
        return cast(TaskDetail, self.request("GET", f"/tasks/{_seg(name)}"))

    def create_task(self, name: str, **settings: Any) -> TaskDetail:
        """Create a task; ``settings`` are the other fields of the create body (``embedder``,
        ``classifier``). Raises ``ConflictError`` if the name is taken."""
        return cast(TaskDetail, self.request("POST", "/tasks", json_body={"name": name, **settings}))

    def delete_task(self, name: str, *, if_match: int | None = None) -> None:
        """Delete a task and its images (only if it is still at revision ``if_match``, if given)."""
        self.request("DELETE", f"/tasks/{_seg(name)}", headers=_if_match(if_match))

    # ---- data ----------------------------------------------------------
    def add_examples(self, task: str, label: str, images: Iterable[Image], *, new_class: bool = False,
                     if_match: int | None = None) -> AddResult:
        """Add labelled examples, creating the class if needed: a new name close to an existing
        class's (``beagle`` for ``Beagle``) raises ``ValidationError`` (CLASS_NAME_NEAR_MISS)
        unless ``new_class``. Images are deduplicated by content, so adding them again is safe:
        ``added`` lists the new examples, ``unchanged`` those already there, ``moved`` those moved
        from another class. Images that can't be added are listed in ``errors``, with the reason
        for a whole batch the server refused (``file`` None); raises ``BadRequestError`` if no
        image at all could be added. ``if_match``: only if the task is still at that revision."""
        return cast(AddResult, self._upload(f"/tasks/{_seg(task)}/examples", images,
                                            {"label": label, **({"new_class": "true"} if new_class else {})},
                                            if_match))

    def add_to_inbox(self, task: str, images: Iterable[Image], *, if_match: int | None = None) -> InboxAdded:
        """Add unlabelled images to the task's inbox; reported as by ``add_examples``
        (``unchanged``: waiting already), and ``BadRequestError`` is raised if no image could be
        added (e.g. all are examples already)."""
        return cast(InboxAdded, self._upload(f"/tasks/{_seg(task)}/inbox", images, {}, if_match))

    def _upload(self, path: str, images: Iterable[Image], fields: dict[str, str],
                if_match: int | None) -> dict[str, Any]:
        """POST ``images`` a batch at a time, all one undo step (``if_match`` goes with the first);
        ``added``, ``unchanged``, ``moved``, ``created_classes`` and ``errors`` are combined,
        ``task`` is the last. A file that can't be read is listed in ``errors``. A batch refused
        with 400 or 413 (nothing in it added) adds its reason to ``errors``; if every batch was
        refused, the first refusal is raised (``BadRequestError`` if no file could be read)."""
        out: dict[str, Any] = {"added": [], "unchanged": [], "errors": []}
        refused: ApiError | None = None
        fields = {**fields, "undo_group": uuid.uuid4().hex}  # the batches are one undo step
        start = 0
        for batch in _batches(images, self.batch_bytes()):
            loaded, unread = _read(batch, start)
            start += len(batch)
            out["errors"] += [{"file": name, "error": why} for _, name, why in unread]
            if not loaded:
                continue
            try:
                r = self.request("POST", path, fields=fields, files=loaded, headers=_if_match(if_match))
            except ApiError as e:
                if e.status not in {400, 413}:
                    raise
                refused = refused or e
                out["errors"] += [{"file": None, "error": e.detail}]
                continue
            if_match = None  # the first batch changed the revision
            _merge(out, r)
        if "task" not in out:
            if refused is None and out["errors"]:
                first = out["errors"][0]
                refused = BadRequestError(400, f"No image could be read: {first['file']}: {first['error']}",
                                          "NO_IMAGES_ADDED")
            raise refused or ValueError("No images given")
        return out

    def examples(self, task: str, *, label: str | None = None, tag: str | None = None, offset: int = 0,
                 limit: int | None = None) -> ExamplePage:
        """One page of the task's examples (``limit`` from ``offset``; the server's default page is
        100, its largest 1000), optionally only of class ``label`` and/or with ``tag``."""
        params = {"label": label, "tag": tag, "offset": offset or None, "limit": limit}
        return cast(ExamplePage, self.request("GET", f"/tasks/{_seg(task)}/examples", params=params))

    def iter_examples(self, task: str, label: str | None = None, tag: str | None = None,
                      page_size: int = 1000) -> Iterator[Example]:
        """Every example of the task (of ``label`` and/or with ``tag``), a page at a time."""
        offset = 0
        while True:
            page = self.examples(task, label=label, tag=tag, offset=offset, limit=page_size)
            yield from page["examples"]
            offset += len(page["examples"])
            if not page["examples"] or offset >= page["total"]:
                return

    def inbox(self, task: str, sort: str = "uncertain", *, offset: int = 0, limit: int | None = None) -> InboxPage:
        """One page of the inbox with the model's suggestion for each image (``sort``: uncertain,
        confident, novel, added): ``limit`` items from ``offset`` (the server's default page is
        100, its largest 1000); ``total`` counts them all, and ``iter_inbox`` walks every page."""
        params = {"sort": sort, "offset": offset or None, "limit": limit}
        return cast(InboxPage, self.request("GET", f"/tasks/{_seg(task)}/inbox", params=params))

    def iter_inbox(self, task: str, sort: str = "uncertain", page_size: int = 1000) -> Iterator[InboxEntry]:
        """Every inbox image with its suggestion, in ``sort`` order, a page at a time. Labelling
        images while iterating shifts the pages: collect the labels first, then ``accept_inbox``."""
        offset = 0
        while True:
            page = self.inbox(task, sort, offset=offset, limit=page_size)
            yield from page["items"]
            offset += len(page["items"])
            if not page["items"] or offset >= page["total"]:
                return

    def accept_inbox(self, task: str, labels: dict[str, str], *, new_class: bool = False,
                     if_match: int | None = None) -> TaskDetail:
        """Label inbox images (``{id: class}``), moving them to the task's examples; a new class
        name close to an existing class's needs ``new_class`` (as for ``add_examples``)."""
        body: dict[str, Any] = {"labels": labels, **({"new_class": True} if new_class else {})}
        return cast(TaskDetail, self.request("POST", f"/tasks/{_seg(task)}/inbox/accept", json_body=body,
                                             headers=_if_match(if_match)))

    # ---- inference -----------------------------------------------------
    def predict(self, task: str, images: Iterable[Image], *, top_k: int | None = None, evidence: int = 0,
                revision: int | None = None) -> list[Prediction]:
        """Classify images; each result has ``label``, ``confidence``, ``scores``, ``revision``, in
        the order given. An image that can't be read (here or by the server) has ``error`` set,
        ``label``, ``top_label``, ``confidence`` and ``similarity`` None and ``scores`` empty.

        ``revision`` predicts with an earlier version of the task that the server still keeps.
        """
        params = {"top_k": top_k, "evidence": evidence or None, "revision": revision}
        out: list[Prediction] = []
        start = 0
        for batch in _batches(images, self.batch_bytes()):
            loaded, unread = _read(batch, start)
            start += len(batch)
            preds = iter(self.request("POST", f"/tasks/{_seg(task)}/predict", params=params, files=loaded)
                         if loaded else [])
            errors = {i: _failed(name, why) for i, name, why in unread}
            out += [errors[i] if i in errors else next(preds) for i in range(len(batch))]
        return out

    def job(self, job_id: str) -> Job[Any]:
        """The background job ``job_id``, as it is now (``NotFoundError`` if the server doesn't know it)."""
        return Job(self, self.request("GET", f"/jobs/{_seg(job_id)}"))

    def start_evaluate(self, task: str, folds: int | None = None) -> Job[EvalReport]:
        """Start cross-validating the task in the background (leave-one-out unless ``folds``);
        ``.wait()`` returns the report. A task that can't be evaluated raises at once (``BadRequestError``)."""
        return Job(self, self.request("POST", f"/tasks/{_seg(task)}/evaluate/jobs", params={"folds": folds}))

    def start_compare(self, task: str, embedders: Sequence[str], folds: int | None = None) -> Job[CompareReport]:
        """Start cross-validating the task with each of ``embedders`` in the background; ``.wait()``
        returns their comparison (``best`` is the most accurate)."""
        body: dict[str, Any] = {"embedders": list(embedders), **({"folds": folds} if folds is not None else {})}
        return Job(self, self.request("POST", f"/tasks/{_seg(task)}/compare/jobs", json_body=body))

    def evaluate(self, task: str, folds: int | None = None, *, timeout: float | None = None, poll: float = 1.0,
                 on_progress: Callable[[JobInfo], None] | None = None) -> EvalReport:
        """Cross-validate the task and return the report (accuracy, per-class metrics, mistakes...),
        as a background job (``start_evaluate``), so a long evaluation isn't cut off by the request
        timeout; see ``Job.wait`` for ``timeout``, ``poll`` and ``on_progress`` and the errors."""
        return _waited(self.start_evaluate(task, folds), timeout, on_progress, poll)

    def compare(self, task: str, embedders: Sequence[str], folds: int | None = None, *,  # pylint: disable=too-many-arguments
                timeout: float | None = None, poll: float = 1.0,
                on_progress: Callable[[JobInfo], None] | None = None) -> CompareReport:
        """Compare ``embedders`` on the task's own labelled examples and return the report, as
        ``evaluate`` does."""
        return _waited(self.start_compare(task, embedders, folds), timeout, on_progress, poll)


def _waited[R](job: Job[R], timeout: float | None, on_progress: Callable[[JobInfo], None] | None, poll: float) -> R:
    """``job.wait(...)``; Ctrl-C says how to come back to the job, which goes on on the server."""
    try:
        return job.wait(timeout, on_progress, poll)
    except KeyboardInterrupt:
        raise KeyboardInterrupt(f"Stopped waiting for job {job.id}, which goes on on the server: "
                                f"Client.job({job.id!r}).wait() waits again, .cancel() stops it") from None


def _merge(out: dict[str, Any], batch: dict[str, Any]) -> None:
    """Add one batch's upload result to ``out``: lists are joined (each created class once),
    anything else (``task``) replaced."""
    for key, value in batch.items():
        if isinstance(value, list):
            prev = out.get(key, [])
            out[key] = prev + [v for v in value if key != "created_classes" or v not in prev]
        else:
            out[key] = value


def _if_match(revision: int | None) -> dict[str, str]:
    """The If-Match header naming ``revision`` (none if None)."""
    return {} if revision is None else {"If-Match": f'"{revision}"'}


def _seg(s: str) -> str:
    """``s`` quoted as one path segment."""
    return quote(s, safe="")


def _text(s: str) -> str:
    """``s`` as valid Unicode: a file name that isn't UTF-8 (its bytes escaped as lone surrogates)
    has each such byte replaced by U+FFFD."""
    try:
        return s.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    except UnicodeEncodeError:  # a surrogate not from a file name's bytes
        return s.encode("utf-8", "replace").decode("utf-8")


def _multipart(fields: dict[str, str], files: Sequence[tuple[str, bytes]]) -> tuple[bytes, str]:
    """A multipart/form-data body with ``fields`` and each ``(name, bytes)`` image as a ``files``
    part, and its content type."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{_text(v)}\r\n'.encode())
    for name, data in files:
        name = _text(name)
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        safe = name.replace('"', "%22").replace("\r", "").replace("\n", "")
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{safe}"\r\n'
                     f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
