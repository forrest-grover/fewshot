"""REST API (FastAPI). Run with ``fewshot serve``; docs at /docs.

The built web UI (web/dist) is served at / when present. Server settings come from environment
variables, read when this module is imported (``fewshot serve`` sets them from its options):

``FEWSHOT_TOKEN``
    If set, every /api request except /api/login and /api/health needs this token, as a
    ``Bearer`` header or the cookie /api/login sets.
``FEWSHOT_ALLOW_ANY_MODEL=1``
    Let API callers choose any embedder spec. By default only presets, registered embedders and
    specs existing tasks already use are accepted, so a caller can't make the server download
    arbitrary models.
``FEWSHOT_WARM=1``
    At startup, load the models existing tasks use (in the background).
``FEWSHOT_MAX_UPLOAD_MB`` / ``FEWSHOT_MAX_REQUEST_MB``
    Largest image per upload (default 50) and largest request body (default 1024).
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .classifiers import CLASSIFIERS
from .embedders import (
    DEFAULT_SPEC,
    PRESETS,
    check_embedder,
    get_embedder,
    is_registered,
    loaded_embedders,
    max_models,
    spec_supports_text,
)
from .evaluate import evaluate_task
from .pipeline import Pipeline, Prediction
from .store import Task, Workspace, thumbnail

log = logging.getLogger(__name__)

TOKEN = os.environ.get("FEWSHOT_TOKEN") or None
ALLOW_ANY_MODEL = os.environ.get("FEWSHOT_ALLOW_ANY_MODEL") == "1"
WARM = os.environ.get("FEWSHOT_WARM") == "1"
MAX_REQUEST_BYTES = int(float(os.environ.get("FEWSHOT_MAX_REQUEST_MB", "1024")) * 1024 * 1024)

ws = Workspace()
pipe = Pipeline(ws)


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    if WARM:
        threading.Thread(target=warm_models, name="fewshot-warm", daemon=True).start()
    yield


app = FastAPI(title="fewshot", version="0.1.0", lifespan=_lifespan)


def warm_models() -> list[str]:
    """Load the embedders the most tasks use (up to the loaded-model limit), or the default one if
    there are no tasks, so the first request doesn't wait for a model; returns the specs loaded."""
    counts = Counter(t.config.embedder for t in ws.list_tasks())
    specs = [s for s, _ in counts.most_common(max_models())] or [DEFAULT_SPEC]
    loaded = []
    for spec in specs:
        try:
            get_embedder(spec)
        except Exception:  # pylint: disable=broad-exception-caught  # a bad model mustn't stop the server
            log.exception("Could not preload embedder %r", spec)
        else:
            loaded.append(spec)
    return loaded


# ---- access control -----------------------------------------------------
AUTH_COOKIE = "fewshot_token"
OPEN_PATHS = {"/api/login", "/api/health"}  # reachable without the token


def _token_ok(given: str | None) -> bool:
    return TOKEN is None or (given is not None and secrets.compare_digest(given.encode(), TOKEN.encode()))


@app.middleware("http")
async def _require_token(request: Request, call_next: Any) -> Response:
    path = request.url.path
    if TOKEN is not None and path.startswith("/api/") and path not in OPEN_PATHS:
        auth = request.headers.get("authorization", "")
        given = auth[7:] if auth.lower().startswith("bearer ") else request.cookies.get(AUTH_COOKIE)
        if not _token_ok(given):
            return JSONResponse({"detail": "Not signed in"}, 401, headers={"WWW-Authenticate": "Bearer"})
    response: Response = await call_next(request)
    return response


class _BodyTooLarge(Exception):
    pass


class LimitBody:  # pylint: disable=too-few-public-methods  # an ASGI middleware is just __call__
    """Refuse (413) request bodies over ``limit`` bytes, by Content-Length or while streaming, so an
    oversized upload isn't spooled to disk first."""

    def __init__(self, inner: ASGIApp, limit: int) -> None:
        self.inner, self.limit = inner, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.inner(scope, receive, send)
            return
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.limit:
            await self._refuse(scope, receive, send)
            return
        seen = 0
        started = False

        async def counted() -> Message:
            nonlocal seen
            msg = await receive()
            seen += len(msg.get("body", b""))
            if seen > self.limit:
                raise _BodyTooLarge
            return msg

        async def tracked(msg: Message) -> None:
            nonlocal started
            started = started or msg["type"] == "http.response.start"
            await send(msg)

        try:
            await self.inner(scope, counted, tracked)
        except _BodyTooLarge:
            if not started:
                await self._refuse(scope, receive, send)

    async def _refuse(self, scope: Scope, receive: Receive, send: Send) -> None:
        mb = self.limit / 1024 / 1024
        await JSONResponse({"detail": f"Request is larger than the {mb:g} MB limit"}, 413)(scope, receive, send)


app.add_middleware(LimitBody, limit=MAX_REQUEST_BYTES)  # added last, so it runs first


def _task(name: str) -> Task:
    try:
        return ws.get(name)
    except KeyError as e:
        raise _missing(e) from e


def _missing(e: KeyError) -> HTTPException:
    return HTTPException(404, str(e.args[0]))


def _bad(e: Exception) -> HTTPException:
    return HTTPException(400, str(e.args[0]) if e.args else str(e))


def _check_embedder(spec: str) -> None:
    """Load the embedder now so a bad spec fails here (400), not at predict time.

    Unless ``ALLOW_ANY_MODEL``, a spec that isn't a preset, registered, or already used by a task
    is refused (403) before anything is downloaded.
    """
    if not spec.strip():
        raise HTTPException(400, "Embedder spec is empty")
    if not (ALLOW_ANY_MODEL or spec in {p.spec for p in PRESETS} or is_registered(spec)
            or any(t.config.embedder == spec for t in ws.list_tasks())):
        raise HTTPException(403, f"Embedder {spec!r} is not a preset. To use other models, start the "
                                 "server with --allow-any-model (FEWSHOT_ALLOW_ANY_MODEL=1).")
    try:
        check_embedder(spec)
    except ValueError as e:
        raise _bad(e) from e


# Largest accepted upload per image; FEWSHOT_MAX_UPLOAD_MB overrides it.
MAX_UPLOAD_BYTES = int(float(os.environ.get("FEWSHOT_MAX_UPLOAD_MB", "50")) * 1024 * 1024)


def _read_upload(f: UploadFile) -> bytes | None:
    """The uploaded file's bytes, or None if it is over ``MAX_UPLOAD_BYTES`` (read no further)."""
    data = f.file.read(MAX_UPLOAD_BYTES + 1)
    return None if len(data) > MAX_UPLOAD_BYTES else data


def _too_large() -> str:
    return f"File is larger than the {MAX_UPLOAD_BYTES / 1024 / 1024:g} MB upload limit"


class _Body(BaseModel):
    """Request bodies refuse unknown fields, so a misspelt setting is an error, not a no-op."""

    model_config = ConfigDict(extra="forbid")


def _detail(t: Task) -> dict[str, Any]:
    c = t.config
    counts = t.summary()["counts"]
    return {
        "name": c.name,
        "embedder": c.embedder,
        "supports_text": spec_supports_text(c.embedder),
        "classifier": c.classifier,
        "classifier_params": c.classifier_params,
        "text_weight": c.text_weight,
        "reject_below": c.reject_below,
        "revision": c.revision,
        "classes": [{"name": k.name, "description": k.description, "count": counts.get(k.name, 0)} for k in c.classes],
        "examples": [e.model_dump() for e in c.examples],
    }


# ---- metadata -----------------------------------------------------------
@app.get("/api/health")
@app.get("/healthz", include_in_schema=False)
def health() -> dict[str, Any]:
    """Liveness check; also lists loaded models and running jobs. Needs no token."""
    with _jobs_lock:
        running = sum(j.status == "running" for j in _jobs.values())
    return {"status": "ok", "auth_required": TOKEN is not None, "models_loaded": loaded_embedders(),
            "jobs_running": running}


class Login(_Body):
    """Body of POST /api/login."""

    token: str


@app.post("/api/login", status_code=204)
def login(body: Login, response: Response) -> None:
    """Check the server token and remember it in an HttpOnly cookie (30 days)."""
    if not _token_ok(body.token):
        raise HTTPException(401, "Wrong token")
    if TOKEN is not None:
        response.set_cookie(AUTH_COOKIE, TOKEN, max_age=30 * 86400, httponly=True, samesite="strict")


@app.post("/api/logout", status_code=204)
def logout(response: Response) -> None:
    """Forget the login cookie."""
    response.delete_cookie(AUTH_COOKIE)


@app.get("/api/models")
def models() -> dict[str, Any]:
    """List embedder presets and classifier names."""
    return {
        "embedders": [p.__dict__ for p in PRESETS],
        "classifiers": sorted(CLASSIFIERS),
    }


# ---- tasks --------------------------------------------------------------
class TaskCreate(_Body):
    """Body of POST /api/tasks."""

    name: str
    embedder: str | None = None
    classifier: str = "prototype"


class TaskSettings(_Body):
    """Body of PATCH /api/tasks/{name}; omitted fields are left unchanged."""

    embedder: str | None = None
    classifier: str | None = None
    classifier_params: dict[str, Any] | None = None
    text_weight: float | None = None
    reject_below: float | None = None
    clear_reject: bool = False


@app.get("/api/tasks")
def list_tasks() -> list[dict[str, Any]]:
    """Summaries of all tasks."""
    return [t.summary() for t in ws.list_tasks()]


@app.post("/api/tasks", status_code=201)
def create_task(body: TaskCreate) -> dict[str, Any]:
    """Create a task, validating the embedder spec first."""
    kw = {"classifier": body.classifier}
    if body.embedder is not None:
        _check_embedder(body.embedder)
        kw["embedder"] = body.embedder
    try:
        return _detail(ws.create(body.name, **kw))
    except ValueError as e:
        raise _bad(e) from e


@app.get("/api/tasks/{name}")
def get_task(name: str) -> dict[str, Any]:
    """Full detail of one task."""
    return _detail(_task(name))


@app.patch("/api/tasks/{name}")
def update_task(name: str, body: TaskSettings) -> dict[str, Any]:
    """Change task settings; switching classifier resets its params unless new ones are given."""
    t = _task(name)
    kw = body.model_dump(exclude_none=True, exclude={"clear_reject"})
    if body.clear_reject:
        if body.reject_below is not None:
            raise HTTPException(400, "Give either reject_below or clear_reject, not both")
        kw["reject_below"] = None
    if "classifier" in kw and "classifier_params" not in kw and kw["classifier"] != t.config.classifier:
        kw["classifier_params"] = {}
    if "embedder" in kw and kw["embedder"] != t.config.embedder:
        _check_embedder(kw["embedder"])
    try:
        t.update_settings(**kw)
    except ValueError as e:
        raise _bad(e) from e
    return _detail(t)


@app.delete("/api/tasks/{name}", status_code=204)
def delete_task(name: str) -> None:
    """Delete a task and its images."""
    _task(name)
    ws.delete(name)


# ---- classes ------------------------------------------------------------
class ClassBody(_Body):
    """Body of POST .../classes."""

    name: str
    description: str | None = None


class ClassPatch(_Body):
    """Body of PATCH .../classes/{label}: rename and/or set the description."""

    name: str | None = None
    description: str | None = None


@app.post("/api/tasks/{name}/classes", status_code=201)
def add_class(name: str, body: ClassBody) -> dict[str, Any]:
    """Add a class, or update the description of an existing one."""
    t = _task(name)
    try:
        t.upsert_class(body.name, body.description)
    except ValueError as e:
        raise _bad(e) from e
    return _detail(t)


@app.patch("/api/tasks/{name}/classes/{label}")
def patch_class(name: str, label: str, body: ClassPatch) -> dict[str, Any]:
    """Rename a class (relabelling its examples) and/or set its description."""
    t = _task(name)
    if label not in t.class_names():
        raise HTTPException(404, f"No class {label!r}")
    if body.name is not None and body.name.strip() != label:
        new = body.name.strip()
        try:
            t.rename_class(label, new)
        except ValueError as e:
            raise _bad(e) from e
        label = new
    if body.description is not None:
        t.upsert_class(label, body.description)
    return _detail(t)


class Describe(_Body):
    """Body of POST .../describe."""

    template: str = "a photo of a {}"
    overwrite: bool = False


@app.post("/api/tasks/{name}/describe")
def describe_classes(name: str, body: Describe) -> dict[str, Any]:
    """Fill class descriptions from a template such as 'a photo of a {}'."""
    t = _task(name)
    try:
        t.describe_classes(body.template, body.overwrite)
    except ValueError as e:
        raise _bad(e) from e
    return _detail(t)


@app.delete("/api/tasks/{name}/classes/{label}")
def delete_class(name: str, label: str) -> dict[str, Any]:
    """Delete a class and all of its examples."""
    t = _task(name)
    try:
        t.delete_class(label)
    except KeyError as e:
        raise _missing(e) from e
    return _detail(t)


# ---- examples -----------------------------------------------------------
@app.post("/api/tasks/{name}/examples", status_code=201)
def add_examples(name: str, label: str = Form(...), files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Upload images as examples of ``label``; unreadable files are reported in ``errors``.

    Returns 400 when the label is blank or no file could be added.
    """
    t = _task(name)
    if not label.strip():
        raise HTTPException(400, "Class name is empty")
    added: list[dict[str, Any]] = []
    errors: list[dict[str, str | None]] = []
    readable: list[UploadFile] = []  # in step with add_examples' results

    def items() -> Iterator[tuple[bytes, str, str]]:
        for f in files:
            data = _read_upload(f)
            if data is None:
                errors.append({"file": f.filename, "error": _too_large()})
                continue
            readable.append(f)
            yield data, label, f.filename or ""

    results = t.add_examples(items())
    for f, r in zip(readable, results, strict=True):
        if isinstance(r, ValueError):
            errors.append({"file": f.filename, "error": str(r)})
        else:
            added.append(r.model_dump())
    if not added:
        raise HTTPException(400, "No images added: " + "; ".join(f"{e['file']}: {e['error']}" for e in errors))
    return {"added": added, "errors": errors, "task": _detail(t)}


class Relabel(_Body):
    """Body of PATCH .../examples/{eid}."""

    label: str


@app.patch("/api/tasks/{name}/examples/{eid}")
def relabel(name: str, eid: str, body: Relabel) -> dict[str, Any]:
    """Move an example to another (possibly new) class."""
    t = _task(name)
    try:
        t.relabel(eid, body.label)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _detail(t)


@app.delete("/api/tasks/{name}/examples/{eid}")
def delete_example(name: str, eid: str) -> dict[str, Any]:
    """Delete one example."""
    t = _task(name)
    try:
        t.delete_example(eid)
    except KeyError as e:
        raise _missing(e) from e
    return _detail(t)


class ExampleIds(_Body):
    """Body of POST .../examples/delete."""

    ids: list[str] = Field(min_length=1)


class BulkRelabel(ExampleIds):
    """Body of PATCH .../examples."""

    label: str


@app.patch("/api/tasks/{name}/examples")
def relabel_many(name: str, body: BulkRelabel) -> dict[str, Any]:
    """Move several examples to another (possibly new) class; an unknown id changes nothing (404)."""
    t = _task(name)
    try:
        t.relabel_many(body.ids, body.label)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _detail(t)


@app.post("/api/tasks/{name}/examples/delete")
def delete_examples(name: str, body: ExampleIds) -> dict[str, Any]:
    """Delete several examples; an unknown id deletes nothing (404)."""
    t = _task(name)
    try:
        t.delete_examples(body.ids)
    except KeyError as e:
        raise _missing(e) from e
    return _detail(t)


CACHE_FOREVER = {"Cache-Control": "max-age=31536000, immutable"}  # image ids are content hashes


@app.get("/api/tasks/{name}/examples/{eid}/image", response_model=None)
def example_image(name: str, eid: str,
                  size: int | None = Query(None, ge=16, le=2048)) -> FileResponse | Response:
    """Serve an example's image file, or with ``size`` a JPEG scaled down to fit in that many pixels."""
    t = _task(name)
    try:
        path = t.image_path(eid)
    except KeyError as e:
        raise _missing(e) from e
    if size is None:
        return FileResponse(path, headers=CACHE_FOREVER)
    try:
        data = thumbnail(path.read_bytes(), size)
    except FileNotFoundError as e:  # deleted between the lookup and the read
        raise HTTPException(404, f"No example {eid!r}") from e
    except ValueError as e:
        raise _bad(e) from e
    return Response(data, media_type="image/jpeg", headers=CACHE_FOREVER)


# ---- inference ----------------------------------------------------------
@app.post("/api/tasks/{name}/predict")
def predict(name: str, files: list[UploadFile] = File(...)) -> list[dict[str, Any]]:
    """Classify uploaded images; an unreadable file gets ``error`` set instead of a label."""
    t = _task(name)
    blobs = [_read_upload(f) for f in files]
    try:
        preds = pipe.predict(t, [b if b is not None else b"" for b in blobs])
    except ValueError as e:
        raise _bad(e) from e
    preds = [p if b is not None else Prediction.failed(_too_large()) for b, p in zip(blobs, preds, strict=True)]
    return [{"file": f.filename, **p.to_dict()} for f, p in zip(files, preds, strict=True)]


@app.post("/api/tasks/{name}/evaluate")
def evaluate(name: str, folds: int | None = None) -> dict[str, Any]:
    """Cross-validate the task (leave-one-out unless ``folds`` is given). For a large task, prefer
    POST .../evaluate/jobs, which reports progress and can be cancelled."""
    try:
        return evaluate_task(pipe, _task(name), folds)
    except ValueError as e:
        raise _bad(e) from e


# ---- background jobs ----------------------------------------------------
class _Cancelled(Exception):
    pass


@dataclass
class Job:  # pylint: disable=too-many-instance-attributes  # a record of a job's state, one field each
    """A long-running operation in a background thread, polled through /api/jobs/{id}."""

    task: str
    kind: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "running"  # running | done | failed | cancelled
    stage: str = "starting"
    done: int = 0
    total: int = 0
    result: Any = None
    error: str | None = None
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict[str, Any]:
        """JSON view of the job."""
        return {k: getattr(self, k) for k in ("id", "task", "kind", "status", "stage", "done", "total", "result",
                                              "error")}

    def progress(self, stage: str, done: int, total: int) -> None:
        """Progress callback for the work; raises ``_Cancelled`` once cancellation was asked for."""
        if self.cancel.is_set():
            raise _Cancelled
        self.stage, self.done, self.total = stage, done, total


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
JOBS_KEPT = 100  # finished jobs remembered (oldest are forgotten first)


def _start_job(job: Job, work: Any) -> Job:
    def run() -> None:
        try:
            job.result = work(job.progress)
            job.status = "done"
        except _Cancelled:
            job.status = "cancelled"
        except (ValueError, KeyError) as e:
            job.status, job.error = "failed", str(e.args[0]) if e.args else str(e)
        except Exception as e:  # pylint: disable=broad-exception-caught  # reported to the poller
            log.exception("Job %s failed", job.id)
            job.status, job.error = "failed", f"Internal error: {e}"
        job.finished = time.time()

    with _jobs_lock:
        finished = sorted((j for j in _jobs.values() if j.finished), key=lambda j: j.finished or 0)
        for old in finished[: max(0, len(finished) - JOBS_KEPT + 1)]:
            del _jobs[old.id]
        _jobs[job.id] = job
    threading.Thread(target=run, name=f"fewshot-job-{job.id[:8]}", daemon=True).start()
    return job


def _job(jid: str) -> Job:
    with _jobs_lock:
        job = _jobs.get(jid)
    if job is None:
        raise HTTPException(404, f"No job {jid!r}")
    return job


@app.post("/api/tasks/{name}/evaluate/jobs", status_code=202)
def start_evaluation(name: str, folds: int | None = None) -> dict[str, Any]:
    """Start cross-validating the task in the background; poll GET /api/jobs/{id} for progress
    (stage "embedding" counts images, "evaluating" counts folds) and the report."""
    t = _task(name)
    if folds is not None and folds < 2:
        raise HTTPException(400, f"folds must be at least 2, got {folds}")
    return _start_job(Job(task=name, kind="evaluate"),
                      lambda progress: evaluate_task(pipe, t, folds, progress=progress)).to_dict()


@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict[str, Any]:
    """A job's status, progress and (once done) result."""
    return _job(jid).to_dict()


@app.delete("/api/jobs/{jid}")
def cancel_job(jid: str) -> dict[str, Any]:
    """Ask a running job to stop; it does so at its next progress step (status becomes "cancelled")."""
    job = _job(jid)
    job.cancel.set()
    return job.to_dict()


# ---- web UI -------------------------------------------------------------
def mount_ui(application: FastAPI, dist: Path) -> None:
    """Serve the built single-page app in ``dist`` at / (no-op if it hasn't been built)."""
    if not dist.is_dir():
        return
    dist = dist.resolve()
    application.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @application.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "Not Found")  # unknown API routes must not get the HTML page
        f = dist / path
        if path and f.is_file() and dist in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(dist / "index.html")


mount_ui(app, Path(__file__).resolve().parent.parent / "web" / "dist")
