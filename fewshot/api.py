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
``FEWSHOT_ALLOWED_HOSTS``
    Comma-separated host names the server answers to, besides IP addresses (``*``: any; default:
    localhost). A request with another Host header gets 400, which stops web pages from reaching
    the server through DNS rebinding. ``fewshot serve`` sets it from ``--host``.
``FEWSHOT_ALLOW_URLS=1``
    Accept image URLs (``.../predict/urls``, ``.../examples/urls``): the server downloads them,
    from public addresses only.
``FEWSHOT_MAX_UPLOAD_MB`` / ``FEWSHOT_MAX_REQUEST_MB``
    Largest image per upload (default 50) and largest request body (default 1024).
"""

from __future__ import annotations

import logging
import mimetypes
import os
import secrets
import tempfile
import threading
import time
from collections import Counter, deque
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .archive import export_task as write_export
from .archive import import_task as read_export
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
from .evaluate import compare_embedders, evaluate_task
from .fetch import fetch_url
from .guard import LimitBody, cross_site, host_allowed, host_name
from .images import thumbnail
from .jobs import Job, Jobs, TooManyJobs
from .pipeline import Pipeline, Prediction
from .schemas import (
    AddResult,
    BulkEdit,
    ClassBody,
    ClassPatch,
    ClassThresholds,
    Compare,
    Describe,
    Duplicate,
    DuplicatePair,
    EvalReport,
    ExampleEdit,
    ExampleIds,
    ExamplePage,
    Health,
    InboxAdded,
    InboxLabels,
    InboxPage,
    JobOut,
    LabelledUrls,
    Login,
    ModelsOut,
    PredictionOut,
    Rename,
    TaskCreate,
    TaskDetail,
    TaskSettings,
    TaskSummary,
    Urls,
)
from .store import Task, Workspace
from .text import check_class_name, check_tag

log = logging.getLogger(__name__)

TOKEN = os.environ.get("FEWSHOT_TOKEN") or None
ALLOW_ANY_MODEL = os.environ.get("FEWSHOT_ALLOW_ANY_MODEL") == "1"
WARM = os.environ.get("FEWSHOT_WARM") == "1"
ALLOW_URLS = os.environ.get("FEWSHOT_ALLOW_URLS") == "1"
URL_FETCH_SECONDS = 120.0  # time budget for all of one request's URLs
URL_FETCH_WORKERS = 4  # URLs fetched at once
MAX_JOBS = 4  # background jobs running at once
LOGIN_FAILURES = 10  # wrong tokens per client within LOGIN_WINDOW before /api/login answers 429
LOGIN_WINDOW = 60.0
ALLOWED_HOSTS = {host_name(h)
                 for h in os.environ.get("FEWSHOT_ALLOWED_HOSTS", "localhost,127.0.0.1,::1").split(",") if h.strip()}
MAX_REQUEST_BYTES = int(float(os.environ.get("FEWSHOT_MAX_REQUEST_MB", "1024")) * 1024 * 1024)

ws = Workspace()
pipe = Pipeline(ws)
jobs = Jobs()


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    if WARM:
        threading.Thread(target=warm_models, name="fewshot-warm", daemon=True).start()
    yield


# Operation ids are the function names (predict, add_examples...), which make tidy generated clients.
app = FastAPI(title="fewshot", version="0.1.0", lifespan=_lifespan,
              generate_unique_id_function=lambda route: route.name)


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


SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


@app.middleware("http")
async def _guard(request: Request, call_next: Any) -> Response:
    """Refuse requests for another host name (DNS rebinding) and writes from other sites (CSRF),
    then check the token."""
    if not host_allowed(request.headers.get("host", ""), ALLOWED_HOSTS):
        return JSONResponse({"detail": "Unknown host; see FEWSHOT_ALLOWED_HOSTS"}, 400)
    path = request.url.path
    if request.method not in SAFE_METHODS and path.startswith("/api/") and cross_site(request):
        return JSONResponse({"detail": "Cross-site requests are not allowed"}, 403)
    if TOKEN is not None and path.startswith("/api/") and path not in OPEN_PATHS:
        auth = request.headers.get("authorization", "")
        given = auth[7:] if auth.lower().startswith("bearer ") else request.cookies.get(AUTH_COOKIE)
        if not _token_ok(given):
            return JSONResponse({"detail": "Not signed in"}, 401, headers={"WWW-Authenticate": "Bearer"})
    response: Response = await call_next(request)
    return response


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
    _allowed_embedder(spec)
    try:
        check_embedder(spec)
    except ValueError as e:
        raise _bad(e) from e


def _allowed_embedder(spec: str) -> None:
    """Refuse an empty spec (400), or one ``_check_embedder``'s allowlist doesn't allow (403)."""
    if not spec.strip():
        raise HTTPException(400, "Embedder spec is empty")
    if not (ALLOW_ANY_MODEL or spec in {p.spec for p in PRESETS} or is_registered(spec)
            or any(t.config.embedder == spec for t in ws.list_tasks())):
        raise HTTPException(403, f"Embedder {spec!r} is not a preset. To use other models, start the "
                                 "server with --allow-any-model (FEWSHOT_ALLOW_ANY_MODEL=1).")


# Largest accepted upload per image; FEWSHOT_MAX_UPLOAD_MB overrides it.
MAX_UPLOAD_BYTES = int(float(os.environ.get("FEWSHOT_MAX_UPLOAD_MB", "50")) * 1024 * 1024)


def _read_upload(f: UploadFile) -> bytes | None:
    """The uploaded file's bytes, or None if it is over ``MAX_UPLOAD_BYTES`` (read no further)."""
    data = f.file.read(MAX_UPLOAD_BYTES + 1)
    return None if len(data) > MAX_UPLOAD_BYTES else data


def _too_large() -> str:
    return f"File is larger than the {MAX_UPLOAD_BYTES / 1024 / 1024:g} MB upload limit"


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
        "reject_below_zero_shot": c.reject_below_zero_shot,
        "flip": c.flip,
        "temperature": c.temperature,
        "revision": c.revision,
        "inbox_count": len(c.inbox),
        "can_undo": t.can_undo(),
        "can_redo": t.can_redo(),
        "undo_label": t.undo_label(),
        "redo_label": t.redo_label(),
        "classes": [{"name": k.name, "description": k.description, "count": counts.get(k.name, 0),
                     "reject_below": k.reject_below} for k in c.classes],
        "examples": [e.model_dump() for e in c.examples],
    }


# ---- metadata -----------------------------------------------------------
@app.get("/api/health", response_model=Health)
@app.get("/healthz", include_in_schema=False)
def health() -> dict[str, Any]:
    """Liveness check; also lists loaded models and running jobs. Needs no token."""
    running = len(jobs.running())
    return {"status": "ok", "auth_required": TOKEN is not None, "models_loaded": loaded_embedders(),
            "jobs_running": running, "url_fetch": ALLOW_URLS}


_failed_logins: dict[str, deque[float]] = {}  # client address -> times of recent wrong tokens
_login_lock = threading.Lock()


@app.post("/api/login", status_code=204)
def login(body: Login, request: Request, response: Response) -> None:
    """Check the server token and remember it in an HttpOnly cookie (30 days; Secure over HTTPS).

    After LOGIN_FAILURES wrong tokens from one address within LOGIN_WINDOW seconds, further
    attempts get 429 until the window has passed, so the token can't be guessed quickly.
    """
    client = request.client.host if request.client else "?"
    now = time.monotonic()
    with _login_lock:
        recent = _failed_logins.setdefault(client, deque())
        while recent and now - recent[0] > LOGIN_WINDOW:
            recent.popleft()
        if len(recent) >= LOGIN_FAILURES:
            raise HTTPException(429, "Too many wrong tokens; try again in a minute")
        if not _token_ok(body.token):
            recent.append(now)
            raise HTTPException(401, "Wrong token")
        _failed_logins.pop(client, None)
    if TOKEN is not None:
        https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(AUTH_COOKIE, TOKEN, max_age=30 * 86400, httponly=True, samesite="strict", secure=https)


@app.post("/api/logout", status_code=204)
def logout(response: Response) -> None:
    """Forget the login cookie."""
    response.delete_cookie(AUTH_COOKIE)


@app.get("/api/models", response_model=ModelsOut)
def models() -> dict[str, Any]:
    """List embedder presets and classifier names."""
    return {
        "embedders": [p.__dict__ for p in PRESETS],
        "classifiers": sorted(CLASSIFIERS),
    }


# ---- tasks --------------------------------------------------------------
@app.get("/api/tasks", response_model=list[TaskSummary])
def list_tasks() -> list[dict[str, Any]]:
    """Summaries of all tasks."""
    return [t.summary() for t in ws.list_tasks()]


@app.post("/api/tasks", status_code=201, response_model=TaskDetail)
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


@app.get("/api/tasks/{name}", response_model=TaskDetail)
def get_task(name: str) -> dict[str, Any]:
    """Full detail of one task."""
    return _detail(_task(name))


@app.patch("/api/tasks/{name}", response_model=TaskDetail)
def update_task(name: str, body: TaskSettings) -> dict[str, Any]:
    """Change task settings; switching classifier resets its params unless new ones are given."""
    t = _task(name)
    kw = body.model_dump(exclude_none=True, exclude={"clear_reject", "clear_reject_zero_shot", "clear_temperature"})
    if body.clear_temperature:
        if body.temperature is not None:
            raise HTTPException(400, "Give either temperature or clear_temperature, not both")
        kw["temperature"] = None
    for key in ("reject_below", "reject_below_zero_shot"):
        if getattr(body, key.replace("reject_below", "clear_reject")):
            if getattr(body, key) is not None:
                raise HTTPException(400, f"Give either {key} or {key.replace('reject_below', 'clear_reject')}, "
                                         "not both")
            kw[key] = None
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


@app.post("/api/tasks/{name}/rename", response_model=TaskDetail)
def rename_task(name: str, body: Rename) -> dict[str, Any]:
    """Rename a task (400 if the new name is bad or taken)."""
    _task(name)
    try:
        return _detail(ws.rename(name, body.name))
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e


@app.post("/api/tasks/{name}/duplicate", status_code=201, response_model=TaskDetail)
def duplicate_task(name: str, body: Duplicate) -> dict[str, Any]:
    """Copy a task under a new name, e.g. to try other settings without touching it (400 if the
    name is bad or taken)."""
    _task(name)
    try:
        return _detail(ws.duplicate(name, body.name, body.examples, body.inbox))
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e


@app.post("/api/tasks/{name}/undo", response_model=TaskDetail)
def undo(name: str) -> dict[str, Any]:
    """Undo the task's last change (400 if there is nothing to undo); ``undo_label`` in the task
    says what it was before the call."""
    t = _task(name)
    try:
        t.undo()
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _detail(t)


@app.post("/api/tasks/{name}/redo", response_model=TaskDetail)
def redo(name: str) -> dict[str, Any]:
    """Make the last undone change again (400 if there is nothing to redo, as after any new change)."""
    t = _task(name)
    try:
        t.redo()
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _detail(t)


@app.get("/api/tasks/{name}/export", response_class=StreamingResponse,
         responses={200: {"content": {"application/zip": {}}}})
def export_task(name: str) -> StreamingResponse:
    """Download the task (settings, classes, labels and images) as a zip, for backup or to move it."""
    _task(name)
    buf = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)  # noqa: SIM115  # pylint: disable=consider-using-with  # closed by the response
    try:
        _, skipped = write_export(ws, name, buf)
    except KeyError as e:
        buf.close()
        raise _missing(e) from e
    size = buf.seek(0, os.SEEK_END)
    buf.seek(0)

    def chunks() -> Iterator[bytes]:
        with buf:
            while chunk := buf.read(1024 * 1024):
                yield chunk

    headers = {"Content-Disposition": f'attachment; filename="{name}.zip"', "Content-Length": str(size),
               "X-Fewshot-Skipped": str(skipped)}  # examples left out: their image file was missing
    return StreamingResponse(chunks(), media_type="application/zip", headers=headers)


@app.post("/api/tasks/import", status_code=201, response_model=TaskDetail)
def import_task(file: UploadFile = File(...), name: str | None = Form(None),
                embedder: str | None = Form(None)) -> dict[str, Any]:
    """Create a task from a zip made by .../export, named ``name`` or as exported, optionally
    with another ``embedder`` (say, a preset, when the exported one isn't allowed here)."""
    try:
        return _detail(read_export(ws, file.file, name or None, embedder_ok=_check_embedder, embedder=embedder or None))
    except ValueError as e:
        raise _bad(e) from e


# ---- classes ------------------------------------------------------------
@app.post("/api/tasks/{name}/classes", status_code=201, response_model=TaskDetail)
def add_class(name: str, body: ClassBody) -> dict[str, Any]:
    """Add a class, or update the description of an existing one."""
    t = _task(name)
    try:
        t.upsert_class(body.name, body.description)
    except ValueError as e:
        raise _bad(e) from e
    return _detail(t)


@app.patch("/api/tasks/{name}/classes/{label}", response_model=TaskDetail)
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


@app.post("/api/tasks/{name}/describe", response_model=TaskDetail)
def describe_classes(name: str, body: Describe) -> dict[str, Any]:
    """Fill class descriptions from a template such as 'a photo of a {}'."""
    t = _task(name)
    try:
        t.describe_classes(body.template, body.overwrite)
    except ValueError as e:
        raise _bad(e) from e
    return _detail(t)


@app.put("/api/tasks/{name}/class-thresholds", response_model=TaskDetail)
def set_class_thresholds(name: str, body: ClassThresholds) -> dict[str, Any]:
    """Set or clear the reject thresholds of the named classes (one undoable change); classes not
    named keep theirs. An unknown class (404) or a value outside -1..1 (400) changes nothing."""
    t = _task(name)
    if not body.thresholds:
        raise HTTPException(400, "Give at least one class")
    try:
        t.set_class_thresholds(body.thresholds)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _detail(t)


@app.delete("/api/tasks/{name}/classes/{label}", response_model=TaskDetail)
def delete_class(name: str, label: str) -> dict[str, Any]:
    """Delete a class and all of its examples."""
    t = _task(name)
    try:
        t.delete_class(label)
    except KeyError as e:
        raise _missing(e) from e
    return _detail(t)


# ---- examples -----------------------------------------------------------
@app.get("/api/tasks/{name}/examples", response_model=ExamplePage)
def list_examples(name: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),
                  label: str | None = None, tag: str | None = None) -> dict[str, Any]:
    """One page of the task's examples, optionally only those with ``label`` and/or ``tag``."""
    found = _task(name).find_examples(label, tag)
    return {"total": len(found), "offset": offset,
            "examples": [e.model_dump() for e in found[offset : offset + limit]]}


Blob = tuple[str | None, bytes | None, str | None]  # (file name, bytes or None, error if None)


def _add_blobs(t: Task, label: str, blobs: list[Blob]) -> dict[str, Any]:
    """Add the readable ``blobs`` as examples of ``label``; 400 if none could be added."""
    if not label.strip():
        raise HTTPException(400, "Class name is empty")
    errors: list[dict[str, str | None]] = [{"file": f, "error": err} for f, data, err in blobs if data is None]
    ok = [(f, data) for f, data, _ in blobs if data is not None]
    added: list[dict[str, Any]] = []
    for (f, _), r in zip(ok, t.add_examples((data, label, f or "") for f, data in ok), strict=True):
        if isinstance(r, ValueError):
            errors.append({"file": f, "error": str(r)})
        else:
            added.append(r.model_dump())
    if not added:
        raise HTTPException(400, "No images added: " + "; ".join(f"{e['file']}: {e['error']}" for e in errors))
    return {"added": added, "errors": errors, "task": _detail(t)}


def _uploads(files: list[UploadFile]) -> list[Blob]:
    out: list[Blob] = []
    for f in files:
        data = _read_upload(f)
        # Browsers (and fewshot.client) send a '"' in a file name as %22, per the HTML standard;
        # the multipart parser leaves it encoded.
        name = f.filename.replace("%22", '"') if f.filename else f.filename
        out.append((name, data, None if data is not None else _too_large()))
    return out


def _downloads(urls: list[str]) -> list[Blob]:
    """Fetch image URLs for the server (public addresses only); 403 unless ALLOW_URLS."""
    if not ALLOW_URLS:
        raise HTTPException(403, "This server doesn't fetch URLs; start it with --allow-urls "
                                 "(FEWSHOT_ALLOW_URLS=1)")
    # A few at a time, with a budget for the whole request: at most MAX_REQUEST_BYTES in all and
    # URL_FETCH_SECONDS overall, so one request can't hold gigabytes or a worker for hours.
    deadline = time.monotonic() + URL_FETCH_SECONDS
    total = 0
    budget_lock = threading.Lock()

    def one(url: str) -> Blob:
        nonlocal total
        left = deadline - time.monotonic()
        if left <= 0:
            return url, None, f"Not fetched: the request's {URL_FETCH_SECONDS:g} s budget for URLs ran out"
        try:
            data = fetch_url(url, MAX_UPLOAD_BYTES, timeout=min(20.0, left), public_only=True)
        except ValueError as e:
            return url, None, str(e)
        with budget_lock:
            total += len(data)
            over = total > MAX_REQUEST_BYTES
        if over:
            mb = MAX_REQUEST_BYTES / 1024 / 1024
            return url, None, f"Not used: the images together exceed the {mb:.3g} MB request limit"
        return url, data, None

    with ThreadPoolExecutor(max_workers=URL_FETCH_WORKERS) as pool:
        return list(pool.map(one, urls))


@app.post("/api/tasks/{name}/examples", status_code=201, response_model=AddResult)
def add_examples(name: str, label: str = Form(...), files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Upload images as examples of ``label``; unreadable files are reported in ``errors``.

    Returns 400 when the label is blank or no file could be added.
    """
    t = _task(name)
    return _add_blobs(t, label, _uploads(files))


@app.post("/api/tasks/{name}/examples/urls", status_code=201, response_model=AddResult)
def add_examples_by_url(name: str, body: LabelledUrls) -> dict[str, Any]:
    """Download images (server started with --allow-urls) and add them as examples of ``label``."""
    t = _task(name)
    try:
        label = check_class_name(body.label)  # before downloading anything
    except ValueError as e:
        raise _bad(e) from e
    return _add_blobs(t, label, _downloads(body.urls))


def _check_tags(tags: list[str]) -> None:
    """Raise ``ValueError`` for the first bad tag."""
    for tag in tags:
        check_tag(tag)


@app.patch("/api/tasks/{name}/examples/{eid}", response_model=TaskDetail)
def edit_example(name: str, eid: str, body: ExampleEdit) -> dict[str, Any]:
    """Move an example to another (possibly new) class, and/or set its tags or note."""
    t = _task(name)
    try:
        t.example(eid)  # 404 for an unknown id, even if the edit is empty
        _check_tags(body.tags or [])  # all checks before any change, so a bad tag changes nothing
        if body.label is not None:
            t.relabel(eid, body.label)
        if body.tags is not None or body.note is not None:
            t.annotate([eid], tags=body.tags, note=body.note)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _detail(t)


@app.delete("/api/tasks/{name}/examples/{eid}", response_model=TaskDetail)
def delete_example(name: str, eid: str) -> dict[str, Any]:
    """Delete one example."""
    t = _task(name)
    try:
        t.delete_example(eid)
    except KeyError as e:
        raise _missing(e) from e
    return _detail(t)


@app.patch("/api/tasks/{name}/examples", response_model=TaskDetail)
def edit_examples(name: str, body: BulkEdit) -> dict[str, Any]:
    """Move several examples to another (possibly new) class and/or tag them; an unknown id
    changes nothing (404)."""
    t = _task(name)
    try:
        for eid in body.ids:
            t.example(eid)  # 404 for an unknown id, even if the edit is empty
        _check_tags(body.add_tags + body.remove_tags)
        if body.label is not None:
            t.relabel_many(body.ids, body.label)
        if body.add_tags or body.remove_tags:
            t.annotate(body.ids, add_tags=body.add_tags, remove_tags=body.remove_tags)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _detail(t)


@app.post("/api/tasks/{name}/examples/delete", response_model=TaskDetail)
def delete_examples(name: str, body: ExampleIds) -> dict[str, Any]:
    """Delete several examples; an unknown id deletes nothing (404)."""
    t = _task(name)
    try:
        t.delete_examples(body.ids)
    except KeyError as e:
        raise _missing(e) from e
    return _detail(t)


CACHE_FOREVER = {"Cache-Control": "max-age=31536000, immutable"}  # image ids are content hashes
IMAGE_HEADERS = {"X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'; sandbox"}


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
        # Only ever served as an image (never e.g. HTML from a crafted import), and never sniffed.
        kind = mimetypes.guess_type(path.name)[0] or ""
        safe = kind if kind.startswith("image/") and kind != "image/svg+xml" else "application/octet-stream"
        return FileResponse(path, media_type=safe, headers={**CACHE_FOREVER, **IMAGE_HEADERS})
    try:
        data = thumbnail(path.read_bytes(), size)
    except FileNotFoundError as e:  # deleted between the lookup and the read
        raise HTTPException(404, f"No example {eid!r}") from e
    except ValueError as e:
        raise _bad(e) from e
    return Response(data, media_type="image/jpeg", headers={**CACHE_FOREVER, **IMAGE_HEADERS})


# ---- inbox --------------------------------------------------------------
def _add_inbox(t: Task, blobs: list[Blob]) -> dict[str, Any]:
    """Put the readable ``blobs`` in the inbox; 400 if none could be added."""
    errors: list[dict[str, str | None]] = [{"file": f, "error": err} for f, data, err in blobs if data is None]
    ok = [(f, data) for f, data, _ in blobs if data is not None]
    added: list[dict[str, Any]] = []
    for (f, _), r in zip(ok, t.add_to_inbox((data, f or "") for f, data in ok), strict=True):
        if isinstance(r, ValueError):
            errors.append({"file": f, "error": str(r)})
        else:
            added.append(r.model_dump())
    if not added:
        why = "; ".join(f"{e['file']}: {e['error']}" for e in errors)
        raise HTTPException(400, f"Nothing added to the inbox: {why}")
    return {"added": added, "errors": errors, "task": _detail(t)}


@app.post("/api/tasks/{name}/inbox", status_code=201, response_model=InboxAdded)
def add_to_inbox(name: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Upload unlabelled images to the task's inbox, to label later with the model's suggestions."""
    return _add_inbox(_task(name), _uploads(files))


@app.post("/api/tasks/{name}/inbox/urls", status_code=201, response_model=InboxAdded)
def add_urls_to_inbox(name: str, body: Urls) -> dict[str, Any]:
    """Download images (server started with --allow-urls) into the task's inbox."""
    t = _task(name)
    return _add_inbox(t, _downloads(body.urls))


@app.get("/api/tasks/{name}/inbox", response_model=InboxPage)
def list_inbox(name: str, sort: str = Query("uncertain", pattern="^(uncertain|confident|novel|added)$"),
               offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    """The inbox with suggestions: ``uncertain`` first (the most useful to label), ``confident``
    first (to accept in bulk), ``novel`` first (unlike any example), or as ``added``."""
    t = _task(name)
    try:
        items = pipe.inbox_suggestions(t, sort)
    except ValueError as e:
        raise _bad(e) from e
    return {"total": len(items), "offset": offset, "items": items[offset : offset + limit]}


@app.post("/api/tasks/{name}/inbox/accept", response_model=TaskDetail)
def accept_inbox(name: str, body: InboxLabels) -> dict[str, Any]:
    """Label inbox images, moving them to the examples (one undo step); 404 if an id isn't waiting."""
    t = _task(name)
    try:
        t.accept(body.labels)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _detail(t)


@app.post("/api/tasks/{name}/inbox/dismiss", response_model=TaskDetail)
def dismiss_inbox(name: str, body: ExampleIds) -> dict[str, Any]:
    """Remove images from the inbox without labelling them (undo brings them back)."""
    t = _task(name)
    try:
        t.dismiss(body.ids)
    except KeyError as e:
        raise _missing(e) from e
    return _detail(t)


# ---- inference ----------------------------------------------------------
TOP_K = Query(None, ge=1, description="Return only this many of the highest scores per image")
EVIDENCE = Query(0, ge=0, le=20, description="Also return this many of the most similar labelled examples")
REVISION = Query(None, ge=0, description="Predict with this earlier version of the task (one kept for undo)")


def _predict(t: Task, blobs: list[Blob], top_k: int | None, evidence: int = 0,  # pylint: disable=too-many-arguments,too-many-positional-arguments
             revision: int | None = None) -> list[dict[str, Any]]:
    try:
        cfg = None if revision is None else t.version(revision)
    except KeyError as e:
        raise _missing(e) from e
    try:
        preds = pipe.predict(t, [data if data is not None else b"" for _, data, _ in blobs], top_k=top_k,
                             evidence=evidence, cfg=cfg)
    except ValueError as e:
        raise _bad(e) from e
    return [{"file": f, **(p if data is not None else Prediction.failed(err or "")).to_dict()}
            for (f, data, err), p in zip(blobs, preds, strict=True)]


@app.post("/api/tasks/{name}/predict", response_model=list[PredictionOut])
def predict(name: str, files: list[UploadFile] = File(...), top_k: int | None = TOP_K,
            evidence: int = EVIDENCE, revision: int | None = REVISION) -> list[dict[str, Any]]:
    """Classify uploaded images; an unreadable file gets ``error`` set instead of a label. Each
    prediction says which ``revision`` of the task made it."""
    t = _task(name)
    return _predict(t, _uploads(files), top_k, evidence, revision)


@app.post("/api/tasks/{name}/predict/urls", response_model=list[PredictionOut])
def predict_urls(name: str, body: Urls, top_k: int | None = TOP_K, evidence: int = EVIDENCE,
                 revision: int | None = REVISION) -> list[dict[str, Any]]:
    """Download images (server started with --allow-urls) and classify them; ``file`` is the URL."""
    t = _task(name)
    return _predict(t, _downloads(body.urls), top_k, evidence, revision)


@app.get("/api/tasks/{name}/duplicates", response_model=list[DuplicatePair])
def duplicates(name: str, threshold: float = Query(0.95, ge=0.5, le=1.0),
               limit: int = Query(200, ge=1, le=5000)) -> list[dict[str, Any]]:
    """Pairs of near-identical examples (cosine similarity >= ``threshold``), most similar first."""
    try:
        return pipe.near_duplicates(_task(name), threshold, limit)
    except ValueError as e:
        raise _bad(e) from e


@app.post("/api/tasks/{name}/evaluate", response_model=EvalReport)
def evaluate(name: str, folds: int | None = None) -> dict[str, Any]:
    """Cross-validate the task (leave-one-out unless ``folds`` is given). For a large task, prefer
    POST .../evaluate/jobs, which reports progress and can be cancelled."""
    t = _task(name)
    try:
        return _evaluated(t, evaluate_task(pipe, t, folds))
    except ValueError as e:
        raise _bad(e) from e


def _evaluated(t: Task, report: dict[str, Any]) -> dict[str, Any]:
    """``report``, after recording its headline for the dashboard."""
    t.record_evaluation(report)
    return report


# ---- background jobs ----------------------------------------------------
def _job(jid: str) -> Job:
    job = jobs.get(jid)
    if job is None:
        raise HTTPException(404, f"No job {jid!r}")
    return job


@app.post("/api/tasks/{name}/evaluate/jobs", status_code=202, response_model=JobOut)
def start_evaluation(name: str, folds: int | None = None) -> dict[str, Any]:
    """Start cross-validating the task in the background; poll GET /api/jobs/{id} for progress
    (stage "embedding" counts images, "evaluating" counts folds) and the report."""
    t = _task(name)
    if folds is not None and folds < 2:
        raise HTTPException(400, f"folds must be at least 2, got {folds}")
    kind = "evaluate" if folds is None else f"evaluate:{folds}"
    running = jobs.running()
    same = next((j for j in running if (j.task, j.kind) == (name, kind) and j.revision == t.config.revision), None)
    if same is not None:  # the same evaluation is already running: share it
        return same.to_dict()
    job = Job(task=name, kind=kind, revision=t.config.revision)
    return _start(job, lambda progress: _evaluated(t, evaluate_task(pipe, t, folds, progress=progress)))


def _start(job: Job, work: Callable[[Callable[[str, int, int], None]], Any]) -> dict[str, Any]:
    """Start ``job``; 429 if ``MAX_JOBS`` are running."""
    try:
        return jobs.start(job, work, limit=MAX_JOBS).to_dict()
    except TooManyJobs:
        raise HTTPException(429, f"{MAX_JOBS} jobs are already running; try again when one has finished") from None


MAX_COMPARED = 8  # embedders per comparison


@app.post("/api/tasks/{name}/compare/jobs", status_code=202, response_model=JobOut)
def start_comparison(name: str, body: Compare) -> dict[str, Any]:
    """Start cross-validating the task with each of ``embedders`` in the background (the same
    allowlist as choosing a task's embedder applies); the job's result compares their accuracy.
    Stages are like "2/3 embedding"."""
    t = _task(name)
    specs = list(dict.fromkeys(s.strip() for s in body.embedders if s.strip()))
    if not 1 <= len(specs) <= MAX_COMPARED:
        raise HTTPException(400, f"Give between 1 and {MAX_COMPARED} embedders")
    if body.folds is not None and body.folds < 2:
        raise HTTPException(400, f"folds must be at least 2, got {body.folds}")
    if not t.config.examples:
        raise HTTPException(400, "Task has no examples to evaluate")
    for s in specs:
        _allowed_embedder(s)  # loaded by the job: a model that fails is reported in its result
    job = Job(task=name, kind="compare", revision=t.config.revision)
    return _start(job, lambda progress: compare_embedders(pipe, t, specs, body.folds, progress=progress))


@app.get("/api/jobs/{jid}", response_model=JobOut)
def get_job(jid: str) -> dict[str, Any]:
    """A job's status, progress and (once done) result."""
    return _job(jid).to_dict()


@app.delete("/api/jobs/{jid}", response_model=JobOut)
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
