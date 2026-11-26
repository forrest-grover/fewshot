"""REST API (FastAPI). Run with ``fewshot serve``; docs at /docs.

``create_app(settings)`` builds the server for one ``ServerSettings`` (see
``fewshot.server_settings`` for each setting and its environment variable); ``fewshot serve`` uses
it. ``uvicorn fewshot.api:app`` gets the same server, built from the environment when ``app`` is
first used. The built web UI (web/dist) is served at / when present.

Requests are answered from two pools of threads: cheap ones (task details, edits) from
Starlette's, and heavy ones (predictions, evaluations, uploads, imports, exports, thumbnails,
inbox suggestions, and creating a task or changing its settings, which may load a model) from the
server's own ``workers`` threads, so heavy work queues there and can't keep the server from
answering anything else; ``/healthz`` needs no thread at all. Responses are validated and
serialised in those threads too (``_json``), not on the event loop as FastAPI would.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import mimetypes
import os
import tempfile
import threading
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import services
from .access import SESSION_COOKIE, TOO_MANY_TOKENS, allow_no_token_from_env, check_token, client_address, guard
from .archive import export_task as write_export
from .archive import import_task as read_export
from .auth import SESSION_DAYS, Sessions, WrongTokens
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
from .guard import LimitBody, body_limit, without_input
from .images import thumbnail
from .inbox import chunked
from .jobs import Job, Jobs, TooManyJobs
from .pipeline import Pipeline
from .responses import json_response
from .schemas import (
    AcceptConfident,
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
from .server_settings import ServerSettings, requested_processes
from .store import Task, Workspace
from .text import check_class_name

log = logging.getLogger(__name__)

URL_FETCH_SECONDS = 120.0  # time budget for all of one request's URLs
URL_FETCH_WORKERS = 4  # URLs fetched at once
MAX_JOBS = 4  # background jobs running at once
PREDICT_CHUNK = 64  # uploaded images read and classified at a time
T = TypeVar("T")


# ---- the server's state ---------------------------------------------------
class Server:  # pylint: disable=too-many-instance-attributes  # the server's parts, one each
    """Everything one app serves: its settings, workspace, pipeline, jobs and access control."""

    def __init__(self, settings: ServerSettings, workspace: Workspace | None = None,
                 allow_no_token: bool = False) -> None:
        self.settings = settings
        self.allow_no_token = allow_no_token  # see access.guard
        self.ws = workspace or Workspace(settings.home)
        self.pipe = Pipeline(self.ws)
        self.jobs = Jobs()
        self.wrong_tokens = WrongTokens()
        self.sessions = Sessions(settings.token) if settings.token else None
        self.warned_proxy = False  # see access._note_untrusted_proxy
        self.warned_exposed = False  # see access.guard
        self._heavy: ThreadPoolExecutor | None = None  # made when first needed, after each shutdown too
        self._heavy_lock = threading.Lock()

    @property
    def heavy(self) -> ThreadPoolExecutor:
        """The threads for heavy work (``settings.workers`` of them)."""
        with self._heavy_lock:
            if self._heavy is None:
                self._heavy = ThreadPoolExecutor(self.settings.workers, thread_name_prefix="fewshot-work")
            return self._heavy

    def stop_heavy(self) -> None:
        """Stop the heavy-work threads, dropping work not started yet (the app is shutting down); the
        next heavy request, if the app is started again, gets new ones."""
        with self._heavy_lock:
            pool, self._heavy = self._heavy, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        """``fn(*args)`` on the heavy-work threads, so it waits there, not in Starlette's pool."""
        return await asyncio.get_running_loop().run_in_executor(self.heavy, functools.partial(fn, *args))

    async def render(self, model: Any, fn: Callable[..., Any], *args: Any, status: int = 200) -> Response:
        """``fn(*args)`` on the heavy-work threads, as JSON validated as ``model`` (see ``_json``)."""
        return await self.run(lambda: json_response(model, fn(*args), status))

    def task(self, name: str) -> Task:
        """Task ``name``; 404 if there is none."""
        try:
            return self.ws.get(name)
        except KeyError as e:
            raise _missing(e) from e

    def warm_models(self) -> list[str]:
        """Load the embedders the most tasks use (up to the loaded-model limit), or the default one
        if there are no tasks, so the first request doesn't wait for a model; returns those loaded."""
        counts = Counter(t.config.embedder for t in self.ws.list_tasks())
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

    def allowed_embedder(self, spec: str) -> None:
        """Refuse an empty spec (400), or (403) one that isn't a preset, registered, or used by a
        task already, unless the server allows any model. Specs in use are checked last, as that
        reads every task."""
        if not spec.strip():
            raise HTTPException(400, "Embedder spec is empty")
        if not (self.settings.allow_any_model or spec in {p.spec for p in PRESETS} or is_registered(spec)
                or any(t.config.embedder == spec for t in self.ws.list_tasks())):
            raise HTTPException(403, f"Embedder {spec!r} is not a preset. To use other models, start the "
                                     "server with --allow-any-model (FEWSHOT_ALLOW_ANY_MODEL=1).")

    def check_embedder(self, spec: str) -> None:
        """``allowed_embedder``, then load the embedder so a bad spec fails now (400), not at predict time."""
        self.allowed_embedder(spec)
        try:
            check_embedder(spec)
        except ValueError as e:
            raise _bad(e) from e


async def _server(request: Request) -> Server:
    server: Server = request.app.state.server
    return server


Srv = Annotated[Server, Depends(_server)]  # async, so resolving it takes no thread


# ---- the app -------------------------------------------------------------
def create_app(settings: ServerSettings | None = None, workspace: Workspace | None = None,
               allow_no_token: bool | None = None) -> FastAPI:
    """The server for ``settings`` (default: from the environment), on ``workspace`` (default:
    ``settings.home``). Without a token, it refuses requests that reach it on an address other
    than loopback, unless ``allow_no_token`` (default: ``FEWSHOT_ALLOW_NO_TOKEN``)."""
    allow = allow_no_token_from_env() if allow_no_token is None else allow_no_token
    server = Server(settings or ServerSettings.from_env(), workspace, allow)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if (n := requested_processes()) > 1:
            log.warning("This server was started with %d processes, which isn't supported: logged-out sessions, "
                        "wrong-token limits and background jobs are kept in each process's memory, so a request "
                        "can miss what another process knows (a job's progress, a logout). Run one process; "
                        "FEWSHOT_WORKERS sets its threads for heavy work.", n)
        if server.settings.warm:
            threading.Thread(target=server.warm_models, name="fewshot-warm", daemon=True).start()
        yield
        server.stop_heavy()

    # Operation ids are the function names (predict, add_examples...), which make tidy generated clients.
    app = FastAPI(title="fewshot", version="0.1.0", lifespan=lifespan,
                  generate_unique_id_function=lambda route: route.name)
    app.state.server = server
    app.include_router(router)
    app.middleware("http")(guard)
    app.add_exception_handler(RequestValidationError, without_input)
    # Added last, so it runs first.
    app.add_middleware(LimitBody, limit=functools.partial(body_limit, server.settings.max_request_bytes))
    mount_ui(app, Path(__file__).resolve().parent.parent / "web" / "dist")
    return app


@functools.cache
def _default_app() -> FastAPI:
    try:
        settings = ServerSettings.from_env()
    except ValueError as e:
        raise SystemExit(f"fewshot: {e}") from None
    return create_app(settings)


def __getattr__(name: str) -> Any:
    """``app``: the server built from the environment, for ``uvicorn fewshot.api:app``. Built on
    first use, so importing this module reads no settings and touches no workspace."""
    if name == "app":
        return _default_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


router = APIRouter()


def _missing(e: KeyError) -> HTTPException:
    return HTTPException(404, str(e.args[0]))


def _bad(e: Exception) -> HTTPException:
    return HTTPException(400, str(e.args[0]) if e.args else str(e))


def _task_json(t: Task, status: int = 200) -> Response:
    """The task's detail as a response (see ``_json``)."""
    return json_response(TaskDetail, _detail(t), status)


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
        "names_as_text": c.names_as_text,
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
@router.get("/api/health", response_model=Health)
@router.get("/healthz", include_in_schema=False)
async def health(srv: Srv) -> dict[str, Any]:
    """Liveness check; also lists loaded models and running jobs. Needs no token, and no thread, so
    it answers even while every thread is busy."""
    return {"status": "ok", "auth_required": srv.settings.token is not None, "models_loaded": loaded_embedders(),
            "jobs_running": len(srv.jobs.running()), "jobs_cancelling": len(srv.jobs.stopping()),
            "url_fetch": srv.settings.allow_urls,
            "max_request_bytes": srv.settings.max_request_bytes, "max_upload_bytes": srv.settings.max_upload_bytes}


@router.post("/api/login", status_code=204)
async def login(srv: Srv, body: Login, request: Request, response: Response) -> None:
    """Check the server token and start a session: an HttpOnly, SameSite=Strict cookie (Secure over
    HTTPS) holding a signed session id, never the token, valid for 30 days or until logout.

    After LOGIN_FAILURES wrong tokens from one address within LOGIN_WINDOW seconds (here or as
    Bearer headers), further wrong ones from it get 429 until it has sent none for that long; the
    right token is always accepted.
    """
    refused = check_token(srv, client_address(request), body.token)
    if refused == 429:
        raise HTTPException(429, TOO_MANY_TOKENS)
    if refused:
        raise HTTPException(401, "Wrong token")
    if srv.sessions is not None:
        https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(SESSION_COOKIE, srv.sessions.issue(), max_age=SESSION_DAYS * 86400, httponly=True,
                            samesite="strict", secure=https)


@router.post("/api/logout", status_code=204)
async def logout(srv: Srv, request: Request, response: Response) -> None:
    """End the session: the cookie is deleted, and its session refused from now on."""
    if srv.sessions is not None:
        srv.sessions.revoke(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE)


@router.get("/api/models", response_model=ModelsOut)
async def models() -> dict[str, Any]:
    """List embedder presets and classifier names."""
    return {
        "embedders": [p.__dict__ for p in PRESETS],
        "classifiers": sorted(CLASSIFIERS),
    }


# ---- tasks --------------------------------------------------------------
@router.get("/api/tasks", response_model=list[TaskSummary])
def list_tasks(srv: Srv) -> Response:
    """Summaries of all tasks."""
    return json_response(list[TaskSummary], [t.summary() for t in srv.ws.list_tasks()])


def _create_task(srv: Server, body: TaskCreate) -> Response:
    try:
        return _task_json(services.create_task(srv.ws, body.name, body.embedder, body.classifier,
                                               embedder_ok=srv.check_embedder), 201)
    except ValueError as e:
        raise _bad(e) from e


@router.post("/api/tasks", status_code=201, response_model=TaskDetail)
async def create_task(srv: Srv, body: TaskCreate) -> Response:
    """Create a task, validating the embedder spec (which loads the model) first."""
    return await srv.run(_create_task, srv, body)


@router.get("/api/tasks/{name}", response_model=TaskDetail)
def get_task(srv: Srv, name: str) -> Response:
    """Full detail of one task."""
    return _task_json(srv.task(name))


def _update_task(srv: Server, name: str, body: TaskSettings) -> Response:
    t = srv.task(name)
    flags = {"reject_below": "clear_reject", "reject_below_zero_shot": "clear_reject_zero_shot",
             "temperature": "clear_temperature"}
    values = body.model_dump(exclude_none=True, exclude=set(flags.values()))
    clear = {k: flag for k, flag in flags.items() if getattr(body, flag)}
    try:
        services.change_settings(t, values, clear, embedder_ok=srv.check_embedder)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _task_json(t)


@router.patch("/api/tasks/{name}", response_model=TaskDetail)
async def update_task(srv: Srv, name: str, body: TaskSettings) -> Response:
    """Change task settings. ``classifier_params`` replaces the params; switching classifier starts
    its params afresh unless they are given. A temperature of 1 means none. A new embedder is
    loaded first, on the heavy-work threads."""
    return await srv.run(_update_task, srv, name, body)


@router.delete("/api/tasks/{name}", status_code=204)
def delete_task(srv: Srv, name: str) -> None:
    """Delete a task and its images."""
    srv.task(name)
    try:
        srv.ws.delete(name)
    except KeyError as e:
        raise _missing(e) from e


@router.post("/api/tasks/{name}/rename", response_model=TaskDetail)
def rename_task(srv: Srv, name: str, body: Rename) -> Response:
    """Rename a task (400 if the new name is bad or taken)."""
    srv.task(name)
    try:
        return _task_json(srv.ws.rename(name, body.name))
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e


@router.post("/api/tasks/{name}/duplicate", status_code=201, response_model=TaskDetail)
def duplicate_task(srv: Srv, name: str, body: Duplicate) -> Response:
    """Copy a task under a new name, e.g. to try other settings without touching it (400 if the
    name is bad or taken)."""
    srv.task(name)
    try:
        return _task_json(srv.ws.duplicate(name, body.name, body.examples, body.inbox), 201)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e


@router.post("/api/tasks/{name}/undo", response_model=TaskDetail)
def undo(srv: Srv, name: str) -> Response:
    """Undo the task's last change (400 if there is nothing to undo); ``undo_label`` in the task
    says what it was before the call."""
    t = srv.task(name)
    try:
        t.undo()
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _task_json(t)


@router.post("/api/tasks/{name}/redo", response_model=TaskDetail)
def redo(srv: Srv, name: str) -> Response:
    """Make the last undone change again (400 if there is nothing to redo, as after any new change)."""
    t = srv.task(name)
    try:
        t.redo()
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _task_json(t)


def _export(srv: Server, name: str) -> StreamingResponse:
    srv.task(name)
    buf = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)  # noqa: SIM115  # pylint: disable=consider-using-with  # closed by the response
    try:
        _, skipped = write_export(srv.ws, name, buf)
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


@router.get("/api/tasks/{name}/export", response_class=StreamingResponse,
            responses={200: {"content": {"application/zip": {}}}})
async def export_task(srv: Srv, name: str) -> StreamingResponse:
    """Download the task (settings, classes, labels and images) as a zip, for backup or to move it."""
    return await srv.run(_export, srv, name)


def _import(srv: Server, file: UploadFile, name: str | None, embedder: str | None) -> dict[str, Any]:
    try:
        return _detail(read_export(srv.ws, file.file, name, embedder_ok=srv.check_embedder, embedder=embedder))
    except ValueError as e:
        raise _bad(e) from e


@router.post("/api/tasks/import", status_code=201, response_model=TaskDetail)
async def import_task(srv: Srv, file: UploadFile = File(...), name: str | None = Form(None),
                      embedder: str | None = Form(None)) -> Response:
    """Create a task from a zip made by .../export, named ``name`` or as exported, optionally
    with another ``embedder`` (say, a preset, when the exported one isn't allowed here)."""
    return await srv.render(TaskDetail, _import, srv, file, name or None, embedder or None, status=201)


# ---- classes ------------------------------------------------------------
@router.post("/api/tasks/{name}/classes", status_code=201, response_model=TaskDetail)
def add_class(srv: Srv, name: str, body: ClassBody) -> Response:
    """Add a class, or update the description of an existing one."""
    t = srv.task(name)
    try:
        t.upsert_class(body.name, body.description)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _task_json(t, 201)


@router.patch("/api/tasks/{name}/classes/{label}", response_model=TaskDetail)
def patch_class(srv: Srv, name: str, label: str, body: ClassPatch) -> Response:
    """Rename a class (relabelling its examples) and/or set its description, as one change (one
    undo step)."""
    t = srv.task(name)
    try:
        t.edit_class(label, body.name, body.description)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _task_json(t)


@router.post("/api/tasks/{name}/describe", response_model=TaskDetail)
def describe_classes(srv: Srv, name: str, body: Describe) -> Response:
    """Fill class descriptions from a template such as 'a photo of a {}'."""
    t = srv.task(name)
    try:
        t.describe_classes(body.template, body.overwrite)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _task_json(t)


@router.put("/api/tasks/{name}/class-thresholds", response_model=TaskDetail)
def set_class_thresholds(srv: Srv, name: str, body: ClassThresholds) -> Response:
    """Set or clear the reject thresholds of the named classes (one undoable change); classes not
    named keep theirs. An unknown class (404) or a value outside -1..1 (400) changes nothing."""
    t = srv.task(name)
    if not body.thresholds:
        raise HTTPException(400, "Give at least one class")
    try:
        t.set_class_thresholds(body.thresholds)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _task_json(t)


@router.delete("/api/tasks/{name}/classes/{label}", response_model=TaskDetail)
def delete_class(srv: Srv, name: str, label: str) -> Response:
    """Delete a class and all of its examples."""
    t = srv.task(name)
    try:
        t.delete_class(label)
    except KeyError as e:
        raise _missing(e) from e
    return _task_json(t)


# ---- examples -----------------------------------------------------------
@router.get("/api/tasks/{name}/examples", response_model=ExamplePage)
def list_examples(srv: Srv, name: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),  # pylint: disable=too-many-arguments,too-many-positional-arguments
                  label: str | None = None, tag: str | None = None) -> Response:
    """One page of the task's examples, optionally only those with ``label`` and/or ``tag``."""
    found = srv.task(name).find_examples(label, tag)
    return json_response(ExamplePage, {"total": len(found), "offset": offset,
                               "examples": [e.model_dump() for e in found[offset : offset + limit]]})


def _uploads(srv: Server, files: list[UploadFile]) -> Iterator[services.Blob]:
    """The uploaded files, each read only when it is needed, so a big upload is never held in
    memory whole; one over the upload limit carries an error instead of its bytes."""
    limit = srv.settings.max_upload_bytes
    for f in files:
        # Browsers (and fewshot.client) send a '"' in a file name as %22, per the HTML standard;
        # the multipart parser leaves it encoded.
        name = f.filename.replace("%22", '"') if f.filename else ""
        data = f.file.read(limit + 1)
        if len(data) > limit:
            yield services.Blob(name, None, f"File is larger than the {limit / 1024 / 1024:g} MB upload limit")
        else:
            yield services.Blob(name, data)


def _downloads(srv: Server, urls: list[str]) -> list[services.Blob]:
    """Fetch image URLs for the server (public addresses only); 403 unless it allows URLs. A few
    at a time, with a budget for the whole request: at most the request size limit in all and
    ``URL_FETCH_SECONDS`` overall, so one request can't hold gigabytes or a thread for hours."""
    if not srv.settings.allow_urls:
        raise HTTPException(403, "This server doesn't fetch URLs; start it with --allow-urls "
                                 "(FEWSHOT_ALLOW_URLS=1)")
    return services.fetch_urls(urls, srv.settings.max_upload_bytes, public_only=True, seconds=URL_FETCH_SECONDS,
                               max_total=srv.settings.max_request_bytes, workers=URL_FETCH_WORKERS)


UNDO_GROUP = Form(None, max_length=64, description=(
    "Any id the client chooses for one upload it sends in several requests: consecutive uploads "
    "to the task with the same id, with no other change in between, are one undo step"))


def _errors(r: services.Added) -> list[dict[str, str]]:
    return [{"file": f, "error": why} for f, why in r.errors]


def _add_examples(srv: Server, name: str, label: str, blobs: Iterator[services.Blob] | list[services.Blob],
                  undo_group: str | None = None) -> dict[str, Any]:
    """Add the readable ``blobs`` as examples of ``label``; 400 if none could be added."""
    t = srv.task(name)
    try:
        r = services.add_examples(t, label, blobs, undo_group)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    if not r.added:
        raise HTTPException(400, "No images added: " + "; ".join(f"{f}: {why}" for f, why in r.errors))
    return {"added": [e.model_dump() for e in r.added], "errors": _errors(r), "task": _detail(t)}


@router.post("/api/tasks/{name}/examples", status_code=201, response_model=AddResult)
async def add_examples(srv: Srv, name: str, label: str = Form(...), files: list[UploadFile] = File(...),
                       undo_group: str | None = UNDO_GROUP) -> Response:
    """Upload images as examples of ``label``; unreadable files are reported in ``errors``.

    Returns 400 when the label is blank or no file could be added.
    """
    return await srv.render(AddResult, _add_examples, srv, name, label, _uploads(srv, files), undo_group or None,
                            status=201)


def _add_examples_by_url(srv: Server, name: str, body: LabelledUrls) -> dict[str, Any]:
    srv.task(name)
    try:
        check_class_name(body.label)  # before downloading anything
    except ValueError as e:
        raise _bad(e) from e
    return _add_examples(srv, name, body.label, _downloads(srv, body.urls))


@router.post("/api/tasks/{name}/examples/urls", status_code=201, response_model=AddResult)
async def add_examples_by_url(srv: Srv, name: str, body: LabelledUrls) -> Response:
    """Download images (server started with --allow-urls) and add them as examples of ``label``."""
    return await srv.render(AddResult, _add_examples_by_url, srv, name, body, status=201)


@router.patch("/api/tasks/{name}/examples/{eid}", response_model=TaskDetail)
def edit_example(srv: Srv, name: str, eid: str, body: ExampleEdit) -> Response:
    """Move an example to another (possibly new) class, and/or set its tags or note, as one change
    (one undo step)."""
    t = srv.task(name)
    try:
        t.example(eid)  # 404 for an unknown id, even if the edit is empty
        if body.label is not None or body.tags is not None or body.note is not None:
            t.edit_examples([eid], label=body.label, tags=body.tags, note=body.note)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _task_json(t)


@router.delete("/api/tasks/{name}/examples/{eid}", response_model=TaskDetail)
def delete_example(srv: Srv, name: str, eid: str) -> Response:
    """Delete one example."""
    t = srv.task(name)
    try:
        t.delete_example(eid)
    except KeyError as e:
        raise _missing(e) from e
    return _task_json(t)


@router.patch("/api/tasks/{name}/examples", response_model=TaskDetail)
def edit_examples(srv: Srv, name: str, body: BulkEdit) -> Response:
    """Move several examples to another (possibly new) class and/or tag them, as one change (one
    undo step); an unknown id changes nothing (404)."""
    t = srv.task(name)
    try:
        for eid in body.ids:
            t.example(eid)  # 404 for an unknown id, even if the edit is empty
        if body.label is not None or body.add_tags or body.remove_tags:
            t.edit_examples(body.ids, label=body.label, add_tags=body.add_tags, remove_tags=body.remove_tags)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _task_json(t)


@router.post("/api/tasks/{name}/examples/delete", response_model=TaskDetail)
def delete_examples(srv: Srv, name: str, body: ExampleIds) -> Response:
    """Delete several examples; an unknown id deletes nothing (404)."""
    t = srv.task(name)
    try:
        t.delete_examples(body.ids)
    except KeyError as e:
        raise _missing(e) from e
    return _task_json(t)


CACHE_FOREVER = {"Cache-Control": "max-age=31536000, immutable"}  # image ids are content hashes
IMAGE_HEADERS = {"X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'; sandbox"}


def _image(srv: Server, name: str, eid: str, size: int | None) -> FileResponse | Response:
    try:
        path = srv.task(name).image_path(eid)
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


@router.get("/api/tasks/{name}/examples/{eid}/image", response_model=None)
async def example_image(srv: Srv, name: str, eid: str,
                        size: int | None = Query(None, ge=16, le=2048)) -> FileResponse | Response:
    """Serve an example's image file, or with ``size`` a JPEG scaled down to fit in that many pixels."""
    return await srv.run(_image, srv, name, eid, size)


# ---- inbox --------------------------------------------------------------
def _add_inbox(srv: Server, name: str, blobs: Iterator[services.Blob] | list[services.Blob],
               undo_group: str | None = None) -> dict[str, Any]:
    """Put the readable ``blobs`` in the inbox; 400 if none could be added."""
    t = srv.task(name)
    try:
        r = services.add_to_inbox(t, blobs, undo_group)
    except KeyError as e:
        raise _missing(e) from e
    if not r.added:
        why = "; ".join(f"{f}: {why}" for f, why in r.errors)
        raise HTTPException(400, f"Nothing added to the inbox: {why}")
    return {"added": [x.model_dump() for x in r.added], "errors": _errors(r), "task": _detail(t)}


@router.post("/api/tasks/{name}/inbox", status_code=201, response_model=InboxAdded)
async def add_to_inbox(srv: Srv, name: str, files: list[UploadFile] = File(...),
                       undo_group: str | None = UNDO_GROUP) -> Response:
    """Upload unlabelled images to the task's inbox, to label later with the model's suggestions."""
    return await srv.render(InboxAdded, _add_inbox, srv, name, _uploads(srv, files), undo_group or None,
                            status=201)


def _add_inbox_by_url(srv: Server, name: str, urls: list[str]) -> dict[str, Any]:
    srv.task(name)  # 404 before downloading anything
    return _add_inbox(srv, name, _downloads(srv, urls))


@router.post("/api/tasks/{name}/inbox/urls", status_code=201, response_model=InboxAdded)
async def add_urls_to_inbox(srv: Srv, name: str, body: Urls) -> Response:
    """Download images (server started with --allow-urls) into the task's inbox."""
    return await srv.render(InboxAdded, _add_inbox_by_url, srv, name, body.urls, status=201)


def _inbox_page(srv: Server, name: str, sort: str, offset: int, limit: int) -> dict[str, Any]:
    try:
        items = srv.pipe.inbox_suggestions(srv.task(name), sort)
    except ValueError as e:
        raise _bad(e) from e
    return {"total": len(items), "offset": offset, "items": items[offset : offset + limit]}


@router.get("/api/tasks/{name}/inbox", response_model=InboxPage)
async def list_inbox(srv: Srv, name: str, sort: str = Query("uncertain", pattern="^(uncertain|confident|novel|added)$"),
                     offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)) -> Response:
    """The inbox with suggestions: ``uncertain`` first (the most useful to label), ``confident``
    first (to accept in bulk), ``novel`` first (unlike any example), or as ``added``."""
    return await srv.render(InboxPage, _inbox_page, srv, name, sort, offset, limit)


@router.post("/api/tasks/{name}/inbox/accept", response_model=TaskDetail)
def accept_inbox(srv: Srv, name: str, body: InboxLabels) -> Response:
    """Label inbox images, moving them to the examples (one undo step); 404 if an id isn't waiting."""
    t = srv.task(name)
    try:
        t.accept(body.labels)
    except KeyError as e:
        raise _missing(e) from e
    except ValueError as e:
        raise _bad(e) from e
    return _task_json(t)


def _accept_confident(srv: Server, name: str, body: AcceptConfident) -> dict[str, Any]:
    t = srv.task(name)
    try:
        services.accept_confident(srv.pipe, t, body.min_confidence, body.ids)
    except KeyError as e:
        raise _missing(e) from e
    return _detail(t)


@router.post("/api/tasks/{name}/inbox/accept-confident", response_model=TaskDetail)
async def accept_confident(srv: Srv, name: str, body: AcceptConfident) -> Response:
    """Label as the model suggests every inbox image (of ``ids``, if given) it is at least
    ``min_confidence`` sure of and doesn't reject, in one undo step."""
    return await srv.render(TaskDetail, _accept_confident, srv, name, body)


@router.post("/api/tasks/{name}/inbox/dismiss", response_model=TaskDetail)
def dismiss_inbox(srv: Srv, name: str, body: ExampleIds) -> Response:
    """Remove images from the inbox without labelling them (undo brings them back)."""
    t = srv.task(name)
    try:
        t.dismiss(body.ids)
    except KeyError as e:
        raise _missing(e) from e
    return _task_json(t)


# ---- inference ----------------------------------------------------------
TOP_K = Query(None, ge=1, description="Return only this many of the highest scores per image")
EVIDENCE = Query(0, ge=0, le=20, description="Also return this many of the most similar labelled examples")
REVISION = Query(None, ge=0, description="Predict with this earlier version of the task (one kept for undo)")


def _predict(srv: Server, name: str, blobs: Iterator[services.Blob] | list[services.Blob],  # pylint: disable=too-many-arguments,too-many-positional-arguments
             top_k: int | None, evidence: int, revision: int | None) -> list[dict[str, Any]]:
    """Classify ``blobs`` ``PREDICT_CHUNK`` at a time, all with the same version of the task."""
    t = srv.task(name)
    try:
        cfg = t.config if revision is None else t.version(revision)
    except KeyError as e:
        raise _missing(e) from e
    out: list[dict[str, Any]] = []
    for part in chunked(blobs, PREDICT_CHUNK):
        try:
            preds = services.predict(srv.pipe, t, part, top_k=top_k, evidence=evidence, cfg=cfg)
        except ValueError as e:
            raise _bad(e) from e
        out += [{"file": b.name, **p.to_dict()} for b, p in zip(part, preds, strict=True)]
    return out


@router.post("/api/tasks/{name}/predict", response_model=list[PredictionOut])
async def predict(srv: Srv, name: str, files: list[UploadFile] = File(...),  # pylint: disable=too-many-arguments,too-many-positional-arguments
                  top_k: int | None = TOP_K, evidence: int = EVIDENCE,
                  revision: int | None = REVISION) -> Response:
    """Classify uploaded images; an unreadable file gets ``error`` set instead of a label. Each
    prediction says which ``revision`` of the task made it."""
    return await srv.render(list[PredictionOut], _predict, srv, name, _uploads(srv, files), top_k, evidence, revision)


def _predict_urls(srv: Server, name: str, urls: list[str], top_k: int | None, evidence: int,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                  revision: int | None) -> list[dict[str, Any]]:
    srv.task(name)  # 404 before downloading anything
    return _predict(srv, name, _downloads(srv, urls), top_k, evidence, revision)


@router.post("/api/tasks/{name}/predict/urls", response_model=list[PredictionOut])
async def predict_urls(srv: Srv, name: str, body: Urls, top_k: int | None = TOP_K,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                       evidence: int = EVIDENCE, revision: int | None = REVISION) -> Response:
    """Download images (server started with --allow-urls) and classify them; ``file`` is the URL."""
    return await srv.render(list[PredictionOut], _predict_urls, srv, name, body.urls, top_k, evidence, revision)


def _duplicates(srv: Server, name: str, threshold: float, limit: int) -> list[dict[str, Any]]:
    try:
        return srv.pipe.near_duplicates(srv.task(name), threshold, limit)
    except ValueError as e:
        raise _bad(e) from e


@router.get("/api/tasks/{name}/duplicates", response_model=list[DuplicatePair])
async def duplicates(srv: Srv, name: str, threshold: float = Query(0.95, ge=0.5, le=1.0),
                     limit: int = Query(200, ge=1, le=5000)) -> Response:
    """Pairs of near-identical examples (cosine similarity >= ``threshold``), most similar first."""
    return await srv.render(list[DuplicatePair], _duplicates, srv, name, threshold, limit)


def _evaluate(srv: Server, name: str, folds: int | None) -> dict[str, Any]:
    t = srv.task(name)
    try:
        return services.evaluate(srv.pipe, t, folds)
    except ValueError as e:
        raise _bad(e) from e


@router.post("/api/tasks/{name}/evaluate", response_model=EvalReport)
async def evaluate(srv: Srv, name: str, folds: int | None = None) -> Response:
    """Cross-validate the task (leave-one-out unless ``folds`` is given). For a large task, prefer
    POST .../evaluate/jobs, which reports progress and can be cancelled."""
    return await srv.render(EvalReport, _evaluate, srv, name, folds)


# ---- background jobs ----------------------------------------------------
def _job(srv: Server, jid: str) -> Job:
    job = srv.jobs.get(jid)
    if job is None:
        raise HTTPException(404, f"No job {jid!r}: it finished long ago, or the server was restarted since")
    return job


def _start(srv: Server, job: Job, work: Callable[[Callable[[str, int, int], None]], Any]) -> Response:
    """Start ``job``, or return the same one already running; 429 if ``MAX_JOBS`` are running or
    being cancelled."""
    try:
        return json_response(JobOut, srv.jobs.start(job, work, limit=MAX_JOBS).to_dict(), 202)
    except TooManyJobs:
        raise HTTPException(429, f"{MAX_JOBS} jobs are already running or being cancelled; try again when one "
                                 "has finished") from None


@router.post("/api/tasks/{name}/evaluate/jobs", status_code=202, response_model=JobOut)
def start_evaluation(srv: Srv, name: str, folds: int | None = None) -> Response:
    """Start cross-validating the task in the background; poll GET /api/jobs/{id} for progress
    (stage "embedding" counts images, "evaluating" counts folds) and the report. The same
    evaluation of the same version already running is shared rather than started again."""
    t = srv.task(name)
    if folds is not None and folds < 2:
        raise HTTPException(400, f"folds must be at least 2, got {folds}")
    job = Job(task=name, kind="evaluate" if folds is None else f"evaluate:{folds}", revision=t.config.revision,
              uid=t.config.uid)
    return _start(srv, job, lambda progress: services.evaluate(srv.pipe, t, folds, progress))


MAX_COMPARED = 8  # embedders per comparison


@router.post("/api/tasks/{name}/compare/jobs", status_code=202, response_model=JobOut)
def start_comparison(srv: Srv, name: str, body: Compare) -> Response:
    """Start cross-validating the task with each of ``embedders`` in the background (the same
    allowlist as choosing a task's embedder applies); the job's result compares their accuracy.
    Stages are like "2/3 embedding"."""
    from .evaluate import compare_embedders  # pylint: disable=import-outside-toplevel  # with evaluate's other imports

    t = srv.task(name)
    try:
        specs = services.check_specs(body.embedders, MAX_COMPARED)
    except ValueError as e:
        raise _bad(e) from e
    if body.folds is not None and body.folds < 2:
        raise HTTPException(400, f"folds must be at least 2, got {body.folds}")
    if not t.config.examples:
        raise HTTPException(400, "Task has no examples to evaluate")
    for s in specs:
        srv.allowed_embedder(s)  # loaded by the job: a model that fails is reported in its result
    job = Job(task=name, kind="compare", revision=t.config.revision, uid=t.config.uid,
              key=f"{body.folds}:" + "\n".join(specs))
    return _start(srv, job, lambda progress: compare_embedders(srv.pipe, t, specs, body.folds, progress=progress))


@router.get("/api/jobs/{jid}", response_model=JobOut)
def get_job(srv: Srv, jid: str) -> Response:
    """A job's status, progress and (once done) result."""
    return json_response(JobOut, _job(srv, jid).to_dict())


@router.delete("/api/jobs/{jid}", response_model=JobOut)
def cancel_job(srv: Srv, jid: str) -> Response:
    """Ask a running job to stop: its status becomes "cancelling", then "cancelled" at its next
    progress step. Until then it still counts towards the job limit."""
    job = _job(srv, jid)
    job.request_cancel()
    return json_response(JobOut, job.to_dict())


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
