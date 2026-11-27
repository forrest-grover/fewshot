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
import contextvars
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

from fastapi import APIRouter, Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, errors, services
from .access import SESSION_COOKIE, TOO_MANY_TOKENS, check_token, client_address, guard
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
)
from .errors import ApiProblem, from_error, problem_body, refusals
from .exceptions import TaskNotFound, message
from .guard import LimitBody, body_limit
from .images import thumbnail
from .inbox import chunked
from .jobs import Job, Jobs, TooManyJobs
from .models import check_evaluable
from .pipeline import Pipeline
from .responses import ERRORS, PRECONDITION, UI_HEADERS, UPLOAD, json_response
from .routing import (
    ADDED,
    IF_MATCH,
    INBOX_ADDED,
    JOB_STARTED,
    MULTIPART,
    TAGS,
    added_status,
    amend_openapi,
    check_label,
    edited_json,
    etag,
    file_errors,
    json_only,
    known_params,
    task_detail,
    task_json,
    uploads,
)
from .schemas import (
    CLEAR_FLAGS,
    CLEARABLE,
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
    InboxSort,
    JobOut,
    LabelledUrls,
    Login,
    ModelsOut,
    PredictionOut,
    Rename,
    TaskCreate,
    TaskDetail,
    TaskEdited,
    TaskSettings,
    TaskSummary,
    Urls,
)
from .server_settings import ServerSettings, requested_processes
from .store import Task, Workspace
from .text import closest

log = logging.getLogger(__name__)

URL_FETCH_SECONDS = 120.0  # time budget for all of one request's URLs
URL_FETCH_WORKERS = 4  # URLs fetched at once
MAX_JOBS = 4  # background jobs running at once
PREDICT_CHUNK = 64  # uploaded images read and classified at a time
API_VERSION = "1"  # within one, responses only gain fields (see the README's compatibility policy)
POLL_SECONDS = 1  # Retry-After of a running job
JOBS_RETRY_SECONDS = 5  # Retry-After when too many jobs are running
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
        ctx = contextvars.copy_context()  # the request's (see ``preconditions``), as Starlette's pool does
        return await asyncio.get_running_loop().run_in_executor(self.heavy, ctx.run, functools.partial(fn, *args))

    async def render(self, model: Any, fn: Callable[..., Any], *args: Any, status: int = 200) -> Response:
        """``fn(*args)`` on the heavy-work threads, as JSON validated as ``model`` (see ``_json``)."""
        return await self.run(lambda: json_response(model, fn(*args), status))

    def task(self, name: str) -> Task:
        """Task ``name``; 404 if there is none, naming the closest ones."""
        try:
            return self.ws.get(name)
        except TaskNotFound as e:
            near = closest(name, [p.name for p in self.ws.task_folders()])
            hint = f"; did you mean {' or '.join(map(repr, near))}? Or create it" if near else "; create it"
            raise ApiProblem(404, errors.TASK_NOT_FOUND, f"No task {name!r}{hint} with POST /api/tasks "
                             "(fewshot.client: create_task)", task=name, similar_to=near) from e
        except KeyError as e:
            raise from_error(e) from e

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
            raise ApiProblem(400, errors.INVALID_EMBEDDER, "Embedder spec is empty")
        if not (self.settings.allow_any_model or spec in {p.spec for p in PRESETS} or is_registered(spec)
                or any(t.config.embedder == spec for t in self.ws.list_tasks())):
            raise ApiProblem(403, errors.EMBEDDER_NOT_ALLOWED, f"Embedder {spec!r} is not a preset (see GET "
                             "/api/models). To use other models, start the server with --allow-any-model "
                             "(FEWSHOT_ALLOW_ANY_MODEL=1).")

    def check_embedder(self, spec: str) -> None:
        """``allowed_embedder``, then load the embedder so a bad spec fails now (400), not at predict time."""
        self.allowed_embedder(spec)
        try:
            check_embedder(spec)
        except ValueError as e:
            raise ApiProblem(400, errors.INVALID_EMBEDDER, message(e)) from e


async def _server(request: Request) -> Server:
    server: Server = request.app.state.server
    return server


Srv = Annotated[Server, Depends(_server)]  # async, so resolving it takes no thread


# ---- the app -------------------------------------------------------------
def create_app(settings: ServerSettings | None = None, workspace: Workspace | None = None,
               allow_no_token: bool | None = None) -> FastAPI:
    """The server for ``settings`` (default: from the environment), on ``workspace`` (default:
    ``settings.home``). Without a token, it refuses requests that reach it on an address other
    than loopback, unless ``allow_no_token`` (default: the settings', from ``FEWSHOT_ALLOW_NO_TOKEN``)."""
    settings = settings or ServerSettings.from_env()
    server = Server(settings, workspace, settings.allow_no_token if allow_no_token is None else allow_no_token)

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
    app = FastAPI(title="fewshot", version=__version__, lifespan=lifespan, openapi_tags=TAGS,
                  description="Few-shot image classification. Errors are application/problem+json with a stable "
                              "`code`; see the README for the routes, the status policy and the codes.",
                  generate_unique_id_function=lambda route: route.name)
    app.state.server = server
    app.include_router(router)
    app.middleware("http")(guard)
    errors.install(app)
    amend_openapi(app)
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


router = APIRouter(responses=ERRORS, dependencies=[Depends(known_params), Depends(json_only)])


# ---- metadata -----------------------------------------------------------
@router.get("/api/health", response_model=Health, tags=["system"], summary="Server status and version")
@router.get("/healthz", include_in_schema=False)
async def health(srv: Srv) -> dict[str, Any]:
    """Liveness check; also gives the server's ``version`` and ``api_version``, and lists loaded
    models and running jobs. Needs no token, and no thread, so it answers even while every thread
    is busy."""
    return {"status": "ok", "version": __version__, "api_version": API_VERSION,
            "auth_required": srv.settings.token is not None, "models_loaded": loaded_embedders(),
            "jobs_running": len(srv.jobs.running()), "jobs_cancelling": len(srv.jobs.stopping()),
            "url_fetch": srv.settings.allow_urls,
            "max_request_bytes": srv.settings.max_request_bytes, "max_upload_bytes": srv.settings.max_upload_bytes}


@router.post("/api/login", status_code=204, response_class=Response, tags=["auth"], summary="Sign in with the token")
async def login(srv: Srv, body: Login, request: Request, response: Response) -> None:
    """Check the server token and start a session: an HttpOnly, SameSite=Strict cookie (Secure over
    HTTPS) holding a signed session id, never the token, valid for 30 days or until logout.

    After LOGIN_FAILURES wrong tokens from one address within LOGIN_WINDOW seconds (here or as
    Bearer headers), further wrong ones from it get 429 until it has sent none for that long; the
    right token is always accepted.
    """
    refused = check_token(srv, client_address(request), body.token)
    if refused == 429:
        raise ApiProblem(429, errors.TOO_MANY_WRONG_TOKENS, TOO_MANY_TOKENS, headers={"Retry-After": "60"})
    if refused:
        raise ApiProblem(401, errors.WRONG_TOKEN, "Wrong token")
    if srv.sessions is not None:
        https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(SESSION_COOKIE, srv.sessions.issue(), max_age=SESSION_DAYS * 86400, httponly=True,
                            samesite="strict", secure=https)


@router.post("/api/logout", status_code=204, response_class=Response, tags=["auth"], summary="End the session")
async def logout(srv: Srv, request: Request, response: Response) -> None:
    """End the session: the cookie is deleted, and its session refused from now on."""
    if srv.sessions is not None:
        srv.sessions.revoke(request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE)


@router.get("/api/models", response_model=ModelsOut, tags=["system"], summary="List embedder presets and classifiers")
async def models() -> dict[str, Any]:
    """List embedder presets and classifier names."""
    return {
        "embedders": [p.__dict__ for p in PRESETS],
        "classifiers": sorted(CLASSIFIERS),
    }


# ---- tasks --------------------------------------------------------------
@router.get("/api/tasks", response_model=list[TaskSummary], tags=["tasks"], summary="List tasks")
def list_tasks(srv: Srv) -> Response:
    """Summaries of all tasks."""
    return json_response(list[TaskSummary], [t.summary() for t in srv.ws.list_tasks()])


def _create_task(srv: Server, body: TaskCreate) -> Response:
    with refusals():
        return task_json(services.create_task(srv.ws, body.name, body.embedder, body.classifier,
                                               embedder_ok=srv.check_embedder), 201)


@router.post("/api/tasks", status_code=201, response_model=TaskDetail, tags=["tasks"], summary="Create a task")
async def create_task(srv: Srv, body: TaskCreate) -> Response:
    """Create a task, validating the embedder spec (which loads the model) first; 409
    TASK_EXISTS if the name is taken."""
    return await srv.run(_create_task, srv, body)


@router.get("/api/tasks/{name}", response_model=TaskDetail, tags=["tasks"], summary="Get a task")
def get_task(srv: Srv, name: str) -> Response:
    """Full detail of one task. The ETag header is its revision, for If-Match on changes."""
    return task_json(srv.task(name))


def _update_task(srv: Server, name: str, body: TaskSettings) -> Response:
    t = srv.task(name)
    given = body.model_fields_set - set(CLEAR_FLAGS.values())
    nulls = sorted(k for k in given if getattr(body, k) is None and k not in CLEARABLE)
    if nulls:
        raise RequestValidationError([{"type": "null_not_allowed", "loc": ("body", k), "input": None,
                                       "msg": f"{k} can't be null: omit it to leave it unchanged"} for k in nulls])
    values = {k: getattr(body, k) for k in given if getattr(body, k) is not None}
    clear = {k: k for k in given if getattr(body, k) is None}
    clear.update({k: flag for k, flag in CLEAR_FLAGS.items() if getattr(body, flag)})
    with refusals(task=name):
        services.change_settings(t, values, clear, embedder_ok=srv.check_embedder)
    out = task_json(t)
    if body.model_fields_set & set(CLEAR_FLAGS.values()):
        out.headers["Deprecation"] = "true"  # null clears now
    return out


@router.patch("/api/tasks/{name}", response_model=TaskDetail, tags=["tasks"], summary="Change a task's settings",
              dependencies=IF_MATCH, responses=PRECONDITION)
async def update_task(srv: Srv, name: str, body: TaskSettings) -> Response:
    """Change task settings, as a JSON Merge Patch: omitted fields are unchanged, and null clears
    ``reject_below``, ``reject_below_zero_shot`` or ``temperature`` (null for another field is a
    422). ``classifier_params`` replaces the params; switching classifier starts its params afresh
    unless they are given. A temperature of 1 means none. A new embedder is loaded first, on the
    heavy-work threads."""
    return await srv.run(_update_task, srv, name, body)


@router.delete("/api/tasks/{name}", status_code=204, response_class=Response, tags=["tasks"],
               summary="Delete a task", dependencies=IF_MATCH, responses=PRECONDITION)
def delete_task(srv: Srv, name: str) -> None:
    """Delete a task and its images (this can't be undone)."""
    srv.task(name)
    with refusals():
        srv.ws.delete(name)


@router.post("/api/tasks/{name}/rename", response_model=TaskDetail, tags=["tasks"], summary="Rename a task",
             dependencies=IF_MATCH, responses=PRECONDITION)
def rename_task(srv: Srv, name: str, body: Rename) -> Response:
    """Rename a task: 400 if the new name is bad, 409 TASK_EXISTS if it is taken."""
    srv.task(name)
    with refusals():
        return task_json(srv.ws.rename(name, body.name))


@router.post("/api/tasks/{name}/duplicate", status_code=201, response_model=TaskDetail, tags=["tasks"],
             summary="Copy a task under a new name")
def duplicate_task(srv: Srv, name: str, body: Duplicate) -> Response:
    """Copy a task under a new name, e.g. to try other settings without touching it: 400 if the
    name is bad, 409 TASK_EXISTS if it is taken."""
    srv.task(name)
    with refusals():
        return task_json(srv.ws.duplicate(name, body.name, body.examples, body.inbox), 201)


@router.post("/api/tasks/{name}/undo", response_model=TaskDetail, tags=["tasks"], summary="Undo the last change",
             dependencies=IF_MATCH, responses=PRECONDITION)
def undo(srv: Srv, name: str) -> Response:
    """Undo the task's last change (409 NOTHING_TO_UNDO if there is none); ``undo_label`` in the
    task says what it was before the call."""
    t = srv.task(name)
    with refusals():
        t.undo()
    return task_json(t)


@router.post("/api/tasks/{name}/redo", response_model=TaskDetail, tags=["tasks"], summary="Redo the last undone change",
             dependencies=IF_MATCH, responses=PRECONDITION)
def redo(srv: Srv, name: str) -> Response:
    """Make the last undone change again (409 NOTHING_TO_REDO if there is none, as after any new change)."""
    t = srv.task(name)
    with refusals():
        t.redo()
    return task_json(t)


def _export(srv: Server, name: str) -> StreamingResponse:
    srv.task(name)
    # A large export spills into the workspace, not the often small, in-memory /tmp (a container's
    # tmpfs), unless this server can only read the workspace.
    spill = str(srv.ws.root) if os.access(srv.ws.root, os.W_OK) else None
    buf = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024, dir=spill)  # noqa: SIM115  # pylint: disable=consider-using-with  # closed by the response
    try:
        _, skipped = write_export(srv.ws, name, buf)
    except KeyError as e:
        buf.close()
        raise from_error(e) from e
    size = buf.seek(0, os.SEEK_END)
    buf.seek(0)

    def chunks() -> Iterator[bytes]:
        with buf:
            while chunk := buf.read(1024 * 1024):
                yield chunk

    headers = {"Content-Disposition": f'attachment; filename="{name}.zip"', "Content-Length": str(size),
               "X-Fewshot-Skipped": str(skipped)}  # examples left out: their image file was missing
    return StreamingResponse(chunks(), media_type="application/zip", headers=headers)


@router.get("/api/tasks/{name}/export", response_class=StreamingResponse, tags=["tasks"],
            summary="Export a task as a zip", responses={200: {"content": {"application/zip": {}}}})
async def export_task(srv: Srv, name: str) -> StreamingResponse:
    """Download the task (settings, classes, labels and images) as a zip, for backup or to move it."""
    return await srv.run(_export, srv, name)


def _import(srv: Server, file: UploadFile, name: str | None, embedder: str | None) -> Response:
    with refusals():
        return task_json(read_export(srv.ws, file.file, name, embedder_ok=srv.check_embedder, embedder=embedder),
                          201)


@router.post("/api/tasks/import", status_code=201, response_model=TaskDetail, tags=["tasks"],
             summary="Import a task from a zip", dependencies=MULTIPART, responses=UPLOAD)
async def import_task(srv: Srv, file: UploadFile = File(...), name: str | None = Form(None),
                      embedder: str | None = Form(None)) -> Response:
    """Create a task from a zip made by .../export, named ``name`` or as exported (409
    TASK_EXISTS if that name is taken), optionally with another ``embedder`` (say, a preset,
    when the exported one isn't allowed here)."""
    return await srv.run(_import, srv, file, name or None, embedder or None)


# ---- classes ------------------------------------------------------------
@router.post("/api/tasks/{name}/classes", status_code=201, response_model=TaskDetail, tags=["classes"],
             summary="Add a class (or set its description)", dependencies=IF_MATCH,
             responses={200: {"model": TaskDetail, "description": "The class existed: its description was set"},
                        **PRECONDITION})
def add_class(srv: Srv, name: str, body: ClassBody) -> Response:
    """Add a class (201), or update the description of an existing one (200)."""
    t = srv.task(name)
    existed = body.name.strip() in t.class_names()
    with refusals():
        t.upsert_class(body.name, body.description)
    return task_json(t, 200 if existed else 201)


@router.patch("/api/tasks/{name}/classes/{label}", response_model=TaskDetail, tags=["classes"],
              summary="Rename a class or set its description", dependencies=IF_MATCH, responses=PRECONDITION)
def patch_class(srv: Srv, name: str, label: str, body: ClassPatch) -> Response:
    """Rename a class (relabelling its examples; 409 CLASS_EXISTS if the new name is taken)
    and/or set its description, as one change (one undo step)."""
    t = srv.task(name)
    with refusals():
        t.edit_class(label, body.name, body.description)
    return task_json(t)


@router.post("/api/tasks/{name}/describe", response_model=TaskDetail, tags=["classes"],
             summary="Describe classes from a template", dependencies=IF_MATCH, responses=PRECONDITION)
def describe_classes(srv: Srv, name: str, body: Describe) -> Response:
    """Fill class descriptions from a template such as 'a photo of a {}'."""
    t = srv.task(name)
    with refusals():
        t.describe_classes(body.template, body.overwrite)
    return task_json(t)


@router.put("/api/tasks/{name}/class-thresholds", response_model=TaskDetail, tags=["classes"],
            summary="Set classes' reject thresholds", dependencies=IF_MATCH, responses=PRECONDITION)
def set_class_thresholds(srv: Srv, name: str, body: ClassThresholds) -> Response:
    """Set or clear the reject thresholds of the named classes (one undoable change); classes not
    named keep theirs. An unknown class (404) or a value outside -1..1 (422) changes nothing."""
    t = srv.task(name)
    with refusals():
        t.set_class_thresholds(body.thresholds)
    return task_json(t)


@router.delete("/api/tasks/{name}/classes/{label}", response_model=TaskDetail, tags=["classes"],
               summary="Delete a class and its examples", dependencies=IF_MATCH, responses=PRECONDITION)
def delete_class(srv: Srv, name: str, label: str) -> Response:
    """Delete a class and all of its examples (undo brings them back)."""
    t = srv.task(name)
    with refusals():
        t.delete_class(label)
    return task_json(t)


# ---- examples -----------------------------------------------------------
@router.get("/api/tasks/{name}/examples", response_model=ExamplePage, tags=["examples"],
            summary="List examples, a page at a time")
def list_examples(srv: Srv, name: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),  # pylint: disable=too-many-arguments,too-many-positional-arguments
                  label: str | None = None, tag: str | None = None) -> Response:
    """One page of the task's examples, optionally only those with ``label`` (404 CLASS_NOT_FOUND
    if the task has no such class) and/or ``tag``; ``total`` counts every match."""
    t = srv.task(name)
    if label is not None and label not in t.class_names():
        near = closest(label, t.class_names())
        raise ApiProblem(404, errors.CLASS_NOT_FOUND, f"No class {label!r} in {name}"
                         + (f"; did you mean {' or '.join(map(repr, near))}?" if near else ""),
                         task=name, label=label, similar_to=near)
    found = t.find_examples(label, tag)
    return json_response(ExamplePage, {"total": len(found), "offset": offset,
                               "examples": [e.model_dump() for e in found[offset : offset + limit]]})


def _downloads(srv: Server, urls: list[str]) -> list[services.Blob]:
    """Fetch image URLs for the server (public addresses only); 403 unless it allows URLs. A few
    at a time, with a budget for the whole request: at most the request size limit in all and
    ``URL_FETCH_SECONDS`` overall, so one request can't hold gigabytes or a thread for hours."""
    if not srv.settings.allow_urls:
        raise ApiProblem(403, errors.URL_FETCH_DISABLED, "This server doesn't fetch URLs; start it with "
                         "--allow-urls (FEWSHOT_ALLOW_URLS=1), or upload the images")
    return services.fetch_urls(urls, srv.settings.max_upload_bytes, public_only=True, seconds=URL_FETCH_SECONDS,
                               max_total=srv.settings.max_request_bytes, workers=URL_FETCH_WORKERS)


UNDO_GROUP = Form(None, max_length=64, description=(
    "Any id the client chooses for one upload it sends in several requests: consecutive uploads "
    "to the task with the same id, with no other change in between, are one undo step"))
NEW_CLASS = Form(False, description="Create the class even if its name is close to an existing class's (which is "
                                    "otherwise refused as a likely typo, 422 CLASS_NAME_NEAR_MISS)")
def _add_examples(srv: Server, name: str, label: str,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                  blobs: Iterator[services.Blob] | list[services.Blob], undo_group: str | None = None,
                  new_class: bool = False) -> Response:
    """Add the readable ``blobs`` as examples of ``label`` (see ``routing.added_status``)."""
    t = srv.task(name)
    check_label(t, label, new_class, "/label")
    with refusals():
        r = services.add_examples(t, label, blobs, undo_group, new_class=new_class)
    status = added_status(r, "No images added")
    return json_response(AddResult, {
        "added": [e.model_dump() for e in r.new], "task": task_detail(t), "errors": file_errors(r),
        "unchanged": [e.model_dump() for e in r.unchanged],
        "moved": [{"id": e.id, "previous_label": was} for e, was in r.moved], "created_classes": r.created},
        status, etag(t))


@router.post("/api/tasks/{name}/examples", status_code=201, response_model=AddResult, tags=["examples"],
             summary="Upload labelled images", dependencies=[*MULTIPART, *IF_MATCH],
             responses={**ADDED, **UPLOAD, **PRECONDITION})
async def add_examples(srv: Srv, name: str, label: str = Form(...), files: list[UploadFile] = File(...),  # pylint: disable=too-many-arguments,too-many-positional-arguments
                       undo_group: str | None = UNDO_GROUP, new_class: bool = NEW_CLASS) -> Response:
    """Upload images as examples of ``label``, creating the class if needed (a new name close to
    an existing class's needs ``new_class``). Images are deduplicated by content, so sending an
    upload again is safe: 201 if anything was new, 200 if every image was there already, 400
    NO_IMAGES_ADDED if none could be read; unreadable files are listed in ``errors``."""
    return await srv.run(_add_examples, srv, name, label, uploads(srv, files), undo_group or None, new_class)


def _add_examples_by_url(srv: Server, name: str, body: LabelledUrls) -> Response:
    t = srv.task(name)
    check_label(t, body.label, body.new_class, "/label")  # before downloading anything
    return _add_examples(srv, name, body.label, _downloads(srv, body.urls), new_class=body.new_class)


@router.post("/api/tasks/{name}/examples/urls", status_code=201, response_model=AddResult, tags=["examples"],
             summary="Add labelled images by URL", dependencies=IF_MATCH, responses={**ADDED, **PRECONDITION})
async def add_examples_by_url(srv: Srv, name: str, body: LabelledUrls) -> Response:
    """Download images (server started with --allow-urls) and add them as examples of ``label``,
    as uploads are added."""
    return await srv.run(_add_examples_by_url, srv, name, body)


@router.patch("/api/tasks/{name}/examples/{eid}", response_model=TaskEdited, tags=["examples"],
              summary="Relabel, tag or annotate an example", dependencies=IF_MATCH, responses=PRECONDITION)
def edit_example(srv: Srv, name: str, eid: str, body: ExampleEdit) -> Response:
    """Move an example to another (possibly new) class, and/or set its tags or note, as one change
    (one undo step). A new class name close to an existing class's needs ``new_class``;
    ``created_classes`` names the class if this created it."""
    t = srv.task(name)
    before = t.class_names()
    with refusals():
        t.example(eid)  # 404 for an unknown id, even if the edit is empty
    if body.label is not None:
        check_label(t, body.label, body.new_class, "/label")
    with refusals():
        if body.label is not None or body.tags is not None or body.note is not None:
            t.edit_examples([eid], label=body.label, tags=body.tags, note=body.note)
    return edited_json(t, before)


@router.delete("/api/tasks/{name}/examples/{eid}", response_model=TaskDetail, tags=["examples"],
               summary="Delete an example", dependencies=IF_MATCH, responses=PRECONDITION)
def delete_example(srv: Srv, name: str, eid: str) -> Response:
    """Delete one example."""
    t = srv.task(name)
    with refusals():
        t.delete_example(eid)
    return task_json(t)


@router.patch("/api/tasks/{name}/examples", response_model=TaskEdited, tags=["examples"],
              summary="Relabel or tag several examples", dependencies=IF_MATCH, responses=PRECONDITION)
def edit_examples(srv: Srv, name: str, body: BulkEdit) -> Response:
    """Move several examples to another (possibly new) class and/or tag them, as one change (one
    undo step); an unknown id changes nothing (404). A new class name close to an existing
    class's needs ``new_class``; ``created_classes`` names the class if this created it."""
    t = srv.task(name)
    before = t.class_names()
    with refusals():
        # 404 for an unknown id, even if the edit is empty; by dict, not one scan per id (minutes for many)
        t._examples(body.ids)  # pylint: disable=protected-access
    if body.label is not None:
        check_label(t, body.label, body.new_class, "/label")
    with refusals():
        if body.label is not None or body.add_tags or body.remove_tags:
            t.edit_examples(body.ids, label=body.label, add_tags=body.add_tags, remove_tags=body.remove_tags)
    return edited_json(t, before)


@router.post("/api/tasks/{name}/examples/delete", response_model=TaskDetail, tags=["examples"],
             summary="Delete several examples", dependencies=IF_MATCH, responses=PRECONDITION)
def delete_examples(srv: Srv, name: str, body: ExampleIds) -> Response:
    """Delete several examples; an unknown id deletes nothing (404)."""
    t = srv.task(name)
    with refusals():
        t.delete_examples(body.ids)
    return task_json(t)


# Image ids are content hashes, so an image never changes; "private": kept by the browser only, not by
# shared caches (it may need the token). Other /api responses get "no-store" (see access.guard).
CACHE_FOREVER = {"Cache-Control": "private, max-age=31536000, immutable"}
IMAGE_HEADERS = {"X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'; sandbox"}


def _image(srv: Server, name: str, eid: str, size: int | None) -> FileResponse | Response:
    with refusals():
        path = srv.task(name).image_path(eid)
    if size is None:
        # Only ever served as an image (never e.g. HTML from a crafted import), and never sniffed.
        kind = mimetypes.guess_type(path.name)[0] or ""
        safe = kind if kind.startswith("image/") and kind != "image/svg+xml" else "application/octet-stream"
        return FileResponse(path, media_type=safe, headers={**CACHE_FOREVER, **IMAGE_HEADERS})
    try:
        data = thumbnail(path.read_bytes(), size)
    except FileNotFoundError as e:  # deleted between the lookup and the read
        raise ApiProblem(404, errors.EXAMPLE_NOT_FOUND, f"No example {eid!r}", example_id=eid) from e
    except ValueError as e:
        raise from_error(e) from e
    return Response(data, media_type="image/jpeg", headers={**CACHE_FOREVER, **IMAGE_HEADERS})


@router.get("/api/tasks/{name}/examples/{eid}/image", response_model=None, tags=["examples"],
            summary="Get an example's image or thumbnail")
async def example_image(srv: Srv, name: str, eid: str,
                        size: int | None = Query(None, ge=16, le=2048)) -> FileResponse | Response:
    """Serve an example's image file, or with ``size`` a JPEG scaled down to fit in that many pixels."""
    return await srv.run(_image, srv, name, eid, size)


# ---- inbox --------------------------------------------------------------
def _add_inbox(srv: Server, name: str, blobs: Iterator[services.Blob] | list[services.Blob],
               undo_group: str | None = None) -> Response:
    """Put the readable ``blobs`` in the inbox (see ``routing.added_status``)."""
    t = srv.task(name)
    with refusals():
        r = services.add_to_inbox(t, blobs, undo_group)
    status = added_status(r, "Nothing added to the inbox")
    return json_response(InboxAdded, {"added": [x.model_dump() for x in r.new],
                                      "unchanged": [x.model_dump() for x in r.unchanged],
                                      "errors": file_errors(r), "task": task_detail(t)}, status, etag(t))


@router.post("/api/tasks/{name}/inbox", status_code=201, response_model=InboxAdded, tags=["inbox"],
             summary="Upload unlabelled images", dependencies=[*MULTIPART, *IF_MATCH],
             responses={**INBOX_ADDED, **UPLOAD, **PRECONDITION})
async def add_to_inbox(srv: Srv, name: str, files: list[UploadFile] = File(...),
                       undo_group: str | None = UNDO_GROUP) -> Response:
    """Upload unlabelled images to the task's inbox, to label later with the model's suggestions.
    Deduplicated by content, as examples are: 201 if anything was new, 200 if every image was
    waiting already; an image that is an example already is listed in ``errors``."""
    return await srv.run(_add_inbox, srv, name, uploads(srv, files), undo_group or None)


def _add_inbox_by_url(srv: Server, name: str, urls: list[str]) -> Response:
    srv.task(name)  # 404 before downloading anything
    return _add_inbox(srv, name, _downloads(srv, urls))


@router.post("/api/tasks/{name}/inbox/urls", status_code=201, response_model=InboxAdded, tags=["inbox"],
             summary="Add unlabelled images by URL", dependencies=IF_MATCH, responses={**INBOX_ADDED, **PRECONDITION})
async def add_urls_to_inbox(srv: Srv, name: str, body: Urls) -> Response:
    """Download images (server started with --allow-urls) into the task's inbox."""
    return await srv.run(_add_inbox_by_url, srv, name, body.urls)


def _inbox_page(srv: Server, name: str, sort: str, offset: int, limit: int) -> dict[str, Any]:
    with refusals():
        items = srv.pipe.inbox_suggestions(srv.task(name), sort)
    return {"total": len(items), "offset": offset, "items": items[offset : offset + limit]}


@router.get("/api/tasks/{name}/inbox", response_model=InboxPage, tags=["inbox"],
            summary="List the inbox with suggestions")
async def list_inbox(srv: Srv, name: str, sort: InboxSort = "uncertain",
                     offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)) -> Response:
    """One page (``limit`` items from ``offset``; ``total`` counts all) of the inbox with
    suggestions: ``uncertain`` first (the most useful to label), ``confident`` first (to accept
    in bulk), ``novel`` first (unlike any example), or as ``added``."""
    return await srv.render(InboxPage, _inbox_page, srv, name, sort, offset, limit)


@router.post("/api/tasks/{name}/inbox/accept", response_model=TaskEdited, tags=["inbox"],
             summary="Label inbox images", dependencies=IF_MATCH, responses=PRECONDITION)
def accept_inbox(srv: Srv, name: str, body: InboxLabels) -> Response:
    """Label inbox images, moving them to the examples (one undo step); 404 if an id isn't
    waiting. A new class name close to an existing class's needs ``new_class``;
    ``created_classes`` names the classes this created."""
    t = srv.task(name)
    before = t.class_names()
    first: dict[str, str] = {}  # each label, with the first id given it
    for iid, label in body.labels.items():
        first.setdefault(label, iid)
    for label, iid in first.items():
        check_label(t, label, body.new_class, "/labels/" + errors.escape(iid))
    with refusals():
        t.accept(body.labels)
    return edited_json(t, before)


def _accept_confident(srv: Server, name: str, body: AcceptConfident) -> Response:
    t = srv.task(name)
    with refusals():
        services.accept_confident(srv.pipe, t, body.min_confidence, body.ids)
    return task_json(t)


@router.post("/api/tasks/{name}/inbox/accept-confident", response_model=TaskDetail, tags=["inbox"],
             summary="Accept the confident suggestions", dependencies=IF_MATCH, responses=PRECONDITION)
async def accept_confident(srv: Srv, name: str, body: AcceptConfident) -> Response:
    """Label as the model suggests every inbox image (of ``ids``, if given) it is at least
    ``min_confidence`` sure of and doesn't reject, in one undo step."""
    return await srv.run(_accept_confident, srv, name, body)


@router.post("/api/tasks/{name}/inbox/dismiss", response_model=TaskDetail, tags=["inbox"],
             summary="Dismiss inbox images", dependencies=IF_MATCH, responses=PRECONDITION)
def dismiss_inbox(srv: Srv, name: str, body: ExampleIds) -> Response:
    """Remove images from the inbox without labelling them (undo brings them back)."""
    t = srv.task(name)
    with refusals():
        t.dismiss(body.ids)
    return task_json(t)


# ---- inference ----------------------------------------------------------
TOP_K = Query(None, ge=1, description="Return only this many of the highest scores per image")
EVIDENCE = Query(0, ge=0, le=20, description="Also return this many of the most similar labelled examples")
REVISION = Query(None, ge=0, description="Predict with this earlier version of the task (one kept for undo)")
FOLDS = Query(None, ge=2, description="Cross-validation folds (default: leave-one-out up to 100 or 300 examples, "
                                      "then 10- or 5-fold)")


def _predict(srv: Server, name: str, blobs: Iterator[services.Blob] | list[services.Blob],  # pylint: disable=too-many-arguments,too-many-positional-arguments
             top_k: int | None, evidence: int, revision: int | None) -> list[dict[str, Any]]:
    """Classify ``blobs`` ``PREDICT_CHUNK`` at a time, all with the same version of the task."""
    t = srv.task(name)
    with refusals():
        cfg = t.config if revision is None else t.version(revision)
    out: list[dict[str, Any]] = []
    for part in chunked(blobs, PREDICT_CHUNK):
        with refusals():
            preds = services.predict(srv.pipe, t, part, top_k=top_k, evidence=evidence, cfg=cfg)
        out += [{"file": b.name, **p.to_dict()} for b, p in zip(part, preds, strict=True)]
    return out


@router.post("/api/tasks/{name}/predict", response_model=list[PredictionOut], tags=["predict"],
             summary="Classify uploaded images", dependencies=MULTIPART, responses=UPLOAD)
async def predict(srv: Srv, name: str, files: list[UploadFile] = File(...),  # pylint: disable=too-many-arguments,too-many-positional-arguments
                  top_k: int | None = TOP_K, evidence: int = EVIDENCE,
                  revision: int | None = REVISION) -> Response:
    """Classify uploaded images, in the order sent; an unreadable file gets ``error`` set and its
    other fields empty (null scores). Each prediction says which ``revision`` of the task made it."""
    return await srv.render(list[PredictionOut], _predict, srv, name, uploads(srv, files), top_k, evidence, revision)


def _predict_urls(srv: Server, name: str, urls: list[str], top_k: int | None, evidence: int,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                  revision: int | None) -> list[dict[str, Any]]:
    srv.task(name)  # 404 before downloading anything
    return _predict(srv, name, _downloads(srv, urls), top_k, evidence, revision)


@router.post("/api/tasks/{name}/predict/urls", response_model=list[PredictionOut], tags=["predict"],
             summary="Classify images by URL")
async def predict_urls(srv: Srv, name: str, body: Urls, top_k: int | None = TOP_K,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                       evidence: int = EVIDENCE, revision: int | None = REVISION) -> Response:
    """Download images (server started with --allow-urls) and classify them; ``file`` is the URL."""
    return await srv.render(list[PredictionOut], _predict_urls, srv, name, body.urls, top_k, evidence, revision)


def _duplicates(srv: Server, name: str, threshold: float, limit: int) -> list[dict[str, Any]]:
    with refusals():
        return srv.pipe.near_duplicates(srv.task(name), threshold, limit)


@router.get("/api/tasks/{name}/duplicates", response_model=list[DuplicatePair], tags=["predict"],
            summary="Find near-duplicate examples")
async def duplicates(srv: Srv, name: str, threshold: float = Query(0.95, ge=0.5, le=1.0),
                     limit: int = Query(200, ge=1, le=5000)) -> Response:
    """Pairs of near-identical examples (cosine similarity >= ``threshold``), most similar first."""
    return await srv.render(list[DuplicatePair], _duplicates, srv, name, threshold, limit)


def _evaluate(srv: Server, name: str, folds: int | None) -> dict[str, Any]:
    t = srv.task(name)
    with refusals():
        return services.evaluate(srv.pipe, t, folds)


@router.post("/api/tasks/{name}/evaluate", response_model=EvalReport, tags=["evaluate"],
             summary="Cross-validate a task (synchronous)")
async def evaluate(srv: Srv, name: str, folds: int | None = FOLDS) -> Response:
    """Cross-validate the task and answer with the report, which can take minutes for a large
    task: prefer POST .../evaluate/jobs, which reports progress and can be cancelled. 400
    NO_EXAMPLES for a task without examples."""
    return await srv.render(EvalReport, _evaluate, srv, name, folds)


# ---- background jobs ----------------------------------------------------
def _job(srv: Server, jid: str) -> Job:
    job = srv.jobs.get(jid)
    if job is None:
        raise ApiProblem(404, errors.JOB_NOT_FOUND, f"No job {jid!r}: it finished long ago, or the server was "
                         "restarted since", job_id=jid)
    return job


def _job_json(job: Job, status: int = 200) -> Response:
    """The job as a response; while it runs, Retry-After says when to poll again."""
    out = job.to_dict()
    out["problem"] = None if job.code is None else problem_body(
        job.error_status or 500, job.code, job.error or "", f"/api/jobs/{job.id}", job_id=job.id, task=job.task)
    headers = {"Retry-After": str(POLL_SECONDS)} if out["status"] in {"running", "cancelling"} else {}
    if status == 202:
        headers["Location"] = f"/api/jobs/{job.id}"
    return json_response(JobOut, out, status, headers)


def _start(srv: Server, job: Job, work: Callable[[Callable[[str, int, int], None]], Any]) -> Response:
    """Start ``job``, or return the same one already running; 429 if ``MAX_JOBS`` are running or
    being cancelled."""
    try:
        return _job_json(srv.jobs.start(job, work, limit=MAX_JOBS), 202)
    except TooManyJobs:
        raise ApiProblem(429, errors.TOO_MANY_JOBS, f"{MAX_JOBS} jobs are already running or being cancelled; try "
                         "again when one has finished", headers={"Retry-After": str(JOBS_RETRY_SECONDS)}) from None


@router.post("/api/tasks/{name}/evaluate/jobs", status_code=202, response_model=JobOut, tags=["evaluate", "jobs"],
             summary="Start cross-validating a task", responses=JOB_STARTED)
def start_evaluation(srv: Srv, name: str, folds: int | None = FOLDS) -> Response:
    """Start cross-validating the task in the background; poll the job its Location header names
    (GET /api/jobs/{id}) for progress (stage "embedding" counts images, "evaluating" counts folds)
    and the report. A task that can't be evaluated is refused at once (400 NO_EXAMPLES). The same
    evaluation of the same version already running is shared rather than started again."""
    t = srv.task(name)
    with refusals():
        check_evaluable(t.config)
    job = Job(task=name, kind="evaluate" if folds is None else f"evaluate:{folds}", revision=t.config.revision,
              uid=t.config.uid)
    return _start(srv, job, lambda progress: services.evaluate(srv.pipe, t, folds, progress))


MAX_COMPARED = 8  # embedders per comparison


@router.post("/api/tasks/{name}/compare/jobs", status_code=202, response_model=JobOut, tags=["evaluate", "jobs"],
             summary="Start comparing embedders on a task", responses=JOB_STARTED)
def start_comparison(srv: Srv, name: str, body: Compare) -> Response:
    """Start cross-validating the task with each of ``embedders`` in the background (the same
    allowlist as choosing a task's embedder applies); the job's result compares their accuracy.
    Stages are like "2/3 embedding". Refused at once as POST .../evaluate/jobs is."""
    from .evaluate import compare_embedders  # pylint: disable=import-outside-toplevel  # with evaluate's other imports

    t = srv.task(name)
    with refusals():
        specs = services.check_specs(body.embedders, MAX_COMPARED)
        check_evaluable(t.config)
    for s in specs:
        srv.allowed_embedder(s)  # loaded by the job: a model that fails is reported in its result
    job = Job(task=name, kind="compare", revision=t.config.revision, uid=t.config.uid,
              key=f"{body.folds}:" + "\n".join(specs))
    return _start(srv, job, lambda progress: compare_embedders(srv.pipe, t, specs, body.folds, progress=progress))


@router.get("/api/jobs/{jid}", response_model=JobOut, tags=["jobs"], summary="Get a job's progress and result")
def get_job(srv: Srv, jid: str) -> Response:
    """A job's status, progress and (once done) result; while it runs, Retry-After says when to
    poll again. A failed job has ``error`` and ``problem`` (with its ``code``)."""
    return _job_json(_job(srv, jid))


@router.delete("/api/jobs/{jid}", response_model=JobOut, tags=["jobs"], summary="Cancel a job")
def cancel_job(srv: Srv, jid: str) -> Response:
    """Ask a running job to stop: its status becomes "cancelling", then "cancelled" at its next
    progress step. Until then it still counts towards the job limit."""
    job = _job(srv, jid)
    job.request_cancel()
    return _job_json(job)


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
            raise ApiProblem(404, errors.ROUTE_NOT_FOUND,  # unknown API routes must not get the HTML page
                             f"No route /{path}; see /docs for the API's routes")
        f = dist / path
        if path and f.is_file() and dist in f.resolve().parents:
            return FileResponse(f, headers=UI_HEADERS)
        return FileResponse(dist / "index.html", headers={**UI_HEADERS, "Cache-Control": "no-cache"})
