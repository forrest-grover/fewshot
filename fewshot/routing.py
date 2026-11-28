"""What the API's routes share besides the server itself: the checks that run before them
(unknown query parameters, the kind of body, If-Match), reading uploads and reporting what they
added, the near-miss class name check, a task's detail and ETag, and OpenAPI metadata (tags,
extra responses, and what FastAPI can't declare: see ``amend_openapi``)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI, Header, Request, UploadFile, params
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool

from . import errors, preconditions, services
from .embedders import spec_supports_text
from .errors import ApiProblem, from_error
from .exceptions import ClassNameNearMiss
from .responses import json_response
from .schemas import AddResult, InboxAdded, TaskDetail, TaskEdited
from .store import Task
from .text import check_class_name, closest

if TYPE_CHECKING:
    from .server import Server

log = logging.getLogger(__name__)

TAGS = [
    {"name": "tasks", "description": "Create, configure, rename, copy, export and import tasks; undo and redo"},
    {"name": "classes", "description": "A task's classes: names, descriptions and reject thresholds"},
    {"name": "examples", "description": "Labelled images: upload (deduplicated by content), relabel, tag, delete"},
    {"name": "inbox", "description": "Unlabelled images with the model's suggestions, to label"},
    {"name": "predict", "description": "Classify images, and find near-duplicate examples"},
    {"name": "evaluate", "description": "Cross-validate a task, or compare embedders on it"},
    {"name": "jobs", "description": "Background jobs: poll for progress and results, or cancel"},
    {"name": "auth", "description": "Sessions for a server with a token (scripts send Authorization: Bearer)"},
    {"name": "system", "description": "Health, version and the models available"},
]


async def known_params(request: Request) -> None:
    """422 for a query parameter the route doesn't take (``?topk=`` for ``?top_k=``), as for
    an unknown body field: a typo must not be silently ignored."""
    route = request.scope.get("route")
    if not isinstance(route, APIRoute) or not request.query_params:
        return
    known = query_names(route)
    unknown = [k for k in dict.fromkeys(request.query_params.keys()) if k not in known]
    if unknown:
        raise RequestValidationError([{
            "type": "unknown_parameter", "loc": ("query", k), "input": request.query_params[k],
            "msg": f"unknown parameter {k!r}" + (f"; did you mean {near[0]!r}?" if (near := closest(k, known)) else
                                                 f"; this route takes {', '.join(known) or 'none'}")}
            for k in unknown])


async def json_only(request: Request) -> None:
    """415 for a route that takes a JSON body sent a body of another kind, or of none (a form, or
    ``curl -d`` without ``-H 'content-type: application/json'``), as upload routes refuse JSON: the
    framework would read it as no body at all, or one that isn't an object (422)."""
    route = request.scope.get("route")
    kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
    sent = request.headers.get("content-length", "0") != "0" or "transfer-encoding" in request.headers
    is_json = kind == "application/json" or (kind.startswith("application/") and kind.endswith("+json"))
    if is_json or not (kind or sent) or not isinstance(route, APIRoute) or not takes_json(route):
        return
    raise ApiProblem(415, errors.UNSUPPORTED_MEDIA_TYPE, f"{request.url.path} expects a JSON body sent as "
                     f"application/json, not {kind or 'one without a content type'} (with curl: -H "
                     "'content-type: application/json')")


def takes_json(route: APIRoute) -> bool:
    """Whether ``route`` takes a JSON body (not a form, not none)."""
    return bool(route.dependant.body_params) and not any(
        isinstance(p.field_info, params.Form) for p in route.dependant.body_params)


def query_names(route: APIRoute) -> list[str]:
    """The query parameters ``route`` takes, its dependencies' among them (kept on the route)."""
    names: list[str] | None = getattr(route, "fewshot_query_names", None)
    if names is None:
        found: set[str] = set()
        todo = [route.dependant]
        while todo:
            d = todo.pop()
            found.update(p.alias for p in d.query_params)
            todo += d.dependencies
        names = sorted(found)
        route.fewshot_query_names = names  # type: ignore[attr-defined]
    return names


def etag(t: Task) -> dict[str, str]:
    """The ETag of the task as it is now: its revision, which If-Match may name (see ``IF_MATCH``)."""
    return {"ETag": f'"{t.config.revision}"'}


def task_json(t: Task, status: int = 200) -> Response:
    """The task's detail as a response (see ``responses.json_response``), with its ETag."""
    return json_response(TaskDetail, task_detail(t), status, etag(t))


def edited_json(t: Task, before: list[str]) -> Response:
    """The task's detail after a change that may create classes, naming the classes it has that
    it didn't have ``before`` (see ``TaskEdited``)."""
    known = set(before)
    created = [c for c in t.class_names() if c not in known]
    return json_response(TaskEdited, {**task_detail(t), "created_classes": created}, 200, etag(t))


async def _if_match(name: str, request: Request, if_match: str | None = Header(
        None, alias="If-Match", description='Change the task only if it is still at this revision: its ETag, '
                                            'such as "12" (412 PRECONDITION_FAILED if it moved on)')) -> None:
    """412 if the request has an If-Match header that doesn't name the task's current revision
    (as its ETag gives it), so a client can change a task only as it last saw it. Checked now, and
    again by the change itself under the task's write lock (see ``preconditions``), so of several
    changes sent with the same If-Match at once only one is made. Without If-Match, changes are made
    whatever the revision. Async, so ``expect`` holds for the request's whole context."""
    if if_match is None:
        return
    preconditions.expect(name, if_match)
    srv: Server = request.app.state.server
    t = await run_in_threadpool(srv.task, name)
    preconditions.require(name, if_match, t.config.revision)  # also when the request changes nothing


IF_MATCH = [Depends(_if_match)]


async def _multipart(request: Request) -> None:
    """415 for an upload route sent something other than a multipart form (say, JSON)."""
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise ApiProblem(415, errors.UNSUPPORTED_MEDIA_TYPE, f"{request.url.path} expects multipart/form-data with "
                         "the image files (as 'files') and its other fields as form fields")


MULTIPART = [Depends(_multipart)]


def uploads(srv: Server, files: list[UploadFile]) -> Iterator[services.Blob]:
    """The uploaded files, each read only when it is needed, so a big upload is never held in
    memory whole; one over the upload limit carries an error instead of its bytes. A file that
    can't be read any more (closed as the request was abandoned) stops the upload with a 400 and
    a log line: the batch being read isn't stored, batches stored before it stay."""
    limit = srv.settings.max_upload_bytes
    for i, f in enumerate(files):
        # Browsers (and fewshot.client) send a '"' in a file name as %22, per the HTML standard;
        # the multipart parser leaves it encoded.
        name = f.filename.replace("%22", '"') if f.filename else ""
        try:
            data = f.file.read(limit + 1)
        except (ValueError, OSError) as e:  # ValueError: read of a closed file
            log.warning("Stopped an upload at file %d of %d (%s): it can't be read any more (%s), most likely as "
                        "the client went away; batches stored before it are kept", i + 1, len(files), name, e)
            raise ApiProblem(400, errors.UPLOAD_INTERRUPTED, f"The upload stopped at file {i + 1} of {len(files)} "
                             f"({name}): it could no longer be read; images before it may have been stored") from None
        if len(data) > limit:
            yield services.Blob(name, None, f"File is larger than the {limit / 1024 / 1024:g} MB upload limit")
        else:
            yield services.Blob(name, data)


def file_errors(r: services.Added) -> list[dict[str, str]]:
    """Each file that couldn't be used, and why."""
    return [{"file": f, "error": why} for f, why in r.errors]


def added_status(r: services.Added, what: str) -> int:
    """201 if anything new was stored, 200 if every readable image was there already, else 400
    (no image could be read, or for the inbox every one is an example already), listing each
    file and why in ``errors``."""
    if r.new:
        return 201
    if r.added:
        return 200
    raise ApiProblem(400, errors.NO_IMAGES_ADDED, f"{what}: " + "; ".join(f"{f}: {why}" for f, why in r.errors),
                     errors=file_errors(r))


def check_label(t: Task, label: str, new_class: bool, pointer: str) -> None:
    """Refuse (422) a new class name close to an existing class's, unless ``new_class`` (see
    ``services.check_new_class``); ``pointer`` locates the label in the request."""
    try:
        services.check_new_class(t, check_class_name(label), new_class)
    except ClassNameNearMiss as e:
        raise from_error(e, pointer=pointer) from e
    except ValueError as e:
        raise from_error(e) from e


def task_detail(t: Task) -> dict[str, Any]:
    """The task as ``TaskDetail``: its settings, classes (with their counts) and examples."""
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
        "undo_group": t.undo_group(),
        "redo_label": t.redo_label(),
        "classes": [{"name": k.name, "description": k.description, "count": counts.get(k.name, 0),
                     "reject_below": k.reject_below} for k in c.classes],
        "examples": [e.model_dump() for e in c.examples],
    }


# Answers some routes give besides their own, for the schema.
ADDED: dict[int | str, dict[str, Any]] = {
    200: {"model": AddResult, "description": "Nothing new: every image was there already (see unchanged, moved)"}}


INBOX_ADDED: dict[int | str, dict[str, Any]] = {
    200: {"model": InboxAdded, "description": "Nothing new: every image was waiting already"}}


JOB_STARTED: dict[int | str, dict[str, Any]] = {202: {"headers": {
    "Location": {"description": "The job, to poll", "schema": {"type": "string"}},
    "Retry-After": {"description": "Seconds to wait before polling", "schema": {"type": "integer"}}}}}


# Responses that carry a task, and so its ETag.
WITH_ETAG = {"TaskDetail", "TaskEdited", "AddResult", "InboxAdded"}
REF = "#/components/schemas/"


def amend_openapi(app: FastAPI) -> None:
    """Make ``app``'s schema say what the server sends, which FastAPI can't declare: errors as
    ``application/problem+json``, the ETag of responses that carry a task, and 415 for a route
    whose body is of the wrong kind (see ``json_only``)."""
    build = app.openapi

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            for item in build()["paths"].values():
                for op in item.values():
                    _amend(op)
        assert app.openapi_schema is not None
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]


def _amend(op: dict[str, Any]) -> None:
    responses: dict[str, dict[str, Any]] = op["responses"]
    if "application/json" in op.get("requestBody", {}).get("content", {}):
        responses.setdefault("415", {"description": "Not an application/json body (application/problem+json)",
                                     "content": {"application/json": {"schema": {"$ref": REF + "Problem"}}}})
    for status, r in responses.items():
        content = r.get("content", {})
        ref = content.get("application/json", {}).get("schema", {}).get("$ref", "")
        if status[0] in "45" and ref == REF + "Problem":
            r["content"] = {errors.PROBLEM_JSON: content.pop("application/json"), **content}
        elif status[0] == "2" and ref.removeprefix(REF) in WITH_ETAG:
            r.setdefault("headers", {})["ETag"] = {"schema": {"type": "string"}, "description": (
                'The task\'s revision, such as "12", to send back as If-Match')}
