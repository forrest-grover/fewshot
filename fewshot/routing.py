"""What the API's routes share besides the server itself: the checks that run before them
(unknown query parameters, If-Match, multipart bodies), the near-miss class name check, a task's
detail and ETag, and OpenAPI metadata (tags, extra responses)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from fastapi.routing import APIRoute

from . import errors, services
from .embedders import spec_supports_text
from .errors import ApiProblem, from_error
from .exceptions import ClassNameNearMiss
from .responses import json_response
from .schemas import AddResult, InboxAdded, TaskDetail
from .store import Task
from .text import check_class_name, closest

if TYPE_CHECKING:
    from .api import Server

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


def _if_match(name: str, request: Request) -> None:
    """412 if the request has an If-Match header that doesn't name the task's current revision
    (as its ETag gives it), so a client can change a task only as it last saw it. Checked when the
    request arrives; without If-Match, changes are made whatever the revision."""
    given = request.headers.get("if-match")
    if given is None:
        return
    srv: Server = request.app.state.server
    current = srv.task(name).config.revision
    tags = [g.strip().removeprefix("W/").strip('"') for g in given.split(",")]
    if "*" not in tags and str(current) not in tags:
        raise ApiProblem(412, errors.PRECONDITION_FAILED, f"Task {name!r} is at revision {current}, not {given}: it "
                         f"changed since; get it again (GET /api/tasks/{name}) and redo the change if it still applies",
                         task=name, revision=current)


IF_MATCH = [Depends(_if_match)]


async def _multipart(request: Request) -> None:
    """415 for an upload route sent something other than a multipart form (say, JSON)."""
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise ApiProblem(415, errors.UNSUPPORTED_MEDIA_TYPE, f"{request.url.path} expects multipart/form-data with "
                         "the image files (as 'files') and its other fields as form fields")


MULTIPART = [Depends(_multipart)]


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
