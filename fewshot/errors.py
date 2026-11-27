"""The REST API's errors: one RFC 9457 envelope (``application/problem+json``) with a stable
``code`` for every error the server answers, its own and the framework's alike.

Every error body is ``{type, title, status, detail, instance, code}``, plus members naming what it
concerns (``task``, ``label``, ``example_id``, ``job_id``, ``limit_bytes``, ``similar_to``...) and,
for a request that doesn't validate (422), ``errors``: one entry per problem, each with a JSON
``pointer`` into the body or the ``parameter`` concerned, a ``detail`` and a ``code``. ``detail``
is always one line for people; ``code`` is the contract for programs (``CODES`` lists them).
"""

from __future__ import annotations

import contextlib
import http
import logging
from collections.abc import Iterator, Mapping
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from .exceptions import ClassNameNearMiss, error_code, http_status, message

log = logging.getLogger(__name__)

PROBLEM_JSON = "application/problem+json"
PROBLEM_TYPES = "https://fewshot.dev/problems/"

# ---- the codes ------------------------------------------------------------------
TASK_NOT_FOUND = "TASK_NOT_FOUND"
TASK_UNREADABLE = "TASK_UNREADABLE"
TASK_EXISTS = "TASK_EXISTS"
CLASS_NOT_FOUND = "CLASS_NOT_FOUND"
CLASS_EXISTS = "CLASS_EXISTS"
CLASS_NAME_NEAR_MISS = "CLASS_NAME_NEAR_MISS"
EXAMPLE_NOT_FOUND = "EXAMPLE_NOT_FOUND"
REVISION_NOT_FOUND = "REVISION_NOT_FOUND"
JOB_NOT_FOUND = "JOB_NOT_FOUND"
NOT_FOUND = "NOT_FOUND"
NOTHING_TO_UNDO = "NOTHING_TO_UNDO"
NOTHING_TO_REDO = "NOTHING_TO_REDO"
INVALID_REQUEST = "INVALID_REQUEST"
VALIDATION_FAILED = "VALIDATION_FAILED"
UNKNOWN_CLASSIFIER = "UNKNOWN_CLASSIFIER"
INVALID_EMBEDDER = "INVALID_EMBEDDER"
EMBEDDER_NOT_ALLOWED = "EMBEDDER_NOT_ALLOWED"
URL_FETCH_DISABLED = "URL_FETCH_DISABLED"
NO_IMAGES_ADDED = "NO_IMAGES_ADDED"
NO_EXAMPLES = "NO_EXAMPLES"
UPLOAD_INTERRUPTED = "UPLOAD_INTERRUPTED"
PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
NOT_SIGNED_IN = "NOT_SIGNED_IN"
WRONG_TOKEN = "WRONG_TOKEN"
TOO_MANY_WRONG_TOKENS = "TOO_MANY_WRONG_TOKENS"
CROSS_SITE = "CROSS_SITE"
UNKNOWN_HOST = "UNKNOWN_HOST"
NO_SERVER_TOKEN = "NO_SERVER_TOKEN"
TOO_MANY_JOBS = "TOO_MANY_JOBS"
ROUTE_NOT_FOUND = "ROUTE_NOT_FOUND"
METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
PRECONDITION_FAILED = "PRECONDITION_FAILED"
INTERNAL = "INTERNAL"

# code -> (status, title): the table the README lists.
CODES: dict[str, tuple[int, str]] = {
    INVALID_REQUEST: (400, "The request can't be done as asked"),
    INVALID_EMBEDDER: (400, "The embedder can't be used"),
    UNKNOWN_CLASSIFIER: (400, "Unknown classifier"),
    NO_IMAGES_ADDED: (400, "No image could be added"),
    NO_EXAMPLES: (400, "The task has no examples"),
    UPLOAD_INTERRUPTED: (400, "The upload stopped part-way"),
    UNKNOWN_HOST: (400, "Unknown host"),
    NOT_SIGNED_IN: (401, "Not signed in"),
    WRONG_TOKEN: (401, "Wrong token"),
    EMBEDDER_NOT_ALLOWED: (403, "Embedder not allowed on this server"),
    URL_FETCH_DISABLED: (403, "This server doesn't fetch URLs"),
    CROSS_SITE: (403, "Cross-site request refused"),
    NO_SERVER_TOKEN: (403, "This server needs a token to be reached from elsewhere"),
    TASK_NOT_FOUND: (404, "No such task"),
    TASK_UNREADABLE: (404, "The task can't be read"),
    CLASS_NOT_FOUND: (404, "No such class"),
    EXAMPLE_NOT_FOUND: (404, "No such example"),
    REVISION_NOT_FOUND: (404, "That revision isn't kept"),
    JOB_NOT_FOUND: (404, "No such job"),
    NOT_FOUND: (404, "Not found"),
    ROUTE_NOT_FOUND: (404, "Not Found"),
    METHOD_NOT_ALLOWED: (405, "Method Not Allowed"),
    TASK_EXISTS: (409, "A task of that name exists"),
    CLASS_EXISTS: (409, "A class of that name exists"),
    NOTHING_TO_UNDO: (409, "Nothing to undo"),
    NOTHING_TO_REDO: (409, "Nothing to redo"),
    PRECONDITION_FAILED: (412, "The task changed since that revision"),
    PAYLOAD_TOO_LARGE: (413, "Request too large"),
    UNSUPPORTED_MEDIA_TYPE: (415, "Unsupported request content type"),
    VALIDATION_FAILED: (422, "The request doesn't validate"),
    CLASS_NAME_NEAR_MISS: (422, "Class name close to an existing class's"),
    TOO_MANY_WRONG_TOKENS: (429, "Too many wrong tokens"),
    TOO_MANY_JOBS: (429, "Too many jobs running"),
    INTERNAL: (500, "Internal Server Error"),
}
# Codes that mean no more than their HTTP status: their type is about:blank (RFC 9457 4.2.1).
PLAIN = {ROUTE_NOT_FOUND, METHOD_NOT_ALLOWED, INTERNAL}
# The code of an error that has none of its own, by status.
BY_STATUS = {400: INVALID_REQUEST, 401: NOT_SIGNED_IN, 404: ROUTE_NOT_FOUND, 405: METHOD_NOT_ALLOWED,
             413: PAYLOAD_TOO_LARGE, 415: UNSUPPORTED_MEDIA_TYPE, 422: VALIDATION_FAILED, 500: INTERNAL}
NEW_CLASS_HINT = "Send new_class=true to create it"
OWN_TYPES = {"unknown_parameter", "null_not_allowed"}  # 422 problems the server finds itself, their msg whole


class ApiProblem(HTTPException):
    """An error answered as a problem: its ``status``, stable ``code``, one-line ``detail``, and
    ``fields``, the extension members (``task``, ``label``, ``similar_to``...; None ones are left out)."""

    def __init__(self, status: int, code: str, detail: str, *, headers: Mapping[str, str] | None = None,
                 **fields: Any) -> None:
        super().__init__(status, " ".join(detail.split()), dict(headers) if headers else None)
        self.code = code
        self.fields = {k: v for k, v in fields.items() if v is not None}


def problem_body(status: int, code: str, detail: str, instance: str, **fields: Any) -> dict[str, Any]:
    """The problem document for an error: ``type`` from its ``code`` (``about:blank`` for codes
    that only repeat the status), ``title``, ``status``, one-line ``detail``, ``instance`` (the
    request path), ``code``, and ``fields`` that aren't None."""
    known = CODES.get(code)
    plain = code in PLAIN or known is None
    try:
        phrase = http.HTTPStatus(status).phrase
    except ValueError:  # pragma: no cover  # every status the server sends is a standard one
        phrase = "Error"
    return {"type": "about:blank" if plain else PROBLEM_TYPES + code.lower().replace("_", "-"),
            "title": phrase if plain or known is None else known[1], "status": status,
            "detail": " ".join(detail.split()), "instance": instance, "code": code,
            **{k: v for k, v in fields.items() if v is not None}}


def problem_response(status: int, code: str, detail: str, instance: str, headers: Mapping[str, str] | None = None,
                     **fields: Any) -> JSONResponse:
    """``problem_body`` as an ``application/problem+json`` response."""
    return JSONResponse(problem_body(status, code, detail, instance, **fields), status, headers=headers,
                        media_type=PROBLEM_JSON)


def from_error(e: KeyError | ValueError, **fields: Any) -> ApiProblem:
    """A domain error (see ``exceptions``) as a problem: its status, code and fields, with
    ``fields`` added. A likely typo of a class name says how to create it anyway."""
    detail = message(e)
    if isinstance(e, ClassNameNearMiss):
        detail = f"{detail} {NEW_CLASS_HINT}"
    return ApiProblem(http_status(e), error_code(e), detail, **{**getattr(e, "fields", {}), **fields})


@contextlib.contextmanager
def refusals(**fields: Any) -> Iterator[None]:
    """Answer a ``KeyError`` or ``ValueError`` raised inside as its problem (see ``from_error``)."""
    try:
        yield
    except (KeyError, ValueError) as e:
        raise from_error(e, **fields) from e


# ---- 422: requests that don't validate ------------------------------------------------
def escape(part: Any) -> str:
    """``part`` escaped as one JSON pointer segment (RFC 6901)."""
    return str(part).replace("~", "~0").replace("/", "~1")


def _short(value: Any) -> str | None:
    """``value`` for a message, if it is short and plain (a big bad body isn't sent back)."""
    if isinstance(value, bool | int | float) or value is None:
        return "null" if value is None else str(value).lower() if isinstance(value, bool) else str(value)
    if isinstance(value, str) and len(value) <= 80:
        return repr(value)
    return None


def violation(error: Mapping[str, Any]) -> dict[str, Any]:
    """One entry of a 422's ``errors``, from a pydantic error: a JSON ``pointer`` into the body, or
    the query, path or header ``parameter``; a ``detail`` naming the field, the rule and the value
    received; and pydantic's error type as ``code``. ``loc`` and ``msg`` are pydantic's, kept for
    clients written before ``pointer`` and ``detail`` (deprecated)."""
    loc = list(error.get("loc", ()))
    where, path = (loc[0], loc[1:]) if loc else ("body", [])
    kind = str(error.get("type", "invalid"))
    msg = str(error.get("msg", "Is invalid"))
    name = ".".join(map(str, path)) or ("the body" if where == "body" else str(where))
    if kind == "missing":
        detail = f"{name} is required"
    elif kind == "extra_forbidden":
        detail = f"unknown field {name!r}"
    elif kind.startswith(("value_error", "assertion_error")):
        detail = msg.split(", ", 1)[1] if msg.startswith(("Value error, ", "Assertion failed, ")) else msg
    elif kind in OWN_TYPES:
        detail = msg
    else:
        rule = msg[0].lower() + msg[1:] if msg else "is invalid"
        detail = f"{name} {rule.replace('input should be', 'must be', 1)}"
        value = error.get("input")
        got = value if where != "body" and isinstance(value, str) and len(value) <= 80 else _short(value)
        if got is not None:  # a query or path parameter is text: shown as sent
            detail += f"; got {got}"
    out: dict[str, Any] = {"detail": detail, "code": kind}
    if where == "body":
        out["pointer"] = "".join(f"/{escape(p)}" for p in path)
    else:
        out["parameter"] = name
    return {**out, "loc": [str(x) if not isinstance(x, int) else x for x in loc], "msg": msg}


def violations_detail(errors: list[dict[str, Any]]) -> str:
    """One line summing up a 422's ``errors``."""
    shown = "; ".join(e["detail"] for e in errors[:3])
    more = f"; and {len(errors) - 3} more" if len(errors) > 3 else ""
    return f"{len(errors)} problem{'s' if len(errors) != 1 else ''} with the request: {shown}{more}"


# ---- the handlers -------------------------------------------------------------------
async def http_problem(request: Request, exc: Exception) -> JSONResponse:
    """Any ``HTTPException`` (ours, or the framework's 404, 405 and body-parsing 400) as a problem."""
    assert isinstance(exc, StarletteHTTPException)
    if isinstance(exc, ApiProblem):
        code, fields = exc.code, exc.fields
    else:
        code, fields = BY_STATUS.get(exc.status_code, f"HTTP_{exc.status_code}"), {}
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    if code == ROUTE_NOT_FOUND and detail == "Not Found":
        detail = f"No route {request.url.path}; see /docs for the API's routes"
    elif code == METHOD_NOT_ALLOWED and detail == "Method Not Allowed":
        detail = f"{request.method} isn't allowed on {request.url.path}"
    return problem_response(exc.status_code, code, detail, request.url.path, exc.headers, **fields)


async def validation_problem(request: Request, exc: Exception) -> JSONResponse:
    """422 for a request that doesn't validate, listing every problem in ``errors``."""
    assert isinstance(exc, RequestValidationError)
    errors = [violation(e) for e in exc.errors()]
    return problem_response(422, VALIDATION_FAILED, violations_detail(errors), request.url.path, errors=errors)


async def internal_problem(request: Request, exc: Exception) -> JSONResponse:
    """500 for an unexpected error, noted in the log (the server then logs its traceback, as
    Starlette raises it on), while the body says only that it happened: the details can hold
    server paths."""
    log.error("Internal error answering %s %s: %s", request.method, request.url.path, type(exc).__name__)
    return problem_response(500, INTERNAL, "Internal error; see the server log", request.url.path)


def install(app: FastAPI) -> None:
    """Answer every error of ``app`` as a problem."""
    app.add_exception_handler(StarletteHTTPException, http_problem)
    app.add_exception_handler(RequestValidationError, validation_problem)
    app.add_exception_handler(Exception, internal_problem)
