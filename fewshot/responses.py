"""JSON responses built where they are made, not on the event loop; the errors every route may
answer, and the headers of the UI's pages.

FastAPI validates and serialises an endpoint's result against its ``response_model`` on the event
loop, where a large task's response (10 MB for 50,000 examples) stalls every other request. The
endpoints return ``json_response`` instead, built in their own thread, and still declare
``response_model`` for the OpenAPI schema.
"""

from __future__ import annotations

import functools
from typing import Any, get_origin

from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

RENDER_CHUNK = 1000  # list items validated and serialised at a time (see render)


@functools.cache
def _adapter(model: Any) -> TypeAdapter[Any]:
    return TypeAdapter(model)


def _big(content: Any) -> bool:
    """Whether ``content`` holds a list longer than ``RENDER_CHUNK`` (directly, or in a dict)."""
    if isinstance(content, list):
        return len(content) > RENDER_CHUNK
    return isinstance(content, dict) and any(_big(v) for v in content.values())


def render(model: Any, content: Any) -> bytes:
    """``content`` validated as ``model`` and serialised to JSON, as FastAPI does for a
    ``response_model``. A long list (of a list model, or in a field of a model) is done
    ``RENDER_CHUNK`` items at a time: pydantic holds the GIL for a whole call, and one call on
    50,000 examples would hold it for most of 100 ms, stalling the event loop."""
    if get_origin(model) is list and isinstance(content, list) and len(content) > RENDER_CHUNK:
        adapter = _adapter(model)
        parts = (adapter.dump_json(adapter.validate_python(content[i : i + RENDER_CHUNK]))[1:-1]
                 for i in range(0, len(content), RENDER_CHUNK))
        return b"[" + b",".join(parts) + b"]"
    if isinstance(model, type) and issubclass(model, BaseModel) and isinstance(content, dict) and _big(content):
        fields = model.model_fields
        if all(k in content or not f.is_required() for k, f in fields.items()):  # else: let pydantic say why
            return b"{" + b",".join(
                f'"{k}":'.encode() + render(f.annotation, content[k] if k in content else f.get_default(
                    call_default_factory=True)) for k, f in fields.items()) + b"}"
    adapter = _adapter(model)
    return adapter.dump_json(adapter.validate_python(content))


class Violation(BaseModel):
    """One problem of a request that doesn't validate (an entry of a 422's ``errors``)."""

    pointer: str | None = None  # JSON pointer into the body ("" is the whole body)
    parameter: str | None = None  # or the query, path or header parameter
    detail: str  # what is wrong, naming the value received
    code: str  # the kind of problem (pydantic's error type, or unknown_parameter)
    loc: list[str | int] = Field(json_schema_extra={"deprecated": True})  # pydantic's: use pointer or parameter
    msg: str = Field(json_schema_extra={"deprecated": True})  # pydantic's message: use detail


class Problem(BaseModel):
    """An error (RFC 9457, sent as ``application/problem+json``). ``code`` is stable and meant for
    programs; ``detail`` is one line for people. The other members are present when they apply."""

    model_config = ConfigDict(extra="allow")  # clients must ignore members they don't know

    type: str  # https://fewshot.dev/problems/<code>, or about:blank for plain HTTP errors
    title: str  # the code's (or status's) short summary
    status: int
    detail: str
    instance: str  # the request path
    code: str
    task: str | None = None
    label: str | None = None
    example_id: str | None = None
    job_id: str | None = None
    revision: int | None = None  # the task's current revision (412)
    limit_bytes: int | None = None  # the size limit passed (413)
    similar_to: list[str] | None = None  # the closest existing names
    pointer: str | None = None  # the body member concerned
    errors: list[Violation] | None = None  # every problem of a request that doesn't validate (422)


def json_response(model: Any, content: Any, status: int = 200, headers: dict[str, str] | None = None) -> Response:
    """``content`` as a JSON response validated as ``model`` (see ``render``), made in the
    calling thread. FastAPI would do this on the event loop, where a large task's response (10 MB
    for 50,000 examples) would stall every other request; endpoints return this instead, and
    still declare ``response_model`` for the OpenAPI schema."""
    return Response(render(model, content), status, headers=headers, media_type="application/json")


def _problem(description: str) -> dict[str, Any]:
    return {"model": Problem, "description": description + " (application/problem+json)"}


# What every route may answer: the access checks and body limits (``access``, ``guard``) and the
# errors of the routes themselves (see the README's status policy).
ERRORS: dict[int | str, dict[str, Any]] = {
    400: _problem("Well-formed, but can't be done as asked (an unreadable image, a bad name)"),
    401: _problem("Not signed in, or a wrong token (a server with a token)"),
    403: _problem("Cross-site request, a server without a token reached from another machine, or a "
                  "setting this server refuses"),
    404: _problem("No such task, class, example, job or route"),
    409: _problem("Conflicts with the current state: a name taken, nothing to undo or redo"),
    413: _problem("Request body over its size limit"),
    422: _problem("The request doesn't validate: errors lists every problem"),
    429: _problem("Too many wrong tokens from this address, or too many jobs running; see Retry-After"),
    500: _problem("Internal error"),
}
PRECONDITION: dict[int | str, dict[str, Any]] = {412: _problem("If-Match doesn't name the task's revision")}
UPLOAD: dict[int | str, dict[str, Any]] = {415: _problem("Not a multipart/form-data body")}

# The built UI loads only its own scripts and styles (no inline ones; React sets styles through
# the DOM, which style-src allows), shows thumbnails from the API, and previews files picked for
# prediction as blob: URLs. No other site may frame it (clickjacking).
UI_CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
          "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
UI_HEADERS = {"Content-Security-Policy": UI_CSP, "X-Frame-Options": "DENY", "X-Content-Type-Options": "nosniff"}
