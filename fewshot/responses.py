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
from pydantic import BaseModel, TypeAdapter

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


class Problem(BaseModel):
    """An error: what FastAPI's ``HTTPException`` and the server's access checks answer."""

    detail: str


def json_response(model: Any, content: Any, status: int = 200) -> Response:
    """``content`` as a JSON response validated as ``model`` (see ``render``), made in the
    calling thread. FastAPI would do this on the event loop, where a large task's response (10 MB
    for 50,000 examples) would stall every other request; endpoints return this instead, and
    still declare ``response_model`` for the OpenAPI schema."""
    return Response(render(model, content), status, media_type="application/json")


# What the access checks and body limits (``access``, ``guard``) answer on any route, for the schema.
ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": Problem, "description": "Not signed in (a server with a token)"},
    403: {"model": Problem, "description": "Cross-site request, or a server without a token reached from "
                                           "another machine"},
    413: {"model": Problem, "description": "Request body over its size limit"},
    429: {"model": Problem, "description": "Too many wrong tokens from this address"},
}

# The built UI loads only its own scripts and styles (no inline ones; React sets styles through
# the DOM, which style-src allows), shows thumbnails from the API, and previews files picked for
# prediction as blob: URLs. No other site may frame it (clickjacking).
UI_CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
          "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
UI_HEADERS = {"Content-Security-Policy": UI_CSP, "X-Frame-Options": "DENY", "X-Content-Type-Options": "nosniff"}
