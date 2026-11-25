"""REST API (FastAPI). Run with ``fewshot serve``; docs at /docs.

The built web UI (web/dist) is served at / when present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from .classifiers import CLASSIFIERS
from .embedders import PRESETS, check_embedder, spec_supports_text
from .evaluate import evaluate_task
from .pipeline import Pipeline, Prediction
from .store import Task, Workspace, thumbnail

ws = Workspace()
pipe = Pipeline(ws)
app = FastAPI(title="fewshot", version="0.1.0")


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
    """Load the embedder now so a bad spec fails here (400), not at predict time."""
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
    for f in files:
        data = _read_upload(f)
        if data is None:
            errors.append({"file": f.filename, "error": _too_large()})
            continue
        try:
            added.append(t.add_example(data, label, f.filename or "").model_dump())
        except ValueError as e:
            errors.append({"file": f.filename, "error": str(e)})
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
    """Cross-validate the task (leave-one-out unless ``folds`` is given)."""
    try:
        return evaluate_task(pipe, _task(name), folds)
    except ValueError as e:
        raise _bad(e) from e


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
