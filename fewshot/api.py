"""REST API (FastAPI). Run with ``fewshot serve``; docs at /docs.

The built web UI (web/dist) is served at / when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .classifiers import CLASSIFIERS
from .embedders import PRESETS, get_embedder, spec_supports_text
from .evaluate import evaluate_task
from .pipeline import Pipeline
from .store import Task, Workspace

ws = Workspace()
pipe = Pipeline(ws)
app = FastAPI(title="fewshot", version="0.1.0")


def _task(name: str) -> Task:
    try:
        return ws.get(name)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]))


def _bad(e: Exception) -> HTTPException:
    return HTTPException(400, str(e.args[0]) if e.args else str(e))


def _check_embedder(spec: str) -> None:
    """Load the embedder now so a bad spec fails here, not at predict time."""
    try:
        get_embedder(spec)
    except Exception as e:  # unknown model names raise a variety of errors
        msg = str(e).split(". Available:")[0]  # open_clip appends every known model name
        raise HTTPException(400, f"Could not load embedder {spec!r}: {msg[:300]}")


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
def models():
    return {
        "embedders": [p.__dict__ for p in PRESETS],
        "classifiers": sorted(CLASSIFIERS),
    }


# ---- tasks --------------------------------------------------------------
class TaskCreate(BaseModel):
    name: str
    embedder: Optional[str] = None
    classifier: str = "prototype"


class TaskSettings(BaseModel):
    embedder: Optional[str] = None
    classifier: Optional[str] = None
    classifier_params: Optional[dict[str, Any]] = None
    text_weight: Optional[float] = None
    reject_below: Optional[float] = None
    clear_reject: bool = False


@app.get("/api/tasks")
def list_tasks():
    return [t.summary() for t in ws.list_tasks()]


@app.post("/api/tasks", status_code=201)
def create_task(body: TaskCreate):
    kw = {"classifier": body.classifier}
    if body.embedder:
        _check_embedder(body.embedder)
        kw["embedder"] = body.embedder
    try:
        return _detail(ws.create(body.name, **kw))
    except ValueError as e:
        raise _bad(e)


@app.get("/api/tasks/{name}")
def get_task(name: str):
    return _detail(_task(name))


@app.patch("/api/tasks/{name}")
def update_task(name: str, body: TaskSettings):
    t = _task(name)
    kw = body.model_dump(exclude_none=True, exclude={"clear_reject"})
    if body.clear_reject:
        kw["reject_below"] = None
    if "classifier" in kw and "classifier_params" not in kw and kw["classifier"] != t.config.classifier:
        kw["classifier_params"] = {}
    if kw.get("embedder") and kw["embedder"] != t.config.embedder:
        _check_embedder(kw["embedder"])
    try:
        t.update_settings(**kw)
    except ValueError as e:
        raise _bad(e)
    return _detail(t)


@app.delete("/api/tasks/{name}", status_code=204)
def delete_task(name: str):
    _task(name)
    ws.delete(name)


# ---- classes ------------------------------------------------------------
class ClassBody(BaseModel):
    name: str
    description: Optional[str] = None


class ClassPatch(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@app.post("/api/tasks/{name}/classes", status_code=201)
def add_class(name: str, body: ClassBody):
    t = _task(name)
    try:
        t.upsert_class(body.name, body.description)
    except ValueError as e:
        raise _bad(e)
    return _detail(t)


@app.patch("/api/tasks/{name}/classes/{label}")
def patch_class(name: str, label: str, body: ClassPatch):
    t = _task(name)
    try:
        if body.description is not None:
            t.upsert_class(label, body.description)
        if body.name and body.name != label:
            t.rename_class(label, body.name)
    except (KeyError, ValueError) as e:
        raise _bad(e)
    return _detail(t)


class Describe(BaseModel):
    template: str = "a photo of a {}"
    overwrite: bool = False


@app.post("/api/tasks/{name}/describe")
def describe_classes(name: str, body: Describe):
    t = _task(name)
    try:
        t.describe_classes(body.template, body.overwrite)
    except ValueError as e:
        raise _bad(e)
    return _detail(t)


@app.delete("/api/tasks/{name}/classes/{label}")
def delete_class(name: str, label: str):
    t = _task(name)
    try:
        t.delete_class(label)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]))
    return _detail(t)


# ---- examples -----------------------------------------------------------
@app.post("/api/tasks/{name}/examples", status_code=201)
def add_examples(name: str, label: str = Form(...), files: list[UploadFile] = File(...)):
    t = _task(name)
    added, errors = [], []
    for f in files:
        try:
            added.append(t.add_example(f.file.read(), label, f.filename or "").model_dump())
        except ValueError as e:
            errors.append({"file": f.filename, "error": str(e)})
    return {"added": added, "errors": errors, "task": _detail(t)}


class Relabel(BaseModel):
    label: str


@app.patch("/api/tasks/{name}/examples/{eid}")
def relabel(name: str, eid: str, body: Relabel):
    t = _task(name)
    try:
        t.relabel(eid, body.label)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]))
    return _detail(t)


@app.delete("/api/tasks/{name}/examples/{eid}")
def delete_example(name: str, eid: str):
    t = _task(name)
    try:
        t.delete_example(eid)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]))
    return _detail(t)


@app.get("/api/tasks/{name}/examples/{eid}/image")
def example_image(name: str, eid: str):
    t = _task(name)
    try:
        return FileResponse(t.image_path(eid), headers={"Cache-Control": "max-age=31536000, immutable"})
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]))


# ---- inference ----------------------------------------------------------
@app.post("/api/tasks/{name}/predict")
def predict(name: str, files: list[UploadFile] = File(...)):
    t = _task(name)
    blobs = [f.file.read() for f in files]
    try:
        preds = pipe.predict(t, blobs)
    except ValueError as e:
        raise _bad(e)
    return [{"file": f.filename, **p.to_dict()} for f, p in zip(files, preds)]


@app.post("/api/tasks/{name}/evaluate")
def evaluate(name: str, folds: Optional[int] = None):
    try:
        return evaluate_task(pipe, _task(name), folds)
    except ValueError as e:
        raise _bad(e)


# ---- web UI -------------------------------------------------------------
_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = _dist / path
        if path and f.is_file() and _dist in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(_dist / "index.html")
