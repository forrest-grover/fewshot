"""Command-line interface: ``fewshot --help``."""

from __future__ import annotations

import itertools
import json
import os
import shutil
import socket
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from .cli_common import (
    JSON_OPT,
    ROOT_OPT,
    Source,
)
from .cli_common import (
    add_files as _add_files,
)
from .cli_common import (
    echo_json as _echo_json,
)
from .cli_common import (
    error_text as _error_text,
)
from .cli_common import (
    expand as _expand,
)
from .cli_common import (
    fail as _fail,
)
from .cli_common import (
    get_task as _task,
)
from .cli_common import (
    images_in as _images_in,
)
from .cli_common import (
    make_pipeline as _pipe,
)
from .cli_common import (
    progress as _progress,
)
from .cli_common import (
    read_source as _read,
)
from .cli_common import (
    workspace as _ws,
)
from .cli_common import (
    write_csv as _write_csv,
)
from .cli_data import ex_app, inbox_app
from .embedders import DEFAULT_SPEC, PRESETS, check_embedder, spec_supports_text
from .fetch import is_url
from .store import Workspace
from .text import check_class_name

if TYPE_CHECKING:
    from .pipeline import Pipeline, Prediction

app = typer.Typer(help="Embedding-based few-shot image classification.", no_args_is_help=True)
cls_app = typer.Typer(help="Manage a task's classes.", no_args_is_help=True)
app.add_typer(cls_app, name="class")
cache_app = typer.Typer(help="Inspect and prune the embedding cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")
app.add_typer(ex_app, name="example")
app.add_typer(inbox_app, name="inbox")

def _check_embedder(spec: str) -> None:
    try:
        check_embedder(spec)
    except ValueError as e:
        _fail(str(e))


@app.command()
def models(as_json: bool = JSON_OPT) -> None:
    """List preset embedders (any openclip:/timm: spec also works)."""
    if as_json:
        _echo_json([p.__dict__ for p in PRESETS])
        return
    for p in PRESETS:
        typer.echo(f"{p.spec}\n    {p.label}. {p.notes}")


@app.command("create")
def create(
    name: str,
    embedder: str | None = typer.Option(None, help="Embedder spec; see `fewshot models`"),
    classifier: str = typer.Option("prototype", help="prototype | knn | logreg"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Create a task."""
    kw = {"classifier": classifier}
    if embedder is not None:
        _check_embedder(embedder)
        kw["embedder"] = embedder
    try:
        t = _ws(root).create(name, **kw)
    except ValueError as e:
        _fail(str(e))
    typer.echo(f"Created task {t.name} at {t.path}")


@app.command("list")
def list_tasks(as_json: bool = JSON_OPT, root: Path | None = ROOT_OPT) -> None:
    """List tasks."""
    if as_json:
        _echo_json([t.summary() for t in _ws(root).list_tasks()])
        return
    for t in _ws(root).list_tasks():
        s = t.summary()
        ev = s["last_evaluation"]
        acc = "      -" if not ev or ev["accuracy"] is None else f"acc {ev['accuracy']:.2f}"
        stale = "*" if ev and ev["revision"] != s["revision"] else " "
        typer.echo(f"{s['name']:24} {s['n_classes']:3} classes {s['n_examples']:5} examples {s['n_inbox']:4} inbox  "
                   f"{acc}{stale} {s['classifier']:9} {s['embedder']}")
        for w in s["warnings"]:
            typer.secho(f"    ! {w}", fg="yellow")


@app.command()
def show(name: str, as_json: bool = JSON_OPT, root: Path | None = ROOT_OPT) -> None:
    """Show a task's settings and per-class example counts."""
    t = _task(root, name)
    c = t.config
    counts = t.summary()["counts"]
    if as_json:
        _echo_json({**c.model_dump(exclude={"examples", "classes"}),
                    "classes": [{**k.model_dump(), "count": counts.get(k.name, 0)} for k in c.classes]})
        return
    typer.echo(f"{c.name}\n  embedder:   {c.embedder}\n  classifier: {c.classifier} {c.classifier_params or ''}"
               f"\n  text_weight: {c.text_weight}   reject_below: {c.reject_below}   "
               f"reject_below_zero_shot: {c.reject_below_zero_shot}   flip: {c.flip}   "
               f"temperature: {c.temperature or 1}")
    for cd in c.classes:
        desc = "  " + " | ".join(f'"{line}"' for line in cd.description.splitlines() if line.strip())
        thr = "" if cd.reject_below is None else f"  reject_below {cd.reject_below}"
        typer.echo(f"  {cd.name:24} {counts.get(cd.name, 0):5}{thr}{desc.rstrip()}")


def _parse_params(param: list[str], base: dict[str, Any]) -> dict[str, Any]:
    params = dict(base)
    for p in param:
        k, sep, v = p.partition("=")
        try:
            if not (k and sep):
                raise ValueError("expected KEY=VALUE")
            params[k] = json.loads(v)
        except ValueError as e:
            _fail(f"Bad --param {p!r}: {e}")
    return params


@app.command()
def config(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals  # one per CLI option
    name: str,
    embedder: str | None = None,
    classifier: str | None = None,
    param: list[str] = typer.Option([], help="Classifier param KEY=VALUE (e.g. k=3, scale=20). Repeatable."),
    text_weight: float | None = None,
    reject_below: float | None = typer.Option(None, help="Similarity threshold (-1 to 1) for 'unknown'"),
    clear_reject: bool = typer.Option(False, "--clear-reject", help="Turn off the 'unknown' threshold"),
    reject_below_zero_shot: float | None = typer.Option(
        None, help="Threshold for zero-shot predictions (no example images), e.g. 0.22 for CLIP"),
    clear_reject_zero_shot: bool = typer.Option(False, "--clear-reject-zero-shot"),
    flip: bool | None = typer.Option(None, "--flip/--no-flip",
                                     help="Average each image's embedding with its mirror image's"),
    temperature: float | None = typer.Option(None, help="Confidence calibration (see `fewshot eval`); 1 = none"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Change task settings; switching classifier resets its params unless --param is given."""
    t = _task(root, name)
    if clear_reject and reject_below is not None:
        _fail("Give either --reject-below or --clear-reject, not both")
    if clear_reject_zero_shot and reject_below_zero_shot is not None:
        _fail("Give either --reject-below-zero-shot or --clear-reject-zero-shot, not both")
    kw: dict[str, Any] = {}
    if embedder is not None and embedder != t.config.embedder:
        _check_embedder(embedder)
        kw["embedder"] = embedder
    if classifier and classifier != t.config.classifier:
        kw["classifier"] = classifier
        kw["classifier_params"] = {}
    if param:
        kw["classifier_params"] = _parse_params(param, kw.get("classifier_params", t.config.classifier_params))
    if text_weight is not None:
        kw["text_weight"] = text_weight
    if reject_below is not None:
        kw["reject_below"] = reject_below
    if clear_reject:
        kw["reject_below"] = None
    if reject_below_zero_shot is not None or clear_reject_zero_shot:
        kw["reject_below_zero_shot"] = reject_below_zero_shot
    if flip is not None:
        kw["flip"] = flip
    if temperature is not None:
        kw["temperature"] = None if temperature == 1 else temperature
    try:
        _say_cleared(name, t.update_settings(**kw))
    except ValueError as e:
        _fail(str(e))
    show(name, False, root)


def _say_cleared(name: str, cleared: list[str]) -> None:
    """Tell the user which settings switching embedder cleared, if any."""
    if cleared:
        typer.echo(f"Cleared {', '.join(cleared)}: fitted to the previous embedder "
                   f"(`fewshot undo {name}` restores them)", err=True)


@app.command()
def delete(name: str, yes: bool = typer.Option(False, "--yes", "-y"), root: Path | None = ROOT_OPT) -> None:
    """Delete a task and its images."""
    _task(root, name)
    if not yes:
        typer.confirm(f"Delete task {name} and all its images?", abort=True)
    _ws(root).delete(name)
    typer.echo(f"Deleted task {name}")


@app.command()
def rename(name: str, new_name: str, root: Path | None = ROOT_OPT) -> None:
    """Rename a task."""
    try:
        _ws(root).rename(name, new_name)
    except (KeyError, ValueError) as e:
        _fail(_error_text(e))
    typer.echo(f"Renamed task {name} to {new_name}")


@app.command("copy")
def copy_task(
    name: str,
    new_name: str,
    no_examples: bool = typer.Option(False, "--no-examples", help="Copy only settings and classes"),
    inbox: bool = typer.Option(False, "--inbox", help="Copy the inbox too"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Copy a task under a new name, e.g. to try other settings without touching the original."""
    try:
        t = _ws(root).duplicate(name, new_name, examples=not no_examples, inbox=inbox)
    except (KeyError, ValueError) as e:
        _fail(_error_text(e))
    typer.echo(f"Copied {name} to {t.name} ({len(t.config.examples)} examples)")


@app.command()
def undo(name: str, root: Path | None = ROOT_OPT) -> None:
    """Undo a task's last change (settings, classes or examples); repeat to go further back."""
    t = _task(root, name)
    label = t.undo_label()
    try:
        rev = t.undo()
    except (KeyError, ValueError) as e:
        _fail(_error_text(e))
    typer.echo(f"{name}: undid {label or 'the last change'}, back to revision {rev} "
               f"({len(t.config.examples)} examples)")


@app.command()
def redo(name: str, root: Path | None = ROOT_OPT) -> None:
    """Make the last undone change again (until another change is made)."""
    t = _task(root, name)
    label = t.redo_label()
    try:
        t.redo()
    except (KeyError, ValueError) as e:
        _fail(_error_text(e))
    typer.echo(f"{name}: redid {label or 'the last undone change'} ({len(t.config.examples)} examples)")


@app.command()
def export(name: str, file: Path, root: Path | None = ROOT_OPT) -> None:
    """Save a task (settings, classes, labels and images) as a zip, for backup or another workspace."""
    from .archive import export_task  # pylint: disable=import-outside-toplevel  # keeps --help fast

    ws = _ws(root)
    _task(root, name)
    part = file.with_name(file.name + ".part")  # renamed into place when complete
    try:
        with open(part, "wb") as f:
            n, skipped = export_task(ws, name, f)
        os.replace(part, file)
    except OSError as e:
        part.unlink(missing_ok=True)
        _fail(f"Can't write {file}: {e.strerror or e}")
    typer.echo(f"Exported {name} ({n} examples) to {file}")
    if skipped:
        typer.secho(f"{skipped} examples were left out: their image files are missing", fg="yellow", err=True)


@app.command("import-task")
def import_task(
    file: Path,
    name: str | None = typer.Option(None, help="Name for the task [default: the exported name]"),
    embedder: str | None = typer.Option(None, help="Use this embedder instead of the exported one"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Create a task from a zip made by `fewshot export`."""
    from .archive import import_task as read_export  # pylint: disable=import-outside-toplevel  # keeps --help fast

    if not file.is_file():
        _fail(f"No such file: {file}")
    try:
        if embedder:
            _check_embedder(embedder)
        t = read_export(_ws(root), file, name, embedder=embedder)
    except ValueError as e:
        _fail(str(e))
    typer.echo(f"Imported task {t.name} ({len(t.config.examples)} examples)")


@cls_app.command("add")
def class_add(
    name: str,
    label: str,
    description: str = typer.Option("", "--desc", help="Text prompt, e.g. 'a photo of a cracked weld'"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Add a class, or set an existing class's description."""
    try:
        _task(root, name).upsert_class(label, description)
    except ValueError as e:
        _fail(str(e))


@cls_app.command("describe")
def class_describe(
    name: str,
    template: str = typer.Argument(
        "a photo of a {}", help="'{}' is replaced by the class name (underscores become spaces)"),
    overwrite: bool = typer.Option(False, help="Also replace descriptions that are already set"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Fill class descriptions from a template, e.g. 'a photo of a {}, a type of pet'."""
    try:
        n = _task(root, name).describe_classes(template, overwrite)
    except ValueError as e:
        _fail(str(e))
    typer.echo(f"Set {n} descriptions")


@cls_app.command("threshold", context_settings={"ignore_unknown_options": True})  # takes "-0.5"
def class_threshold(
    name: str,
    label: str,
    value: float | None = typer.Argument(None, min=-1, max=1, help="Similarity under which predictions of "
                                                                     "this class become 'unknown'"),
    clear: bool = typer.Option(False, "--clear", help="Use the task's --reject-below for this class again"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Set a class's own reject threshold, overriding the task's (see `eval` for suggestions)."""
    if (value is None) == (not clear):
        _fail("Give either a threshold or --clear")
    try:
        _task(root, name).set_class_thresholds({label: value})
    except KeyError as e:
        _fail(_error_text(e))


@cls_app.command("rm")
def class_rm(name: str, label: str, root: Path | None = ROOT_OPT) -> None:
    """Delete a class and its examples."""
    try:
        n = _task(root, name).delete_class(label)
    except KeyError as e:
        _fail(_error_text(e))
    typer.echo(f"Removed class {label} and {n} examples")


@app.command()
def add(name: str, label: str, paths: list[str], root: Path | None = ROOT_OPT) -> None:
    """Add images (files, directories or http(s) URLs) as examples of LABEL; exits 1 if none could be added."""
    t = _task(root, name)
    try:
        label = check_class_name(label)
    except ValueError as e:
        _fail(str(e))
    n = _add_files(t, label, _expand(paths))
    typer.echo(f"{name}: {len(t.config.examples)} examples")
    if n == 0:
        _fail("No images were added")


@app.command("import")
def import_dir(
    name: str,
    folder: Path,
    shots: int | None = typer.Option(None, min=1, help="Take at most this many images per class"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Import FOLDER/<class>/<image> into a task; exits 1 if no image could be imported."""
    t = _task(root, name)
    if not folder.is_dir():
        _fail(f"Not a directory: {folder}")
    n = 0
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        n += _add_files(t, d.name, _images_in(d, recursive=False)[:shots])
    typer.echo(f"Imported {n} images into {name}")
    if n == 0:
        _fail(f"No images were imported; expected {folder}/<class>/<image>")


PREDICT_CHUNK = 256  # images classified (and printed) at a time


@app.command()
def predict(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals  # one per CLI option
    name: str,
    paths: list[str] = typer.Argument(None, help="Image files, directories or http(s) URLs"),
    from_file: Path | None = typer.Option(None, "--from-file", help="Read more paths or URLs, one per line, from "
                                                                     "this file ('-' for stdin)"),
    as_json: bool = JSON_OPT,
    as_jsonl: bool = typer.Option(False, "--jsonl", help="Print one JSON object per image as each batch finishes"),
    as_csv: bool = typer.Option(False, "--csv", help="Print CSV: one row per image"),
    top: int = typer.Option(3, min=1, help="Scores to show per image (and to include, with --json/--jsonl)"),
    evidence: int = typer.Option(0, min=0, max=20, help="Also show this many of the most similar labelled examples"),
    revision: int | None = typer.Option(None, help="Use this earlier version of the task (one kept for undo)"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Classify images (files, directories or http(s) URLs), PREDICT_CHUNK at a time: text, CSV and
    JSON lines are printed as each batch finishes, and --from-file is read as it arrives, so a long
    list can be piped through. A path in it that can't be read gets an error row."""
    t = _task(root, name)
    if sum((as_json, as_jsonl, as_csv)) > 1:
        _fail("Give at most one of --json, --jsonl and --csv")
    if not paths and from_file is None:
        _fail("Give images to classify, or --from-file")
    named = _expand(paths) if paths else []
    fmt = "json" if as_json else "jsonl" if as_jsonl else "csv" if as_csv else "text"
    try:
        cfg = None if revision is None else t.version(revision)
    except KeyError as e:
        _fail(_error_text(e))
    pipe = _pipe(_ws(root))
    files: list[Source] = []  # every file so far (all of them for --json, else the last batch)
    collected: list[Prediction] = []
    done = failed = 0
    sources = itertools.chain(named, _listed(from_file) if from_file is not None else [])
    with _progress("Classifying") as update:
        while part := list(itertools.islice(sources, PREDICT_CHUNK)):
            images, fetch_errors = _download(part)
            try:
                preds = pipe.predict(t, images, top_k=top if fmt in {"json", "jsonl"} else None, evidence=evidence,
                                     cfg=cfg)
            except ValueError as e:
                _fail(str(e))
            preds = _with_fetch_errors(preds, fetch_errors)
            failed += sum(p.error is not None for p in preds)
            if fmt == "json":
                files += part
                collected += preds
            else:
                _output_predictions(part, preds, fmt, top, header=done == 0)
            done += len(part)
            update(done, len(named) if from_file is None else done)
    if not done:
        _fail("No images to classify")
    if fmt == "json":
        _output_predictions(files, collected, fmt, top)
    if failed:
        typer.secho(f"{failed} of {done} images could not be read", fg="red", err=True)
        raise typer.Exit(1)


def _listed(from_file: Path) -> Iterator[Source]:
    """The images a --from-file list names, read line by line as it arrives (``-``: stdin): URLs,
    files (a missing one too, so it gets an error row) and the images in folders."""
    if str(from_file) == "-":
        yield from _named_in(sys.stdin)
        return
    try:
        f = from_file.open(encoding="utf-8")
    except OSError as e:
        _fail(f"Can't read {from_file}: {e.strerror or e}")
    with f:
        yield from _named_in(f)


def _named_in(lines: Iterable[str]) -> Iterator[Source]:
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if is_url(line):
            yield line
        elif Path(line).is_dir():
            yield from _images_in(Path(line))
        else:
            yield Path(line)


@app.command()
def compare(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    name: str,
    embedder: list[str] = typer.Option([], "--embedder", "-e", help="An embedder to try (repeatable); the task's "
                                                                   "own is always included"),
    presets: bool = typer.Option(False, "--presets", help="Also try every preset (see `fewshot models`)"),
    folds: int | None = typer.Option(None, min=2, help="Number of folds [default: as for eval]"),
    as_json: bool = JSON_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """Cross-validate the task with other embedders, to choose a model on its own labelled examples."""
    from .evaluate import compare_embedders  # pylint: disable=import-outside-toplevel  # torch-backed

    t = _task(root, name)
    specs = [t.config.embedder, *embedder, *([p.spec for p in PRESETS] if presets else [])]
    if len(set(specs)) < 2:
        _fail("Give another --embedder to compare with, or --presets")
    try:
        with _progress("Comparing") as update:
            r = compare_embedders(_pipe(_ws(root)), t, specs, folds,
                                  progress=lambda _stage, done, total: update(done, total))
    except ValueError as e:
        _fail(str(e))
    if as_json:
        _echo_json(r)
        return
    for x in r["results"]:
        mark = ("*" if x["embedder"] == r["best"] else " ") + ("<" if x["embedder"] == r["current"] else " ")
        if x.get("error"):
            typer.echo(f"{mark} {'failed':>8}  {x['embedder']}: {x['error']}")
        elif x["accuracy"] is None:
            typer.echo(f"{mark} {'-':>8}  {x['embedder']}: nothing scored")
        else:
            typer.echo(f"{mark} {x['accuracy']:8.3f}  macro-F1 {x['macro_f1']:.3f}  {x['seconds']:7.1f}s  "
                       f"{x['embedder']}")
    typer.echo(f"* most accurate, < current ({r['n_examples']} examples)")
    if r["best"] and r["best"] != r["current"]:
        typer.echo(f"To switch: fewshot config {name} --embedder {r['best']}")


@app.command()
def watch(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    name: str,
    folder: Path = typer.Argument(..., help="Folder new images arrive in"),
    out: Path = typer.Option(..., help="Where images go: OUT/<class>/, OUT/_unknown/ (rejected or not confident "
                                       "enough) or OUT/_error/"),
    copy: bool = typer.Option(False, "--copy", help="Copy instead of moving (then each file is handled once per run)"),
    min_confidence: float | None = typer.Option(None, min=0, max=1, help="Below this, an image goes to _unknown"),
    interval: float = typer.Option(2.0, min=0.1, help="Seconds between looks at the folder"),
    once: bool = typer.Option(False, "--once", help="Handle what is there now and stop"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Classify images as they arrive in FOLDER and sort them into OUT, logging each to OUT/predictions.jsonl.

    A file is handled once its size has stayed the same for one interval (so half-copied files wait);
    one that can't be read yet is retried until it is WATCH_SETTLE seconds old, then goes to _error.
    The task is re-read every time, so labelling more examples improves the sorting as it runs.
    """
    if not folder.is_dir():
        _fail(f"Not a directory: {folder}")
    ws = _ws(root)
    _task(root, name)
    pipe = _pipe(ws)
    last: dict[Path, int] = {}  # files seen in the last look, and their sizes
    done: set[Path] = set()  # with --copy, files already copied
    try:
        while True:
            sizes = _sizes(folder)
            ready = [f for f, n in sizes.items() if f not in done and n > 0 and (once or last.get(f) == n)]
            last, done = sizes, done & sizes.keys()  # forget files that went away
            try:
                done |= _sort_files(ws, pipe, name, ready, out, copy, min_confidence,
                                    settle=0 if once else WATCH_SETTLE) if ready else set()
            except ValueError as e:  # the task can't predict yet (no examples): wait for some
                if once:
                    _fail(str(e))
                typer.secho(f"Waiting: {e}", fg="yellow", err=True)
            if once:
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo("Stopped")


WATCH_SETTLE = 30.0  # seconds after its last change that an unreadable file is given up on


def _sizes(folder: Path) -> dict[Path, int]:
    """The images in ``folder`` and their sizes, skipping any gone since they were listed."""
    out = {}
    for f in _images_in(folder, recursive=False):
        try:
            out[f] = f.stat().st_size
        except OSError:
            continue
    return out


def _sort_files(ws: Workspace, pipe: Pipeline, name: str, files: list[Path], out: Path,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                copy: bool, min_confidence: float | None, settle: float = 0) -> set[Path]:
    """Classify ``files`` with the task's current version and move (or copy) each into its folder
    under ``out``, appending a JSON line per file to ``out/predictions.jsonl``. Returns the files
    handled. An unreadable file changed less than ``settle`` seconds ago may still be arriving and
    is left for next time. Raises ``ValueError`` if the task can't predict yet (no examples)."""
    try:
        preds = pipe.predict(ws.get(name), files)
    except KeyError:
        _fail(f"Task {name!r} is gone (renamed or deleted); stopping")
    handled = set()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.jsonl", "a", encoding="utf-8") as log:
        for f, p in zip(files, preds, strict=True):
            try:
                if p.error and time.time() - f.stat().st_mtime < settle:
                    continue
                dest = _place(f, out / _folder_for(p, min_confidence), copy)
            except OSError as e:  # removed meanwhile, or the destination can't be written
                typer.secho(f"{f.name}: {e}", fg="yellow", err=True)
                continue
            handled.add(f)
            log.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "file": str(f), "to": str(dest),
                                  "task": name, **p.to_dict()}) + "\n")
            typer.echo(f"{f.name} -> {dest.parent.name}")
    return handled


def _folder_for(p: Prediction, min_confidence: float | None) -> str:
    """The folder under OUT a prediction sends its image to."""
    unsure = p.rejected or (min_confidence is not None and p.confidence < min_confidence)
    return "_error" if p.error else "_unknown" if unsure or not p.label else p.label


def _place(f: Path, folder: Path, copy: bool) -> Path:
    """Move (or copy) ``f`` into ``folder`` under a free name; returns where it went."""
    dest = _free_name(folder / f.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    (shutil.copy2 if copy else shutil.move)(str(f), str(dest))
    return dest


def _free_name(path: Path) -> Path:
    """``path``, or ``name-1.ext``, ``name-2.ext``... if it is taken: files are never overwritten."""
    n = 0
    candidate = path
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
    return candidate


def _with_fetch_errors(preds: list[Prediction], errors: dict[int, str]) -> list[Prediction]:
    """``preds`` with the URLs that couldn't be downloaded failed with the real reason."""
    from .pipeline import Prediction  # pylint: disable=import-outside-toplevel,redefined-outer-name  # torch-backed

    return [Prediction.failed(errors[i]) if i in errors else p for i, p in enumerate(preds)]


def _output_predictions(files: list[Source], preds: list[Prediction], fmt: str, top: int, header: bool = True) -> None:
    """Print predictions as JSON, JSON lines, CSV (with a ``header`` row) or text."""
    if fmt == "json":
        _echo_json([{"file": str(f), **p.to_dict()} for f, p in zip(files, preds, strict=True)])
    elif fmt == "jsonl":
        for f, p in zip(files, preds, strict=True):
            typer.echo(json.dumps({"file": str(f), **p.to_dict()}))
    elif fmt == "csv":
        _write_csv(["file", "label", "labels", "top_label", "confidence", "similarity", "rejected", "error", "scores"],
                   [[str(f), p.label or "", ";".join(p.labels), p.top_label, f"{p.confidence:.4f}",
                     f"{p.similarity:.4f}", p.rejected, p.error or "",
                     ";".join(f"{c}={v:.4f}" for c, v in list(p.scores.items())[:top])]
                    for f, p in zip(files, preds, strict=True)], header)
    else:
        _print_predictions(files, preds, top)


def _download(files: list[Source]) -> tuple[list[bytes | Path], dict[int, str]]:
    """Files as they are and URLs downloaded; a URL that can't be fetched becomes empty bytes, with
    its error by position in the second result."""
    images: list[bytes | Path] = []
    errors: dict[int, str] = {}
    for i, f in enumerate(files):
        try:
            images.append(f if isinstance(f, Path) else _read(f))
        except ValueError as e:
            images.append(b"")
            errors[i] = str(e)
    return images, errors


def _print_predictions(files: list[Source], preds: list[Prediction], top: int) -> None:
    for f, p in zip(files, preds, strict=True):
        if p.error is not None:
            typer.secho(f"{f}: error: {p.error}", fg="yellow", err=True)
            continue
        best = list(p.scores.items())[:top]
        label = ", ".join(p.labels) if p.label is not None else f"unknown (closest: {p.top_label})" + (
            f"; matches {', '.join(p.labels)}" if p.labels else "")
        scores = "  ".join(f"{c}={v:.2f}" for c, v in best)
        typer.echo(f"{f}: {label}  sim={p.similarity:.3f}  {scores}")
        for e in p.evidence:
            typer.echo(f"    like {e['id'][:12]} ({e['label']}, sim={e['similarity']:.3f})")


def _print_eval(r: dict[str, Any]) -> None:
    acc = r["accuracy"]
    if acc is None:
        typer.echo("Nothing scored")
    else:
        typer.echo(f"{r['method']}: accuracy {acc:.3f} on {r['n_scored']}/{r['n_examples']} examples, "
                   f"macro-F1 {r['macro_f1']:.3f}")
        topk = "  ".join(f"top-{k} {v:.3f}" for k, v in r["top_k_accuracy"].items())
        if topk:
            typer.echo(f"  {topk}")
    if r["suggested_reject_below"] is not None:
        typer.echo(f"Suggested --reject-below {r['suggested_reject_below']} (keeps 95% of correct predictions)")
    if cal := r["calibration"]:
        typer.echo(f"Calibration error {cal['ece']:.3f}; --temperature {cal['temperature']} would make it "
                   f"{cal['ece_calibrated']:.3f}")
    suggested = r["suggested_class_thresholds"]
    for c, m in r["per_class"].items():
        rec = "-" if m["recall"] is None else f"{m['recall']:.2f}"
        prec = "-" if m["precision"] is None else f"{m['precision']:.2f}"
        thr = f"  suggested threshold {suggested[c]}" if c in suggested else ""
        typer.echo(f"  {c:24} n={m['support']:<4} recall={rec:5} precision={prec:5}{thr}")
    if suggested:
        typer.echo("(--apply-class-thresholds sets the suggested per-class thresholds)")
    if r["mistakes"]:
        typer.echo(f"{len(r['mistakes'])} mistakes; most confident:")
        for m in r["mistakes"][:10]:
            typer.echo(f"  {m['id']}  {m['true']} -> {m['pred']} ({m['confidence']:.2f})")


@app.command("eval")
def eval_cmd(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    name: str,
    folds: int | None = typer.Option(
        None, help="Number of folds [default: leave-one-out up to 300 examples, else 5]"),
    as_json: bool = JSON_OPT,
    as_csv: bool = typer.Option(False, "--csv", help="Print CSV: each example's held-out prediction"),
    apply_class_thresholds: bool = typer.Option(
        False, "--apply-class-thresholds", help="Set each class's reject threshold to the one suggested"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Cross-validate a task on its own examples."""
    # evaluate imports the torch-backed pipeline; importing it lazily keeps `fewshot --help` fast.
    from .evaluate import evaluate_task  # pylint: disable=import-outside-toplevel

    t = _task(root, name)
    try:
        with _progress("Evaluating") as update:
            r = evaluate_task(_pipe(_ws(root)), t, folds, progress=lambda _stage, done, total: update(done, total))
    except ValueError as e:
        _fail(str(e))
    if apply_class_thresholds and r["suggested_class_thresholds"]:
        current = r["revision"] == t.config.revision
        try:
            t.set_class_thresholds(r["suggested_class_thresholds"])
        except KeyError as e:  # a class was renamed or deleted meanwhile
            _fail(f"Thresholds not set: {e.args[0]}")
        typer.echo(f"Set the thresholds of {len(r['suggested_class_thresholds'])} classes", err=True)
        if current:  # thresholds don't change what evaluation measures, so the result still holds
            r = {**r, "revision": t.config.revision}
    t.record_evaluation(r)
    if as_json:
        _echo_json(r)
    elif as_csv:
        _write_csv(["id", "true", "pred", "correct", "confidence", "similarity"],
                   [[p["id"], p["true"], p["pred"], p["correct"], p["confidence"], p["similarity"]]
                    for p in r["predictions"]])
    else:
        _print_eval(r)


@app.command()
def dupes(
    name: str,
    threshold: float = typer.Option(0.95, min=0.5, max=1.0, help="Cosine similarity at or above which "
                                                                  "two examples count as near-duplicates"),
    limit: int = typer.Option(200, min=1, help="Show at most this many pairs"),
    as_json: bool = JSON_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """List near-duplicate examples, most similar first; pairs with different labels are marked."""
    t = _task(root, name)
    pairs = _pipe(_ws(root)).near_duplicates(t, threshold, limit)
    if as_json:
        _echo_json(pairs)
        return
    for d in pairs:
        mark = "" if d["same_label"] else "  DIFFERENT LABELS"
        typer.echo(f"{d['similarity']:.3f}  {d['a']} ({d['labels'][0]})  {d['b']} ({d['labels'][1]}){mark}")
    typer.echo(f"{len(pairs)} pairs at similarity >= {threshold}")


def _bench_runs(specs: list[str], classifiers: list[str], shots: list[int], text_template: str | None
                ) -> Iterator[tuple[str, str, int, str | None]]:
    """Yield (embedder, classifier, shot, text_template) combinations; text is used only by CLIP embedders,
    and 0-shot runs need it, so they are skipped otherwise."""
    for e in specs:
        tt = text_template if spec_supports_text(e) else None
        for c in classifiers:
            for s in shots:
                if s > 0 or tt:
                    yield e, c, s, tt


def _bench_one(pipe: Pipeline, folder: Path, run: tuple[str, str, int, str | None], opts: dict[str, Any]) -> str:
    # evaluate imports the torch-backed pipeline; importing it lazily keeps `fewshot --help` fast.
    from .evaluate import benchmark  # pylint: disable=import-outside-toplevel

    e, c, s, tt = run
    try:
        r = benchmark(pipe, folder, e, c, shot=s, text_template=tt, progress=lambda m: typer.echo(m, err=True),
                      **opts)
    except ValueError as err:
        _fail(str(err))
    return (f"{e:42} {c:9} {opts['way']}-way {s}-shot{' +text' if tt else '      '}  "
            f"acc {r['accuracy']:.3f} ± {r['ci95']:.3f}")


@app.command()
def bench(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    folder: Path,
    embedder: list[str] = typer.Option(None, help="Embedder spec(s) to compare [default: CLIP ViT-B/32]"),
    classifier: list[str] = typer.Option(["prototype"], help="Classifier(s) to compare"),
    way: int = typer.Option(5, min=2),
    shot: list[int] = typer.Option([1, 5], min=0, help="Shot count(s)"),
    query: int = typer.Option(15, min=1),
    episodes: int = typer.Option(200, min=1),
    text_template: list[str] = typer.Option(
        None, help="e.g. 'a photo of a {}' to use class names as text (CLIP only). Repeat to average "
                   "several prompts."),
    flip: bool = typer.Option(False, "--flip", help="Average each image with its mirror image"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Episodic N-way K-shot benchmark over FOLDER/<class>/<image>."""
    if not folder.is_dir():
        _fail(f"Not a directory: {folder}")
    pipe = _pipe(_ws(root))
    opts = {"way": way, "query": query, "episodes": episodes, "flip": flip}
    template = "\n".join(text_template) if text_template else None
    for run in _bench_runs(embedder or [DEFAULT_SPEC], classifier, shot, template):
        typer.echo(_bench_one(pipe, folder, run, opts))


@app.command()
def serve(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    host: str = "127.0.0.1",
    port: int = 8000,
    token: str | None = typer.Option(None, envvar="FEWSHOT_TOKEN", show_envvar=True,
                                     help="Require this token for the API and UI"),
    allow_any_model: bool = typer.Option(False, "--allow-any-model",
                                         help="Let API callers load any embedder spec, not just presets"),
    warm: bool = typer.Option(True, "--warm/--no-warm", help="Load the models tasks use at startup"),
    allow_urls: bool = typer.Option(False, "--allow-urls",
                                    help="Let API callers send image URLs for the server to download "
                                         "(public addresses only)"),
    max_upload_mb: float = typer.Option(50, help="Largest image accepted per upload"),
    max_request_mb: float = typer.Option(1024, help="Largest request body accepted"),
    allowed_host: list[str] = typer.Option(
        [], "--allowed-host", help="Host name clients use to reach the server; others are refused (IP addresses "
                                   "always work). Repeatable; '*' allows any. [default: $FEWSHOT_ALLOWED_HOSTS "
                                   "(comma-separated), else localhost, plus this machine's names when not serving "
                                   "locally]"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Run the REST API and web UI."""
    # uvicorn is only needed here; importing it lazily keeps other commands fast.
    import uvicorn  # pylint: disable=import-outside-toplevel

    if root:
        os.environ["FEWSHOT_HOME"] = str(root)
    local = host in {"127.0.0.1", "localhost", "::1"}
    from_env = [h.strip() for h in os.environ.get("FEWSHOT_ALLOWED_HOSTS", "").split(",") if h.strip()]
    hosts = allowed_host or from_env or ["localhost", "127.0.0.1", "::1"]
    if not (allowed_host or from_env or local):
        hosts += sorted({socket.gethostname().lower(), socket.getfqdn().lower()} - {""})
    # fewshot.api reads its settings from the environment when uvicorn imports it.
    settings = {
        "FEWSHOT_TOKEN": token or "", "FEWSHOT_ALLOW_ANY_MODEL": "1" if allow_any_model else "",
        "FEWSHOT_WARM": "1" if warm else "", "FEWSHOT_ALLOW_URLS": "1" if allow_urls else "",
        "FEWSHOT_ALLOWED_HOSTS": ",".join(hosts),
        "FEWSHOT_MAX_UPLOAD_MB": f"{max_upload_mb:g}", "FEWSHOT_MAX_REQUEST_MB": f"{max_request_mb:g}",
    }
    os.environ.update(settings)
    if not token and not local:
        typer.secho(f"Warning: serving on {host} without --token: anyone who can reach this port can read and "
                    "change every task.", fg="yellow", err=True)
    if not local and "*" not in hosts:
        typer.echo(f"Answering requests by IP address or for: {', '.join(hosts)}. Reaching it by another name "
                   "needs --allowed-host NAME.", err=True)
    uvicorn.run("fewshot.api:app", host=host, port=port)


@cache_app.command("info")
def cache_info(root: Path | None = ROOT_OPT) -> None:
    """Show the cache file's size and how many vectors each model has cached."""
    from .cache import EmbeddingCache  # pylint: disable=import-outside-toplevel  # keeps --help fast

    ws = _ws(root)
    cache = EmbeddingCache(ws.cache_path)
    typer.echo(f"{ws.cache_path}: {_mb(ws.cache_path.stat().st_size)}")
    for ns, n, size in cache.stats():
        typer.echo(f"  {ns}: {n} vectors, {_mb(size)}")


@cache_app.command("prune")
def cache_prune(
    dry_run: bool = typer.Option(False, "--dry-run", help="Only report how many vectors would go"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Delete cached image vectors that no task's examples use (e.g. deleted examples, bench datasets),
    and stored images no task links to any more.

    Text vectors are kept. Refuses to run if a task can't be read, since its examples would be lost.
    """
    from .cache import EmbeddingCache  # pylint: disable=import-outside-toplevel  # keeps --help fast

    ws = _ws(root)
    tasks = ws.list_tasks()
    folders = [p for p in ws.task_folders() if (p / "task.json").exists()]
    if len(tasks) < len(folders):
        unreadable = sorted({p.name for p in folders} - {t.config.name for t in tasks})
        _fail(f"Can't read task(s) {', '.join(unreadable)}; fix or remove them before pruning")
    from .pipeline import FLIP_KEY  # pylint: disable=import-outside-toplevel  # keeps --help fast

    # Images undo or redo may bring back keep their vectors too.
    ids = {e.id for t in tasks for e in [*t.config.examples, *t.config.inbox]}
    ids |= {f.split("/", 1)[1].rsplit(".", 1)[0] for t in tasks for f in t.kept_files()}
    keep = {k for i in ids for k in (i, i + FLIP_KEY)}
    cache = EmbeddingCache(ws.cache_path)
    n = cache.prune(keep, dry_run=dry_run)
    typer.echo(f"{'Would delete' if dry_run else 'Deleted'} {n} cached image vectors; "
               f"{_mb(ws.cache_path.stat().st_size)} on disk")
    if not dry_run:
        typer.echo(f"Deleted {ws.gc_blobs()} stored images no task uses")


@app.command()
def gc(root: Path | None = ROOT_OPT) -> None:
    """Remove what crashes and undo history leave behind: half-made or half-deleted task folders
    (older than 10 minutes), image files no version of their task uses, and stored images no task uses."""
    n = _ws(root).gc()
    typer.echo(f"Removed {n['folders']} leftover folders, {n['images']} unused image files and "
               f"{n['blobs']} unused stored images")


def _mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB"




if __name__ == "__main__":
    app()
