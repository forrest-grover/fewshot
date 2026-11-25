"""Command-line interface: ``fewshot --help``."""

from __future__ import annotations

import contextlib
import csv
import json
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import typer

from .embedders import DEFAULT_SPEC, PRESETS, check_embedder, spec_supports_text
from .fetch import fetch_url, is_url
from .images import IMAGE_EXTS
from .store import Task, Workspace
from .text import check_class_name

if TYPE_CHECKING:
    from .pipeline import Pipeline, Prediction

app = typer.Typer(help="Embedding-based few-shot image classification.", no_args_is_help=True)
cls_app = typer.Typer(help="Manage a task's classes.", no_args_is_help=True)
app.add_typer(cls_app, name="class")
cache_app = typer.Typer(help="Inspect and prune the embedding cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")
ex_app = typer.Typer(help="List, move, tag and delete a task's examples (ids may be shortened to a unique "
                          "prefix of 4+ characters).", no_args_is_help=True)
app.add_typer(ex_app, name="example")
inbox_app = typer.Typer(help="Unlabelled images waiting to be labelled, with the model's suggestions (ids may be "
                             "shortened to a unique prefix of 4+ characters).", no_args_is_help=True)
app.add_typer(inbox_app, name="inbox")

MAX_DOWNLOAD = 50 * 1024 * 1024  # largest image fetched from a URL
JSON_OPT = typer.Option(False, "--json", help="Print JSON")

ROOT_OPT = typer.Option(None, "--root", envvar="FEWSHOT_HOME", help="Workspace directory [default: ./fewshot_data]")


def _fail(msg: str) -> NoReturn:
    typer.secho(f"Error: {msg}", fg="red", err=True)
    raise typer.Exit(1)


def _error_text(e: Exception) -> str:
    # str(KeyError("x")) is "'x'"; the first argument is the readable message.
    return str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)


def _ws(root: Path | None) -> Workspace:
    return Workspace(root)


def _task(root: Path | None, name: str) -> Task:
    try:
        return _ws(root).get(name)
    except KeyError as e:
        _fail(_error_text(e))


def _pipe(ws: Workspace) -> Pipeline:
    # Pipeline imports torch; importing it lazily keeps `fewshot --help` fast.
    from .pipeline import Pipeline  # pylint: disable=import-outside-toplevel

    return Pipeline(ws)


Source = Path | str  # an image file, or an http(s) URL


def _images_in(folder: Path, recursive: bool = True) -> list[Path]:
    """Image files under ``folder``, sorted; folders and broken links with image names are left out."""
    found = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(f for f in found if f.suffix.lower() in IMAGE_EXTS and f.is_file())


def _expand(paths: list[str]) -> list[Source]:
    """URLs and files named on the command line plus the images inside named folders; fails if
    there are none."""
    out: list[Source] = []
    for arg in paths:
        if is_url(arg):
            out.append(arg)
            continue
        p = Path(arg)
        if not p.exists():
            _fail(f"No such file or directory: {p}")
        out += _images_in(p) if p.is_dir() else [p]
    if not out:
        _fail("No images found in " + ", ".join(paths))
    return out


def _read(src: Source) -> bytes:
    """The bytes of a file or URL; ``ValueError`` says why they couldn't be read."""
    if isinstance(src, str):
        return fetch_url(src, MAX_DOWNLOAD)
    try:
        return src.read_bytes()
    except OSError as e:
        raise ValueError(e.strerror or str(e)) from None


def _name(src: Source) -> str:
    return src.name if isinstance(src, Path) else src.rstrip("/").rsplit("/", 1)[-1]


@contextlib.contextmanager
def _progress(what: str) -> Iterator[Callable[[int, int], None]]:
    """A progress bar on stderr while the block runs (only on a terminal); yields ``update(done, total)``."""
    from rich.console import Console  # pylint: disable=import-outside-toplevel  # keeps --help fast
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn  # pylint: disable=import-outside-toplevel

    with Progress(TextColumn(what), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
                  console=Console(stderr=True), transient=True, disable=not sys.stderr.isatty()) as progress:
        tid = progress.add_task(what, total=None)
        yield lambda done, total: progress.update(tid, completed=done, total=total)


def _add_files(task: Task, label: str, files: Sequence[Source]) -> int:
    """Add ``files`` (paths or URLs) as examples of ``label``, skipping unreadable images; returns
    how many were added."""

    def readable(update: Callable[[int, int], None]) -> Iterator[tuple[bytes, str, str]]:
        for i, f in enumerate(files):
            update(i, len(files))
            try:
                data = _read(f)
            except ValueError as e:
                typer.secho(f"skip {f}: {e}", fg="yellow", err=True)
                continue
            names.append(f)
            yield data, label, _name(f)

    names: list[Source] = []  # the files actually read, in step with add_examples' results
    with _progress(f"Adding to {label}") as update:
        results = task.add_examples(readable(update))
    n = 0
    for f, r in zip(names, results, strict=True):
        if isinstance(r, ValueError):
            typer.secho(f"skip {f}: {r}", fg="yellow", err=True)
        else:
            n += 1
    return n


def _echo_json(obj: Any) -> None:
    typer.echo(json.dumps(obj, indent=2))


def _write_csv(header: list[str], rows: list[list[Any]]) -> None:
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)


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
        typer.echo(f"  {cd.name:24} {counts.get(cd.name, 0):5}{desc.rstrip()}")


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
        t.update_settings(**kw)
    except ValueError as e:
        _fail(str(e))
    show(name, False, root)


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


@app.command()
def predict(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    name: str,
    paths: list[str],
    as_json: bool = JSON_OPT,
    as_csv: bool = typer.Option(False, "--csv", help="Print CSV: one row per image"),
    top: int = typer.Option(3, min=1, help="Scores to show per image (and to include, with --json)"),
    evidence: int = typer.Option(0, min=0, max=20, help="Also show this many of the most similar labelled examples"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Classify images (files, directories or http(s) URLs)."""
    t = _task(root, name)
    files = _expand(paths)
    images, fetch_errors = _download(files)
    try:
        with _progress("Embedding") as update:
            preds = _pipe(_ws(root)).predict(t, images, top_k=top if as_json else None, progress=update,
                                             evidence=evidence)
    except ValueError as e:
        _fail(str(e))
    preds = _with_fetch_errors(preds, fetch_errors)
    failed = sum(p.error is not None for p in preds)
    _output_predictions(files, preds, "json" if as_json else "csv" if as_csv else "text", top)
    if failed:
        typer.secho(f"{failed} of {len(preds)} images could not be read", fg="red", err=True)
        raise typer.Exit(1)


def _with_fetch_errors(preds: list[Prediction], errors: dict[int, str]) -> list[Prediction]:
    """``preds`` with the URLs that couldn't be downloaded failed with the real reason."""
    from .pipeline import Prediction  # pylint: disable=import-outside-toplevel,redefined-outer-name  # torch-backed

    return [Prediction.failed(errors[i]) if i in errors else p for i, p in enumerate(preds)]


def _output_predictions(files: list[Source], preds: list[Prediction], fmt: str, top: int) -> None:
    """Print predictions as JSON, CSV or text."""
    if fmt == "json":
        _echo_json([{"file": str(f), **p.to_dict()} for f, p in zip(files, preds, strict=True)])
    elif fmt == "csv":
        _write_csv(["file", "label", "labels", "top_label", "confidence", "similarity", "rejected", "error", "scores"],
                   [[str(f), p.label or "", ";".join(p.labels), p.top_label, f"{p.confidence:.4f}",
                     f"{p.similarity:.4f}", p.rejected, p.error or "",
                     ";".join(f"{c}={v:.4f}" for c, v in list(p.scores.items())[:top])]
                    for f, p in zip(files, preds, strict=True)])
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
        label = ", ".join(p.labels) if p.label is not None else f"unknown (closest: {p.top_label})"
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
    for c, m in r["per_class"].items():
        rec = "-" if m["recall"] is None else f"{m['recall']:.2f}"
        prec = "-" if m["precision"] is None else f"{m['precision']:.2f}"
        typer.echo(f"  {c:24} n={m['support']:<4} recall={rec:5} precision={prec}")
    if r["mistakes"]:
        typer.echo(f"{len(r['mistakes'])} mistakes; most confident:")
        for m in r["mistakes"][:10]:
            typer.echo(f"  {m['id']}  {m['true']} -> {m['pred']} ({m['confidence']:.2f})")


@app.command("eval")
def eval_cmd(
    name: str,
    folds: int | None = typer.Option(
        None, help="Number of folds [default: leave-one-out up to 300 examples, else 5]"),
    as_json: bool = JSON_OPT,
    as_csv: bool = typer.Option(False, "--csv", help="Print CSV: each example's held-out prediction"),
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
        [], "--allowed-host", help="Host name clients use to reach the server; others are refused. Repeatable. "
                                   "[default: localhost names when serving locally, else any]"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Run the REST API and web UI."""
    # uvicorn is only needed here; importing it lazily keeps other commands fast.
    import uvicorn  # pylint: disable=import-outside-toplevel

    if root:
        os.environ["FEWSHOT_HOME"] = str(root)
    local = host in {"127.0.0.1", "localhost", "::1"}
    hosts = allowed_host or (["localhost", "127.0.0.1", "::1"] if local else ["*"])
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


# ---- examples -------------------------------------------------------------
def _ids(t: Task, ids: list[str]) -> list[str]:
    try:
        return t.resolve_ids(ids)
    except KeyError as e:
        _fail(_error_text(e))


@ex_app.command("ls")
def example_ls(
    name: str,
    label: str | None = typer.Option(None, help="Only this class"),
    tag: str | None = typer.Option(None, help="Only examples with this tag"),
    as_json: bool = JSON_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """List examples: id, class, tags, source file and note."""
    found = _task(root, name).find_examples(label, tag)
    if as_json:
        _echo_json([e.model_dump() for e in found])
        return
    for e in found:
        tags = f"  [{', '.join(e.tags)}]" if e.tags else ""
        note = f"  # {e.note}" if e.note else ""
        typer.echo(f"{e.id[:12]}  {e.label:20} {e.source}{tags}{note}")


@ex_app.command("rm")
def example_rm(name: str, ids: list[str], root: Path | None = ROOT_OPT) -> None:
    """Delete examples (undo brings them back)."""
    t = _task(root, name)
    n = t.delete_examples(_ids(t, ids))
    typer.echo(f"Deleted {n} examples")


@ex_app.command("relabel")
def example_relabel(name: str, label: str, ids: list[str], root: Path | None = ROOT_OPT) -> None:
    """Move examples to class LABEL (created if needed)."""
    t = _task(root, name)
    try:
        n = t.relabel_many(_ids(t, ids), label)
    except ValueError as e:
        _fail(str(e))
    typer.echo(f"Moved {n} examples to {label}")


@ex_app.command("tag")
def example_tag(
    name: str,
    ids: list[str],
    add_tag: list[str] = typer.Option([], "--add", help="Tag to add. Repeatable."),
    remove_tag: list[str] = typer.Option([], "--remove", help="Tag to remove. Repeatable."),
    root: Path | None = ROOT_OPT,
) -> None:
    """Add or remove tags on examples."""
    if not add_tag and not remove_tag:
        _fail("Give --add and/or --remove")
    t = _task(root, name)
    try:
        n = t.annotate(_ids(t, ids), add_tags=add_tag, remove_tags=remove_tag)
    except ValueError as e:
        _fail(str(e))
    typer.echo(f"Tagged {n} examples")


@ex_app.command("note")
def example_note(name: str, example_id: str, note: str, root: Path | None = ROOT_OPT) -> None:
    """Set an example's note ("" clears it)."""
    t = _task(root, name)
    try:
        t.annotate(_ids(t, [example_id]), note=note)
    except ValueError as e:
        _fail(str(e))
    typer.echo("Note saved")


# ---- inbox ----------------------------------------------------------------
def _inbox_ids(t: Task, prefixes: list[str]) -> list[str]:
    ids = [x.id for x in t.config.inbox]
    out = []
    for p in prefixes:
        hits = [i for i in ids if i.startswith(p)] if len(p) >= 4 else []
        if len(hits) != 1:
            _fail(f"No inbox image {p!r}" if not hits else f"Inbox id {p!r} is ambiguous")
        out.append(hits[0])
    return out


@inbox_app.command("add")
def inbox_add(name: str, paths: list[str], root: Path | None = ROOT_OPT) -> None:
    """Put unlabelled images (files, directories or http(s) URLs) in the inbox; exits 1 if none could be."""
    t = _task(root, name)
    files = _expand(paths)
    names: list[Source] = []

    def readable(update: Callable[[int, int], None]) -> Iterator[tuple[bytes, str]]:
        for i, f in enumerate(files):
            update(i, len(files))
            try:
                data = _read(f)
            except ValueError as e:
                typer.secho(f"skip {f}: {e}", fg="yellow", err=True)
                continue
            names.append(f)
            yield data, _name(f)

    with _progress("Adding to the inbox") as update:
        results = t.add_to_inbox(readable(update))
    for f, r in zip(names, results, strict=True):
        if isinstance(r, ValueError):
            typer.secho(f"skip {f}: {r}", fg="yellow", err=True)
    n = sum(not isinstance(r, ValueError) for r in results)
    typer.echo(f"{name}: {len(t.config.inbox)} images in the inbox")
    if n == 0:
        _fail("Nothing was added to the inbox")


@inbox_app.command("ls")
def inbox_ls(
    name: str,
    sort: str = typer.Option("uncertain", help="uncertain (most useful to label first) | confident | novel | added"),
    limit: int = typer.Option(50, min=1, help="Show at most this many"),
    as_json: bool = JSON_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """List inbox images with the model's suggestion, in the order to label them."""
    t = _task(root, name)
    try:
        items = _pipe(_ws(root)).inbox_suggestions(t, sort)[:limit]
    except ValueError as e:
        _fail(str(e))
    if as_json:
        _echo_json(items)
        return
    for i in items:
        p = i.get("prediction")
        hint = "" if p is None else (f"unknown (closest: {p['top_label']})" if p["rejected"] else p["label"])
        stats = "" if p is None else f"conf={p['confidence']:.2f} margin={i['margin']:.2f} sim={p['similarity']:.3f}"
        typer.echo(f"{i['id'][:12]}  {hint:24} {stats}  {i['source']}")
    typer.echo(f"{len(t.config.inbox)} images in the inbox")


@inbox_app.command("accept")
def inbox_accept(
    name: str,
    ids: list[str] = typer.Argument(None, help="Inbox ids to label as --label"),
    label: str | None = typer.Option(None, help="Class for the given ids"),
    min_confidence: float | None = typer.Option(
        None, min=0, max=1, help="Instead, accept every suggestion at least this confident (not rejected)"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Label inbox images as examples: the given ids as --label, or every suggestion above --min-confidence."""
    t = _task(root, name)
    if (min_confidence is None) == (label is None) or (label is not None and not ids):
        _fail("Give ids with --label, or --min-confidence alone")
    if label is not None:
        labels = dict.fromkeys(_inbox_ids(t, ids or []), label)
    else:
        items = _pipe(_ws(root)).inbox_suggestions(t, "confident")
        labels = {i["id"]: i["prediction"]["label"] for i in items if i.get("prediction")
                  and not i["prediction"]["rejected"] and i["prediction"]["confidence"] >= (min_confidence or 0)}
    if not labels:
        typer.echo("Nothing to accept")
        return
    try:
        n = t.accept(labels)
    except (KeyError, ValueError) as e:
        _fail(_error_text(e))
    typer.echo(f"Labelled {n} inbox images; {len(t.config.inbox)} left")


@inbox_app.command("dismiss")
def inbox_dismiss(name: str, ids: list[str], root: Path | None = ROOT_OPT) -> None:
    """Remove images from the inbox without labelling them (undo brings them back)."""
    t = _task(root, name)
    n = t.dismiss(_inbox_ids(t, ids))
    typer.echo(f"Dismissed {n} images; {len(t.config.inbox)} left")


if __name__ == "__main__":
    app()
