"""Command-line interface: ``fewshot --help``."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import typer

from .embedders import DEFAULT_SPEC, PRESETS, spec_supports_text
from .store import IMAGE_EXTS, Task, Workspace

if TYPE_CHECKING:
    from .pipeline import Pipeline, Prediction

app = typer.Typer(help="Embedding-based few-shot image classification.", no_args_is_help=True)
cls_app = typer.Typer(help="Manage a task's classes.", no_args_is_help=True)
app.add_typer(cls_app, name="class")

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


def _expand(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if not p.exists():
            _fail(f"No such file or directory: {p}")
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        else:
            out.append(p)
    return out


def _add_files(task: Task, label: str, files: list[Path]) -> int:
    """Add ``files`` as examples of ``label``, skipping unreadable images; returns how many were added."""
    n = 0
    for f in files:
        try:
            task.add_example(f.read_bytes(), label, f.name)
        except ValueError as e:
            typer.secho(f"skip {f}: {e}", fg="yellow", err=True)
        else:
            n += 1
    return n


@app.command()
def models() -> None:
    """List preset embedders (any openclip:/timm: spec also works)."""
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
    if embedder:
        kw["embedder"] = embedder
    try:
        t = _ws(root).create(name, **kw)
    except ValueError as e:
        _fail(str(e))
    typer.echo(f"Created task {t.name} at {t.path}")


@app.command("list")
def list_tasks(root: Path | None = ROOT_OPT) -> None:
    """List tasks."""
    for t in _ws(root).list_tasks():
        s = t.summary()
        typer.echo(f"{s['name']:24} {s['n_classes']:3} classes {s['n_examples']:5} examples  "
                   f"{s['classifier']:9} {s['embedder']}")


@app.command()
def show(name: str, root: Path | None = ROOT_OPT) -> None:
    """Show a task's settings and per-class example counts."""
    t = _task(root, name)
    c = t.config
    typer.echo(f"{c.name}\n  embedder:   {c.embedder}\n  classifier: {c.classifier} {c.classifier_params or ''}"
               f"\n  text_weight: {c.text_weight}   reject_below: {c.reject_below}")
    counts = t.summary()["counts"]
    for cd in c.classes:
        desc = f'  "{cd.description}"' if cd.description else ""
        typer.echo(f"  {cd.name:24} {counts.get(cd.name, 0):5}{desc}")


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
def config(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    name: str,
    embedder: str | None = None,
    classifier: str | None = None,
    param: list[str] = typer.Option([], help="Classifier param KEY=VALUE (e.g. k=3, scale=20). Repeatable."),
    text_weight: float | None = None,
    reject_below: float | None = typer.Option(None, help="Similarity threshold for 'unknown'; negative clears it"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Change task settings."""
    t = _task(root, name)
    kw: dict[str, Any] = {}
    if embedder:
        kw["embedder"] = embedder
    if classifier:
        kw["classifier"] = classifier
        kw["classifier_params"] = {}
    if param:
        kw["classifier_params"] = _parse_params(param, kw.get("classifier_params", t.config.classifier_params))
    if text_weight is not None:
        kw["text_weight"] = text_weight
    if reject_below is not None:
        kw["reject_below"] = None if reject_below < 0 else reject_below
    try:
        t.update_settings(**kw)
    except ValueError as e:
        _fail(str(e))
    show(name, root)


@app.command()
def delete(name: str, yes: bool = typer.Option(False, "--yes", "-y"), root: Path | None = ROOT_OPT) -> None:
    """Delete a task and its images."""
    _task(root, name)
    if not yes:
        typer.confirm(f"Delete task {name} and all its images?", abort=True)
    _ws(root).delete(name)
    typer.echo(f"Deleted task {name}")


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
def add(name: str, label: str, paths: list[Path], root: Path | None = ROOT_OPT) -> None:
    """Add images (files or directories) as examples of LABEL."""
    t = _task(root, name)
    _add_files(t, label, _expand(paths))
    typer.echo(f"{name}: {len(t.config.examples)} examples")


@app.command("import")
def import_dir(
    name: str,
    folder: Path,
    shots: int | None = typer.Option(None, min=1, help="Take at most this many images per class"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Import FOLDER/<class>/<image> into a task."""
    t = _task(root, name)
    if not folder.is_dir():
        _fail(f"Not a directory: {folder}")
    n = 0
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)[:shots]
        n += _add_files(t, d.name, files)
    typer.echo(f"Imported {n} images into {name}")


@app.command()
def predict(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    name: str,
    paths: list[Path],
    as_json: bool = typer.Option(False, "--json"),
    top: int = typer.Option(3, help="Scores to show per image"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Classify images (files or directories)."""
    t = _task(root, name)
    files = _expand(paths)
    try:
        preds = _pipe(_ws(root)).predict(t, files)
    except ValueError as e:
        _fail(str(e))
    failed = sum(p.error is not None for p in preds)
    if as_json:
        typer.echo(json.dumps([{"file": str(f), **p.to_dict()} for f, p in zip(files, preds, strict=True)], indent=2))
    else:
        _print_predictions(files, preds, top)
    if failed:
        typer.secho(f"{failed} of {len(preds)} images could not be read", fg="red", err=True)
        raise typer.Exit(1)


def _print_predictions(files: list[Path], preds: list[Prediction], top: int) -> None:
    for f, p in zip(files, preds, strict=True):
        if p.error is not None:
            typer.secho(f"{f}: error: {p.error}", fg="yellow", err=True)
            continue
        best = sorted(p.scores.items(), key=lambda kv: -kv[1])[:top]
        label = p.label if p.label is not None else f"unknown (closest: {p.top_label})"
        scores = "  ".join(f"{c}={v:.2f}" for c, v in best)
        typer.echo(f"{f}: {label}  sim={p.similarity:.3f}  {scores}")


def _print_eval(r: dict[str, Any]) -> None:
    acc = r["accuracy"]
    if acc is None:
        typer.echo("Nothing scored")
    else:
        typer.echo(f"{r['method']}: accuracy {acc:.3f} on {r['n_scored']}/{r['n_examples']} examples")
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
    as_json: bool = typer.Option(False, "--json"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Cross-validate a task on its own examples."""
    # evaluate imports the torch-backed pipeline; importing it lazily keeps `fewshot --help` fast.
    from .evaluate import evaluate_task  # pylint: disable=import-outside-toplevel

    t = _task(root, name)
    try:
        r = evaluate_task(_pipe(_ws(root)), t, folds)
    except ValueError as e:
        _fail(str(e))
    if as_json:
        typer.echo(json.dumps(r, indent=2))
    else:
        _print_eval(r)


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


def _bench_one(pipe: Pipeline, folder: Path, run: tuple[str, str, int, str | None], opts: dict[str, int]) -> str:
    # evaluate imports the torch-backed pipeline; importing it lazily keeps `fewshot --help` fast.
    from .evaluate import benchmark  # pylint: disable=import-outside-toplevel

    e, c, s, tt = run
    try:
        r = benchmark(pipe, folder, e, c, shot=s, text_template=tt, progress=lambda m: typer.echo(m, err=True), **opts)
    except ValueError as err:
        _fail(str(err))
    return (f"{e:42} {c:9} {opts['way']}-way {s}-shot{' +text' if tt else '      '}  "
            f"acc {r['accuracy']:.3f} ± {r['ci95']:.3f}")


@app.command()
def bench(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    folder: Path,
    embedder: list[str] = typer.Option(None, help="Embedder spec(s) to compare [default: CLIP ViT-B/32]"),
    classifier: list[str] = typer.Option(["prototype"], help="Classifier(s) to compare"),
    way: int = 5,
    shot: list[int] = typer.Option([1, 5], help="Shot count(s)"),
    query: int = 15,
    episodes: int = 200,
    text_template: str | None = typer.Option(
        None, help="e.g. 'a photo of a {}' to use class names as text (CLIP only)"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Episodic N-way K-shot benchmark over FOLDER/<class>/<image>."""
    if not folder.is_dir():
        _fail(f"Not a directory: {folder}")
    pipe = _pipe(_ws(root))
    opts = {"way": way, "query": query, "episodes": episodes}
    for run in _bench_runs(embedder or [DEFAULT_SPEC], classifier, shot, text_template):
        typer.echo(_bench_one(pipe, folder, run, opts))


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    root: Path | None = ROOT_OPT,
) -> None:
    """Run the REST API and web UI."""
    # uvicorn is only needed here; importing it lazily keeps other commands fast.
    import uvicorn  # pylint: disable=import-outside-toplevel

    if root:
        os.environ["FEWSHOT_HOME"] = str(root)
    uvicorn.run("fewshot.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
