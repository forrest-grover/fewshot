"""Command-line interface: ``fewshot --help``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from .store import IMAGE_EXTS, Workspace

app = typer.Typer(help="Embedding-based few-shot image classification.", no_args_is_help=True)
cls_app = typer.Typer(help="Manage a task's classes.", no_args_is_help=True)
app.add_typer(cls_app, name="class")

RootOpt = typer.Option(None, "--root", envvar="FEWSHOT_HOME", help="Workspace directory [default: ./fewshot_data]")


def _ws(root: Optional[Path]) -> Workspace:
    return Workspace(root)


def _pipe(ws: Workspace):
    from .pipeline import Pipeline  # imports torch; keep `fewshot --help` fast

    return Pipeline(ws)


def _expand(paths: List[Path]) -> List[Path]:
    out = []
    for p in paths:
        if not p.exists():
            _fail(f"No such file or directory: {p}")
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        else:
            out.append(p)
    return out


def _fail(msg: str) -> None:
    typer.secho(f"Error: {msg}", fg="red", err=True)
    raise typer.Exit(1)


@app.command()
def models():
    """List preset embedders (any openclip:/timm: spec also works)."""
    from .embedders import PRESETS

    for p in PRESETS:
        typer.echo(f"{p.spec}\n    {p.label}. {p.notes}")


@app.command("create")
def create(
    name: str,
    embedder: Optional[str] = typer.Option(None, help="Embedder spec; see `fewshot models`"),
    classifier: str = typer.Option("prototype", help="prototype | knn | logreg"),
    root: Optional[Path] = RootOpt,
):
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
def list_tasks(root: Optional[Path] = RootOpt):
    """List tasks."""
    for t in _ws(root).list_tasks():
        s = t.summary()
        typer.echo(f"{s['name']:24} {s['n_classes']:3} classes {s['n_examples']:5} examples  {s['classifier']:9} {s['embedder']}")


@app.command()
def show(name: str, root: Optional[Path] = RootOpt):
    """Show a task's settings and per-class example counts."""
    t = _ws(root).get(name)
    c = t.config
    typer.echo(f"{c.name}\n  embedder:   {c.embedder}\n  classifier: {c.classifier} {c.classifier_params or ''}"
               f"\n  text_weight: {c.text_weight}   reject_below: {c.reject_below}")
    counts = t.summary()["counts"]
    for cd in c.classes:
        desc = f'  "{cd.description}"' if cd.description else ""
        typer.echo(f"  {cd.name:24} {counts.get(cd.name, 0):5}{desc}")


@app.command()
def config(
    name: str,
    embedder: Optional[str] = None,
    classifier: Optional[str] = None,
    param: List[str] = typer.Option([], help="Classifier param KEY=VALUE (e.g. k=3, scale=20). Repeatable."),
    text_weight: Optional[float] = None,
    reject_below: Optional[float] = typer.Option(None, help="Similarity threshold for 'unknown'; negative clears it"),
    root: Optional[Path] = RootOpt,
):
    """Change task settings."""
    t = _ws(root).get(name)
    kw = {}
    if embedder:
        kw["embedder"] = embedder
    if classifier:
        kw["classifier"] = classifier
        kw["classifier_params"] = {}
    if param:
        params = dict(kw.get("classifier_params", t.config.classifier_params))
        for p in param:
            k, _, v = p.partition("=")
            params[k] = json.loads(v)
        kw["classifier_params"] = params
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
def delete(name: str, yes: bool = typer.Option(False, "--yes", "-y"), root: Optional[Path] = RootOpt):
    """Delete a task and its images."""
    if not yes:
        typer.confirm(f"Delete task {name} and all its images?", abort=True)
    _ws(root).delete(name)


@cls_app.command("add")
def class_add(name: str, label: str, description: str = typer.Option("", "--desc", help="Text prompt, e.g. 'a photo of a cracked weld'"), root: Optional[Path] = RootOpt):
    """Add a class, or set an existing class's description."""
    _ws(root).get(name).upsert_class(label, description)


@cls_app.command("describe")
def class_describe(
    name: str,
    template: str = typer.Argument("a photo of a {}", help="'{}' is replaced by the class name (underscores become spaces)"),
    overwrite: bool = typer.Option(False, help="Also replace descriptions that are already set"),
    root: Optional[Path] = RootOpt,
):
    """Fill class descriptions from a template, e.g. 'a photo of a {}, a type of pet'."""
    t = _ws(root).get(name)
    n = t.describe_classes(template, overwrite)
    typer.echo(f"Set {n} descriptions")


@cls_app.command("rm")
def class_rm(name: str, label: str, root: Optional[Path] = RootOpt):
    """Delete a class and its examples."""
    n = _ws(root).get(name).delete_class(label)
    typer.echo(f"Removed class {label} and {n} examples")


@app.command()
def add(name: str, label: str, paths: List[Path], root: Optional[Path] = RootOpt):
    """Add images (files or directories) as examples of LABEL."""
    t = _ws(root).get(name)
    files = _expand(paths)
    for f in files:
        try:
            t.add_example(f.read_bytes(), label, f.name)
        except ValueError as e:
            typer.secho(f"skip {f}: {e}", fg="yellow", err=True)
    typer.echo(f"{name}: {len(t.config.examples)} examples")


@app.command("import")
def import_dir(
    name: str,
    folder: Path,
    shots: Optional[int] = typer.Option(None, help="Take at most this many images per class"),
    root: Optional[Path] = RootOpt,
):
    """Import FOLDER/<class>/<image> into a task."""
    t = _ws(root).get(name)
    n = 0
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)[:shots]
        for f in files:
            t.add_example(f.read_bytes(), d.name, f.name)
            n += 1
    typer.echo(f"Imported {n} images into {name}")


@app.command()
def predict(
    name: str,
    paths: List[Path],
    as_json: bool = typer.Option(False, "--json"),
    top: int = typer.Option(3, help="Scores to show per image"),
    root: Optional[Path] = RootOpt,
):
    """Classify images (files or directories)."""
    ws = _ws(root)
    files = _expand(paths)
    try:
        preds = _pipe(ws).predict(ws.get(name), files)
    except ValueError as e:
        _fail(str(e))
    if as_json:
        typer.echo(json.dumps([{"file": str(f), **p.to_dict()} for f, p in zip(files, preds)], indent=2))
        return
    for f, p in zip(files, preds):
        best = sorted(p.scores.items(), key=lambda kv: -kv[1])[:top]
        label = p.label if p.label is not None else f"unknown (closest: {p.top_label})"
        typer.echo(f"{f}: {label}  sim={p.similarity:.3f}  " + "  ".join(f"{c}={v:.2f}" for c, v in best))


@app.command("eval")
def eval_cmd(
    name: str,
    folds: Optional[int] = typer.Option(None, help="Number of folds [default: leave-one-out up to 300 examples, else 5]"),
    as_json: bool = typer.Option(False, "--json"),
    root: Optional[Path] = RootOpt,
):
    """Cross-validate a task on its own examples."""
    from .evaluate import evaluate_task

    ws = _ws(root)
    try:
        r = evaluate_task(_pipe(ws), ws.get(name), folds)
    except ValueError as e:
        _fail(str(e))
    if as_json:
        typer.echo(json.dumps(r, indent=2))
        return
    acc = r["accuracy"]
    typer.echo(f"{r['method']}: accuracy {acc:.3f} on {r['n_scored']}/{r['n_examples']} examples" if acc is not None else "Nothing scored")
    for c, m in r["per_class"].items():
        rec = "-" if m["recall"] is None else f"{m['recall']:.2f}"
        prec = "-" if m["precision"] is None else f"{m['precision']:.2f}"
        typer.echo(f"  {c:24} n={m['support']:<4} recall={rec:5} precision={prec}")
    if r["mistakes"]:
        typer.echo(f"{len(r['mistakes'])} mistakes; most confident:")
        for m in r["mistakes"][:10]:
            typer.echo(f"  {m['id']}  {m['true']} -> {m['pred']} ({m['confidence']:.2f})")


@app.command()
def bench(
    folder: Path,
    embedder: List[str] = typer.Option(None, help="Embedder spec(s) to compare [default: CLIP ViT-B/32]"),
    classifier: List[str] = typer.Option(["prototype"], help="Classifier(s) to compare"),
    way: int = 5,
    shot: List[int] = typer.Option([1, 5], help="Shot count(s)"),
    query: int = 15,
    episodes: int = 200,
    text_template: Optional[str] = typer.Option(None, help="e.g. 'a photo of a {}' to use class names as text (CLIP only)"),
    root: Optional[Path] = RootOpt,
):
    """Episodic N-way K-shot benchmark over FOLDER/<class>/<image>."""
    from .embedders import DEFAULT_SPEC, spec_supports_text
    from .evaluate import benchmark

    pipe = _pipe(_ws(root))
    for e in embedder or [DEFAULT_SPEC]:
        tt = text_template if spec_supports_text(e) else None
        for c in classifier:
            for s in shot:
                if s == 0 and not tt:
                    continue
                r = benchmark(pipe, folder, e, c, way, s, query, episodes, tt, progress=lambda m: typer.echo(m, err=True))
                typer.echo(f"{e:42} {c:9} {way}-way {s}-shot{' +text' if tt else '      '}  acc {r['accuracy']:.3f} ± {r['ci95']:.3f}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    root: Optional[Path] = RootOpt,
):
    """Run the REST API and web UI."""
    import os

    import uvicorn

    if root:
        os.environ["FEWSHOT_HOME"] = str(root)
    uvicorn.run("fewshot.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
