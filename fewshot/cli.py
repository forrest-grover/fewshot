"""Command-line interface: ``fewshot --help``."""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from typing import Any

import typer

from . import cli_common, cli_eval, cli_label, cli_predict, services
from .classifiers import CLASSIFIERS, SETTING_MAX
from .cli_common import (
    CLASSIFIER_HELP,
    FAILED,
    FLAGS,
    JSON_OPT,
    ROOT_OPT,
    TASK_ARG,
    ClassifierName,
    command_group,
    error_text,
    examples,
    explained,
    fail,
    get_task,
    note,
    on_off,
    say,
    usage,
    workspace,
)
from .cli_common import echo_json as _echo_json
from .cli_data import ex_app, inbox_app
from .cli_label import cls_app
from .cli_serve import serve
from .embedders import PRESETS

README = "https://github.com/forrest-grover/fewshot#cli"
ROOT_EPILOG = f"""\b
Environment:
  FEWSHOT_HOME        the workspace, as --root [default: ./fewshot_data]
  FEWSHOT_DEVICE      the torch device models run on, as --device
  FEWSHOT_MAX_MODELS  models kept loaded at once [default: 2]
  FEWSHOT_QUIET=1     as -q;  FEWSHOT_DEBUG=1: as --debug
  HF_HOME             where model weights are downloaded [default: ~/.cache/huggingface]
  HF_HUB_OFFLINE=1    never download: use the models already downloaded

\b
Exit status: 0 done; 1 failed, or failed for some inputs (each is reported);
2 called wrongly (a usage error); 130 interrupted (Ctrl-C).

Every command takes -h, -q and --debug. More: {README}"""

app = command_group("Few-shot image classification from a handful of labelled examples per class.\n\n"
                    "A workspace holds tasks; a task holds classes, labelled example images, an inbox of "
                    "unlabelled ones and its settings. Start with `fewshot create NAME`, then `fewshot import` "
                    "or `fewshot add`, then `fewshot eval` and `fewshot predict`.", epilog=ROOT_EPILOG)
cache_app = command_group("Inspect and prune the embedding cache.")
app.add_typer(cls_app, name="class", rich_help_panel="Labelling")
app.add_typer(ex_app, name="example", rich_help_panel="Labelling")
app.add_typer(inbox_app, name="inbox", rich_help_panel="Labelling")
app.add_typer(cache_app, name="cache", rich_help_panel="Maintenance")
for _part in (cli_label.commands, cli_predict.commands, cli_eval.commands):
    app.registered_commands += _part.registered_commands
app.command(rich_help_panel="Server", epilog=examples(
    "fewshot serve                                  # http://127.0.0.1:8000, API docs at /docs",
    "FEWSHOT_TOKEN=$(openssl rand -hex 16) fewshot serve --host 0.0.0.0"))(serve)
ex_app.command("add", help="Add images as labelled examples of CLASS (the same as `fewshot add`).",
               epilog=examples("fewshot example add pets Beagle a.jpg b.jpg"))(cli_label.add)


def _version(value: bool) -> None:
    """--version: print the versions results depend on, and exit."""
    if not value:
        return
    have = {}
    for dist in ("fewshot", "torch", "open_clip_torch"):
        try:
            have[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:  # pragma: no cover  # installed with fewshot
            have[dist] = "not installed"
    typer.echo(f"fewshot {have['fewshot']} (torch {have['torch']}, open_clip {have['open_clip_torch']}, "
               f"python {platform.python_version()})")
    raise typer.Exit()


@app.callback()
def run_command(  # pylint: disable=too-many-arguments  # one per root option
    _version_opt: bool = typer.Option(False, "--version", is_eager=True, callback=_version,
                                      help="Print the versions of fewshot, torch, open_clip and Python, and exit"),
    device: str | None = typer.Option(None, "--device", envvar="FEWSHOT_DEVICE", show_envvar=True,
                                      help="Torch device models run on, e.g. cpu, cuda:1, mps [default: CUDA, then "
                                           "Apple's GPU, then the CPU]"),
    quiet: bool = typer.Option(False, "-q", "--quiet", envvar="FEWSHOT_QUIET", show_envvar=True,
                               help="Print no status messages, only results and errors"),
    debug: bool = typer.Option(False, "--debug", envvar="FEWSHOT_DEBUG", show_envvar=True,
                               help="Show tracebacks and the warnings of the libraries used"),
) -> None:
    """Before every command: set the run's options; read and write the standard streams with file
    names' bytes kept as they are, as the C locale does, so a name that isn't UTF-8 (escaped as
    lone surrogates in Python) is printed as its bytes, and read back from a piped list, instead of
    raising ``UnicodeError``. (The group's help is set on ``app``, which takes precedence.)"""
    FLAGS.reset()
    FLAGS.quiet, FLAGS.debug = quiet, debug
    if device:
        os.environ["FEWSHOT_DEVICE"] = device
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # a text stream (not a plain StringIO)
            stream.reconfigure(errors="surrogateescape")


def main() -> None:
    """Entry point of the ``fewshot`` command: the CLI, with file system errors (an unreadable
    workspace, a full disk) reported as an error message rather than a traceback, and Ctrl-C as
    "Interrupted" with status 130."""
    try:
        app()
    except OSError as e:
        if FLAGS.debug:
            raise
        where = f"{e.filename}: " if e.filename else ""
        typer.secho(f"Error: {where}{e.strerror or e}", fg="red", err=True)
        sys.exit(FAILED)
    except KeyboardInterrupt:  # outside any command (each command reports its own)
        typer.echo("\nInterrupted", err=True)
        sys.exit(130)


# ---- tasks ----------------------------------------------------------------------
@app.command(rich_help_panel="Tasks", epilog=examples("fewshot models", "fewshot models --json"))
def models(as_json: bool = JSON_OPT) -> None:
    """List the preset embedders (any openclip: or timm: spec works too)."""
    if as_json:
        _echo_json([p.__dict__ for p in PRESETS])
        return
    for p in PRESETS:
        typer.echo(f"{p.spec}\n    {p.label}. {p.notes}")


EMBEDDER_HELP = "Embedder spec; see `fewshot models`"


@app.command("create", rich_help_panel="Tasks", epilog=examples(
    "fewshot create pets", "fewshot create pets -e timm:vit_small_patch14_dinov2.lvd142m --classifier knn"))
def create(
    name: str = typer.Argument(..., metavar="NAME", help="The new task's name: letters, digits, '_', '-', '.'",
                               show_default=False),
    embedder: str | None = typer.Option(None, "--embedder", "-e", help=f"{EMBEDDER_HELP} [default: "
                                                                       f"{PRESETS[0].spec}]"),
    classifier: ClassifierName = typer.Option("gda", help=CLASSIFIER_HELP),
    root: Path | None = ROOT_OPT,
) -> None:
    """Create a task (a set of classes and their labelled examples)."""
    try:
        t = services.create_task(workspace(root, create=True), name, embedder, classifier)
    except ValueError as e:
        fail(explained(e))
    say(f"Created task {t.name} at {t.path}")
    say(f"Next: `fewshot import {t.name} FOLDER` (FOLDER/<class>/<image>) or `fewshot add {t.name} CLASS IMAGE...`")


@app.command("list", rich_help_panel="Tasks", epilog=examples("fewshot list", "fewshot list --json"))
def list_tasks(as_json: bool = JSON_OPT, root: Path | None = ROOT_OPT) -> None:
    """List tasks, with their size, inbox and last evaluation (alias: ls)."""
    ws = workspace(root)
    tasks = [t.summary() for t in ws.list_tasks()]
    if as_json:
        _echo_json(tasks)
    if not tasks:
        say(f"No tasks in {ws.root}; create one with `fewshot create NAME`")
    if as_json or not tasks:
        return
    width = max(4, *(len(s["name"]) for s in tasks))
    if cli_common.stdout_is_terminal():
        typer.echo(f"{'TASK':{width}}  CLASSES  EXAMPLES  INBOX  ACCURACY  CLASSIFIER  EMBEDDER")
    for s in tasks:
        ev = s["last_evaluation"]
        acc = "-" if not ev or ev["accuracy"] is None else f"{ev['accuracy']:.2f}"
        stale = "*" if ev and ev["revision"] != s["revision"] else " "
        typer.echo(f"{s['name']:{width}}  {s['n_classes']:7}  {s['n_examples']:8}  {s['n_inbox']:5}  {acc:>7}{stale}  "
                   f"{s['classifier']:10}  {s['embedder']}")
    if any(ev and ev["revision"] != s["revision"] for s in tasks if (ev := s["last_evaluation"])):
        say("* evaluated on an older revision; re-run `fewshot eval TASK`")
    for s in tasks:
        for w in s["warnings"]:
            note(f"{s['name']}: {w}")


app.command("ls", hidden=True)(list_tasks)


@app.command(rich_help_panel="Tasks", epilog=examples("fewshot show pets", "fewshot show pets --json"))
def show(name: str = TASK_ARG, as_json: bool = JSON_OPT, root: Path | None = ROOT_OPT) -> None:
    """Show a task's settings and its classes with their example counts."""
    t = get_task(root, name)
    c = t.config
    counts = t.summary()["counts"]
    if as_json:
        _echo_json({**c.model_dump(exclude={"examples", "classes"}),
                    "classes": [{**k.model_dump(), "count": counts.get(k.name, 0)} for k in c.classes]})
        return
    params = " ".join(f"{k}={v}" for k, v in c.classifier_params.items())
    temperature = "none (1)" if c.temperature is None else f"{c.temperature:g}"
    typer.echo(f"{c.name}\n  embedder:    {c.embedder}\n  classifier:  {c.classifier} {params}".rstrip())
    typer.echo(f"  text_weight: {c.text_weight:g}   reject_below: {_number(c.reject_below)}   "
               f"reject_below_zero_shot: {_number(c.reject_below_zero_shot)}   flip: {on_off(c.flip)}   "
               f"names: {on_off(c.names_as_text)}   temperature: {temperature}")
    width = max((len(k.name) for k in c.classes), default=0)
    for cd in c.classes:
        desc = "  " + " | ".join(f'"{line}"' for line in cd.description.splitlines() if line.strip())
        thr = "" if cd.reject_below is None else f"  reject_below {cd.reject_below:g}"
        typer.echo(f"  {cd.name:{width}}  {counts.get(cd.name, 0):5}{thr}{desc.rstrip()}")


def _number(v: float | None) -> str:
    return "none" if v is None else f"{v:g}"


def _params_help() -> str:
    """Each classifier's own parameters, for --param's help."""
    own = {n: [p for p in inspect.signature(c).parameters if p not in SETTING_MAX] for n, c in CLASSIFIERS.items()}
    return "; ".join(f"{n}: {', '.join(ps) if ps else 'none'}" for n, ps in sorted(own.items()))


PARAM_HELP = f"Classifier parameter KEY=VALUE, repeatable ({_params_help()})"


def _parse_params(param: list[str]) -> dict[str, Any]:
    params = {}
    for p in param:
        k, sep, v = p.partition("=")
        try:
            if not (k and sep):
                raise ValueError("expected KEY=VALUE")
            params[k] = json.loads(v)
        except ValueError as e:
            usage(f"Bad --param {p!r}: {e}")
    return params


@app.command(rich_help_panel="Tasks", epilog=examples(
    "fewshot config pets --classifier knn --param k=3",
    "fewshot config pets --reject-below 0.7 --flip",
    "fewshot config pets -e openclip:PE-Core-L-14-336@meta"))
def config(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals  # one per CLI option
    name: str = TASK_ARG,
    embedder: str | None = typer.Option(None, "--embedder", "-e", help=f"{EMBEDDER_HELP}; switching clears the "
                                                                       "temperature and thresholds"),
    classifier: ClassifierName | None = typer.Option(None, help=CLASSIFIER_HELP + " (see `fewshot compare`)"),
    param: list[str] = typer.Option([], help=PARAM_HELP),
    text_weight: float | None = typer.Option(None, min=0, max=SETTING_MAX["text_weight"],
                                             help="Weight of class descriptions against example images "
                                                  "(text-capable embedders) [default: 1]"),
    reject_below: float | None = typer.Option(None, min=-1, max=1,
                                              help="Similarity threshold (-1 to 1) for 'unknown'"),
    clear_reject: bool = typer.Option(False, "--clear-reject", help="Turn off the 'unknown' threshold"),
    reject_below_zero_shot: float | None = typer.Option(
        None, min=-1, max=1,
        help="Threshold for zero-shot predictions (no example images), e.g. 0.22 for CLIP"),
    clear_reject_zero_shot: bool = typer.Option(False, "--clear-reject-zero-shot",
                                                help="Turn off the zero-shot threshold"),
    flip: bool | None = typer.Option(None, "--flip/--no-flip",
                                     help="Average each image's embedding with its mirror image's"),
    names: bool | None = typer.Option(None, "--names/--no-names",
                                      help="Use class names as text for classes without a description (when "
                                           "the names match the examples)"),
    temperature: float | None = typer.Option(None, min=0.01, max=100,
                                             help="Confidence calibration (see `fewshot eval`); 1 = none"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Change a task's settings, then show them.

    --param sets classifier parameters on top of the current ones; switching classifier starts its
    parameters afresh (then --param sets some). Every change can be undone with `fewshot undo`.
    """
    t = get_task(root, name)
    values = {"embedder": embedder, "classifier": classifier or None, "text_weight": text_weight,
              "reject_below": reject_below, "reject_below_zero_shot": reject_below_zero_shot, "flip": flip,
              "names_as_text": names, "temperature": temperature}
    clears = {"reject_below": clear_reject, "reject_below_zero_shot": clear_reject_zero_shot}
    change = services.SettingsChange({k: v for k, v in values.items() if v is not None},
                                     frozenset(k for k, on in clears.items() if on), _parse_params(param))
    try:
        _say_cleared(name, services.apply_settings(t, change))
    except services.SettingConflict as e:
        set_flag, clear_flag = _CONFIG_FLAGS[e.setting]
        usage(f"Give either {set_flag} or {clear_flag}, not both")
    except ValueError as e:
        if str(e).startswith("Unknown parameter"):
            usage(f"{e} (--param)")
        fail(explained(e, name))
    show(name, False, root)


_CONFIG_FLAGS = {"reject_below": ("--reject-below", "--clear-reject"),  # setting: (its option, clearing it)
                 "reject_below_zero_shot": ("--reject-below-zero-shot", "--clear-reject-zero-shot")}


def _say_cleared(name: str, cleared: list[str]) -> None:
    """Tell the user which settings switching embedder or scoring settings cleared, if any."""
    if cleared:
        say(f"Cleared {', '.join(cleared)}: fitted to the previous settings (`fewshot undo {name}` restores them)")


@app.command(rich_help_panel="Tasks", epilog=examples("fewshot delete pets", "fewshot delete pets --yes"))
def delete(name: str = TASK_ARG,
           yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation (needed without a "
                                                               "terminal)"),
           root: Path | None = ROOT_OPT) -> None:
    """Delete a task and its images, for good (alias: rm; this can't be undone)."""
    get_task(root, name)
    if not yes:
        if not cli_common.stdin_is_terminal():
            fail(f"Refusing to delete task {name} without confirmation; pass --yes")
        if not typer.confirm(f"Delete task {name} and all its images? This can't be undone"):
            fail("Not deleted")
    try:
        workspace(root).delete(name)
    except KeyError as e:  # deleted by someone else meanwhile
        fail(error_text(e))
    say(f"Deleted task {name}")


app.command("rm", hidden=True)(delete)


@app.command(rich_help_panel="Tasks", epilog=examples("fewshot rename pets dogs-and-cats"))
def rename(name: str = TASK_ARG,
           new_name: str = typer.Argument(..., metavar="NEW", help="The task's new name", show_default=False),
           root: Path | None = ROOT_OPT) -> None:
    """Rename a task."""
    get_task(root, name)
    try:
        workspace(root).rename(name, new_name)
    except (KeyError, ValueError) as e:
        fail(error_text(e))
    say(f"Renamed task {name} to {new_name}")


@app.command("copy", rich_help_panel="Tasks", epilog=examples("fewshot copy pets pets-knn",
                                                              "fewshot copy pets pets-empty --no-examples"))
def copy_task(
    name: str = TASK_ARG,
    new_name: str = typer.Argument(..., metavar="NEW", help="The copy's name", show_default=False),
    no_examples: bool = typer.Option(False, "--no-examples", help="Copy only settings and classes"),
    inbox: bool = typer.Option(False, "--inbox", help="Copy the inbox too"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Copy a task under a new name, to try other settings without touching it.

    Images are shared, not copied; the copy starts with no undo history.
    """
    get_task(root, name)
    try:
        t = workspace(root).duplicate(name, new_name, examples=not no_examples, inbox=inbox)
    except (KeyError, ValueError) as e:
        fail(error_text(e))
    say(f"Copied {name} to {t.name} ({len(t.config.examples)} examples)")


def _sizes(name: str, root: Path | None) -> tuple[int, int]:
    t = get_task(root, name)
    return len(t.config.examples), len(t.config.classes)


def _delta(before: tuple[int, int], after: tuple[int, int]) -> str:
    """How many examples and classes there were and are, e.g. "64 -> 52 examples"."""
    parts = [f"{before[0]} -> {after[0]} examples" if before[0] != after[0] else f"{after[0]} examples"]
    if before[1] != after[1]:
        parts.append(f"{before[1]} -> {after[1]} classes")
    return ", ".join(parts)


@app.command(rich_help_panel="Tasks", epilog=examples("fewshot undo pets", "fewshot undo pets && fewshot undo pets"))
def undo(name: str = TASK_ARG, root: Path | None = ROOT_OPT) -> None:
    """Undo a task's last change (settings, classes or examples).

    Repeat to go further back (the last 20 versions are kept); `fewshot redo` makes it again.
    """
    t = get_task(root, name)
    label, before = t.undo_label(), _sizes(name, root)
    try:
        rev = t.undo()
    except (KeyError, ValueError) as e:
        fail(error_text(e))
    say(f"{name}: undid {label or 'the last change'}, back to revision {rev} ({_delta(before, _sizes(name, root))})")


@app.command(rich_help_panel="Tasks", epilog=examples("fewshot redo pets"))
def redo(name: str = TASK_ARG, root: Path | None = ROOT_OPT) -> None:
    """Make the last undone change again (until another change is made)."""
    t = get_task(root, name)
    label, before = t.redo_label(), _sizes(name, root)
    try:
        t.redo()
    except (KeyError, ValueError) as e:
        fail(error_text(e))
    say(f"{name}: redid {label or 'the last undone change'} ({_delta(before, _sizes(name, root))})")


@app.command(rich_help_panel="Tasks", epilog=examples("fewshot export pets pets.zip",
                                                      "fewshot export pets - | ssh server fewshot import-task -"))
def export(name: str = TASK_ARG,
           file: Path = typer.Argument(..., metavar="ZIP", help="The zip file to write; '-' for stdout",
                                       show_default=False),
           root: Path | None = ROOT_OPT) -> None:
    """Save a task (settings, classes, labels, inbox and images) as a zip.

    For a backup, or to move it to another workspace with `fewshot import-task`. Exits 1 if
    examples had to be left out (their images are missing).
    """
    from .archive import export_task  # pylint: disable=import-outside-toplevel  # keeps --help fast

    ws = workspace(root)
    get_task(root, name)
    if str(file) == "-":
        if cli_common.stdout_is_terminal():
            fail("Refusing to write a zip to the terminal; redirect it, e.g. `fewshot export TASK - > task.zip`")
        sys.stdout.flush()
        n, skipped = export_task(ws, name, sys.stdout.buffer)
        sys.stdout.buffer.flush()
    else:
        if file.is_dir() or not file.name:
            usage(f"{file} is a folder; give the zip file's name, e.g. {file / (name + '.zip')}")
        n, skipped = _export_to(ws, name, file, export_task)
    say(f"Exported {name} ({n} examples) to {'stdout' if str(file) == '-' else file}")
    if skipped:
        fail(f"Left out {skipped} of {n + skipped} examples: their image files are missing")


def _export_to(ws: Any, name: str, file: Path, export_task: Any) -> tuple[int, int]:
    part = file.with_name(file.name + ".part")  # renamed into place when complete
    try:
        with open(part, "wb") as f:
            n, skipped = export_task(ws, name, f)
        os.replace(part, file)
    except OSError as e:
        part.unlink(missing_ok=True)
        fail(f"Can't write {file}: {e.strerror or e}")
    return n, skipped


@app.command("import-task", rich_help_panel="Tasks", epilog=examples(
    "fewshot import-task pets.zip", "fewshot import-task pets.zip --name pets-copy",
    "cat pets.zip | fewshot import-task -"))
def import_task(
    file: Path = typer.Argument(..., metavar="ZIP", help="A zip made by `fewshot export`; '-' for stdin",
                                show_default=False),
    name: str | None = typer.Option(None, help="Name for the task [default: the exported name]"),
    embedder: str | None = typer.Option(None, "--embedder", "-e", help="Use this embedder instead of the exported "
                                                                       "one"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Create a task from a zip made by `fewshot export`."""
    from .archive import import_task as read_export  # pylint: disable=import-outside-toplevel  # keeps --help fast

    if str(file) != "-" and not file.is_file():
        fail(f"No such file: {file}")
    try:
        with _zip_input(file) as src:
            t = read_export(workspace(root, create=True), src, name, embedder=embedder or None)
    except ValueError as e:
        msg = explained(e)
        fail(f"{msg}; pass --name NEW" if msg.endswith("already exists") else msg)
    say(f"Imported task {t.name} ({len(t.config.examples)} examples)")


@contextlib.contextmanager
def _zip_input(file: Path) -> Iterator[Any]:
    """The zip to import: ``file``, or for ``-`` stdin copied to a temporary file (a zip is read
    from its end)."""
    if str(file) != "-":
        yield file
        return
    with tempfile.SpooledTemporaryFile(max_size=64 * 2**20) as f:
        shutil.copyfileobj(sys.stdin.buffer, f)
        f.seek(0)
        yield f


# ---- maintenance -----------------------------------------------------------------
@app.command(rich_help_panel="Maintenance", epilog=examples("fewshot gc --dry-run", "fewshot gc"))
def gc(dry_run: bool = typer.Option(False, "--dry-run", help="Only report what would be removed"),
       as_json: bool = JSON_OPT, root: Path | None = ROOT_OPT) -> None:
    """Remove what crashes and undo history leave behind.

    That is: half-made or half-deleted task folders (older than 10 minutes), image files no kept
    version of their task uses, and stored images no task uses.
    """
    n = workspace(root).gc(dry_run=dry_run)
    if as_json:
        _echo_json({"dry_run": dry_run, **n})
        return
    say(f"{'Would remove' if dry_run else 'Removed'} {n['folders']} leftover folders, {n['images']} unused image "
        f"files and {n['blobs']} unused stored images")


@cache_app.command("info", epilog=examples("fewshot cache info", "fewshot cache info --json"))
def cache_info(as_json: bool = JSON_OPT, root: Path | None = ROOT_OPT) -> None:
    """Show the cache file's size and how many vectors each model has cached."""
    from .cache import EmbeddingCache  # pylint: disable=import-outside-toplevel  # keeps --help fast

    ws = workspace(root)
    exists = ws.cache_path.is_file()  # opening it would make it (and the workspace)
    stats = EmbeddingCache(ws.cache_path).stats() if exists else []
    size = ws.cache_path.stat().st_size if exists else 0
    if as_json:
        _echo_json({"path": str(ws.cache_path), "bytes": size,
                    "models": [{"namespace": ns, "vectors": n, "bytes": b} for ns, n, b in stats]})
    elif not exists:
        say(f"No cache at {ws.cache_path}")
    else:
        typer.echo(f"{ws.cache_path}: {_mb(size)}")
        for ns, n, b in stats:
            typer.echo(f"  {ns}: {n} vectors, {_mb(b)}")


@cache_app.command("prune", epilog=examples("fewshot cache prune --dry-run", "fewshot cache prune --vacuum"))
def cache_prune(
    dry_run: bool = typer.Option(False, "--dry-run", help="Only report how many vectors would go"),
    vacuum: bool = typer.Option(False, "--vacuum", help="Shrink the cache file afterwards (slow for a large cache)"),
    as_json: bool = JSON_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """Delete cached vectors and stored images that no task uses.

    That is: image vectors of deleted examples and `bench` datasets, and stored images no task
    links to any more. Text vectors are kept. Refuses to run if a task can't be read, since its
    examples would be lost. The space freed is reused by the cache; --vacuum gives it back to the
    disk, which rewrites the whole file and keeps other processes from writing to the cache meanwhile.
    """
    from .cache import prune_unused  # pylint: disable=import-outside-toplevel  # keeps --help fast

    ws = workspace(root)
    if not ws.cache_path.is_file() and (dry_run or not ws.root.is_dir()):  # a look shouldn't make one
        n, blobs = 0, 0
        say(f"No cache at {ws.cache_path}: nothing to prune")
    else:
        try:
            n, blobs = prune_unused(ws, dry_run=dry_run, vacuum=vacuum)
        except ValueError as e:
            fail(str(e))
    size = ws.cache_path.stat().st_size if ws.cache_path.is_file() else 0
    if as_json:
        _echo_json({"dry_run": dry_run, "vectors": n, "stored_images": blobs, "cache_bytes": size})
    elif ws.cache_path.is_file():
        say(f"{'Would delete' if dry_run else 'Deleted'} {n} cached image vectors; {_mb(size)} on disk")
        if not dry_run:
            say(f"Deleted {blobs} stored images no task uses")


def _mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


if __name__ == "__main__":
    main()
