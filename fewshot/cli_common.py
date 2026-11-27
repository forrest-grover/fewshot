"""Helpers shared by the command-line modules: the command classes, options, messages, errors,
task lookup, inputs and output."""

from __future__ import annotations

import contextlib
import csv
import difflib
import json
import logging
import re
import sys
import time
import traceback
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, get_args

import typer
from typer.core import TyperCommand, TyperGroup

from . import services
from .classifiers import NotReady
from .embedders import report_loading
from .fetch import fetch_url, is_url
from .folders import image_files, readable_name
from .inbox import ALREADY_LABELLED
from .services import Added, Blob
from .store import Task, Workspace

if TYPE_CHECKING:
    from .pipeline import Pipeline

MAX_DOWNLOAD = 50 * 1024 * 1024  # largest image fetched from a URL
ClassifierName = Literal["gda", "knn", "logreg", "prototype"]  # classifiers.CLASSIFIERS (a test keeps them equal)
CLASSIFIER_LIST = " | ".join(get_args(ClassifierName))
InboxSort = Literal["uncertain", "confident", "novel", "added"]  # Pipeline.inbox_suggestions's orders
JSON_OPT = typer.Option(False, "--json", help="Print JSON")
ROOT_OPT = typer.Option(None, "--root", envvar="FEWSHOT_HOME", help="Workspace directory [default: ./fewshot_data]")
TASK_ARG = typer.Argument(..., metavar="TASK", help="Task name; see `fewshot list`", show_default=False)
CLASS_ARG = typer.Argument(..., metavar="CLASS", help="Class name", show_default=False)
IMAGES_ARG = typer.Argument(..., metavar="IMAGE", help="Image files, folders or http(s) URLs", show_default=False)
IDS_ARG = typer.Argument(..., metavar="ID", help="Example ids, or unique prefixes of 4+ characters",
                         show_default=False)
INBOX_IDS_ARG = typer.Argument(..., metavar="ID", help="Inbox image ids, or unique prefixes of 4+ characters",
                               show_default=False)
STRICT_OPT = typer.Option(False, "--strict", help="Exit 1 if any image was skipped as already there")
NEW_CLASS_OPT = typer.Option(False, "--new-class", help="Create CLASS even if its name is close to an existing "
                                                       "class's (without it, a near miss such as 'beagel' for "
                                                       "'Beagle' is refused as a likely typo)")
CLASSIFIER_HELP = ("How examples are turned into predictions: gda (Gaussian discriminant analysis), prototype, "
                   "knn or logreg")
ID_LEN = 12  # characters of an id shown in human output; any unique prefix of 4+ is accepted as input

# Exit statuses: 1 when the command failed (or failed for some of its inputs), 2 for a usage error
# (options that don't go together, missing arguments), as for the usage errors Typer reports itself;
# 130 when interrupted (Ctrl-C).
FAILED, USAGE, INTERRUPTED = 1, 2, 130
PANELS = ("Tasks", "Labelling", "Predicting", "Evaluating", "Maintenance", "Server")  # the root help's sections
HELP_OPTIONS: dict[str, Any] = {"help_option_names": ["-h", "--help"]}


@dataclass
class Flags:
    """Options that hold for the whole run: set by the root command and by each command's own -q/--debug."""

    quiet: bool = False
    debug: bool = False
    interrupt_note: str = ""  # what to say after "Interrupted"; set by commands that know what was saved

    def reset(self) -> None:
        """Back to the defaults."""
        self.quiet = self.debug = False
        self.interrupt_note = ""


FLAGS = Flags()


def shared_options() -> list[Any]:
    """The -q/--quiet and --debug options every command takes (the root command too), as Typer
    makes them."""
    holder = typer.Typer(add_completion=False)

    @holder.command()
    def options(  # pragma: no cover  # never run: only its parameters are used
        quiet: bool = typer.Option(False, "-q", "--quiet", envvar="FEWSHOT_QUIET", show_envvar=True,
                                   help="Print no status messages, only results and errors"),
        debug: bool = typer.Option(False, "--debug", envvar="FEWSHOT_DEBUG", show_envvar=True,
                                   help="Show tracebacks and the warnings of the libraries used"),
    ) -> None:
        del quiet, debug

    return list(typer.main.get_command(holder).params)


class Command(TyperCommand):
    """A command with the shared options, and a clean exit (status 130) when interrupted."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.params.extend(shared_options())

    def invoke(self, ctx: Any) -> Any:
        FLAGS.quiet |= ctx.params.pop("quiet")  # a command's option adds to the root's, never turns it off
        FLAGS.debug |= ctx.params.pop("debug")
        try:
            with library_output():
                return super().invoke(ctx)
        except KeyboardInterrupt:
            interrupted()
        finally:
            FLAGS.reset()  # back to the defaults for whatever runs next in this process


QUIET_LOGGERS = ("huggingface_hub", "timm", "open_clip", "httpx", "urllib3", "PIL")  # chatty libraries


@contextlib.contextmanager
def library_output() -> Iterator[None]:
    """While a command runs: model loads and downloads announced, and (without --debug) the
    warnings and log messages of the libraries used hidden."""
    report_loading(say)
    levels = {name: logging.getLogger(name).level for name in QUIET_LOGGERS}
    with warnings.catch_warnings():
        if not FLAGS.debug:
            warnings.filterwarnings("ignore", module=r"(?!fewshot)")
            for name in QUIET_LOGGERS:
                logging.getLogger(name).setLevel(logging.ERROR)
        try:
            yield
        finally:
            report_loading(None)
            for name, level in levels.items():
                logging.getLogger(name).setLevel(level)


def interrupted() -> NoReturn:
    """Say that Ctrl-C stopped the command (and, where the command knows, what it had saved); exit 130."""
    typer.echo("\nInterrupted" + (f"; {FLAGS.interrupt_note}" if FLAGS.interrupt_note else ""), err=True)
    raise typer.Exit(INTERRUPTED)


class Group(TyperGroup):
    """A group of commands whose help lists them under their ``rich_help_panel`` headings (which
    Typer shows only with Rich markup on), each with the first sentence of its help in full."""

    def format_commands(self, ctx: Any, formatter: Any) -> None:
        panels: dict[str, list[tuple[str, str]]] = {}
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is not None and not cmd.hidden:
                panel = getattr(cmd, "rich_help_panel", None) or "Commands"
                panels.setdefault(panel, []).append((name, cmd.get_short_help_str(limit=1000)))
        for panel in sorted(panels, key=lambda p: PANELS.index(p) if p in PANELS else len(PANELS)):
            with formatter.section(panel):
                formatter.write_dl(panels[panel])


class App(typer.Typer):
    """A Typer app whose commands take -h as well as --help and the shared options."""

    def command(self, name: str | None = None, *, cls: type[TyperCommand] | None = None,
                context_settings: dict[Any, Any] | None = None, **kwargs: Any) -> Callable[[Any], Any]:
        return super().command(name, cls=cls or Command, context_settings={**HELP_OPTIONS, **(context_settings or {})},
                               **kwargs)


def command_group(help_text: str, epilog: str | None = None) -> App:
    """A group of commands. Help is printed as written: with Rich markup on, Typer would take
    ``[default: ...]`` and any other text in brackets for markup and drop it."""
    return App(help=help_text, epilog=epilog, no_args_is_help=True, rich_markup_mode=None, cls=Group,
               context_settings=HELP_OPTIONS)


def examples(*lines: str) -> str:
    """A help epilog of example command lines, printed as written (``\\b`` stops Click rewrapping them)."""
    return "\b\nExamples:\n" + "\n".join(f"  {line}" for line in lines)


def classifier_names(names: list[str]) -> list[str]:
    """Check a repeatable --classifier option's values (Typer can't make a list of choices)."""
    for n in names:
        if n not in get_args(ClassifierName):
            raise typer.BadParameter(f"{n!r} is not one of " + ", ".join(map(repr, get_args(ClassifierName))))
    return names


# ---- messages -----------------------------------------------------------------
def say(msg: str) -> None:
    """Print a status message (what a command did, or a hint) on stderr, unless --quiet."""
    if not FLAGS.quiet:
        typer.echo(msg, err=True)


def note(msg: str) -> None:
    """Print a warning on stderr (in yellow), unless --quiet."""
    if not FLAGS.quiet:
        typer.secho(msg, fg="yellow", err=True)


def fail(msg: str, code: int = FAILED) -> NoReturn:
    """Print ``msg`` as an error and exit with status ``code`` (1: the command failed); with
    --debug, after the traceback of the exception being handled, if any."""
    if FLAGS.debug and sys.exc_info()[1] is not None:
        traceback.print_exc()
    typer.secho(f"Error: {msg}", fg="red", err=True)
    raise typer.Exit(code)


def usage(msg: str) -> NoReturn:
    """Print ``msg`` as an error and exit with status 2: the command was called wrongly."""
    fail(msg, USAGE)


def error_text(e: Exception) -> str:
    """The readable message of an error: for a KeyError its first argument, not its repr."""
    return str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)


MODEL_ERRORS = ("Could not load embedder", "Bad embedder spec", "Unknown embedder backend", "Embedder spec is empty")


def explained(e: Exception, task: str = "TASK") -> str:
    """The message of ``e`` with the next step to take, for the errors that have one."""
    msg = error_text(e)
    if isinstance(e, NotReady):
        return (f"{msg}; add examples with `fewshot add {task} CLASS IMAGE...` or describe classes with "
                f"`fewshot class add {task} CLASS --desc ...`")
    if msg.startswith("Task has no examples"):
        return f"{msg}; add some with `fewshot add {task} CLASS IMAGE...` or `fewshot import {task} FOLDER`"
    if msg.startswith(MODEL_ERRORS) and "HF_HUB_OFFLINE" not in msg:
        return f"{msg}; see `fewshot models`"
    return msg


def fail_on(e: Exception, task: str = "TASK") -> NoReturn:
    """Fail with ``e``'s message and the next step to take (see ``explained``)."""
    fail(explained(e, task))


def did_you_mean(name: str, names: Sequence[str]) -> str:
    """``"; did you mean 'x'?"`` for the names closest to ``name`` (the same but for case first), or ``""``."""
    same_case = [n for n in names if n.lower() == name.lower()]
    close = same_case or difflib.get_close_matches(name, names, n=3, cutoff=0.6)
    return f"; did you mean {' or '.join(map(repr, close))}?" if close else ""


def _edits(a: str, b: str) -> int:
    """The edit (Levenshtein) distance between ``a`` and ``b``."""
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def near_miss(name: str, names: Sequence[str]) -> str | None:
    """An existing class name that ``name`` (not one of them) is probably a typo of: the same but
    for case, or within 2 edits (1 for names of 4 characters or fewer)."""
    allowed = 1 if len(name) <= 4 else 2
    for n in names:
        if n.lower() == name.lower() or _edits(n.lower(), name.lower()) <= allowed:
            return n
    return None


def check_new_class(t: Task, label: str, new_class: bool) -> bool:
    """Before labelling images as ``label``: whether it is a new class. Unless ``new_class``, a
    new name close to an existing class's is refused as a likely typo."""
    names = t.class_names()
    if label in names:
        return False
    if not new_class and (close := near_miss(label, names)):
        fail(f"No class {label!r} in {t.name}; did you mean {close!r}? Pass --new-class to create it")
    return True


def check_class(t: Task, label: str) -> None:
    """Exit with an error (and the closest names) unless ``t`` has class ``label``."""
    if label not in t.class_names():
        fail(f"No class {label!r} in {t.name}{did_you_mean(label, t.class_names())}; see `fewshot class ls {t.name}`")


def stdin_is_terminal() -> bool:
    """Whether stdin is a terminal (someone to ask, or to tell what is read from it)."""
    return sys.stdin.isatty()


def stdout_is_terminal() -> bool:
    """Whether stdout is a terminal (a person reading, not a program or a file)."""
    return sys.stdout.isatty()


# ---- the workspace and tasks ---------------------------------------------------------
def workspace(root: Path | None, create: bool = False) -> Workspace:
    """The workspace at ``root`` (default from FEWSHOT_HOME or ./fewshot_data). Unless ``create``
    (for commands that make tasks), a missing one isn't made: it reads as empty, so a mistyped
    --root doesn't leave a new folder behind. Making one is announced."""
    ws = Workspace(root, create=False)
    if create and not (ws.root / "tasks").is_dir():
        ws = Workspace(root, create=True)
        say(f"Created workspace {ws.root} (set FEWSHOT_HOME or --root to use another)")
    return ws


def get_task(root: Path | None, name: str) -> Task:
    """Task ``name``, or exit with an error saying where it was looked for and what there is."""
    ws = workspace(root)
    try:
        return ws.get(name)
    except KeyError as e:
        msg = error_text(e)
        if not msg.startswith("No task"):  # there, but unreadable
            fail(msg)
        names = [p.name for p in ws.task_folders()]
        there = f"tasks: {', '.join(names[:10])}{', ...' if len(names) > 10 else ''}" if names else "no tasks"
        fail(f"No task {name!r} in {ws.root} ({there}){did_you_mean(name, names)}; run `fewshot list` "
             "or set --root/FEWSHOT_HOME")


def make_pipeline(ws: Workspace) -> Pipeline:
    """A pipeline for ``ws``; imported lazily, since it imports torch and `fewshot --help` should be fast."""
    from .pipeline import Pipeline  # pylint: disable=import-outside-toplevel

    return Pipeline(ws)


# ---- inputs ----------------------------------------------------------------------
Source = Path | str  # an image file, or an http(s) URL


def images_in(folder: Path, recursive: bool = True) -> list[Path]:
    """Image files under ``folder``, sorted; hidden and system files and folders, folders and broken
    links with image names are left out."""
    return image_files(folder, recursive)


def expand(paths: list[str]) -> list[Source]:
    """URLs and files named on the command line plus the images inside named folders; a path that
    doesn't exist is kept, to fail as an unreadable image does. Fails if there are no images."""
    out: list[Source] = []
    for arg in paths:
        p = Path(arg)
        out += [arg] if is_url(arg) else images_in(p) if p.is_dir() else [p]
    if not out:
        fail("No images found in " + ", ".join(paths))
    return out


def read_source(src: Source) -> bytes:
    """The bytes of a file or URL; ``ValueError`` says why they couldn't be read."""
    if isinstance(src, str):
        return fetch_url(src, MAX_DOWNLOAD)
    try:
        return src.read_bytes()
    except OSError as e:
        raise ValueError(e.strerror or str(e)) from None


def source_name(src: Source) -> str:
    """The file name of a path or URL (as valid Unicode, see ``readable_name``)."""
    return readable_name(src.name if isinstance(src, Path) else src.rstrip("/").rsplit("/", 1)[-1])


Update = Callable[..., None]  # update(done, total, stage=None)
STAGES = {"embedding": "Embedding", "evaluating": "Cross-validating fold"}  # the library's stage names, shown
PLAIN_AFTER = 3.0  # seconds after which a run without a terminal prints its progress every 10%


@contextlib.contextmanager
def progress(what: str) -> Iterator[Update]:
    """Progress on stderr while the block runs: a bar on a terminal, else (for runs longer than a
    few seconds) a line every 10%, unless --quiet. Yields ``update(done, total, stage=None)``,
    where ``stage`` (``"embedding"``, ``"evaluating"`` or any text) replaces ``what``."""
    from rich.console import Console  # pylint: disable=import-outside-toplevel  # keeps --help fast
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn  # pylint: disable=import-outside-toplevel

    tty = sys.stderr.isatty()
    start, shown_pct = time.monotonic(), {"": -1}

    with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
                  console=Console(stderr=True), transient=True, disable=not tty or FLAGS.quiet) as shown:
        tid = shown.add_task(what, total=None)

        def update(done: int, total: int | None, stage: str | None = None) -> None:
            model, _, step = (stage or "").rpartition(" ")  # compare's stages are like "2/3 embedding"
            label = STAGES.get(step, stage or what) + (f" (model {model})" if model and step in STAGES else "")
            shown.update(tid, completed=done, total=total, description=label)
            if tty or FLAGS.quiet or not total or time.monotonic() - start < PLAIN_AFTER:
                return
            pct = 100 * done // total // 10 * 10
            if pct > shown_pct.get(label, -1):
                shown_pct[label] = pct
                typer.echo(f"{label} {done}/{total} ({pct}%)", err=True)

        yield update


def blobs(files: Sequence[Source], update: Update) -> Iterator[Blob]:
    """Each file or URL read (lazily, as it is needed), named by its path or URL, or why it can't
    be read; ``update`` counts them."""
    for i, f in enumerate(files):
        update(i, len(files))
        try:
            yield Blob(str(f), read_source(f), source=source_name(f))
        except ValueError as e:
            yield Blob(str(f), None, str(e))


# ---- adding images ----------------------------------------------------------------
@dataclass
class Outcome:
    """What adding images did, counted: ``added`` (new and moved), how many were ``moved`` from which
    classes, ``skipped`` (there already: an example of the same class, or for the inbox already
    labelled or waiting) and ``failed`` (unreadable)."""

    added: int
    moved: Counter[str]
    skipped: int
    failed: int


def report_failed(result: Added) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split the images an add left out into those already labelled (skipped) and those that
    failed, printing each failure; returns (skipped, failed)."""
    skipped = [(f, why) for f, why in result.errors if why.startswith(ALREADY_LABELLED)]
    failed = [(f, why) for f, why in result.errors if not why.startswith(ALREADY_LABELLED)]
    for f, why in failed:
        typer.secho(f"skip {f}: {why}", fg="yellow", err=True)
    return skipped, failed


def outcome(result: Added) -> Outcome:
    """Count what adding examples did (see ``Outcome``), printing each image that failed."""
    skipped, failed = report_failed(result)
    unchanged = {id(e) for e in result.unchanged}
    return Outcome(sum(id(e) not in unchanged for e in result.added), Counter(was for _, was in result.moved),
                   len(skipped) + len(result.unchanged), len(failed))


def moved_text(moved: Counter[str]) -> str:
    """``" (3 moved from Basset_Hound, 1 from Beagle)"``, or ``""`` if none moved."""
    if not moved:
        return ""
    if len(moved) == 1:
        (c, n), = moved.items()
        return f" ({n} moved from {c})"
    return f" ({moved.total()} moved: " + ", ".join(f"{n} from {c}" for c, n in sorted(moved.items())) + ")"


def finish_adding(o: Outcome, strict: bool, nothing: str, verb: str = "added") -> None:
    """Exit as adding images should: 1 if any failed, or (with ``strict``) any were skipped; if
    nothing was added, 0 when all were there already (saying ``nothing``) and 1 otherwise."""
    if o.added == 0 and o.failed == 0:
        say(nothing)
    if o.failed:
        fail(f"No images were {verb}" if o.added == 0 else f"Not {verb}: {o.failed} of "
             f"{o.added + o.skipped + o.failed} images (see above)")
    if strict and o.skipped:
        fail(f"Skipped {o.skipped} images that were there already (--strict)")


def add_files(task: Task, label: str, files: Sequence[Source]) -> Outcome:
    """Add ``files`` (paths or URLs) as examples of ``label``, skipping unreadable images and
    printing each; returns what was done."""
    FLAGS.interrupt_note = f"images stored before it are kept (`fewshot undo {task.name}` takes them back)"
    with progress(f"Adding to {label}") as update:
        result = services.add_examples(task, label, blobs(files, update))
    return outcome(result)


# ---- output -----------------------------------------------------------------------
def echo_json(obj: Any) -> None:
    """Print ``obj`` as indented JSON."""
    typer.echo(json.dumps(obj, indent=2))


def one_format(**flags: bool) -> str:
    """The output format chosen among flags such as ``json=as_json, csv=as_csv`` (``"text"`` if
    none); giving more than one is a usage error."""
    chosen = [name for name, on in flags.items() if on]
    if len(chosen) > 1:
        names = [f"--{n}" for n in flags]
        usage(f"Give at most one of {', '.join(names[:-1])} and {names[-1]}")
    return chosen[0] if chosen else "text"


def on_off(v: bool | None) -> str:
    """A setting as shown to people: on, off or none."""
    return "none" if v is None else "on" if v else "off"


PLAIN_NUMBER = re.compile(r"-?\d+(\.\d+)?(e[-+]?\d+)?", re.IGNORECASE)  # read as that number


def csv_cell(v: Any) -> Any:
    """One CSV cell, by the rule the web UI's CSV export follows: booleans as ``true``/``false``,
    as in JSON, and text a spreadsheet would read as a formula (starting with = + - @, tab or
    carriage return, as a file or class name might) with a leading ' so it stays text; a plain
    negative number is left as it is, so it stays a number."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, str) and v[:1] in {"=", "+", "-", "@", "\t", "\r"} and not PLAIN_NUMBER.fullmatch(v):
        return "'" + v
    return v


def write_csv(header: list[str], rows: list[list[Any]], with_header: bool = True) -> None:
    """Print ``rows`` as CSV, after ``header`` unless ``with_header`` is False; see ``csv_cell``
    for how cells are written."""
    w = csv.writer(sys.stdout, lineterminator="\n")
    if with_header:
        w.writerow(header)
    w.writerows([csv_cell(v) for v in row] for row in rows)
