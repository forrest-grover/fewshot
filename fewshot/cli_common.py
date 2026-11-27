"""Helpers shared by the command-line modules: options, errors, task lookup, inputs, output."""

from __future__ import annotations

import contextlib
import csv
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, get_args

import typer

from . import services
from .fetch import fetch_url, is_url
from .folders import image_files, readable_name
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


def command_group(help_text: str) -> typer.Typer:
    """A group of commands. Help is printed as written: with Rich markup on, Typer would take
    ``[default: ...]`` and any other text in brackets for markup and drop it."""
    return typer.Typer(help=help_text, no_args_is_help=True, rich_markup_mode=None)


# Exit statuses: 1 when the command failed (or failed for some of its inputs), 2 for a usage error
# (options that don't go together, missing arguments), as for the usage errors Typer reports itself.
FAILED, USAGE = 1, 2


def classifier_names(names: list[str]) -> list[str]:
    """Check a repeatable --classifier option's values (Typer can't make a list of choices)."""
    for n in names:
        if n not in get_args(ClassifierName):
            raise typer.BadParameter(f"{n!r} is not one of " + ", ".join(map(repr, get_args(ClassifierName))))
    return names


def fail(msg: str, code: int = FAILED) -> NoReturn:
    """Print ``msg`` as an error and exit with status ``code`` (1: the command failed)."""
    typer.secho(f"Error: {msg}", fg="red", err=True)
    raise typer.Exit(code)


def usage(msg: str) -> NoReturn:
    """Print ``msg`` as an error and exit with status 2: the command was called wrongly."""
    fail(msg, USAGE)


def error_text(e: Exception) -> str:
    """The readable message of an error: for a KeyError its first argument, not its repr."""
    return str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)


def workspace(root: Path | None, create: bool = False) -> Workspace:
    """The workspace at ``root`` (default from FEWSHOT_HOME or ./fewshot_data). Unless ``create``
    (for commands that make tasks), a missing one isn't made: it reads as empty, so a mistyped
    --root doesn't leave a new folder behind."""
    return Workspace(root, create=create)


def get_task(root: Path | None, name: str) -> Task:
    """Task ``name``, or exit with an error if there is none."""
    try:
        return workspace(root).get(name)
    except KeyError as e:
        fail(error_text(e))


def make_pipeline(ws: Workspace) -> Pipeline:
    """A pipeline for ``ws``; imported lazily, since it imports torch and `fewshot --help` should be fast."""
    from .pipeline import Pipeline  # pylint: disable=import-outside-toplevel

    return Pipeline(ws)


Source = Path | str  # an image file, or an http(s) URL


def images_in(folder: Path, recursive: bool = True) -> list[Path]:
    """Image files under ``folder``, sorted; hidden and system files and folders, folders and broken
    links with image names are left out."""
    return image_files(folder, recursive)


def expand(paths: list[str]) -> list[Source]:
    """URLs and files named on the command line plus the images inside named folders; fails if
    there are none."""
    out: list[Source] = []
    for arg in paths:
        if is_url(arg):
            out.append(arg)
            continue
        p = Path(arg)
        if not p.exists():
            fail(f"No such file or directory: {p}")
        out += images_in(p) if p.is_dir() else [p]
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


@contextlib.contextmanager
def progress(what: str) -> Iterator[Callable[[int, int], None]]:
    """A progress bar on stderr while the block runs (only on a terminal); yields ``update(done, total)``."""
    from rich.console import Console  # pylint: disable=import-outside-toplevel  # keeps --help fast
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn  # pylint: disable=import-outside-toplevel

    with Progress(TextColumn(what), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
                  console=Console(stderr=True), transient=True, disable=not sys.stderr.isatty()) as shown:
        tid = shown.add_task(what, total=None)
        yield lambda done, total: shown.update(tid, completed=done, total=total)


def blobs(files: Sequence[Source], update: Callable[[int, int], None]) -> Iterator[Blob]:
    """Each file or URL read (lazily, as it is needed), named by its path or URL, or why it can't
    be read; ``update`` counts them."""
    for i, f in enumerate(files):
        update(i, len(files))
        try:
            yield Blob(str(f), read_source(f), source=source_name(f))
        except ValueError as e:
            yield Blob(str(f), None, str(e))


def report_skipped(result: Added) -> int:
    """Print each image an add left out and why, and how many it moved from other classes;
    returns how many were added (moved ones included)."""
    for f, why in result.errors:
        typer.secho(f"skip {f}: {why}", fg="yellow", err=True)
    if result.moved:
        froms = Counter(was for _, was in result.moved)
        typer.echo(f"{len(result.moved)} of them moved from other classes ("
                   + ", ".join(f"{n} from {c}" for c, n in sorted(froms.items())) + ")", err=True)
    return len(result.added)


def add_files(task: Task, label: str, files: Sequence[Source]) -> tuple[int, int]:
    """Add ``files`` (paths or URLs) as examples of ``label``, skipping unreadable images; returns
    how many were added and how many were left out."""
    with progress(f"Adding to {label}") as update:
        result = services.add_examples(task, label, blobs(files, update))
    return report_skipped(result), len(result.errors)


def echo_json(obj: Any) -> None:
    """Print ``obj`` as indented JSON."""
    typer.echo(json.dumps(obj, indent=2))


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
