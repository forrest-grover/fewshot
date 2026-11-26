"""Helpers shared by the command-line modules: options, errors, task lookup, inputs, output."""

from __future__ import annotations

import contextlib
import csv
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import typer

from .fetch import fetch_url, is_url
from .images import IMAGE_EXTS
from .store import Task, Workspace

if TYPE_CHECKING:
    from .pipeline import Pipeline

MAX_DOWNLOAD = 50 * 1024 * 1024  # largest image fetched from a URL
JSON_OPT = typer.Option(False, "--json", help="Print JSON")

ROOT_OPT = typer.Option(None, "--root", envvar="FEWSHOT_HOME", help="Workspace directory [default: ./fewshot_data]")


def fail(msg: str) -> NoReturn:
    """Print ``msg`` as an error and exit with status 1."""
    typer.secho(f"Error: {msg}", fg="red", err=True)
    raise typer.Exit(1)


def error_text(e: Exception) -> str:
    """The readable message of an error: for a KeyError its first argument, not its repr."""
    return str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)


def workspace(root: Path | None) -> Workspace:
    """The workspace at ``root`` (default from FEWSHOT_HOME or ./fewshot_data)."""
    return Workspace(root)


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
    """Image files under ``folder``, sorted; folders and broken links with image names are left out."""
    found = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(f for f in found if f.suffix.lower() in IMAGE_EXTS and f.is_file())


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
    """The file name of a path or URL."""
    return src.name if isinstance(src, Path) else src.rstrip("/").rsplit("/", 1)[-1]


@contextlib.contextmanager
def progress(what: str) -> Iterator[Callable[[int, int], None]]:
    """A progress bar on stderr while the block runs (only on a terminal); yields ``update(done, total)``."""
    from rich.console import Console  # pylint: disable=import-outside-toplevel  # keeps --help fast
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn  # pylint: disable=import-outside-toplevel

    with Progress(TextColumn(what), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
                  console=Console(stderr=True), transient=True, disable=not sys.stderr.isatty()) as shown:
        tid = shown.add_task(what, total=None)
        yield lambda done, total: shown.update(tid, completed=done, total=total)


def read_all(files: Sequence[Source], update: Callable[[int, int], None],
             read: list[Source]) -> Iterator[tuple[Source, bytes]]:
    """``(file, bytes)`` of each file or URL, skipping (with a message) those that can't be read;
    the ones read are also appended to ``read``."""
    for i, f in enumerate(files):
        update(i, len(files))
        try:
            data = read_source(f)
        except ValueError as e:
            typer.secho(f"skip {f}: {e}", fg="yellow", err=True)
            continue
        read.append(f)
        yield f, data


def report_skipped(read: list[Source], results: Sequence[object]) -> int:
    """Print the files a store call refused (results that are ValueErrors); returns how many weren't."""
    n = 0
    for f, r in zip(read, results, strict=True):
        if isinstance(r, ValueError):
            typer.secho(f"skip {f}: {r}", fg="yellow", err=True)
        else:
            n += 1
    return n


def add_files(task: Task, label: str, files: Sequence[Source]) -> int:
    """Add ``files`` (paths or URLs) as examples of ``label``, skipping unreadable images; returns
    how many were added."""
    read: list[Source] = []  # the files actually read, in step with add_examples' results
    with progress(f"Adding to {label}") as update:
        results = task.add_examples((data, label, source_name(f)) for f, data in read_all(files, update, read))
    return report_skipped(read, results)


def echo_json(obj: Any) -> None:
    """Print ``obj`` as indented JSON."""
    typer.echo(json.dumps(obj, indent=2))


def write_csv(header: list[str], rows: list[list[Any]], with_header: bool = True) -> None:
    """Print ``rows`` as CSV, after ``header`` unless ``with_header`` is False."""
    w = csv.writer(sys.stdout, lineterminator="\n")
    if with_header:
        w.writerow(header)
    w.writerows(rows)
