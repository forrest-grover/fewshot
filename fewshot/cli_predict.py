"""`fewshot predict` and `fewshot watch`: classifying images with a task."""

from __future__ import annotations

import itertools
import json
import shutil
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from . import cli_common, services
from .classifiers import NotReady
from .cli_common import FLAGS as _FLAGS
from .cli_common import (
    ID_LEN,
    MAX_DOWNLOAD,
    ROOT_OPT,
    TASK_ARG,
    App,
    Source,
    examples,
    fail,
    fail_on,
    get_task,
    images_in,
    make_pipeline,
    note,
    one_format,
    progress,
    say,
    usage,
    workspace,
    write_csv,
)
from .cli_common import echo_json as _echo_json
from .cli_common import error_text as _error_text
from .cli_common import expand as _expand
from .fetch import is_url
from .store import Workspace

if TYPE_CHECKING:
    from .pipeline import Pipeline, Prediction

commands = App()
PREDICT_CHUNK = 256  # images classified (and printed) at a time


@commands.command(rich_help_panel="Predicting", help=f"""Classify images with a task.

    Images are classified {PREDICT_CHUNK} at a time: text, CSV and JSON lines are printed as each
    batch finishes, and --from-file is read as it arrives, so a long list can be piped through.
    An image that can't be read (a missing file, a broken image, a URL that fails) doesn't stop
    the others: it gets a row with "error" set (in text, a line on stderr), and the command exits
    1 at the end.""", epilog=examples(
    "fewshot predict pets new/ https://example.com/cat.jpg",
    "fewshot predict pets new/ --json                 # also --jsonl, --csv",
    "find photos -name '*.jpg' | fewshot predict pets --from-file - --jsonl",
    "fewshot predict pets new/ --revision 41          # with an earlier version of the task"))
def predict(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals  # one per CLI option
    name: str = TASK_ARG,
    paths: list[str] = typer.Argument(None, metavar="IMAGE", help="Image files, folders or http(s) URLs",
                                      show_default=False),
    from_file: Path | None = typer.Option(None, "--from-file", help="Read more paths or URLs, one per line, from "
                                                                     "this file ('-' for stdin)"),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON list, once every image is classified"),
    as_jsonl: bool = typer.Option(False, "--jsonl", help="Print one JSON object per image as each batch finishes"),
    as_csv: bool = typer.Option(False, "--csv", help="Print CSV: one row per image"),
    top: int = typer.Option(3, min=1, help="Scores to show per image (and to include, with --json/--jsonl)"),
    evidence: int = typer.Option(0, min=0, max=20, help="Also show this many of the most similar labelled examples"),
    revision: int | None = typer.Option(None, help="Use this earlier version of the task (one kept for undo)"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Classify images; the help text is in the decorator, to name PREDICT_CHUNK's value."""
    t = get_task(root, name)
    fmt = one_format(json=as_json, jsonl=as_jsonl, csv=as_csv)
    if not paths and from_file is None:
        usage("Give images to classify, or --from-file")
    named = _expand(paths) if paths else []
    try:
        cfg = None if revision is None else t.version(revision)
    except KeyError as e:
        fail(_error_text(e))
    _FLAGS.interrupt_note = "no task was changed"
    pipe = make_pipeline(workspace(root))
    files: list[Source] = []  # every file so far (all of them for --json, else the last batch)
    collected: list[Prediction] = []
    done = failed = 0
    sources = itertools.chain(named, _listed(from_file) if from_file is not None else [])
    first = list(itertools.islice(sources, 1))  # waits for the first line of a list, before any progress bar
    sources = itertools.chain(first, sources)
    with progress("Classifying") as update:
        while part := list(itertools.islice(sources, PREDICT_CHUNK)):
            try:
                preds = services.predict(pipe, t, _loaded(part), top_k=top if fmt in {"json", "jsonl"} else None,
                                         evidence=evidence, cfg=cfg)
            except ValueError as e:
                fail_on(e, name)
            failed += sum(p.error is not None for p in preds)
            if fmt == "json":
                files += part
                collected += preds
            else:
                _output_predictions(part, preds, fmt, top, header=done == 0)
            done += len(part)
            update(done, len(named) if from_file is None else None)
    if not done:
        fail("No images to classify")
    if fmt == "json":
        _output_predictions(files, collected, fmt, top)
    if failed:
        fail(f"{failed} of {done} images could not be read")


def _listed(from_file: Path) -> Iterator[Source]:
    """The images a --from-file list names, read line by line as it arrives (``-``: stdin): URLs,
    files (a missing one too, so it gets an error row) and the images in folders."""
    if str(from_file) == "-":
        if cli_common.stdin_is_terminal():
            say("Reading image paths from stdin, one per line; Ctrl-D to finish")
        yield from _named_in(sys.stdin)
        return
    try:
        f = from_file.open(encoding="utf-8", errors="surrogateescape")  # names that aren't UTF-8 as on disk
    except OSError as e:
        fail(f"Can't read {from_file}: {e.strerror or e}")
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
            yield from images_in(Path(line))
        else:
            yield Path(line)


def _output_predictions(files: list[Source], preds: list[Prediction], fmt: str, top: int, header: bool = True) -> None:
    """Print predictions as JSON, JSON lines, CSV (with a ``header`` row) or text."""
    if fmt == "json":
        _echo_json([{"file": str(f), **p.to_dict()} for f, p in zip(files, preds, strict=True)])
    elif fmt == "jsonl":
        for f, p in zip(files, preds, strict=True):
            typer.echo(json.dumps({"file": str(f), **p.to_dict()}))
    elif fmt == "csv":
        write_csv(["file", "label", "labels", "top_label", "confidence", "similarity", "rejected", "error", "scores"],
                  [[str(f), p.label or "", ";".join(p.labels), p.top_label, f"{p.confidence:.4f}",
                    f"{p.similarity:.4f}", p.rejected, p.error or "",
                    ";".join(f"{c}={v:.4f}" for c, v in list(p.scores.items())[:top])]
                   for f, p in zip(files, preds, strict=True)], header)
    else:
        _print_predictions(files, preds, top)


def _loaded(files: list[Source]) -> list[services.Blob]:
    """Files as they are (the pipeline reads them a few at a time) and URLs downloaded, several at
    a time; a URL that can't be fetched carries its error."""
    urls = [f for f in files if isinstance(f, str)]
    fetched = iter(services.fetch_urls(urls, MAX_DOWNLOAD) if urls else [])
    return [services.Blob(str(f), f) if isinstance(f, Path) else next(fetched) for f in files]


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
            typer.echo(f"    like {e['id'][:ID_LEN]} ({e['label']}, sim={e['similarity']:.3f})")


# ---- watch -------------------------------------------------------------------------
WATCH_SETTLE = 30.0  # seconds after its last change that an unreadable file is given up on
WATCH_MAX_BACKOFF = 300.0  # longest wait before trying again to move a file that couldn't be moved


@commands.command(rich_help_panel="Predicting", help=f"""Sort images into class folders as they arrive in FOLDER.

    Each image is classified and moved to OUT/<class>/, and logged as a JSON line in
    OUT/predictions.jsonl.

    A file is handled once its size has stayed the same for one interval (so half-copied files
    wait); one that can't be read yet is retried until it is {WATCH_SETTLE:g} seconds old, then goes to
    _error. The task is re-read every time, so labelling more examples improves the sorting as it
    runs. Ctrl-C stops it.""", epilog=examples(
    "fewshot watch pets ./incoming --out ./sorted",
    "fewshot watch pets ./incoming --out ./sorted --copy --once --min-confidence 0.8"))
def watch(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals  # one per CLI option
    name: str = TASK_ARG,
    folder: Path = typer.Argument(..., metavar="FOLDER", help="Folder new images arrive in", show_default=False),
    out: Path = typer.Option(..., help="Where images go: OUT/<class>/, OUT/_unknown/ (rejected or not confident "
                                       "enough) or OUT/_error/ (a class named _unknown, _error or "
                                       "predictions.jsonl goes to OUT/_class_<name>/)"),
    copy: bool = typer.Option(False, "--copy", help="Copy instead of moving (then each file is handled once per run)"),
    min_confidence: float | None = typer.Option(None, min=0, max=1,
                                                help="Below this, an image goes to _unknown"),
    interval: float = typer.Option(2.0, min=0.1, help="Seconds between looks at the folder"),
    once: bool = typer.Option(False, "--once", help="Handle what is there now and stop"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Sort arriving images; the help text is in the decorator, to name WATCH_SETTLE's value."""
    if not folder.is_dir():
        fail(f"Not a directory: {folder}")
    ws = workspace(root)
    get_task(root, name)
    pipe = make_pipeline(ws)
    last: dict[Path, int] = {}  # files seen in the last look, and their sizes
    done: set[Path] = set()  # with --copy, files already copied
    retry = Backoff(interval)
    try:
        while True:
            sizes = _sizes(folder)
            ready = [f for f, n in sizes.items() if f not in done and n > 0 and (once or last.get(f) == n)
                     and not retry.waiting(f)]
            last, done = sizes, done & sizes.keys()  # forget files that went away
            retry.keep(sizes.keys())
            try:
                done |= _sort_files(ws, pipe, name, ready, out, copy, min_confidence,
                                    settle=0 if once else WATCH_SETTLE, retry=retry) if ready else set()
            except NotReady as e:  # the task can't predict yet (no examples): wait for some
                if once:
                    fail_on(e, name)
                note(f"Waiting: {e}")
            except ValueError as e:  # can't ever predict as set up (a bad device, a model that won't load)
                fail_on(e, name)
            if once:
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        say("Stopped")


class Backoff:
    """When to try again to move files whose move failed: after ``interval`` seconds, then twice
    as long after each further failure, up to ``WATCH_MAX_BACKOFF``."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._failed: dict[Path, tuple[int, float]] = {}  # file -> (failures, when to try next)

    def failed(self, f: Path) -> float:
        """Note that moving ``f`` failed again; returns the seconds until it is tried again."""
        n = self._failed.get(f, (0, 0.0))[0] + 1
        wait: float = min(self.interval * 2.0 ** (n - 1), WATCH_MAX_BACKOFF)
        self._failed[f] = (n, time.monotonic() + wait)
        return wait

    def succeeded(self, f: Path) -> None:
        """Forget ``f``'s failures."""
        self._failed.pop(f, None)

    def waiting(self, f: Path) -> bool:
        """Whether ``f`` failed lately and isn't due to be tried again yet."""
        return f in self._failed and time.monotonic() < self._failed[f][1]

    def keep(self, files: Iterable[Path]) -> None:
        """Forget the files not among ``files`` (gone from the folder)."""
        present = set(files)
        self._failed = {f: v for f, v in self._failed.items() if f in present}


def _sizes(folder: Path) -> dict[Path, int]:
    """The images in ``folder`` and their sizes, skipping any gone since they were listed."""
    out = {}
    for f in images_in(folder, recursive=False):
        try:
            out[f] = f.stat().st_size
        except OSError:
            continue
    return out


def _sort_files(ws: Workspace, pipe: Pipeline, name: str, files: list[Path], out: Path,  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
                copy: bool, min_confidence: float | None, settle: float, retry: Backoff) -> set[Path]:
    """Classify ``files`` with the task's current version and move (or copy) each into its folder
    under ``out``, appending a JSON line per file to ``out/predictions.jsonl``. Returns the files
    handled. An unreadable file changed less than ``settle`` seconds ago may still be arriving and
    is left for next time; a file that can't be moved is noted in ``retry``, to wait before trying
    again. Raises ``NotReady`` if the task can't predict yet (no examples), and ``ValueError``
    if it can't predict as set up."""
    try:
        preds = pipe.predict(ws.get(name), files)
    except KeyError:
        fail(f"Task {name!r} is gone (renamed or deleted); stopping")
    handled = set()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.jsonl", "a", encoding="utf-8") as log:
        for f, p in zip(files, preds, strict=True):
            try:
                if p.error and time.time() - f.stat().st_mtime < settle:
                    continue
                dest = _place(f, out / _folder_for(p, min_confidence), copy)
            except OSError as e:  # removed meanwhile, or the destination can't be written
                typer.secho(f"{f.name}: {e}; trying again in {retry.failed(f):g} s", fg="yellow", err=True)
                continue
            retry.succeeded(f)
            handled.add(f)
            log.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "file": str(f), "to": str(dest),
                                  "task": name, **p.to_dict()}) + "\n")
            typer.echo(f"{f.name} -> {dest.parent.name}")
    return handled


WATCH_RESERVED = {"_unknown", "_error", "predictions.jsonl"}  # watch's own names in OUT
CLASS_PREFIX = "_class_"  # put before a class name that would be one of them


def _folder_for(p: Prediction, min_confidence: float | None) -> str:
    """The folder under OUT a prediction sends its image to: its class's, except that a class
    named like one of watch's own (or like such a folder, starting with CLASS_PREFIX) goes to
    CLASS_PREFIX + its name, so no class shares a folder with another or with watch's own."""
    unsure = p.rejected or (min_confidence is not None and p.confidence < min_confidence)
    if p.error:
        return "_error"
    if unsure or not p.label:
        return "_unknown"
    return CLASS_PREFIX + p.label if p.label in WATCH_RESERVED or p.label.startswith(CLASS_PREFIX) else p.label


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
