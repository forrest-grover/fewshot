"""`fewshot example ...` and `fewshot inbox ...`: a task's labelled examples and its inbox."""

from __future__ import annotations

from pathlib import Path

import typer

from .cli_common import (
    JSON_OPT,
    ROOT_OPT,
    Source,
    echo_json,
    error_text,
    expand,
    fail,
    get_task,
    make_pipeline,
    progress,
    read_all,
    report_skipped,
    source_name,
    workspace,
)
from .store import Task

ex_app = typer.Typer(help="List, move, tag and delete a task's examples (ids may be shortened to a unique "
                          "prefix of 4+ characters).", no_args_is_help=True)
inbox_app = typer.Typer(help="Unlabelled images waiting to be labelled, with the model's suggestions (ids may be "
                             "shortened to a unique prefix of 4+ characters).", no_args_is_help=True)


# ---- examples -------------------------------------------------------------
def _ids(t: Task, ids: list[str]) -> list[str]:
    try:
        return t.resolve_ids(ids)
    except KeyError as e:
        fail(error_text(e))


@ex_app.command("ls")
def example_ls(
    name: str,
    label: str | None = typer.Option(None, help="Only this class"),
    tag: str | None = typer.Option(None, help="Only examples with this tag"),
    as_json: bool = JSON_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """List examples: id, class, tags, source file and note."""
    found = get_task(root, name).find_examples(label, tag)
    if as_json:
        echo_json([e.model_dump() for e in found])
        return
    for e in found:
        tags = f"  [{', '.join(e.tags)}]" if e.tags else ""
        note = f"  # {e.note}" if e.note else ""
        typer.echo(f"{e.id[:12]}  {e.label:20} {e.source}{tags}{note}")


@ex_app.command("rm")
def example_rm(name: str, ids: list[str], root: Path | None = ROOT_OPT) -> None:
    """Delete examples (undo brings them back)."""
    t = get_task(root, name)
    n = t.delete_examples(_ids(t, ids))
    typer.echo(f"Deleted {n} examples")


@ex_app.command("relabel")
def example_relabel(name: str, label: str, ids: list[str], root: Path | None = ROOT_OPT) -> None:
    """Move examples to class LABEL (created if needed)."""
    t = get_task(root, name)
    try:
        n = t.relabel_many(_ids(t, ids), label)
    except ValueError as e:
        fail(str(e))
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
        fail("Give --add and/or --remove")
    t = get_task(root, name)
    try:
        n = t.annotate(_ids(t, ids), add_tags=add_tag, remove_tags=remove_tag)
    except ValueError as e:
        fail(str(e))
    typer.echo(f"Tagged {n} examples")


@ex_app.command("note")
def example_note(name: str, example_id: str, note: str, root: Path | None = ROOT_OPT) -> None:
    """Set an example's note ("" clears it)."""
    t = get_task(root, name)
    try:
        t.annotate(_ids(t, [example_id]), note=note)
    except ValueError as e:
        fail(str(e))
    typer.echo("Note saved")


# ---- inbox ----------------------------------------------------------------
def _inbox_ids(t: Task, prefixes: list[str]) -> list[str]:
    ids = [x.id for x in t.config.inbox]
    out = []
    for p in prefixes:
        hits = [i for i in ids if i.startswith(p)] if len(p) >= 4 else []
        if len(hits) != 1:
            fail(f"No inbox image {p!r}" if not hits else f"Inbox id {p!r} is ambiguous")
        out.append(hits[0])
    return out


@inbox_app.command("add")
def inbox_add(name: str, paths: list[str], root: Path | None = ROOT_OPT) -> None:
    """Put unlabelled images (files, directories or http(s) URLs) in the inbox; exits 1 if none could be."""
    t = get_task(root, name)
    files = expand(paths)
    read: list[Source] = []
    with progress("Adding to the inbox") as update:
        results = t.add_to_inbox((data, source_name(f)) for f, data in read_all(files, update, read))
    n = report_skipped(read, results)
    typer.echo(f"{name}: {len(t.config.inbox)} images in the inbox")
    if n == 0:
        fail("Nothing was added to the inbox")


@inbox_app.command("ls")
def inbox_ls(
    name: str,
    sort: str = typer.Option("uncertain", help="uncertain (most useful to label first) | confident | novel | added"),
    limit: int = typer.Option(50, min=1, help="Show at most this many"),
    as_json: bool = JSON_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """List inbox images with the model's suggestion, in the order to label them."""
    t = get_task(root, name)
    try:
        items = make_pipeline(workspace(root)).inbox_suggestions(t, sort)[:limit]
    except ValueError as e:
        fail(str(e))
    if as_json:
        echo_json(items)
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
    ids: list[str] = typer.Argument(None, help="Inbox ids to label as --label (or to choose from)"),
    label: str | None = typer.Option(None, help="Class for the given ids"),
    min_confidence: float | None = typer.Option(
        None, min=0, max=1, help="Instead, accept every suggestion (of the given ids, if any) at least this "
                                 "confident (not rejected)"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Label inbox images as examples: the given ids as --label, or every suggestion (of the given ids,
    if any) above --min-confidence."""
    t = get_task(root, name)
    if (min_confidence is None) == (label is None) or (label is not None and not ids):
        fail("Give ids with --label, or --min-confidence (with or without ids)")
    if label is not None:
        labels = dict.fromkeys(_inbox_ids(t, ids or []), label)
    else:
        wanted = set(_inbox_ids(t, ids)) if ids else None
        items = make_pipeline(workspace(root)).inbox_suggestions(t, "confident")
        labels = {i["id"]: i["prediction"]["label"] for i in items if i.get("prediction")
                  and (wanted is None or i["id"] in wanted)
                  and not i["prediction"]["rejected"] and i["prediction"]["confidence"] >= (min_confidence or 0)}
    if not labels:
        typer.echo("Nothing to accept")
        return
    try:
        n = t.accept(labels)
    except (KeyError, ValueError) as e:
        fail(error_text(e))
    typer.echo(f"Labelled {n} inbox images; {len(t.config.inbox)} left")


@inbox_app.command("dismiss")
def inbox_dismiss(name: str, ids: list[str], root: Path | None = ROOT_OPT) -> None:
    """Remove images from the inbox without labelling them (undo brings them back)."""
    t = get_task(root, name)
    n = t.dismiss(_inbox_ids(t, ids))
    typer.echo(f"Dismissed {n} images; {len(t.config.inbox)} left")
