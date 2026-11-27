"""`fewshot example ...` and `fewshot inbox ...`: a task's labelled examples and its inbox."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from . import services
from .cli_common import (
    CLASS_ARG,
    FLAGS,
    ID_LEN,
    IDS_ARG,
    IMAGES_ARG,
    INBOX_IDS_ARG,
    JSON_OPT,
    NEW_CLASS_OPT,
    ROOT_OPT,
    STRICT_OPT,
    TASK_ARG,
    InboxSort,
    Outcome,
    blobs,
    check_new_class,
    command_group,
    echo_json,
    error_text,
    examples,
    expand,
    fail,
    fail_on,
    finish_adding,
    get_task,
    make_pipeline,
    progress,
    report_failed,
    say,
    usage,
    workspace,
)
from .store import Task

ex_app = command_group("List, move, tag and delete a task's examples.\n\nIds may be shortened to a unique "
                       "prefix of 4 or more characters, such as the 12 that lists show.")
inbox_app = command_group("Unlabelled images waiting to be labelled, with the model's suggestions.\n\nIds may be "
                          "shortened to a unique prefix of 4 or more characters, such as the 12 that lists show.")


# ---- examples -------------------------------------------------------------
def _ids(t: Task, ids: list[str], inbox: bool = False) -> list[str]:
    try:
        return t.resolve_ids(ids, inbox=inbox)
    except KeyError as e:
        fail(error_text(e))


@ex_app.command("ls", epilog=examples("fewshot example ls pets --label Beagle",
                                       "fewshot example ls pets --tag blurry --json"))
def example_ls(
    name: str = TASK_ARG,
    label: str | None = typer.Option(None, help="Only this class"),
    tag: str | None = typer.Option(None, help="Only examples with this tag"),
    as_json: bool = JSON_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """List examples: id, class, source file, tags and note (alias: list)."""
    found = get_task(root, name).find_examples(label, tag)
    if as_json:
        echo_json([e.model_dump() for e in found])
        return
    width = max((len(e.label) for e in found), default=0)
    for e in found:
        tags = f"  [{', '.join(e.tags)}]" if e.tags else ""
        note = f"  # {e.note}" if e.note else ""
        typer.echo(f"{e.id[:ID_LEN]}  {e.label:{width}}  {e.source}{tags}{note}")


ex_app.command("list", hidden=True)(example_ls)


@ex_app.command("rm", epilog=examples("fewshot example rm pets 3f2a9c01 7be41d22"))
def example_rm(name: str = TASK_ARG, ids: list[str] = IDS_ARG, root: Path | None = ROOT_OPT) -> None:
    """Delete examples (alias: delete; undo brings them back)."""
    t = get_task(root, name)
    try:
        n = t.delete_examples(_ids(t, ids))
    except KeyError as e:  # the task was deleted meanwhile
        fail(error_text(e))
    say(f"Deleted {n} examples (undo with `fewshot undo {name}`)")


ex_app.command("delete", hidden=True)(example_rm)


@ex_app.command("relabel", epilog=examples("fewshot example relabel pets Beagle 3f2a9c01 7be41d22"))
def example_relabel(name: str = TASK_ARG, label: str = CLASS_ARG, ids: list[str] = IDS_ARG,
                    new_class: bool = NEW_CLASS_OPT, root: Path | None = ROOT_OPT) -> None:
    """Move examples to another class.

    CLASS is created if needed, unless its name is close to an existing class's (a likely typo):
    then --new-class is needed.
    """
    t = get_task(root, name)
    new = check_new_class(t, label.strip(), new_class)
    try:
        n = t.relabel_many(_ids(t, ids), label)
    except (KeyError, ValueError) as e:  # KeyError: the task was deleted meanwhile
        fail(error_text(e))
    if new:
        say(f"Created class {label.strip()} in {name}")
    say(f"Moved {n} examples to {label.strip()}")


@ex_app.command("tag", epilog=examples("fewshot example tag pets 3f2a9c01 --add blurry --remove todo"))
def example_tag(
    name: str = TASK_ARG,
    ids: list[str] = IDS_ARG,
    add_tag: list[str] = typer.Option([], "--add", help="Tag to add. Repeatable."),
    remove_tag: list[str] = typer.Option([], "--remove", help="Tag to remove. Repeatable."),
    root: Path | None = ROOT_OPT,
) -> None:
    """Add or remove tags on examples."""
    if not add_tag and not remove_tag:
        usage("Give --add and/or --remove")
    t = get_task(root, name)
    try:
        n = t.annotate(_ids(t, ids), add_tags=add_tag, remove_tags=remove_tag)
    except (KeyError, ValueError) as e:  # KeyError: the task was deleted meanwhile
        fail(error_text(e))
    say(f"Tagged {n} examples")


@ex_app.command("note", epilog=examples("fewshot example note pets 3f2a9c01 'cropped from a group photo'",
                                        "fewshot example note pets 3f2a9c01 ''   # clears it"))
def example_note(
    name: str = TASK_ARG,
    example_id: str = typer.Argument(..., metavar="ID", help="Example id, or a unique prefix of 4+ characters",
                                     show_default=False),
    note: str = typer.Argument(..., metavar="NOTE", help="The note; '' clears it", show_default=False),
    root: Path | None = ROOT_OPT,
) -> None:
    """Set an example's note ("" clears it)."""
    t = get_task(root, name)
    try:
        t.annotate(_ids(t, [example_id]), note=note)
    except (KeyError, ValueError) as e:  # KeyError: the task was deleted meanwhile
        fail(error_text(e))
    say("Note saved" if note.strip() else "Note cleared")


# ---- inbox ----------------------------------------------------------------
@inbox_app.command("add", epilog=examples("fewshot inbox add pets ./unlabelled/"))
def inbox_add(name: str = TASK_ARG, paths: list[str] = IMAGES_ARG, strict: bool = STRICT_OPT,
              root: Path | None = ROOT_OPT) -> None:
    """Put unlabelled images in the inbox, for labelling later.

    Images that are labelled already or waiting in the inbox already are skipped. Exits 1 if any
    image couldn't be read or added (it is reported, and the others are added), or with --strict
    if any was skipped.
    """
    t = get_task(root, name)
    files = expand(paths)
    waiting = {x.id for x in t.config.inbox}
    FLAGS.interrupt_note = f"images stored before it are kept (`fewshot undo {name}` takes them back)"
    with progress("Adding to the inbox") as update:
        result = services.add_to_inbox(t, blobs(files, update))
    skipped, failed = report_failed(result)
    new = sum(x.id not in waiting for x in result.added)
    o = Outcome(new, Counter(), len(skipped) + len(result.added) - new, len(failed))
    if new:
        there = f", skipped {o.skipped} labelled or waiting already" if o.skipped else ""
        say(f"Added {new} images to the inbox of {name}{there}; {len(t.config.inbox)} waiting")
    finish_adding(o, strict, f"Nothing new: all {o.skipped} images are labelled or in the inbox already")


@inbox_app.command("ls", epilog=examples("fewshot inbox ls pets", "fewshot inbox ls pets --sort confident --json"))
def inbox_ls(
    name: str = TASK_ARG,
    sort: InboxSort = typer.Option("uncertain", help="uncertain: most useful to label first; confident; novel: "
                                                     "unlike any example; added: in the order added"),
    limit: int = typer.Option(50, min=1, help="Show at most this many"),
    as_json: bool = JSON_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """List inbox images with the model's suggestions, most useful first (alias: list)."""
    t = get_task(root, name)
    try:
        items = make_pipeline(workspace(root)).inbox_suggestions(t, sort)[:limit]
    except ValueError as e:  # the model can't be loaded, say
        fail_on(e, name)
    if as_json:
        echo_json(items)
        return
    for i in items:
        p = i.get("prediction")
        hint = "" if p is None else (f"unknown (closest: {p['top_label']})" if p["rejected"] else p["label"])
        stats = "" if p is None else f"conf={p['confidence']:.2f} margin={i['margin']:.2f} sim={p['similarity']:.3f}"
        typer.echo(f"{i['id'][:ID_LEN]}  {hint:24} {stats}  {i['source']}")
    say(f"{len(t.config.inbox)} images in the inbox")


inbox_app.command("list", hidden=True)(inbox_ls)


@inbox_app.command("accept", epilog=examples("fewshot inbox accept pets --min-confidence 0.9",
                                             "fewshot inbox accept pets 3f2a9c01 7be41d22 --label Beagle"))
def inbox_accept(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    name: str = TASK_ARG,
    ids: list[str] = typer.Argument(None, metavar="ID", help="Inbox ids to label as --label (or to choose from)",
                                    show_default=False),
    label: str | None = typer.Option(None, help="Class for the given ids"),
    min_confidence: float | None = typer.Option(
        None, min=0, max=1,
        help="Instead, accept every suggestion (of the given ids, if any) at least this "
             "confident (not rejected)"),
    new_class: bool = NEW_CLASS_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """Label inbox images as examples.

    Either the given ids as --label, or every suggestion (of the given ids, if any) at least
    --min-confidence confident. A new --label close to an existing class's name needs --new-class.
    """
    t = get_task(root, name)
    if (min_confidence is None) == (label is None) or (label is not None and not ids):
        usage("Give ids with --label, or --min-confidence (with or without ids)")
    if label is not None:
        new = check_new_class(t, label.strip(), new_class)
        labels = dict.fromkeys(_ids(t, ids or [], inbox=True), label)
    else:
        new = False
        wanted = _ids(t, ids, inbox=True) if ids else None
        try:
            labels = services.confident_suggestions(make_pipeline(workspace(root)), t, min_confidence or 0.0, wanted)
        except ValueError as e:  # the model can't be loaded, say
            fail_on(e, name)
    if not labels:
        say("Nothing to accept")
        return
    try:
        n = t.accept(labels)
    except (KeyError, ValueError) as e:
        fail(error_text(e))
    if new:
        say(f"Created class {(label or '').strip()} in {name}")
    say(f"Labelled {n} inbox images; {len(t.config.inbox)} left")


@inbox_app.command("dismiss", epilog=examples("fewshot inbox dismiss pets 3f2a9c01"))
def inbox_dismiss(name: str = TASK_ARG, ids: list[str] = INBOX_IDS_ARG, root: Path | None = ROOT_OPT) -> None:
    """Remove images from the inbox without labelling them (undo brings them back)."""
    t = get_task(root, name)
    try:
        n = t.dismiss(_ids(t, ids, inbox=True))
    except KeyError as e:  # the task was deleted meanwhile
        fail(error_text(e))
    say(f"Dismissed {n} images; {len(t.config.inbox)} left")
