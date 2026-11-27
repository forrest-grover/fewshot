"""`fewshot add`, `fewshot import` and `fewshot class ...`: labelling a task's images and managing its classes."""

from __future__ import annotations

from pathlib import Path

import typer

from . import services
from .cli_common import (
    CLASS_ARG,
    IMAGES_ARG,
    JSON_OPT,
    NEW_CLASS_OPT,
    ROOT_OPT,
    STRICT_OPT,
    TASK_ARG,
    App,
    check_class,
    check_new_class,
    command_group,
    echo_json,
    error_text,
    examples,
    expand,
    fail,
    finish_adding,
    get_task,
    moved_text,
    outcome,
    progress,
    say,
    usage,
)
from .cli_common import FLAGS as _FLAGS
from .cli_common import add_files as _add_files
from .folders import class_folders
from .store import Task
from .text import check_class_name, near_miss

commands = App()  # the top-level commands defined here
cls_app = command_group("Manage a task's classes: list, add, describe, rename, delete, set thresholds.")


@commands.command(rich_help_panel="Labelling", epilog=examples(
    "fewshot add pets Beagle a.jpg b.jpg",
    "fewshot add pets Beagle ./more-beagles/ https://example.com/beagle.jpg",
    "fewshot add pets Pug pug.jpg --new-class   # even though 'Pug' is close to an existing 'pug'"))
def add(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    name: str = TASK_ARG,
    label: str = CLASS_ARG,
    paths: list[str] = IMAGES_ARG,
    new_class: bool = NEW_CLASS_OPT,
    strict: bool = STRICT_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """Add images as labelled examples of CLASS.

    CLASS is created if the task doesn't have it (and said so), unless its name is close to an
    existing class's: then it is taken for a typo and refused, unless --new-class is given. An
    image that is an example of another class already moves to CLASS; one that is an example of
    CLASS already is skipped. Exits 1 if any image couldn't be read or added (it is reported, and
    the others are added), or with --strict if any was skipped.
    """
    t = get_task(root, name)
    try:
        label = check_class_name(label)
    except ValueError as e:
        fail(str(e))
    new = check_new_class(t, label, new_class)
    o = _add_files(t, label, expand(paths))
    if new and o.added:
        say(f"Created class {label} in {name}")
    if o.added:
        skipped = f", skipped {o.skipped} already in {label}" if o.skipped else ""
        say(f"Added {o.added} to {label}{moved_text(o.moved)}{skipped}; {name} now has "
            f"{len(t.config.examples)} examples")
    finish_adding(o, strict, f"Nothing new: all {o.skipped} images are already examples of {label} in {name}")


@commands.command("import", rich_help_panel="Labelling", epilog=examples(
    "fewshot import pets ./photos               # ./photos/<class>/<image>",
    "fewshot import pets ./photos --shots 5     # at most 5 images per class",
    "fewshot import pets ./more --new-class     # even though more/pug/ is close to an existing 'Pug'"))
def import_dir(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    name: str = TASK_ARG,
    folder: Path = typer.Argument(..., metavar="FOLDER", help="A folder of class folders: FOLDER/<class>/<image>",
                                  show_default=False),
    shots: int | None = typer.Option(None, min=1, help="Take at most this many images per class"),
    new_class: bool = typer.Option(False, "--new-class", help="Create a class for every folder, even one whose name "
                                                             "is close to an existing class's (without it, a near "
                                                             "miss such as 'beagle/' for 'Beagle' is refused as a "
                                                             "likely typo)"),
    strict: bool = STRICT_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """Import a folder of class folders (FOLDER/<class>/<image>) into a task.

    Each subfolder's name is the class of the images in it; hidden and system folders such as
    __MACOSX and .ipynb_checkpoints are skipped. Classes the task doesn't have are created (and
    said so), but if a folder's name is close to an existing class's, nothing is imported: it is
    taken for a typo unless --new-class is given. The whole import is one undo step. Images that
    are examples of their class already are skipped, so importing the same folder again adds only
    what is new. Exits 1 if any image couldn't be read or imported, or with --strict if any was
    skipped.
    """
    t = get_task(root, name)
    if folder.is_file() and folder.suffix.lower() == ".zip":
        fail(f"{folder} looks like an export; use `fewshot import-task {folder}`")
    if not folder.is_dir():
        fail(f"Not a directory: {folder}")
    before, classes = len(t.config.examples), t.class_names()
    if not new_class:
        _check_folder_names(t, folder, shots)
    _FLAGS.interrupt_note = f"images stored before it are kept (`fewshot undo {name}` takes them back)"
    with progress("Importing") as update:
        o = outcome(services.import_folder(t, folder, shots, update))
    if o.added == 0 and o.failed == 0 and o.skipped == 0:
        fail(f"No images were imported; expected {folder}/<class>/<image>")
    if created := [c for c in t.class_names() if c not in classes]:
        shown = ", ".join(created[:10]) + (f" and {len(created) - 10} more" if len(created) > 10 else "")
        say(f"Created {len(created)} {'class' if len(created) == 1 else 'classes'} in {name}: {shown}")
    if o.added:
        skipped = f", skipped {o.skipped} already in the task" if o.skipped else ""
        say(f"Imported {o.added} images into {name}{moved_text(o.moved)}{skipped}; {name} now has "
            f"{len(t.config.examples)} examples in {len(t.config.classes)} classes")
        if before == 0:
            say(f"Next: `fewshot eval {name}` to measure accuracy, or `fewshot predict {name} IMAGE...`")
    finish_adding(o, strict, f"Nothing new: all {o.skipped} images are already in {name}", "imported")


def _check_folder_names(t: Task, folder: Path, shots: int | None) -> None:
    """Refuse an import whose new classes include names close to an existing class's (likely typos)."""
    names = t.class_names()
    close = [(f, c) for f in class_folders(folder, shots) if f not in names and (c := near_miss(f, names))]
    if close:
        pairs = ", ".join(f"{f!r} (close to {c!r})" for f, c in close)
        fail(f"Nothing imported: {folder} has class folders whose names are close to existing classes of {t.name}: "
             f"{pairs}. Rename the folders to match, or pass --new-class to create them as new classes")


# ---- classes ---------------------------------------------------------------------
@cls_app.command("ls", epilog=examples("fewshot class ls pets", "fewshot class ls pets --json"))
def class_ls(name: str = TASK_ARG, as_json: bool = JSON_OPT, root: Path | None = ROOT_OPT) -> None:
    """List a task's classes: name, examples, threshold and description (alias: list)."""
    t = get_task(root, name)
    counts = t.summary()["counts"]
    rows = [{"name": c.name, "description": c.description, "count": counts.get(c.name, 0),
             "reject_below": c.reject_below} for c in t.config.classes]  # as `show --json` and the API's ClassOut
    if as_json:
        echo_json(rows)
        return
    width = max((len(r["name"]) for r in rows), default=0)
    for r in rows:
        thr = "-" if r["reject_below"] is None else f"{r['reject_below']:g}"
        desc = " | ".join(line for line in r["description"].splitlines() if line.strip())
        typer.echo(f"{r['name']:{width}}  {r['count']:5}  {thr:>6}  {desc}".rstrip())


cls_app.command("list", hidden=True)(class_ls)


@cls_app.command("add", epilog=examples("fewshot class add pets Beagle",
                                        "fewshot class add pets Beagle --desc 'a photo of a beagle, a type of dog'"))
def class_add(
    name: str = TASK_ARG,
    label: str = CLASS_ARG,
    description: str | None = typer.Option(None, "--desc", help="Text prompt, e.g. 'a photo of a cracked weld' "
                                                                "[default: keep an existing class's]"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Add a class, or set an existing class's description.

    A class can have a description and no examples: with a text-capable embedder (CLIP, PE Core,
    SigLIP), it is then predicted zero-shot from its description.
    """
    t = get_task(root, name)
    existed = label.strip() in t.class_names()
    try:
        t.upsert_class(label, description)
    except (KeyError, ValueError) as e:  # KeyError: the task was deleted meanwhile
        fail(error_text(e))
    if not existed:
        say(f"Added class {label.strip()} to {name}")
    elif description is not None:
        say(f"Set the description of {label.strip()}")


@cls_app.command("describe", epilog=examples("fewshot class describe pets",
                                             "fewshot class describe pets 'a photo of a {}, a type of pet'"))
def class_describe(
    name: str = TASK_ARG,
    template: str = typer.Argument(
        "a photo of a {}", metavar="TEMPLATE",
        help="'{}' is replaced by the class name (underscores become spaces)"),
    overwrite: bool = typer.Option(False, help="Also replace descriptions that are already set"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Fill class descriptions from a template.

    Naming the kind of thing ("a photo of a {}, a type of pet") helps most with few examples.
    """
    try:
        n = get_task(root, name).describe_classes(template, overwrite)
    except (KeyError, ValueError) as e:
        fail(error_text(e))
    say(f"Set {n} descriptions" + ("" if n or overwrite else " (all were set; --overwrite replaces them)"))


@cls_app.command("threshold", context_settings={"ignore_unknown_options": True},  # takes "-0.5"
                 epilog=examples("fewshot class threshold pets Beagle 0.72", "fewshot class threshold pets Beagle "
                                 "--clear"))
def class_threshold(
    name: str = TASK_ARG,
    label: str = CLASS_ARG,
    value: float | None = typer.Argument(None, min=-1, max=1, metavar="VALUE",
                                         help="Similarity (-1 to 1) under which predictions of this class become "
                                              "'unknown'"),
    clear: bool = typer.Option(False, "--clear", help="Use the task's --reject-below for this class again"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Set a class's own reject threshold, overriding the task's.

    `fewshot eval` suggests one per class, and --apply-class-thresholds sets them all.
    """
    if (value is None) == (not clear):
        usage("Give either a threshold or --clear")
    t = get_task(root, name)
    check_class(t, label)
    try:
        t.set_class_thresholds({label: value})
    except KeyError as e:
        fail(error_text(e))
    say(f"Cleared the threshold of {label}" if value is None else f"Set the threshold of {label} to {value:g}")


@cls_app.command("rename", epilog=examples("fewshot class rename pets Beagle beagle"))
def class_rename(
    name: str = TASK_ARG,
    label: str = CLASS_ARG,
    new_label: str = typer.Argument(..., metavar="NEW", help="The class's new name", show_default=False),
    root: Path | None = ROOT_OPT,
) -> None:
    """Rename a class, relabelling its examples (one undo step)."""
    t = get_task(root, name)
    check_class(t, label)
    n = sum(e.label == label for e in t.config.examples)
    try:
        t.rename_class(label, new_label)
    except (KeyError, ValueError) as e:
        fail(error_text(e))
    say(f"Renamed class {label} to {new_label.strip()} ({n} examples)")


@cls_app.command("rm", epilog=examples("fewshot class rm pets Beagle"))
def class_rm(name: str = TASK_ARG, label: str = CLASS_ARG, root: Path | None = ROOT_OPT) -> None:
    """Delete a class and its examples (alias: delete; undo brings them back)."""
    t = get_task(root, name)
    check_class(t, label)
    try:
        n = t.delete_class(label)
    except KeyError as e:
        fail(error_text(e))
    say(f"Removed class {label} and {n} examples (undo with `fewshot undo {name}`)")


cls_app.command("delete", hidden=True)(class_rm)
