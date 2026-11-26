"""Undo and redo: stepping through versions, labels, damaged history, pruning, and one undo step
per change."""


import pytest

from fewshot import history
from fewshot.store import Workspace
from tests.conftest import BLUE, GREEN, RED, png, red_green_task, red_task, two_each


# ---- undo ---------------------------------------------------------------
def test_undo_steps_back_through_changes(ws):
    """Each undo restores the version before the last change of any kind; the revision keeps rising."""
    t = red_green_task(ws)
    rev = t.config.revision
    t.update_settings(classifier="knn")
    t.upsert_class("blue", "a blue thing")
    assert t.undo() == rev + 1
    assert t.class_names() == ["red", "green"]
    assert t.config.classifier == "knn"
    assert t.undo() == rev
    assert t.config.classifier == "gda"
    assert t.config.revision == rev + 4  # two changes and two undos
    assert Workspace(ws.root).get("t").config.classifier == "gda"  # saved


def test_undo_brings_back_deleted_images(ws):
    """Deleted examples come back with their image files, which stay until no kept version needs them."""
    t = red_green_task(ws)
    gone = t.config.examples[0]
    t.delete_example(gone.id)
    assert (t.path / gone.file).exists()  # kept for undo
    t.undo()
    assert t.example(gone.id) == gone
    assert t.image_path(gone.id).read_bytes() == png(RED, noise=10, seed=0)
    n = t.delete_class("green")
    assert n == 2
    t.undo()
    assert t.class_names() == ["red", "green"]
    assert len(t.config.examples) == 4


def test_undone_images_go_once_redo_is_impossible(ws):
    """An undone add keeps its image for redo; a new change drops it, unless a kept version
    still uses it."""
    t = red_green_task(ws, n=1)
    new = t.add_example(png(BLUE), "blue")
    t.undo()
    assert (t.path / new.file).exists()  # redo may bring it back
    t.upsert_class("x")
    assert not (t.path / new.file).exists()
    first = t.config.examples[0]
    t.delete_example(first.id)  # an older version has it, this one doesn't
    t.add_example(png(RED, noise=10, seed=0), "red")  # back again
    t.undo()
    t.upsert_class("y")  # clears redo: the file is still named by the kept older version
    assert (t.path / first.file).exists()


def test_redo(ws):
    """Redo makes undone changes again, newest undo first, with images; a new change ends it.
    Each step has a label describing the change."""
    t = red_green_task(ws, n=1)
    new = t.add_example(png(BLUE), "blue")
    t.delete_class("red")
    assert t.undo_label() == "delete class red"
    t.undo()
    assert t.undo_label() == "add 1 example to blue"
    t.undo()
    assert t.undo_label() == "add 1 example to green"
    assert (t.can_redo(), t.redo_label()) == (True, "add 1 example to blue")
    t.redo()
    assert t.example(new.id).label == "blue"
    assert t.image_path(new.id).exists()
    assert t.redo_label() == "delete class red"
    t.redo()
    assert t.class_names() == ["green", "blue"]
    assert not t.can_redo()
    assert t.undo_label() == "delete class red"  # redone changes can be undone again
    with pytest.raises(ValueError, match="Nothing to redo"):
        t.redo()
    t.undo()
    t.upsert_class("z")
    assert not t.can_redo()
    assert t.redo_label() == ""


def test_labels(ws):
    """Every kind of change describes itself for undo."""
    t = red_green_task(ws, n=1)
    ids = [e.id for e in t.config.examples]
    steps = [
        (lambda: t.update_settings(classifier="knn"), "change settings"),
        (lambda: t.upsert_class("blue"), "add class blue"),
        (lambda: t.upsert_class("blue", "a blue thing"), "edit class blue"),
        (lambda: t.describe_classes("a {}", overwrite=True), "describe classes"),
        (lambda: t.rename_class("blue", "navy"), "rename class blue to navy"),
        (lambda: t.relabel_many(ids, "navy"), "move 2 examples to navy"),
        (lambda: t.annotate(ids[:1], add_tags=["x"]), "tag 1 example"),
        (lambda: t.annotate(ids[:1], note="hm"), "edit 1 example"),
        (lambda: t.delete_examples(ids[:1]), "delete 1 example"),
        (lambda: t.add_examples([(png(BLUE, seed=i), f"c{i}", "") for i in range(5)]),
         "add 5 examples to c0, c1, c2, ..."),
    ]
    for step, label in steps:
        step()
        assert t.undo_label() == label
    (t.path / "history").joinpath(sorted((t.path / "history").glob("*.json"))[-1].stem + ".label").write_bytes(b"\xff")
    assert t.undo_label() == ""


def test_broken_redo_version(ws):
    """A damaged redo version is removed with an error, like a damaged undo one."""
    t = red_green_task(ws, n=1)
    t.upsert_class("x")
    t.undo()
    next((t.path / "history" / "redo").glob("*.json")).write_text("{bad")
    with pytest.raises(ValueError, match=r"Can't redo: .* is invalid and was removed"):
        t.redo()
    assert not t.can_redo()


def test_nothing_to_undo_and_broken_history(ws):
    """Undo with no history, a corrupt version, or a missing image is a ValueError that changes nothing."""
    t = ws.create("t", embedder="fake:color")
    with pytest.raises(ValueError, match="Nothing to undo"):
        t.undo()
    assert not t.can_undo()
    e = t.add_example(png(RED), "red")
    t.delete_example(e.id)
    assert t.can_undo()
    (t.path / e.file).unlink()
    with pytest.raises(ValueError, match="1 image files are gone"):
        t.undo()
    snaps = sorted((t.path / "history").glob("*.json"))
    snaps[-1].write_text("{broken")
    with pytest.raises(ValueError, match="is invalid"):
        t.undo()
    assert t.config.examples == []


def test_history_is_pruned_and_old_images_deleted(ws, monkeypatch):
    """Only HISTORY_KEPT versions are kept; an image goes once the last version naming it does,
    but not if a later version drops it again (that one deletes it later)."""
    monkeypatch.setattr(history, "HISTORY_KEPT", 2)
    t = red_green_task(ws, n=1)
    red, green = t.config.examples
    t.delete_example(red.id)
    t.upsert_class("a")
    assert (t.path / red.file).exists()
    t.upsert_class("b")  # the version that had red is pruned now
    assert not (t.path / red.file).exists()
    assert len(list((t.path / "history").glob("*.json"))) == 2

    t.delete_example(green.id)
    t.add_example(png(GREEN, noise=10, seed=0), "green")  # re-added...
    t.delete_example(green.id)  # ...and dropped again
    t.upsert_class("c")  # prunes the first drop; the second still holds the file
    assert (t.path / green.file).exists()
    t.upsert_class("d")
    t.upsert_class("e")
    assert not (t.path / green.file).exists()


def test_pruning_keeps_images_a_redone_version_uses(ws, monkeypatch):
    """Redo records the files it drops, and pruning keeps every file a kept version names: a
    re-added, re-deleted, undone and redone example can still be undone back after pruning."""
    monkeypatch.setattr(history, "HISTORY_KEPT", 3)
    t = red_green_task(ws, n=1)
    red = t.config.examples[0]
    t.delete_example(red.id)
    t.add_example(png(RED, noise=10, seed=0), "red")
    t.delete_example(red.id)
    t.undo()
    t.redo()  # the version kept for undo has red; the redo dropped it
    assert t._gone(t._history()[-1][1]) == {red.file}  # pylint: disable=protected-access
    t.upsert_class("a")
    t.upsert_class("b")  # prunes the first drop of red; the redone version still names it
    assert (t.path / red.file).exists()
    t.undo()
    t.undo()
    t.undo()
    assert t.example(red.id).file == red.file
    t.delete_example(red.id)
    for name in "cdef":
        t.upsert_class(name)
    assert not (t.path / red.file).exists()  # once no kept version names it, it goes


def test_history_survives_files_read_without_validation(ws):
    """A kept version that can't be parsed names no files (it is still listed for undo)."""
    t = red_green_task(ws, n=1)
    t.upsert_class("x")
    (t.path / "history" / "0000000099.json").write_text("not json")
    assert t._files_in(t.path / "history" / "0000000099.json") == set()  # pylint: disable=protected-access


# ---- one undo step per change --------------------------------------------
def test_one_undo_takes_back_a_whole_bulk_add(ws):
    """However many chunks an add_examples call is saved in, one undo removes all of it."""
    t = red_task(ws, n=1)
    t.add_examples([(png(GREEN, noise=10, seed=i), "green", "") for i in range(5)], chunk=2)
    assert len(t.config.examples) == 6
    t.undo()
    assert len(t.config.examples) == 1
    assert len(list((t.path / "images").iterdir())) == 6  # kept for redo...
    t.upsert_class("x")  # ...until another change makes redo impossible
    assert len(list((t.path / "images").iterdir())) == 1


def test_edit_class_renames_and_describes_in_one_undo_step(ws):
    """Renaming a class and describing it is one change: one undo takes back both."""
    t = two_each(ws)
    rev = t.config.revision
    t.edit_class("red", "crimson", "a photo of something crimson")
    assert t.config.revision == rev + 1
    assert (t.config.classes[0].name, t.config.classes[0].description) == ("crimson", "a photo of something crimson")
    assert {e.label for e in t.config.examples} == {"crimson", "blue"}
    t.undo()
    assert (t.config.classes[0].name, t.config.classes[0].description) == ("red", "")
    with pytest.raises(ValueError, match="already exists"):
        t.edit_class("red", "blue")
    with pytest.raises(KeyError):
        t.edit_class("nope", description="x")
    t.edit_class("red", "crimson")
    assert t.undo_label() == "rename class red to crimson"
    with pytest.raises(ValueError, match="already exists"):
        t.rename_class("crimson", "crimson")


def test_edit_examples_relabels_and_tags_in_one_undo_step(ws):
    """Moving and tagging examples is one change: one undo takes back both."""
    t = two_each(ws)
    ids = [e.id for e in t.config.examples if e.label == "red"]
    rev = t.config.revision
    assert t.edit_examples(ids, label="blue", add_tags=["x"], note="n") == 2
    assert t.config.revision == rev + 1
    assert all((e.label, e.tags, e.note) == ("blue", ("x",), "n") for e in t.config.examples if e.id in ids)
    t.undo()
    assert all((e.label, e.tags) == ("red", ()) for e in t.config.examples if e.id in ids)
    t.edit_examples(ids, label="green")
    assert t.undo_label() == "move 2 examples to green"
    with pytest.raises(ValueError, match="contain '/'"):
        t.edit_examples(ids, label="a/b")


# ---- damaged history -----------------------------------------------------
def test_unreadable_gone_list_keeps_the_task_working(ws, monkeypatch):
    """A truncated .gone file (a crash while writing it) doesn't break later edits or deletion;
    its files are kept rather than lost."""
    monkeypatch.setattr(history, "HISTORY_KEPT", 1)
    t = red_task(ws)
    gone = t.config.examples[0]
    t.delete_example(gone.id)
    for f in (t.path / "history").glob("*.gone"):
        f.write_text("")
    t.upsert_class("a")  # prunes the version whose .gone is unreadable
    t.upsert_class("b")
    assert t.class_names() == ["red", "a", "b"]
    assert (t.path / gone.file).exists()  # kept, for gc to judge
    assert t._gone(t.path / "history" / "x.json") == set()  # pylint: disable=protected-access
    (t.path / "history" / "x.gone").write_text('{"not": "a list"}')
    assert t._gone(t.path / "history" / "x.json") == set()  # pylint: disable=protected-access
    ws.delete("t")
    assert not (ws.root / "tasks" / "t").exists()


def test_damaged_version_is_skipped_by_undo(ws):
    """Undo past a damaged version removes it and says so; the next undo goes further back."""
    t = red_task(ws, n=1)
    t.upsert_class("a")
    t.upsert_class("b")
    newest = sorted((t.path / "history").glob("*.json"))[-1]
    newest.write_text("{broken")
    with pytest.raises(ValueError, match="is invalid and was removed; undo again"):
        t.undo()
    t.undo()
    assert t.class_names() == ["red"]
