"""Undo and redo: stepping through versions, labels, damaged history, pruning, and one undo step
per change."""


import json
import os
import shutil

import pytest

from fewshot import history
from fewshot.fsutil import fsync_dir, write_atomic
from fewshot.store import Workspace
from tests.conftest import BLUE, GREEN, RED, png, red_green_task, red_task, two_each

pytestmark = pytest.mark.posix_fs  # also run on macOS: see .github/workflows/ci.yml


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
    with pytest.raises(ValueError, match=r"1 image files of that version are gone.*it was removed"):
        t.undo()
    snaps = sorted((t.path / "history").glob("*.json"))
    snaps[-1].write_text("{broken")
    with pytest.raises(ValueError, match="is invalid"):
        t.undo()
    assert t.config.examples == []
    assert not t.can_undo()


def test_a_step_with_missing_images_is_removed_so_undo_goes_further(ws):
    """An undo step whose version names an image that is gone (removed by hand) is removed when
    it fails, so the next undo reaches the versions before it, instead of failing for ever."""
    t = ws.create("t", embedder="fake:color")
    t.upsert_class("first")
    e = t.add_example(png(RED), "red")
    t.delete_example(e.id)
    (t.path / e.file).unlink()
    with pytest.raises(ValueError, match="was removed"):
        t.undo()
    assert t.undo_label() == "add 1 example to red"  # the add, whose undo needs no image
    t.undo()
    t.undo()
    assert t.class_names() == []


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


def test_a_task_json_put_back_by_hand_is_a_new_version(ws, caplog):
    """A kept version copied over task.json by hand carries an old revision: it is taken as the
    next one, so kept versions survive, undo goes back to it and pruning keeps the newest steps."""
    t = red_task(ws, n=1)
    for i in range(5):
        t.upsert_class(f"c{i}")
    p = t.path
    old = sorted((p / "history").glob("*.json"))[2]
    rev = t.config.revision
    shutil.copy(old, p / "task.json")
    t = Workspace(ws.root).get("t")
    assert t.config.revision == rev  # above every kept version
    assert "taking it as revision" in caplog.text
    rolled_back = t.class_names()
    t.upsert_class("mine")
    assert old.exists()  # its revision's version wasn't replaced
    assert [r for r, _ in t._history()][-1] == rev  # pylint: disable=protected-access
    t.undo()
    assert t.class_names() == rolled_back
    t.undo()
    assert t.class_names() == ["red", "c0", "c1", "c2", "c3"]  # the newest kept before the hand edit


def test_an_evaluation_of_a_later_revision_shows_as_out_of_date_after_a_rollback(ws):
    """A rolled-back task.json takes a revision past the last evaluation's, which then no longer
    reads as current; history removed too makes no difference."""
    t = red_task(ws, n=1)
    t.upsert_class("a")
    t.record_evaluation({"revision": t.config.revision})
    rev = t.config.revision
    shutil.rmtree(t.path / "history")
    data = json.loads((t.path / "task.json").read_text())
    (t.path / "task.json").write_text(json.dumps({**data, "revision": 0}))
    t = Workspace(ws.root).get("t")
    assert t.config.revision == rev + 1
    (t.path / "task.json").write_text(json.dumps({**data, "revision": rev}))
    assert t.refresh()
    assert t.config.revision == rev  # the evaluated revision itself is current


def test_a_save_interrupted_after_keeping_its_version_is_not_a_rollback(ws):
    """A save killed between keeping task.json as a version and replacing it leaves that version
    as task.json itself: the revision stays (a dangling one counts as a later version)."""
    t = red_task(ws, n=1)
    t.upsert_class("a")
    rev = t.config.revision
    os.link(t.path / "task.json", t.path / "history" / f"{rev:010d}.json")
    assert Workspace(ws.root).get("t").config.revision == rev
    (t.path / "history" / f"{rev:010d}.json").unlink()
    (t.path / "history" / f"{rev:010d}.json").symlink_to(t.path / "gone")
    assert Workspace(ws.root).get("t").config.revision == rev + 1


# ---- undo groups ----------------------------------------------------------
def _items(rgb, label, n, seed=0):
    return [(png(rgb, noise=30, seed=seed + i), label, f"{label}{seed + i}.png") for i in range(n)]


def test_an_undo_group_is_one_undo_step(ws):
    """Consecutive adds with the same undo group are one undo step, described as a whole; a
    later add in another group, or after any other change, is a step of its own."""
    t = ws.create("t", embedder="fake:color")
    t.add_examples(_items(RED, "red", 2), undo_group="drop")
    t.add_examples(_items(BLUE, "blue", 3), undo_group="drop")
    assert t.undo_label() == "add 5 examples to red, blue"
    t.add_examples(_items(GREEN, "green", 1), undo_group="other")
    assert t.undo_label() == "add 1 example to green"
    t.add_examples(_items(RED, "red", 1, seed=10), undo_group="drop")  # that group was ended
    assert t.undo_label() == "add 1 example to red"
    t.undo()
    t.undo()
    assert t.undo_label() == "add 5 examples to red, blue"
    t.undo()
    assert not t.config.examples
    assert not t.can_undo()


def test_any_other_change_ends_an_undo_group(ws):
    """A change in between (even one not kept for undo, like a rename) ends the group, and so
    does undoing its step."""
    t = ws.create("t", embedder="fake:color")
    t.add_examples(_items(RED, "red", 1), undo_group="g")
    t.upsert_class("blue")
    t.add_examples(_items(RED, "red", 1, seed=1), undo_group="g")
    assert t.undo_label() == "add 1 example to red"
    t = ws.rename("t", "u")
    t.add_examples(_items(RED, "red", 1, seed=2), undo_group="g")
    assert t.undo_label() == "add 1 example to red"
    t.undo()
    t.add_examples(_items(RED, "red", 1, seed=3), undo_group="g")
    assert t.undo_label() == "add 1 example to red"
    assert len(history.TaskHistory._history(t)) == 4  # pylint: disable=protected-access


def test_an_undo_group_works_across_processes(ws):
    """The group is noted on disk with its step, so another process's add (here: another
    Workspace on the same folder) joins it."""
    t = ws.create("t", embedder="fake:color")
    t.add_examples(_items(RED, "red", 2), undo_group="drop")
    other = Workspace(ws.root).get("t")
    other.add_examples(_items(BLUE, "blue", 2), undo_group="drop")
    assert other.undo_label() == t.undo_label() == "add 4 examples to red, blue"
    t.undo()
    assert not t.config.examples


def test_one_call_is_one_step_described_whole(ws):
    """The chunks of one call are one undo step, labelled with everything the call added."""
    t = ws.create("t", embedder="fake:color")
    t.add_examples(_items(RED, "red", 3) + _items(BLUE, "blue", 2), chunk=2)
    assert t.undo_label() == "add 5 examples to red, blue"
    t.add_to_inbox([(png(GREEN, noise=30, seed=i), f"g{i}.png") for i in range(3)], chunk=2)
    assert t.undo_label() == "add 3 images to the inbox"
    t.undo()
    assert t.undo_label() == "add 5 examples to red, blue"


def test_inbox_uploads_group_too_but_not_with_examples(ws):
    """Inbox adds group like examples; an inbox add and an example add with the same group id
    stay two steps."""
    t = ws.create("t", embedder="fake:color")
    for i in range(2):
        t.add_to_inbox([(png(GREEN, noise=30, seed=10 * i + j), f"g{j}.png") for j in range(2)], undo_group="d")
    assert t.undo_label() == "add 4 images to the inbox"
    t.add_examples(_items(RED, "red", 1), undo_group="d")
    t.undo()
    assert t.undo_label() == "add 4 images to the inbox"
    t.undo()
    assert not t.config.inbox


def test_a_damaged_or_unwritable_group_note_just_ends_the_group(ws, monkeypatch, caplog):
    """A group note that can't be read starts a new step; one that can't be written is logged,
    and the change stands. A step pruned at once gets no note."""
    t = ws.create("t", embedder="fake:color")
    t.add_examples(_items(RED, "red", 1), undo_group="g")
    note = history.TaskHistory._history(t)[-1][1].with_suffix(".group")  # pylint: disable=protected-access
    for bad in ("junk", '{"group": "g"}'):
        note.write_text(bad)
        t.add_examples(_items(RED, "red", 1, seed=len(bad)), undo_group="g")
        assert t.undo_label() == "add 1 example to red"
        note = history.TaskHistory._history(t)[-1][1].with_suffix(".group")  # pylint: disable=protected-access
    real = write_atomic

    def failing(path, data):
        if path.suffix == ".group":
            raise OSError("disk full")
        real(path, data)

    monkeypatch.setattr(history, "_write_atomic", failing)
    t.add_examples(_items(BLUE, "blue", 1), undo_group="h")
    assert "its undo group was not noted" in caplog.text
    assert t.undo_label() == "add 1 example to blue"
    monkeypatch.setattr(history, "_write_atomic", real)
    monkeypatch.setattr(history, "HISTORY_KEPT", 0)
    t.add_examples(_items(GREEN, "green", 1), undo_group="i")
    assert not list((t.path / "history").glob("*.group"))


# ---- reading kept versions cheaply ------------------------------------------
def _count_parses(monkeypatch):
    """Patch ``history._parse_files`` to count its calls; returns the counter (a list of one int)."""
    calls = [0]
    real = history._parse_files  # pylint: disable=protected-access

    def counting(data):
        calls[0] += 1
        return real(data)

    monkeypatch.setattr(history, "_parse_files", counting)
    return calls


def test_saves_with_full_history_read_no_kept_version_unless_a_file_may_go(ws, monkeypatch):
    """Once history is full every recorded save prunes it; a change that dropped no file reads
    none of the kept versions (they were parsed on every save, seconds for a large task)."""
    monkeypatch.setattr(history, "_snapshot_files", history._SnapshotFiles(10**6))  # pylint: disable=protected-access
    t = red_task(ws, n=1)
    for i in range(history.HISTORY_KEPT + 2):
        t.upsert_class(f"c{i}")
    calls = _count_parses(monkeypatch)
    t.upsert_class("more")
    assert calls == [0]
    gone = t.config.examples[0]
    t.delete_example(gone.id)
    for i in range(history.HISTORY_KEPT):
        t.upsert_class(f"d{i}")
    assert calls[0] >= 1  # the step that dropped a file was pruned: the kept versions were read
    assert not (t.path / gone.file).exists()


def test_kept_versions_are_parsed_once_per_process(ws, monkeypatch):
    """The files a kept version names are remembered while it is the same file; a changed file is
    read again, and the memo stays within its bound. Versions naming the same files share a set."""
    monkeypatch.setattr(history, "_snapshot_files", history._SnapshotFiles(100))  # pylint: disable=protected-access
    t = red_green_task(ws, n=1)
    t.upsert_class("a")
    t.upsert_class("b")
    calls = _count_parses(monkeypatch)
    first = t.kept_files()
    assert t.kept_files() == first
    assert calls[0] == len(t._history())  # pylint: disable=protected-access  # each once, then remembered
    snaps = [p for _, p in t._history()]  # pylint: disable=protected-access
    assert t._files_in(snaps[-1]) is t._files_in(snaps[-2])  # pylint: disable=protected-access  # same files: one set
    newest = snaps[-1]
    newest.unlink()
    newest.write_text('{"examples": [{"file": "images\\/0123456789abcdef.png"}]}')  # escaped: parsed as JSON
    assert t._files_in(newest) == {"images/0123456789abcdef.png"}  # pylint: disable=protected-access
    newest.write_text('{"examples": [{"file": 3}], "inbox": 5}')
    assert t._files_in(newest) == set()  # pylint: disable=protected-access
    small = history._SnapshotFiles(1)  # pylint: disable=protected-access
    for p in snaps[:-1]:
        small.files(p)
    assert len(small._sets) == 1  # pylint: disable=protected-access  # the older ones were forgotten
    assert t._files_in(t.path / "history" / "missing.json") == set()  # pylint: disable=protected-access


def test_version_is_parsed_once_for_repeated_use(ws, monkeypatch):
    """``version`` of the same kept revision (say, predicting at it again) parses it once."""
    t = red_green_task(ws, n=1)
    rev = t.config.revision
    t.upsert_class("x")
    first = t.version(rev)
    parsed = []
    real = type(t.config).stored

    def counting(cls, data):  # pylint: disable=unused-argument
        parsed.append(1)
        return real(data)

    monkeypatch.setattr(type(t.config), "stored", classmethod(counting))
    assert t.version(rev) == first
    assert not parsed
    t.upsert_class("y")
    t.version(rev + 1)
    assert parsed == [1]


def test_redo_skips_a_step_a_crashed_undo_left(ws):
    """An undo killed after keeping the current version for redo but before restoring the earlier
    one leaves a redo step that changes nothing: redo skips it rather than adding a no-op step."""
    t = red_green_task(ws)
    with t._writing():  # pylint: disable=protected-access  # the undo's first half, then a crash
        t._snapshot(t.config.revision, [], t.undo_label(), folder="history/redo")  # pylint: disable=protected-access
    t = Workspace(ws.root).get("t")
    assert t.can_redo()
    with pytest.raises(ValueError, match="Nothing to redo"):
        t.redo()
    assert not t.can_redo()
    n = len(t.config.examples)
    t.undo()  # the real step is still there
    assert len(t.config.examples) == n - 1
    t.redo()
    assert len(t.config.examples) == n


def test_moving_redo_aside_is_durable_before_the_commit(ws, monkeypatch):
    """The undone versions are moved aside, and that move synced, before the new task.json
    replaces the old: a power cut can't keep the new change together with the old redo/."""
    t = red_green_task(ws)
    t.undo()
    events = []
    real_fsync, real_replace, real_rename = fsync_dir, os.replace, os.rename

    def fsync(path):
        events.append(("sync", path.name))
        real_fsync(path)

    def replace(src, dst):
        events.append(("replace", os.path.basename(dst)))
        real_replace(src, dst)

    def rename(src, dst):
        events.append(("rename", os.path.basename(dst)[:6]))
        real_rename(src, dst)

    monkeypatch.setattr(history, "_fsync_dir", fsync)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "rename", rename)
    t.upsert_class("x")
    aside, commit = events.index(("rename", ".redo-")), events.index(("replace", "task.json"))
    assert ("sync", "history") in events[aside:commit]
