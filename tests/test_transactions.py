"""A change to a task commits when its task.json is replaced: what other threads see before and
after that point, and what a failure on either side of it leaves behind."""

import multiprocessing
import os
import threading
from typing import Any

import pytest

from fewshot import fsutil, store
from fewshot.models import TaskConfig
from fewshot.store import Workspace
from tests.conftest import BLUE, RED, png

pytestmark = pytest.mark.posix_fs  # also run on macOS: see .github/workflows/ci.yml


@pytest.fixture(name="task")
def task_fixture(ws):
    """Task "t" with one red example."""
    t = ws.create("t")
    t.add_example(png(RED, noise=15, seed=1), "red")
    return t


def test_failure_after_commit_keeps_the_change(task, monkeypatch, caplog):
    """Clean-up failing after task.json was replaced is logged; the change and its images stay."""
    def broken_prune():
        raise OSError("EIO")

    monkeypatch.setattr(task, "_prune_history", broken_prune)
    ex = task.add_example(png(BLUE, noise=15, seed=2), "blue")
    assert "failed to prune history" in caplog.text
    assert (task.path / ex.file).exists()
    assert [e.label for e in task.config.examples] == ["red", "blue"]
    assert store.Workspace(task.path.parent.parent).get("t").config.revision == task.config.revision


def test_interrupt_after_commit_keeps_the_images(task, monkeypatch):
    """Ctrl+C after the commit propagates, but doesn't delete images the saved task names."""
    real = fsutil.fsync_dir

    def interrupted(path):
        if path == task.path:  # the sync after the commit, not the image folders' before it
            raise KeyboardInterrupt
        real(path)

    monkeypatch.setattr(store, "_fsync_dir", interrupted)
    with pytest.raises(KeyboardInterrupt):
        task.add_example(png(BLUE, noise=15, seed=2), "blue")
    monkeypatch.setattr(store, "_fsync_dir", real)
    saved = Workspace(task.path.parent.parent).get("t").config
    assert [e.label for e in saved.examples] == ["red", "blue"]
    assert all((task.path / e.file).exists() for e in saved.examples)
    assert task.config.revision == saved.revision  # memory matches what was saved


def test_failure_before_commit_publishes_nothing(task, monkeypatch):
    """A save that fails before replacing task.json leaves the published config as it was."""
    before = task.config

    def broken_replace(*_):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        task.upsert_class("blue")
    assert task.config is before
    monkeypatch.undo()
    task.upsert_class("green")  # the next revision number is the one that failure didn't use
    assert task.config.revision == before.revision + 1


def test_readers_never_see_a_change_in_progress(task, monkeypatch):
    """Another thread sees the last saved version, unchanged, until the change commits."""
    before = task.config
    n_before = len(before.examples)
    inside, release = threading.Event(), threading.Event()
    real = task._snapshot  # pylint: disable=protected-access

    def slow_snapshot(*args, **kw):
        inside.set()
        assert release.wait(5)
        return real(*args, **kw)

    monkeypatch.setattr(task, "_snapshot", slow_snapshot)
    writer = threading.Thread(target=task.add_example, args=(png(BLUE, noise=15, seed=2), "blue"))
    writer.start()
    assert inside.wait(5)
    seen = task.config
    assert seen is before
    assert len(seen.examples) == n_before
    assert seen.revision == before.revision
    release.set()
    writer.join(5)
    assert len(before.examples) == n_before  # a published snapshot never changes
    assert len(task.config.examples) == n_before + 1
    assert task.config.revision == before.revision + 1


def test_readers_do_not_wait_for_a_writer(tmp_path, monkeypatch):
    """While a writer waits for another process's lock, reads of the task still answer at once."""
    a, b = Workspace(tmp_path / "ws"), Workspace(tmp_path / "ws")
    a.create("t")
    ta = a.get("t")
    with fsutil.file_lock(ta.path / ".lock"):  # the "other process" holds the task
        waiting = threading.Event()
        real_lock = fsutil._lock  # pylint: disable=protected-access

        def lock(fd, shared=False):
            if not shared:
                waiting.set()  # the writer holds b's in-process lock and now waits for the file lock
            real_lock(fd, shared)

        monkeypatch.setattr(fsutil, "_lock", lock)
        writer = threading.Thread(target=b.get("t").upsert_class, args=("blue",))
        writer.start()
        assert waiting.wait(5)
        seen: list[list[str]] = []
        reader = threading.Thread(target=lambda: seen.append(b.get("t").class_names()), daemon=True)
        reader.start()
        reader.join(1)
        assert seen == [[]], "a read waited for the writer"
        assert writer.is_alive(), "the writer didn't wait for the lock"
    writer.join(5)
    assert b.get("t").class_names() == ["blue"]


def _undo_then_crash_mid_save(root):
    """In a child process: undo, then die between moving redo aside and the commit."""
    t = Workspace(root).get("t")
    t.undo()
    real = os.replace

    def crash(src: Any, dst: Any, **kw: Any) -> None:
        if str(dst).endswith("task.json"):
            os._exit(3)  # pylint: disable=protected-access  # like kill -9: no clean-up runs
        real(src, dst, **kw)

    os.replace = crash
    t.relabel_many([e.id for e in t.config.examples], "blue")


def test_a_crash_never_lets_redo_overwrite_a_newer_change(tmp_path):
    """A crash during a save after an undo loses the undone change, never the version saved."""
    root = tmp_path / "ws"
    t = Workspace(root).create("t")
    t.add_example(png(RED, noise=15, seed=1), "red")
    t.add_example(png(RED, noise=15, seed=2), "red")
    p = multiprocessing.get_context("spawn").Process(target=_undo_then_crash_mid_save, args=(root,))
    p.start()
    p.join()
    assert p.exitcode == 3
    after = Workspace(root).get("t")
    assert [e.label for e in after.config.examples] == ["red"]  # the undo was saved, the relabel not
    assert not after.can_redo()
    assert list((after.path / "history").glob(".redo-*"))
    Workspace(root).gc()
    assert not list((after.path / "history").glob(".redo-*"))


def _failing_commit(real: Any) -> Any:
    """``os.replace`` failing only for the commit of task.json, after everything before it worked."""
    def replace(src: Any, dst: Any, **kw: Any) -> None:
        if str(dst).endswith("task.json"):
            raise OSError("disk full")
        real(src, dst, **kw)
    return replace


def test_redo_survives_a_save_that_fails(task, monkeypatch):
    """A change that fails before committing leaves undone changes redoable."""
    task.add_example(png(BLUE, noise=15, seed=2), "blue")
    task.undo()
    monkeypatch.setattr(os, "replace", _failing_commit(os.replace))
    with pytest.raises(OSError, match="disk full"):
        task.upsert_class("green")
    monkeypatch.undo()
    assert task.can_redo()
    task.redo()
    assert [e.label for e in task.config.examples] == ["red", "blue"]


def test_failed_undo_leaves_no_redo(task, monkeypatch):
    """An undo that fails before committing adds nothing to redo."""
    task.add_example(png(BLUE, noise=15, seed=2), "blue")
    monkeypatch.setattr(os, "replace", _failing_commit(os.replace))
    with pytest.raises(OSError, match="disk full"):
        task.undo()
    monkeypatch.undo()
    assert not task.can_redo()
    assert task.undo_label() == "add 1 example to blue"


def test_snapshot_replaces_stale_side_files(task):
    """Side files a crashed attempt left at a revision don't label the version kept there."""
    hist = task.path / "history"
    rev = task.config.revision
    (hist / f"{rev:010d}.label").write_text("something else")
    task.upsert_class("blue")
    assert task.undo_label() == "add class blue"


def test_gc_spares_a_folder_still_being_built(ws, monkeypatch):
    """gc takes an old folder without task.json for a leftover only if no builder holds it."""
    monkeypatch.setattr(store, "STALE_SECONDS", -1)
    p = ws.root / "tasks" / "building"
    p.mkdir()
    with fsutil.dir_lock(p):
        assert ws.gc()["folders"] == 0
        with pytest.raises(ValueError, match="already exists"):
            ws.create("building")
        assert p.exists()
    assert ws.gc()["folders"] == 1
    assert not p.exists()


def test_duplicate_is_locked_while_built(ws, monkeypatch):
    """A slow duplicate can't be removed by gc running alongside it."""
    monkeypatch.setattr(store, "STALE_SECONDS", -1)
    src = ws.create("src")
    src.add_example(png(RED, noise=15, seed=1), "red")
    seen = {}
    real = store.Task._add_file  # pylint: disable=protected-access

    def store_and_gc(self, file, source):
        if self.path.name.startswith(".tmp-"):  # what gc would remove (it waits for the source task)
            seen["gc"] = sum(fsutil.reclaim(p, store.STALE_SECONDS) for p in (ws.root / "tasks").iterdir())
        return real(self, file, source)

    monkeypatch.setattr(store.Task, "_add_file", store_and_gc)
    ws.duplicate("src", "copy")
    assert seen["gc"] == 0
    assert len(ws.get("copy").config.examples) == 1


def test_a_damaged_image_is_stored_again(ws):
    """A stored image cut short (a power cut) is replaced when the same image is added again,
    and doesn't spread to other tasks through the shared store."""
    data = png(RED, noise=15, seed=1)
    a = ws.create("a")
    ex = a.add_example(data, "red")
    with open(a.path / ex.file, "r+b") as f:  # the task's file and the shared copy are one inode
        f.truncate(0)
    b = ws.create("b")
    b.add_example(data, "red")
    assert (b.path / ex.file).read_bytes() == data
    a.delete_example(ex.id)
    a.add_example(data, "red")
    assert (a.path / ex.file).read_bytes() == data


def test_failed_create_leaves_no_folder(ws, monkeypatch):
    """A create that fails part-way frees the name again."""
    monkeypatch.setattr(os, "replace", _failing_commit(os.replace))
    with pytest.raises(OSError, match="disk full"):
        ws.create("t")
    assert not (ws.root / "tasks" / "t").exists()


def test_dir_locked_on_a_missing_folder(tmp_path):
    """A folder that is gone is not being built."""
    assert not fsutil.dir_locked(tmp_path / "gone")


def test_undo_interrupted_after_commit_stays_redoable(task, monkeypatch):
    """An undo that was saved stays redoable even if interrupted while cleaning up after it."""
    task.add_example(png(BLUE, noise=15, seed=2), "blue")
    real = fsutil.fsync_dir

    def interrupted(path):
        if path == task.path:
            raise KeyboardInterrupt
        real(path)

    monkeypatch.setattr(store, "_fsync_dir", interrupted)
    with pytest.raises(KeyboardInterrupt):
        task.undo()
    monkeypatch.undo()
    assert [e.label for e in task.config.examples] == ["red"]
    assert task.redo_label() == "add 1 example to blue"


def _interrupt_after_commit(real: Any) -> Any:
    """``os.replace`` that commits task.json and is then interrupted before returning control."""
    def replace(src: Any, dst: Any, **kw: Any) -> None:
        real(src, dst, **kw)
        if str(dst).endswith("task.json"):
            raise KeyboardInterrupt
    return replace


def test_interrupt_right_after_the_replace_keeps_the_change(task, monkeypatch):
    """Ctrl+C landing just after task.json was replaced, before the save recorded it, still
    counts as committed: the images it names stay and memory shows the change."""
    monkeypatch.setattr(os, "replace", _interrupt_after_commit(os.replace))
    with pytest.raises(KeyboardInterrupt):
        task.add_example(png(BLUE, noise=15, seed=2), "blue")
    monkeypatch.undo()
    saved = Workspace(task.path.parent.parent).get("t").config
    assert [e.label for e in saved.examples] == ["red", "blue"]
    assert all((task.path / e.file).exists() for e in saved.examples)
    assert task.config.revision == saved.revision


def test_undo_interrupted_right_after_the_replace_stays_redoable(task, monkeypatch):
    """The same window during an undo keeps its redo step."""
    task.add_example(png(BLUE, noise=15, seed=2), "blue")
    monkeypatch.setattr(os, "replace", _interrupt_after_commit(os.replace))
    with pytest.raises(KeyboardInterrupt):
        task.undo()
    monkeypatch.undo()
    assert [e.label for e in task.config.examples] == ["red"]
    assert task.redo_label() == "add 1 example to blue"


def test_gc_skips_a_task_deleted_meanwhile(ws, monkeypatch):
    """A task another process deletes while gc runs is skipped, and the rest is still cleaned."""
    ws.create("a")
    b = ws.create("b")
    listed = ws.list_tasks()
    Workspace(ws.root).delete("b")
    monkeypatch.setattr(ws, "list_tasks", lambda: listed)
    assert b not in Workspace(ws.root).list_tasks()
    assert ws.gc()["folders"] == 0


# ---- creates and renames racing each other ------------------------------
def test_a_failed_rename_never_moves_the_task_onto_a_new_task_of_its_old_name(ws, monkeypatch):
    """A rename moves the folder, then its save fails (a full disk). If another process reserved
    the freed old name for a new task meanwhile, the rename doesn't move the task back onto that
    folder (which the other's build would then fail to replace and remove, task and all): the
    task stays under the new name, and the other's create succeeds."""
    t = ws.create("foo")
    t.add_example(png(RED), "a")
    other = Workspace(ws.root)
    reserved = threading.Event()
    go = threading.Event()
    made: list[Any] = []

    def images():
        reserved.set()
        go.wait(5)
        yield from ()

    def create():
        made.append(other.build_task(TaskConfig(name="foo"), images()))

    creator = threading.Thread(target=create)
    real_save = store.Task._save  # pylint: disable=protected-access

    def failing_save(self, *a, **kw):
        if self.path.name == "bar":
            creator.start()
            assert reserved.wait(5)
            raise OSError(28, "No space left on device")
        return real_save(self, *a, **kw)

    monkeypatch.setattr(store.Task, "_save", failing_save)
    with pytest.raises(OSError, match="No space"):
        ws.rename("foo", "bar")
    monkeypatch.setattr(store.Task, "_save", real_save)
    go.set()
    creator.join(5)
    assert made
    assert made[0].config.examples == []  # the new, empty "foo"
    fresh = Workspace(ws.root)
    assert [x.name for x in fresh.list_tasks()] == ["bar", "foo"]
    assert len(fresh.get("bar").config.examples) == 1  # the task survives under the folder's name
    assert len(ws.get("bar").config.examples) == 1


def test_a_failed_rename_moves_the_task_back(ws, monkeypatch):
    """Without a competing create, a rename whose save fails puts the folder back under its name."""
    ws.create("foo").add_example(png(RED), "a")
    real_save = store.Task._save  # pylint: disable=protected-access

    def failing_save(self, *a, **kw):
        if self.path.name == "bar":
            raise OSError(28, "No space left on device")
        return real_save(self, *a, **kw)

    monkeypatch.setattr(store.Task, "_save", failing_save)
    with pytest.raises(OSError, match="No space"):
        ws.rename("foo", "bar")
    assert sorted(p.name for p in (ws.root / "tasks").iterdir()) == ["foo"]
    assert ws.get("foo").path.name == "foo"


def test_an_interrupt_after_a_build_is_in_place_keeps_the_task(ws, monkeypatch):
    """An interrupt arriving just after a create moved its finished folder into place doesn't
    delete the task: the clean-up only ever removes its own empty reservation."""
    real_move = fsutil.replace_reserved

    def interrupted(build, p):
        real_move(build, p)
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "replace_reserved", interrupted)
    with pytest.raises(KeyboardInterrupt):
        ws.create("t")
    monkeypatch.setattr(store, "replace_reserved", real_move)
    assert Workspace(ws.root).get("t").name == "t"


def test_a_late_lock_on_a_deleted_task_leaves_nothing_in_a_new_one_s_folder(ws):
    """An object for a deleted task (an evaluation finishing late, say) that locks it while a new
    task of the same name is being built doesn't create a lock file in the new task's reserved
    folder, which would make the build fail to move into place."""
    t = ws.create("foo")
    t.add_example(png(RED), "a")
    stale = Workspace(ws.root).get("foo")
    ws.delete("foo")
    before = []

    def images():
        stale.record_evaluation({"revision": 1, "accuracy": 1.0})  # does nothing: the task is gone
        with pytest.raises(KeyError):
            stale.upsert_class("x")
        before.append(sorted(p.name for p in (ws.root / "tasks" / "foo").iterdir()))
        yield from ()

    new = ws.build_task(TaskConfig(name="foo"), images())
    assert before == [[]]
    assert new.config.examples == []
    assert not (new.path / "eval.json").exists()


def test_a_stray_lock_file_in_a_reserved_folder_is_cleared(ws, monkeypatch):
    """A lock file that still got into a reserved folder (a lock taken in the instant between a
    deletion and the reservation) is removed so the build moves into place; anything else there
    is an error, and the reservation (with what is in it) is left for gc."""
    t = ws.build_task(TaskConfig(name="foo"), _into(ws, "foo", ".lock"))
    assert (t.path / "task.json").exists()
    with pytest.raises(OSError, match="not empty"):
        ws.build_task(TaskConfig(name="bar"), _into(ws, "bar", "junk"))
    assert (ws.root / "tasks" / "bar" / "junk").exists()
    assert not [p for p in (ws.root / "tasks").iterdir() if p.name.startswith(".tmp-")]
    calls = []
    real = os.replace

    def busy(src, dst):  # a lock file reappears every time: give up after a few tries
        if dst == ws.root / "tasks" / "baz":
            calls.append(dst)
            (dst / ".lock").touch()
        real(src, dst)

    monkeypatch.setattr(os, "replace", busy)
    with pytest.raises(OSError, match="not empty"):
        ws.build_task(TaskConfig(name="baz"))
    assert len(calls) == 3


def _into(ws, name, extra):
    """No images, but first a file ``extra`` in task ``name``'s reserved folder."""
    (ws.root / "tasks" / name / extra).write_text("")
    yield from ()
