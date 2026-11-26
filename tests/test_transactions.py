"""A change to a task commits when its task.json is replaced: what other threads see before and
after that point, and what a failure on either side of it leaves behind."""

import multiprocessing
import os
import threading
import time
from typing import Any

import pytest

from fewshot import fsutil, store
from fewshot.store import Workspace
from tests.conftest import png

RED, BLUE = (220, 30, 30), (30, 30, 220)


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


def test_readers_do_not_wait_for_a_writer(tmp_path):
    """While a writer waits for another process's lock, reads of the task still answer at once."""
    a, b = Workspace(tmp_path / "ws"), Workspace(tmp_path / "ws")
    a.create("t")
    ta = a.get("t")
    with fsutil.file_lock(ta.path / ".lock"):  # the "other process" holds the task
        writer = threading.Thread(target=b.get("t").upsert_class, args=("blue",))
        writer.start()
        time.sleep(0.1)  # the writer now holds b's in-process lock, waiting for the file lock
        seen: list[list[str]] = []
        reader = threading.Thread(target=lambda: seen.append(b.get("t").class_names()), daemon=True)
        reader.start()
        reader.join(1)
        assert seen == [[]], "a read waited for the writer"
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
    real = store.Task._store_image  # pylint: disable=protected-access

    def store_and_gc(self, file, data):
        if self.path.name == "copy":
            seen["gc"] = ws.gc()["folders"]
        return real(self, file, data)

    monkeypatch.setattr(store.Task, "_store_image", store_and_gc)
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
