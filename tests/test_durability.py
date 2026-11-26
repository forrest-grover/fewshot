"""Crash safety: what is flushed to disk (``os.fsync``) before a name points at it, and in which order,
so a power cut can't leave a task.json naming files or folder entries that never reached the disk."""

import os

import pytest

from fewshot import fsutil
from tests.conftest import BLUE, RED, png


class DiskLog:
    """Records, in order, every ``os.fsync`` (by the inode of what was synced) and every
    ``os.replace``/``os.link``/``os.rename`` (by the inode of the source and the destination path)."""

    def __init__(self, monkeypatch):
        self.events: list[tuple[str, int, str]] = []
        for name in ("replace", "link", "rename"):
            real = getattr(os, name)
            monkeypatch.setattr(os, name, self._recording(name, real))
        real_fsync = os.fsync

        def fsync(fd):
            self.events.append(("fsync", os.fstat(fd).st_ino, ""))
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", fsync)

    def _recording(self, name, real):
        def op(src, dst, *args, **kw):
            self.events.append((name, os.stat(src).st_ino, str(dst)))
            return real(src, dst, *args, **kw)
        return op

    def synced(self, inode):
        """Indexes of the fsyncs of inode ``inode``."""
        return [i for i, (what, n, _) in enumerate(self.events) if what == "fsync" and n == inode]

    def moves_to(self, dst, kind=("replace", "link", "rename")):
        """The (index, source inode) of each ``kind`` of move onto path ``dst``."""
        return [(i, n) for i, (what, n, d) in enumerate(self.events) if what in kind and d == str(dst)]


@pytest.fixture(name="disk")
def disk_fixture(monkeypatch):
    """A ``DiskLog`` recording from now on."""
    return DiskLog(monkeypatch)


def ino(path):
    """The inode of ``path``."""
    return os.stat(path).st_ino


def test_a_written_file_is_synced_before_it_is_renamed_into_place(tmp_path, disk):
    """write_atomic: the temporary file's data reaches the disk before os.replace puts it in place."""
    fsutil.write_atomic(tmp_path / "f", b"data")
    [(i, src)] = disk.moves_to(tmp_path / "f", ("replace",))
    assert any(j < i for j in disk.synced(src))


def test_a_created_file_is_synced_before_it_is_linked_into_place(tmp_path, disk):
    """create_exclusive: the temporary file's data reaches the disk before os.link names it."""
    assert fsutil.create_exclusive(tmp_path / "f", b"data")
    [(i, src)] = disk.moves_to(tmp_path / "f", ("link",))
    assert any(j < i for j in disk.synced(src))


def test_fsync_dir_syncs_the_folder_itself(tmp_path, disk):
    """fsync_dir flushes the folder's own entries (its inode), not just opens it."""
    fsutil.fsync_dir(tmp_path)
    assert disk.synced(ino(tmp_path))


def test_a_save_syncs_new_images_and_history_before_task_json_and_the_folder_after(ws, monkeypatch):
    """Adding an example: the image folder, the shared blob folder, the new task.json's data and the
    kept version in history/ are all on disk before task.json is replaced; the task folder (the
    replace itself) is synced after it."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")  # so the next save keeps a version in history/
    disk = DiskLog(monkeypatch)
    ex = t.add_example(png(BLUE), "blue")
    [(commit, new)] = disk.moves_to(t.path / "task.json", ("replace",))
    before = {n for i, (what, n, _) in enumerate(disk.events) if what == "fsync" and i < commit}
    after = {n for i, (what, n, _) in enumerate(disk.events) if what == "fsync" and i > commit}
    blob = t._blob(ex.file)  # pylint: disable=protected-access
    assert blob is not None
    assert ino(blob) == ino(t.path / ex.file)  # stored once, linked into the task
    assert {new, ino(t.path / "images"), ino(blob.parent), ino(t.path / "history")} <= before
    assert ino(t.path) in after
    assert ino(t.path) not in before


def test_an_undo_syncs_history_after_forgetting_the_step(ws, monkeypatch):
    """Undo: once the restored version's file is removed from history/, the folder is synced, so a
    power cut can't bring the step back."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    t.add_example(png(BLUE), "blue")
    disk = DiskLog(monkeypatch)
    t.undo()
    [(commit, _)] = disk.moves_to(t.path / "task.json", ("replace",))
    assert any(i > commit for i in disk.synced(ino(t.path / "history")))


def test_creating_renaming_and_deleting_a_task_sync_the_tasks_folder_after(ws, monkeypatch):
    """A task folder moved into place (create, rename) or aside (delete) is followed by a sync of
    the folder holding it."""
    tasks = ws.root / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    disk = DiskLog(monkeypatch)
    t = ws.create("t", embedder="fake:color")
    [(made, _)] = disk.moves_to(tasks / "t", ("replace",))
    assert any(i > made for i in disk.synced(ino(tasks)))
    disk.events.clear()
    ws.rename("t", "u")
    [(moved, _)] = disk.moves_to(tasks / "u", ("replace",))
    assert any(i > moved for i in disk.synced(ino(tasks)))
    disk.events.clear()
    t.destroy()
    [gone] = [i for i, (what, _, d) in enumerate(disk.events) if what == "rename" and ".trash-" in d]
    assert any(i > gone for i in disk.synced(ino(tasks)))
