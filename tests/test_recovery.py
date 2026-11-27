"""Crash safety and clean-up: interrupted renames, creates and deletes, failed adds, and gc."""

import errno
import fcntl
import os
import shutil
import time

import pytest
from typer.testing import CliRunner

from fewshot import archive, cli, fsutil, history, store
from fewshot.store import Workspace
from tests.conftest import GREEN, RED, png, red_task

pytestmark = pytest.mark.posix_fs  # also run on macOS: see .github/workflows/ci.yml


# ---- interrupted operations ----------------------------------------------
def test_interrupted_rename_leaves_a_usable_task(ws):
    """If a rename moved the folder but didn't rewrite task.json, the task opens under the new name."""
    red_task(ws, "old")
    os.rename(ws.root / "tasks" / "old", ws.root / "tasks" / "new")
    fresh = Workspace(ws.root)
    assert [t.name for t in fresh.list_tasks()] == ["new"]
    assert len(fresh.get("new").config.examples) == 2


def _age(path, seconds=3600):
    """Make ``path`` look untouched for ``seconds``."""
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_leftover_folder_from_a_crash_frees_its_name(ws):
    """A task folder without task.json blocks its name only while it may still be being made."""
    (ws.root / "tasks" / "x").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        ws.create("x")
    _age(ws.root / "tasks" / "x")
    assert ws.create("x").name == "x"


def test_delete_moves_the_folder_aside_first(ws, monkeypatch):
    """A delete interrupted after the move leaves a hidden .trash- folder, which listing ignores."""
    red_task(ws)

    def crash(path, *args, **kw):
        raise OSError("power cut")

    monkeypatch.setattr(shutil, "rmtree", crash)
    with pytest.raises(OSError, match="power cut"):
        ws.delete("t")
    leftovers = [p.name for p in (ws.root / "tasks").iterdir()]
    assert len(leftovers) == 1
    assert leftovers[0].startswith(".trash-")
    assert not Workspace(ws.root).list_tasks()


def test_failed_add_removes_its_new_images(ws, monkeypatch):
    """If saving task.json fails, the images that add stored are removed again (not ones that were
    there before)."""
    t = red_task(ws, n=1)
    before = sorted(p.name for p in (t.path / "images").iterdir())
    real = os.replace

    def failing(src, dst):
        if str(dst).endswith("task.json"):
            raise OSError("disk full")
        real(src, dst)

    monkeypatch.setattr(os, "replace", failing)
    existing = (png(RED, noise=10, seed=0), "red", "")  # already stored: must survive
    with pytest.raises(OSError, match="disk full"):
        t.add_examples([(png(GREEN, seed=i), "green", "") for i in range(3)] + [existing])
    assert sorted(p.name for p in (t.path / "images").iterdir()) == before
    assert len(t.config.examples) == 1


# ---- gc ------------------------------------------------------------------
def test_gc(ws, monkeypatch):
    """gc removes old leftover folders, image files no version names, and unused blobs; it keeps
    young leftovers (maybe in progress) and images undo may still need."""
    monkeypatch.setattr(history, "HISTORY_KEPT", 5)
    t = red_task(ws, n=2)
    deleted = t.config.examples[0]
    t.delete_example(deleted.id)  # undo still needs its image
    stray = t.path / "images" / ("ab" * 16 + ".png")
    stray.write_bytes(png(GREEN))
    tasks = ws.root / "tasks"
    for name in (".tmp-old", ".trash-old", "orphan"):
        (tasks / name).mkdir()
        _age(tasks / name)
    (tasks / ".tmp-young").mkdir()
    (ws.blobs / "zz").mkdir(parents=True)
    (ws.blobs / "zz" / "zz.png").write_bytes(b"x")
    _age(ws.blobs / "zz" / "zz.png")
    assert ws.gc() == {"folders": 3, "images": 1, "blobs": 1}
    assert sorted(p.name for p in tasks.iterdir()) == [".tmp-young", "t"]
    assert not stray.exists()
    assert (t.path / deleted.file).exists()
    assert ws.gc() == {"folders": 0, "images": 0, "blobs": 0}


def test_gc_command(tmp_path):
    """`fewshot gc` reports what it removed."""
    root = tmp_path / "ws"
    Workspace(root).create("t")
    (root / "tasks" / "orphan").mkdir()
    _age(root / "tasks" / "orphan")
    r = CliRunner().invoke(cli.app, ["gc", "--root", str(root)])
    assert r.output.strip() == "Removed 1 leftover folders, 0 unused image files and 0 unused stored images"


def test_abandoned_folder_vanishing(tmp_path):
    """A folder that disappears while being checked is not abandoned."""
    assert not fsutil.abandoned(tmp_path / "gone", store.STALE_SECONDS)
    assert archive.MAX_IMPORT_TOTAL > 0


def test_gc_never_deletes_a_folder_taken_back_meanwhile(ws, monkeypatch):
    """A leftover folder judged abandoned may be taken back (its name reused by a create) before
    gc removes it: gc judges it again under its lock, so the new task survives."""
    leftover = ws.root / "tasks" / "pets"
    leftover.mkdir()
    _age(leftover)
    real = fsutil.abandoned

    def racing(p, stale, **kw):
        found = real(p, stale, **kw)
        if found and p == leftover and not kw:  # gc's first look: another process creates "pets"
            monkeypatch.setattr(fsutil, "abandoned", real)
            Workspace(ws.root).create("pets").upsert_class("cat")
        return found

    monkeypatch.setattr(fsutil, "abandoned", racing)
    assert ws.gc()["folders"] == 0
    assert Workspace(ws.root).get("pets").class_names() == ["cat"]
    assert not [p for p in (ws.root / "tasks").iterdir() if p.name.startswith(".")]


def test_claim_dir(tmp_path, monkeypatch):
    """``claim_dir`` holds a folder's lock only if nobody else does and the folder is still there."""
    p = tmp_path / "f"
    with fsutil.claim_dir(p) as held:
        assert not held  # missing
    p.mkdir()
    with fsutil.dir_lock(p), fsutil.claim_dir(p) as held:
        assert not held  # a builder holds it
    with fsutil.claim_dir(p) as held:
        assert held
        assert fsutil.dir_locked(p)
    real = fcntl.flock

    def replaced(fd, op):  # the folder is replaced by another just after it was opened
        p.rename(tmp_path / "old")
        p.mkdir()
        real(fd, op)

    monkeypatch.setattr(fcntl, "flock", replaced)
    with fsutil.claim_dir(p) as held:
        assert not held


def test_a_full_disk_part_way_through_an_add_leaves_no_images(ws, monkeypatch):
    """Storing the third image of a chunk fails (ENOSPC): the two stored before it, named by no
    version, go again, shared copies too."""
    t = red_task(ws, n=1)
    before = sorted(p.name for p in (t.path / "images").iterdir())
    real, calls = store.Task._store_image, []  # pylint: disable=protected-access

    def filling(self, file, data):
        calls.append(file)
        if len(calls) == 3:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(self, file, data)

    monkeypatch.setattr(store.Task, "_store_image", filling)
    with pytest.raises(OSError, match="No space"):
        t.add_examples([(png(GREEN, noise=10, seed=i), "green", "") for i in range(5)])
    assert sorted(p.name for p in (t.path / "images").iterdir()) == before
    assert ws.gc_blobs() == 0


def test_gc_removes_files_half_written_by_a_crash(ws):
    """Temporary files a crashed write left in a task (task.json, an image, a kept version's side
    file) are removed by gc."""
    t = red_task(ws, n=1)
    t.upsert_class("x")
    t.undo()
    stale = [t.path / ".task.json.abc.tmp", t.path / "images" / ".x.png.abc.tmp",
             t.path / "history" / ".0000000001.label.abc.tmp", t.path / "history" / "redo" / ".x.tmp"]
    for f in stale:
        f.write_text("half")
    ws.gc()
    assert not [f for f in stale if f.exists()]
    assert t.class_names() == ["red"]


def test_undo_skips_steps_a_crash_left_that_change_nothing(ws):
    """A save killed after keeping the current version but before replacing task.json, or an undo
    killed after restoring a version but before forgetting it, leaves a step that would restore
    what is already there: undo skips it."""
    t = red_task(ws, n=1)
    t.upsert_class("a")
    t.upsert_class("b")
    rev = t.config.revision
    os.link(t.path / "task.json", t.path / "history" / f"{rev:010d}.json")  # the killed save's version
    t = Workspace(ws.root).get("t")
    t.undo()
    assert t.class_names() == ["red", "a"]
    newest = sorted((t.path / "history").glob("*.json"))[-1]
    kept = newest.read_bytes()
    t.undo()
    newest.write_bytes(kept)  # as if the undo was killed before forgetting the version it restored
    t = Workspace(ws.root).get("t")
    assert t.class_names() == ["red"]
    t.undo()  # skips the restored version
    assert t.class_names() == []
    with pytest.raises(ValueError, match="Nothing to undo"):
        t.undo()


def test_kept_versions_and_undo_are_durable(ws, monkeypatch):
    """The history folder is synced once a version is kept (before the save commits) and once
    undo forgets the version it restored."""
    t = red_task(ws, n=1)
    synced: list[object] = []
    monkeypatch.setattr(history, "_fsync_dir", synced.append)
    t.upsert_class("a")
    assert t.path / "history" in synced
    synced.clear()
    t.undo()
    assert synced.count(t.path / "history") == 1  # the redo version goes in history/redo
    assert t.path / "history" / "redo" in synced


def test_new_files_get_the_umask_not_0600(ws, monkeypatch, tmp_path):
    """tempfile makes files 0600; the store's files get 0666 less the umask, so a group can share
    a workspace."""
    monkeypatch.setattr(fsutil, "_UMASK", 0o002)
    t = red_task(ws, n=0)
    t.add_example(png(RED), "red")
    for f in (t.path / "task.json", t.path / t.config.examples[0].file):
        assert f.stat().st_mode & 0o777 == 0o664
    fsutil.write_atomic(tmp_path / "x", b"data")
    assert (tmp_path / "x").stat().st_mode & 0o777 == 0o664

    def refused(*_):
        raise PermissionError("not allowed")

    monkeypatch.setattr(os, "fchmod", refused)
    with pytest.raises(PermissionError):
        fsutil.write_atomic(tmp_path / "y", b"data")
    assert sorted(p.name for p in tmp_path.iterdir() if p.name != "ws") == ["x"]


def test_a_full_disk_part_way_through_an_inbox_add_leaves_no_images(ws, monkeypatch):
    """As for examples: storing the third inbox image fails, and the two stored before it, named
    by no version, go again (shared copies too)."""
    t = red_task(ws, n=1)
    before = sorted(p.name for p in (t.path / "images").iterdir())
    blobs = sorted(ws.blobs.glob("*/*"))
    real, calls = store.Task._store_image, []  # pylint: disable=protected-access

    def filling(self, file, data):
        calls.append(file)
        if len(calls) == 3:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(self, file, data)

    monkeypatch.setattr(store.Task, "_store_image", filling)
    with pytest.raises(OSError, match="No space"):
        t.add_to_inbox([(png(GREEN, noise=10, seed=i), "") for i in range(5)])
    assert sorted(p.name for p in (t.path / "images").iterdir()) == before
    assert sorted(ws.blobs.glob("*/*")) == blobs
    assert t.config.inbox == []
