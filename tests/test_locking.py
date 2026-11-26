"""Several processes sharing one workspace: file locking, reloading and atomic writes.

Two ``Workspace`` objects on the same root stand in for two processes: each has its own
``Task`` objects and opens its own lock file descriptor, and ``flock`` locks held through
different descriptors exclude each other even within one process.
"""

import contextlib
import errno
import fcntl
import io
import json
import multiprocessing
import os
import shutil
import threading

import numpy as np
import pytest

from fewshot import fsutil
from fewshot.archive import export_task
from fewshot.cache import EmbeddingCache, prune_unused
from fewshot.fsutil import write_atomic
from fewshot.store import Workspace
from tests.conftest import RED, png


@pytest.fixture(name="pair")
def pair_fixture(tmp_path):
    """Two independent workspaces on one root, with task "t" created by the first."""
    a, b = Workspace(tmp_path / "ws"), Workspace(tmp_path / "ws")
    a.create("t")
    return a, b


def on_disk(ws, name="t"):
    """The task's task.json as currently saved."""
    return json.loads((ws.root / "tasks" / name / "task.json").read_text())


def test_writers_do_not_lose_each_others_changes(pair):
    """Each process's change is applied on top of the other's, not over it."""
    a, b = pair
    ta, tb = a.get("t"), b.get("t")
    ta.upsert_class("one")
    tb.upsert_class("two")  # tb's in-memory copy predates "one"
    ta.update_settings(text_weight=0.5)
    assert [c["name"] for c in on_disk(a)["classes"]] == ["one", "two"]
    assert on_disk(a)["text_weight"] == 0.5
    assert on_disk(a)["revision"] == 4
    assert b.get("t").class_names() == ["one", "two"]
    assert b.get("t").config.text_weight == 0.5


def test_get_sees_other_process_changes_and_deletes(pair):
    """A cached task is reloaded when task.json changes, and forgotten when it is deleted."""
    a, b = pair
    tb = b.get("t")
    assert not tb.refresh()
    a.get("t").upsert_class("one")
    assert b.get("t") is tb
    assert tb.class_names() == ["one"]
    a.delete("t")
    with pytest.raises(KeyError, match="No task 't'"):
        b.get("t")
    with pytest.raises(KeyError, match="No task 't'"):
        tb.upsert_class("two")  # a stale handle can't write a deleted task back
    assert not (a.root / "tasks" / "t").exists()
    with pytest.raises(KeyError, match="No task 't'"):
        b.delete("t")


def test_recreated_task_gets_a_new_uid(pair):
    """A task deleted and re-created elsewhere reloads with a different uid (so fits aren't reused)."""
    a, b = pair
    tb = b.get("t")
    uid = tb.config.uid
    a.delete("t")
    a.create("t")
    assert b.get("t") is tb
    assert tb.config.uid != uid


def test_concurrent_threads_across_workspaces(pair):
    """Many threads in two "processes" adding classes at once: every class is kept."""
    a, b = pair
    names = [f"c{i}" for i in range(40)]

    def add(ws, name):
        ws.get("t").upsert_class(name)

    threads = [threading.Thread(target=add, args=(a if i % 2 else b, n)) for i, n in enumerate(names)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sorted(c["name"] for c in on_disk(a)["classes"]) == sorted(names)
    assert on_disk(a)["revision"] == 1 + len(names)


def _add_classes(root, prefix, n):
    ws = Workspace(root)
    for i in range(n):
        ws.get("t").upsert_class(f"{prefix}{i}")


def test_concurrent_processes(tmp_path):
    """Real processes writing the same task keep every change."""
    root = tmp_path / "ws"
    Workspace(root).create("t")
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_add_classes, args=(root, p, 15)) for p in "abc"]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    cfg = Workspace(root).get("t").config
    assert len(cfg.classes) == 45
    assert cfg.revision == 46


def test_failed_save_rolls_back_and_cleans_up(pair, monkeypatch):
    """If writing task.json fails, memory matches disk again and no temp file is left behind."""
    a, _ = pair
    t = a.get("t")
    t.upsert_class("one")

    def broken_replace(*_):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        t.upsert_class("two")
    assert t.class_names() == ["one"]
    assert t.config.revision == 2
    assert sorted(os.listdir(t.path)) == [".lock", "history", "images", "task.json"]
    assert sorted(os.listdir(t.path / "history")) == ["0000000001.json", "0000000001.label"]  # before "one"


def test_delete_files_after_saving(pair, monkeypatch):
    """Image files are removed only once task.json no longer lists them."""
    a, _ = pair
    t = a.get("t")
    t.add_example(png(RED, noise=15, seed=1), "red")
    t.add_example(png(RED, noise=15, seed=2), "red")
    image = t.path / t.config.examples[0].file

    def broken_replace(*_):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        t.delete_example(t.config.examples[0].id)
    with pytest.raises(OSError, match="disk full"):
        t.delete_class("red")
    assert image.exists()
    assert len(t.config.examples) == 2


def test_waiting_writer_follows_a_recreated_task(pair):
    """A writer that waited while the task was deleted and re-created locks the new lock file:
    once the old file's lock is free, it waits for whoever holds the new one."""
    a, b = pair
    tb = b.get("t")
    lock = a.root / "tasks" / "t" / ".lock"
    started, done = threading.Event(), threading.Event()

    def write():
        started.set()
        tb.upsert_class("late")
        done.set()

    with contextlib.ExitStack() as old:
        old.enter_context(fsutil.file_lock(lock))
        th = threading.Thread(target=write)
        th.start()
        started.wait()
        assert not done.wait(0.2)  # blocked on our lock
        shutil.rmtree(lock.parent)
        a.create("t")
        with fsutil.file_lock(lock):  # the new task's lock, held by "another process"
            old.close()  # the deleted file's lock is free now
            assert not done.wait(0.5), "the writer went ahead with the lock of the deleted task"
    th.join()
    assert [c["name"] for c in on_disk(a)["classes"]] == ["late"]


def test_get_rejects_non_task_paths(tmp_path):
    """Names that aren't task folders are unknown tasks, not errors."""
    ws = Workspace(tmp_path / "ws")
    (ws.root / "tasks" / "file").write_text("")
    (ws.root / "tasks" / "empty").mkdir()
    for name in ("file", "empty", "../ws"):
        with pytest.raises(KeyError):
            ws.get(name)


def test_readers_keep_a_stable_snapshot(pair):
    """A config object obtained before a mutation is never changed by it."""
    a, _ = pair
    t = a.get("t")
    t.add_example(png(RED, noise=15, seed=1), "red")
    snapshot = t.config
    t.add_example(png(RED, noise=15, seed=2), "blue")
    t.relabel(snapshot.examples[0].id, "blue")
    assert [e.label for e in snapshot.examples] == ["red"]
    assert [e.label for e in t.config.examples] == ["blue", "blue"]

    t.update_settings(classifier="knn", classifier_params={"k": 3})
    snapshot = t.config
    t.upsert_class("blue", "a blue thing")
    with t._writing():  # pylint: disable=protected-access  # an in-place edit, as mutations make
        t.config.classifier_params["k"] = 5
        t._save()  # pylint: disable=protected-access
    assert [c.description for c in snapshot.classes] == ["", ""]
    assert snapshot.classifier_params == {"k": 3}
    assert t.config.classifier_params == {"k": 5}


def test_lock_file_is_closed_when_locking_fails(tmp_path, monkeypatch):
    """An error while taking the lock doesn't leak the descriptor."""
    closed = []
    real_close = os.close

    def close(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(os, "close", close)

    def broken_flock(*_):
        raise OSError("no locks here")

    monkeypatch.setattr(fcntl, "flock", broken_flock)
    with pytest.raises(OSError, match="no locks here"), fsutil.file_lock(tmp_path / ".lock"):  # pylint: disable=protected-access
        pass  # pragma: no cover
    assert len(closed) == 1


def test_write_atomic_cleans_up(tmp_path, monkeypatch):
    """A failed atomic write leaves neither the file nor a temporary file."""
    def broken(*_):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        write_atomic(tmp_path / "x", b"data")
    assert not list(tmp_path.iterdir())


# ---- reading while holding a task still ------------------------------------
def test_locked_readers_share_the_lock_and_writers_wait(pair):
    """Readers in ``Task.locked`` don't wait for each other; a change waits for them all."""
    a, b = pair
    ta, tb = a.get("t"), b.get("t")
    done = threading.Event()

    def write():
        tb.upsert_class("x")
        done.set()

    with ta.locked() as cfg:
        with Workspace(a.root).get("t").locked():  # another reader gets in at once
            pass
        writer = threading.Thread(target=write)
        writer.start()
        assert not done.wait(0.2)
        assert cfg.classes == []
    writer.join(5)
    assert done.is_set()
    with ta.locked() as cfg:
        assert [c.name for c in cfg.classes] == ["x"]  # the latest saved version


def test_locked_inside_and_around_a_change(pair):
    """Within a change ``locked`` yields the draft; a change inside ``locked`` is refused (its
    exclusive lock would wait for our own shared one forever); a deleted task is a KeyError."""
    a, b = pair
    t = a.get("t")
    with t._writing():  # pylint: disable=protected-access
        t.config.classes = []
        with t.locked() as cfg:
            assert cfg is t.config
    with t.locked(), pytest.raises(RuntimeError, match="while reading it"):
        t.upsert_class("x")
    t.upsert_class("y")  # fine again afterwards
    b.delete("t")
    with pytest.raises(KeyError), t.locked():
        pass  # pragma: no cover


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anything")
def test_reads_need_no_write_access(tmp_path):
    """Export and a dry-run cache prune work in a workspace this user can't write to, with or
    without a lock file (a task never changed has none); changes still fail there."""
    ws = Workspace(tmp_path / "ws")
    ws.create("t").add_example(png(RED), "red")
    ws.create("fresh")
    EmbeddingCache(ws.cache_path).put_many("x", {"k": np.ones(1, np.float32)})
    folders = [p for p in ws.root.rglob("*") if p.is_dir()] + [ws.root]
    for p in folders:
        p.chmod(0o555)
    try:
        assert not (ws.root / "tasks" / "fresh" / ".lock").exists()
        ro = Workspace(ws.root, create=False)
        for name in ("t", "fresh"):
            export_task(ro, name, io.BytesIO())
        assert prune_unused(ro, dry_run=True)[0] == 1
        with pytest.raises(PermissionError):
            ro.get("fresh").upsert_class("x")
    finally:
        for p in folders:
            p.chmod(0o755)


def test_a_read_only_file_system_is_like_a_read_only_folder(tmp_path, monkeypatch):
    """EROFS when creating the lock file is treated as missing write access."""
    real = os.open

    def read_only(path, flags, *args):
        if flags & os.O_CREAT:
            raise OSError(errno.EROFS, "Read-only file system", str(path))
        return real(path, flags, *args)

    monkeypatch.setattr(os, "open", read_only)
    with fsutil.file_lock(tmp_path / ".lock", shared=True):
        pass
    with pytest.raises(PermissionError), fsutil.file_lock(tmp_path / ".lock"):
        pass  # pragma: no cover

    def broken(*_):
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(os, "open", broken)
    with pytest.raises(OSError, match="I/O error"), fsutil.file_lock(tmp_path / ".lock"):
        pass  # pragma: no cover
