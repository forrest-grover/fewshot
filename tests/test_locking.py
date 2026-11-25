"""Several processes sharing one workspace: file locking, reloading and atomic writes.

Two ``Workspace`` objects on the same root stand in for two processes: each has its own
``Task`` objects and opens its own lock file descriptor, and ``flock`` locks held through
different descriptors exclude each other even within one process.
"""

import fcntl
import json
import multiprocessing
import os
import shutil
import threading

import pytest

from fewshot import fsutil
from fewshot.store import Workspace
from tests.conftest import png

RED = (220, 30, 30)


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
    assert len(os.listdir(t.path / "history")) == 1  # just the version before "one"


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
    """A writer that waited while the task was deleted and re-created locks the new lock file."""
    a, b = pair
    tb = b.get("t")
    lock = a.root / "tasks" / "t" / ".lock"
    started, done = threading.Event(), threading.Event()

    def write():
        started.set()
        tb.upsert_class("late")
        done.set()

    with fsutil.file_lock(lock):  # pylint: disable=protected-access
        th = threading.Thread(target=write)
        th.start()
        started.wait()
        assert not done.wait(0.2)  # blocked on our lock
        shutil.rmtree(lock.parent)
        a.create("t")
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
