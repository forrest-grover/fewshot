"""Crash safety and clean-up: damaged history, interrupted renames, creates, imports and deletes,
locked exports, one undo per bulk add, and the gc command."""

import io
import json
import os
import shutil
import threading
import time
import zipfile

import pytest
from typer.testing import CliRunner

from fewshot import archive, cli, history, store
from fewshot.archive import export_task, import_task
from fewshot.fsutil import create_exclusive
from fewshot.images import image_id
from fewshot.store import Workspace
from tests.conftest import png

RED, GREEN = (220, 30, 30), (30, 220, 30)


def _task(ws, name="t", n=2):
    t = ws.create(name, embedder="fake:color")
    for i in range(n):
        t.add_example(png(RED, noise=10, seed=i), "red")
    return t


def _age(path, seconds=3600):
    """Make ``path`` look untouched for ``seconds``."""
    old = time.time() - seconds
    os.utime(path, (old, old))


# ---- damaged history ----------------------------------------------------
def test_unreadable_gone_list_keeps_the_task_working(ws, monkeypatch):
    """A truncated .gone file (a crash while writing it) doesn't break later edits or deletion;
    its files are kept rather than lost."""
    monkeypatch.setattr(history, "HISTORY_KEPT", 1)
    t = _task(ws)
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
    t = _task(ws, n=1)
    t.upsert_class("a")
    t.upsert_class("b")
    newest = sorted((t.path / "history").glob("*.json"))[-1]
    newest.write_text("{broken")
    with pytest.raises(ValueError, match="is invalid and was removed; undo again"):
        t.undo()
    t.undo()
    assert t.class_names() == ["red"]


# ---- interrupted operations ---------------------------------------------
def test_interrupted_rename_leaves_a_usable_task(ws):
    """If a rename moved the folder but didn't rewrite task.json, the task opens under the new name."""
    _task(ws, "old")
    os.rename(ws.root / "tasks" / "old", ws.root / "tasks" / "new")
    fresh = Workspace(ws.root)
    assert [t.name for t in fresh.list_tasks()] == ["new"]
    assert len(fresh.get("new").config.examples) == 2


def test_leftover_folder_from_a_crash_frees_its_name(ws):
    """A task folder without task.json blocks its name only while it may still be being made."""
    (ws.root / "tasks" / "x").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        ws.create("x")
    _age(ws.root / "tasks" / "x")
    assert ws.create("x").name == "x"


def test_delete_moves_the_folder_aside_first(ws, monkeypatch):
    """A delete interrupted after the move leaves a hidden .trash- folder, which listing ignores."""
    _task(ws)

    def crash(path, *args, **kw):
        raise OSError("power cut")

    monkeypatch.setattr(shutil, "rmtree", crash)
    with pytest.raises(OSError, match="power cut"):
        ws.delete("t")
    leftovers = [p.name for p in (ws.root / "tasks").iterdir()]
    assert len(leftovers) == 1
    assert leftovers[0].startswith(".trash-")
    assert not Workspace(ws.root).list_tasks()


def test_failed_import_leaves_nothing(ws, monkeypatch):
    """An import that fails part-way removes its staging folder and frees the name."""
    _task(ws)
    buf = io.BytesIO()
    export_task(ws, "t", buf)
    buf.seek(0)

    def broken(*_args, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(store.Task, "_save", broken)
    with pytest.raises(OSError, match="disk full"):
        import_task(ws, buf, name="copy")
    assert [p.name for p in (ws.root / "tasks").iterdir()] == ["t"]


def test_failed_add_removes_its_new_images(ws, monkeypatch):
    """If saving task.json fails, the images that add stored are removed again (not ones that were
    there before)."""
    t = _task(ws, n=1)
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


# ---- undo, export and import --------------------------------------------
def test_one_undo_takes_back_a_whole_bulk_add(ws):
    """However many chunks an add_examples call is saved in, one undo removes all of it."""
    t = _task(ws, n=1)
    t.add_examples([(png(GREEN, noise=10, seed=i), "green", "") for i in range(5)], chunk=2)
    assert len(t.config.examples) == 6
    t.undo()
    assert len(t.config.examples) == 1
    assert len(list((t.path / "images").iterdir())) == 6  # kept for redo...
    t.upsert_class("x")  # ...until another change makes redo impossible
    assert len(list((t.path / "images").iterdir())) == 1


def test_export_holds_the_task_lock(ws, monkeypatch):
    """A rename from another process waits for an export in progress, so the zip is complete."""
    _task(ws, n=4)
    other = Workspace(ws.root)
    started, renamed = threading.Event(), threading.Event()
    real = zipfile.ZipFile.writestr

    def slow(self, name, data, *args, **kw):
        started.set()
        time.sleep(0.05)
        return real(self, name, data, *args, **kw)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", slow)

    def rename():
        started.wait()
        other.rename("t", "u")
        renamed.set()

    th = threading.Thread(target=rename)
    th.start()
    buf = io.BytesIO()
    assert export_task(ws, "t", buf) == 4
    th.join(5)
    assert renamed.is_set()
    with zipfile.ZipFile(buf) as z:
        assert len([n for n in z.namelist() if n.startswith("images/")]) == 4


def test_import_cleans_up_class_names(ws):
    """Duplicate or padded class names and padded labels are tidied on import."""
    img = png(RED)
    eid = image_id(img)
    cfg = {"name": "t", "classes": [{"name": "cat", "description": "first"}, {"name": " cat"}, {"name": "dog"}],
           "examples": [{"id": eid, "label": " dog ", "file": f"images/{eid}.png"}]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("task.json", json.dumps(cfg))
        z.writestr(f"images/{eid}.png", img)
    buf.seek(0)
    t = import_task(ws, buf)
    assert [(c.name, c.description) for c in t.config.classes] == [("cat", "first"), ("dog", "")]
    assert t.config.examples[0].label == "dog"
    assert t.summary()["counts"] == {"cat": 0, "dog": 1}


# ---- shared image store -------------------------------------------------
def test_no_unused_blob_without_hard_links(ws, monkeypatch):
    """Where hard links fail, the task gets a copy and no blob is left behind."""
    real = os.link

    def no_links(src, dst):
        if "blobs" in str(dst) or "images" in str(dst):
            raise OSError("not supported")
        real(src, dst)

    monkeypatch.setattr(os, "link", no_links)
    t = ws.create("t", embedder="fake:color")
    e = t.add_example(png(RED), "red")
    assert (t.path / e.file).read_bytes() == png(RED)
    assert not list(ws.blobs.glob("*/*"))


def test_blob_link_fails_after_creating_it(ws, monkeypatch):
    """If the blob was made but linking it into the task fails, the blob is removed again."""
    real = os.link

    def link(src, dst):
        if "images" in str(dst):
            raise OSError("cross-device link")
        real(src, dst)

    monkeypatch.setattr(os, "link", link)
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    assert not list(ws.blobs.glob("*/*"))


def test_create_exclusive(tmp_path):
    """create_exclusive never replaces an existing file."""
    assert create_exclusive(tmp_path / "a", b"one")
    assert not create_exclusive(tmp_path / "a", b"two")
    assert (tmp_path / "a").read_bytes() == b"one"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a"]


# ---- gc -----------------------------------------------------------------
def test_gc(ws, monkeypatch):
    """gc removes old leftover folders, image files no version names, and unused blobs; it keeps
    young leftovers (maybe in progress) and images undo may still need."""
    monkeypatch.setattr(history, "HISTORY_KEPT", 5)
    t = _task(ws, n=2)
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
    assert not store._abandoned(tmp_path / "gone")  # pylint: disable=protected-access
    assert archive.MAX_IMPORT_TOTAL > 0
