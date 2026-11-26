"""Crash safety and clean-up: interrupted renames, creates and deletes, failed adds, and gc."""

import os
import shutil
import time

import pytest
from typer.testing import CliRunner

from fewshot import archive, cli, history, store
from fewshot.store import Workspace
from tests.conftest import GREEN, RED, png, red_task


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
