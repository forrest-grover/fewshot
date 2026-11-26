"""The shared image store (blobs/): one copy per workspace hard-linked into tasks, fallbacks and
clean-up."""

import os
from pathlib import Path

import pytest

from fewshot import blobs as blobs_mod
from fewshot import history, store
from fewshot.fsutil import create_exclusive
from fewshot.models import TaskConfig
from tests.conftest import RED, png

pytestmark = pytest.mark.posix_fs  # also run on macOS: see .github/workflows/ci.yml


def test_images_are_stored_once_across_tasks(ws, monkeypatch):
    """The same image in two tasks is one blob hard-linked into both; it goes with the last link."""
    a = ws.create("a", embedder="fake:color")
    b = ws.create("b", embedder="fake:color")
    ea = a.add_example(png(RED), "red")
    b.add_example(png(RED), "red")
    blob = ws.blobs / ea.id[:2] / f"{ea.id}.png"
    assert blob.stat().st_nlink == 3
    assert os.path.samefile(blob, a.path / ea.file)
    ws.delete("a")
    assert blob.stat().st_nlink == 2
    b.delete_example(ea.id)
    monkeypatch.setattr(history, "HISTORY_KEPT", 0)
    b.upsert_class("x")  # prunes the version holding the image
    assert not blob.exists()


def test_duplicate_links_the_images_instead_of_reading_them(ws, monkeypatch):
    """A duplicate's images are more hard links to the source's files (and so to the shared
    copies), made without reading them; without hard links they are copied."""
    t = ws.create("t", embedder="fake:color")
    e = t.add_example(png(RED), "red")
    old = t.add_example(png(RED, noise=5, seed=1), "red")
    t.path.joinpath(old.file).unlink()  # a task-local file (no shared copy) is linked too
    t.path.joinpath(old.file).write_bytes(png(RED, noise=5, seed=1))
    real_read = Path.read_bytes

    def no_reads(p):
        assert "images" not in p.parts, f"read {p}"
        return real_read(p)

    monkeypatch.setattr(Path, "read_bytes", no_reads)
    c = ws.duplicate("t", "c")
    assert os.path.samefile(c.path / e.file, ws.blobs / e.id[:2] / f"{e.id}.png")
    assert os.path.samefile(c.path / old.file, t.path / old.file)
    monkeypatch.setattr(Path, "read_bytes", real_read)
    blobs_mod.link_image(c.path, e.file, t.path / old.file, ws.blobs, set())  # there already: kept
    assert os.path.samefile(c.path / e.file, t.path / e.file)

    def no_links(*_):
        raise OSError("not supported")

    monkeypatch.setattr(os, "link", no_links)
    d = ws.duplicate("t", "d")
    assert (d.path / e.file).read_bytes() == png(RED)
    assert (d.path / e.file).stat().st_nlink == 1


def test_store_falls_back_to_copies(ws, monkeypatch):
    """Without hard links (or if the blob vanishes), the task gets its own copy."""

    def no_links(*_):
        raise OSError("not supported")

    monkeypatch.setattr(os, "link", no_links)
    t = ws.create("t", embedder="fake:color")
    e = t.add_example(png(RED), "red")
    assert (t.path / e.file).read_bytes() == png(RED)
    assert (t.path / e.file).stat().st_nlink == 1


def test_store_link_race(ws, monkeypatch):
    """If another writer links the same file first, that copy is kept."""
    t = ws.create("t", embedder="fake:color")
    real_link = os.link

    def racing_link(src, dst):
        real_link(src, dst)
        if "images" in str(dst):  # not the history snapshot
            raise FileExistsError(dst)

    monkeypatch.setattr(os, "link", racing_link)
    e = t.add_example(png(RED), "red")
    assert (t.path / e.file).read_bytes() == png(RED)


def test_gc_blobs(ws):
    """gc_blobs removes blobs that no task links to."""
    assert ws.gc_blobs() == 0
    t = ws.create("t", embedder="fake:color")
    e = t.add_example(png(RED), "red")
    (t.path / e.file).unlink()
    (ws.blobs / "zz").mkdir()
    assert ws.gc_blobs() == 1
    assert ws.gc_blobs() == 0


def test_without_a_blob_store(tmp_path):
    """A Task made without a workspace store writes plain files."""
    p = tmp_path / "t"
    p.mkdir()
    t = store.Task(p, TaskConfig(name="t", embedder="fake:color"))
    t._save()  # pylint: disable=protected-access
    e = t.add_example(png(RED), "red")
    t.delete_example(e.id)
    assert (p / e.file).exists()  # kept for undo
    t._drop_file(e.file)  # pylint: disable=protected-access
    assert not (p / e.file).exists()


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


def test_short_legacy_ids_stay_out_of_the_shared_store(ws):
    """An image with a 16-character id (older tasks) is copied, not shared through the store."""
    t = ws.create("t", embedder="fake:color")
    assert t._blob("images/" + "ab" * 8 + ".png") is None  # pylint: disable=protected-access
    assert t._blob("images/" + "ab" * 16 + ".png") is not None  # pylint: disable=protected-access
