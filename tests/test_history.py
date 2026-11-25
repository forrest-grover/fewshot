"""Undo history, the shared image store, renaming, export/import, tags and notes."""

import io
import json
import os
import zipfile

import pytest

from fewshot import store
from fewshot.archive import export_task, import_task
from fewshot.store import Example, TaskConfig, Workspace, check_tag, check_template, prompts
from tests.conftest import png

RED, GREEN, BLUE = (220, 30, 30), (30, 220, 30), (30, 30, 220)


def _task(ws, name="t", n=2):
    t = ws.create(name, embedder="fake:color")
    for i in range(n):
        t.add_example(png(RED, noise=10, seed=i), "red")
        t.add_example(png(GREEN, noise=10, seed=i), "green")
    return t


# ---- undo ---------------------------------------------------------------
def test_undo_steps_back_through_changes(ws):
    """Each undo restores the version before the last change of any kind; the revision keeps rising."""
    t = _task(ws)
    rev = t.config.revision
    t.update_settings(classifier="knn")
    t.upsert_class("blue", "a blue thing")
    assert t.undo() == rev + 1
    assert t.class_names() == ["red", "green"]
    assert t.config.classifier == "knn"
    assert t.undo() == rev
    assert t.config.classifier == "prototype"
    assert t.config.revision == rev + 4  # two changes and two undos
    assert Workspace(ws.root).get("t").config.classifier == "prototype"  # saved


def test_undo_brings_back_deleted_images(ws):
    """Deleted examples come back with their image files, which stay until no kept version needs them."""
    t = _task(ws)
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


def test_undo_deletes_images_the_undone_change_added(ws):
    """Undoing an add deletes the new image file, unless an older kept version still uses it."""
    t = _task(ws, n=1)
    new = t.add_example(png(BLUE), "blue")
    t.undo()
    assert not (t.path / new.file).exists()
    first = t.config.examples[0]
    t.delete_example(first.id)  # v1 has it, v2 doesn't
    t.add_example(png(RED, noise=10, seed=0), "red")  # back again in v3
    t.undo()  # back to v2: the file is still named by the kept v1
    assert (t.path / first.file).exists()


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
    with pytest.raises(ValueError, match="1 image files are gone"):
        t.undo()
    snaps = sorted((t.path / "history").glob("*.json"))
    snaps[-1].write_text("{broken")
    with pytest.raises(ValueError, match="is invalid"):
        t.undo()
    assert t.config.examples == []


def test_history_is_pruned_and_old_images_deleted(ws, monkeypatch):
    """Only HISTORY_KEPT versions are kept; an image goes once the last version naming it does,
    but not if a later version drops it again (that one deletes it later)."""
    monkeypatch.setattr(store, "HISTORY_KEPT", 2)
    t = _task(ws, n=1)
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


def test_history_survives_files_read_without_validation(ws):
    """A kept version that can't be parsed names no files (it is still listed for undo)."""
    t = _task(ws, n=1)
    t.upsert_class("x")
    (t.path / "history" / "0000000099.json").write_text("not json")
    assert t._files_in(t.path / "history" / "0000000099.json") == set()  # pylint: disable=protected-access


# ---- shared image store -------------------------------------------------
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
    monkeypatch.setattr(store, "HISTORY_KEPT", 0)
    b.upsert_class("x")  # prunes the version holding the image
    assert not blob.exists()


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
    t = store.Task.new(p, TaskConfig(name="t", embedder="fake:color"))
    e = t.add_example(png(RED), "red")
    t.delete_example(e.id)
    assert (p / e.file).exists()  # kept for undo
    t._drop_file(e.file)  # pylint: disable=protected-access
    assert not (p / e.file).exists()


# ---- rename -------------------------------------------------------------
def test_rename_task(ws):
    """Renaming moves the folder and keeps examples, history and links; the old name is gone."""
    t = _task(ws)
    t.upsert_class("x")
    r = ws.rename("t", "u")
    assert r is t
    assert t.name == "u"
    assert t.path == ws.root / "tasks" / "u"
    assert len(t.config.examples) == 4
    assert t.image_path(t.config.examples[0].id).exists()
    with pytest.raises(KeyError):
        ws.get("t")
    assert Workspace(ws.root).get("u").name == "u"
    t.undo()  # history from before the rename still works, under the new name
    assert t.class_names() == ["red", "green"]
    assert t.name == "u"
    assert ws.rename("u", "u") is t


def test_rename_errors(ws, monkeypatch):
    """A bad or taken new name is a ValueError; a failed save puts the folder back."""
    _task(ws)
    ws.create("other")
    with pytest.raises(ValueError, match="already exists"):
        ws.rename("t", "other")
    with pytest.raises(ValueError, match="Task names use"):
        ws.rename("t", "bad name")
    with pytest.raises(KeyError):
        ws.rename("nope", "x")
    real_replace = os.replace
    calls = []

    def failing_replace(src, dst):
        calls.append(dst)
        if str(dst).endswith("task.json"):
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        ws.rename("t", "u")
    monkeypatch.setattr(os, "replace", real_replace)
    assert ws.get("t").name == "t"
    assert not (ws.root / "tasks" / "u").exists()


# ---- export and import --------------------------------------------------
def _export(ws, name="t"):
    buf = io.BytesIO()
    n = export_task(ws, name, buf)
    buf.seek(0)
    return buf, n


def test_export_and_import(ws, tmp_path):
    """A task exported to a zip imports into another workspace with its settings, labels and images."""
    t = _task(ws)
    t.update_settings(classifier="knn", reject_below=0.5, flip=True)
    t.upsert_class("blue", "a photo of\na blue thing")
    t.annotate([t.config.examples[0].id], tags=["hard"], note="blurry")
    (t.path / t.config.examples[1].file).unlink()  # gone from disk: left out of the export
    buf, n = _export(ws)
    assert n == 3
    other = Workspace(tmp_path / "other")
    u = import_task(other, buf)
    assert u.name == "t"
    assert u.config.classifier == "knn"
    assert u.config.flip
    assert u.config.uid != t.config.uid
    assert u.config.revision == 1
    assert [c.name for c in u.config.classes] == ["red", "green", "blue"]
    assert u.config.examples[0].tags == ("hard",)
    assert len(u.config.examples) == 3
    for e in u.config.examples:
        assert u.image_path(e.id).read_bytes() == t.image_path(e.id).read_bytes()
    assert not u.can_undo()
    buf.seek(0)
    assert import_task(other, buf, name="copy").name == "copy"
    buf.seek(0)
    with pytest.raises(ValueError, match="already exists"):
        import_task(other, buf)


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    buf.seek(0)
    return buf


def test_import_rejects_bad_zips(ws):
    """Bad archives are refused with a ValueError and leave no task folder behind."""
    img = png(RED)
    eid = store.image_id(img)
    good = {"name": "t", "classes": [{"name": "red"}],
            "examples": [{"id": eid, "label": "red", "file": f"images/{eid}.png"}]}
    cases = {
        "Not a task export: File is not a zip": io.BytesIO(b"nope"),
        "no task.json": _zip({"x.txt": "hi"}),
        "task.json is invalid": _zip({"task.json": "{}"}),
        "is missing": _zip({"task.json": json.dumps(good)}),
        "doesn't match its id": _zip({"task.json": json.dumps(good), f"images/{eid}.png": png(BLUE)}),
        "reject_below must be": _zip({"task.json": json.dumps({**good, "reject_below": 3})}),
    }
    for msg, src in cases.items():
        with pytest.raises(ValueError, match=msg):
            import_task(ws, src)
    assert not list((ws.root / "tasks").iterdir())


def test_import_fixes_up_labels_and_duplicates(ws):
    """Examples whose class is missing get it added; repeated ids are kept once."""
    img = png(RED)
    eid = store.image_id(img)
    ex = {"id": eid, "label": "red", "file": f"images/{eid}.png"}
    t = import_task(ws, _zip({"task.json": json.dumps({"name": "t", "examples": [ex, ex]}),
                              f"images/{eid}.png": img}))
    assert t.class_names() == ["red"]
    assert len(t.config.examples) == 1


def test_import_refuses_oversized_members(ws, monkeypatch):
    """Members unpacking beyond the limit are refused."""
    import fewshot.archive  # pylint: disable=import-outside-toplevel

    img = png(RED)
    eid = store.image_id(img)
    zip_ = _zip({"task.json": json.dumps({"name": "t", "examples": [
        {"id": eid, "label": "red", "file": f"images/{eid}.png"}]}), f"images/{eid}.png": img})
    monkeypatch.setattr(fewshot.archive, "MAX_IMPORT_IMAGE", 10)
    with pytest.raises(ValueError, match="larger than 0 MB"):
        import_task(ws, zip_)


def test_import_cleans_up_on_failure(ws, monkeypatch):
    """If writing the new task fails, its folder is removed."""
    _task(ws)
    buf, _ = _export(ws)

    def broken(*_args, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(store.Task, "_store_image", broken)
    with pytest.raises(OSError, match="disk full"):
        import_task(ws, buf, name="copy")
    assert not (ws.root / "tasks" / "copy").exists()


# ---- tags, notes and lookups --------------------------------------------
def test_tags_and_notes(ws):
    """Tags can be replaced, added and removed (kept unique, in order); notes set and cleared."""
    t = _task(ws)
    a, b = (e.id for e in t.config.examples[:2])
    assert t.annotate([a, b], add_tags=["hard", " blurry ", "hard"]) == 2
    assert t.example(a).tags == ("hard", "blurry")
    t.annotate([a], remove_tags=["hard"], note="  check this  ")
    assert t.example(a).tags == ("blurry",)
    assert t.example(a).note == "check this"
    t.annotate([b], tags=["x"])
    assert t.example(b).tags == ("x",)
    assert [e.id for e in t.find_examples(tag="x")] == [b]
    assert [e.id for e in t.find_examples(label="green", tag="hard")] == []
    assert len(t.find_examples()) == 4
    with pytest.raises(KeyError):
        t.annotate(["nope"], tags=[])
    with pytest.raises(ValueError, match="Note is longer"):
        t.annotate([a], note="x" * (store.MAX_NOTE + 1))
    assert Workspace(ws.root).get("t").example(a).note == "check this"


def test_check_tag_and_example_validation():
    """Tags must be short, non-empty and free of commas; a file must be named by its example id."""
    assert check_tag(" ok ") == "ok"
    for bad, msg in (("", "empty"), ("x" * 51, "longer"), ("a,b", "can't contain"), ("a\nb", "can't contain")):
        with pytest.raises(ValueError, match=msg):
            check_tag(bad)
    eid = "ab" * 16
    with pytest.raises(ValueError, match="not named by the example id"):
        Example(id=eid, label="x", file=f"images/{'cd' * 16}.png")
    with pytest.raises(ValueError, match="note is longer"):
        Example(id=eid, label="x", file=f"images/{eid}.png", note="x" * 3000)
    assert Example(id=eid, label="x", file=f"images/{eid}.png", tags=("a", "a")).tags == ("a",)


def test_resolve_ids(ws):
    """Unique prefixes of 4+ characters resolve to full ids."""
    t = _task(ws)
    eid = t.config.examples[0].id
    assert t.resolve_ids([eid[:6], eid]) == [eid, eid]
    with pytest.raises(KeyError, match="No example 'abc'"):
        t.resolve_ids(["abc"])
    with pytest.raises(KeyError, match="No example"):
        t.resolve_ids(["zzzzzz"])
    twins = [Example(id=i, label="red", file=f"images/{i}.png") for i in ("abcd" + "0" * 28, "abcd" + "1" * 28)]
    t.config = t.config.model_copy(update={"examples": twins})  # in memory only
    with pytest.raises(KeyError, match="ambiguous"):
        t.resolve_ids(["abcd"])


def test_prompts_and_templates(ws):
    """Descriptions hold one prompt per line; multi-line templates fill each line."""
    assert prompts(" a \n\n b ") == ["a", "b"]
    check_template("a {}\nthe {}")
    for bad in ("", "no braces", "a {}\nno braces"):
        with pytest.raises(ValueError, match="must contain"):
            check_template(bad)
    t = _task(ws)
    t.describe_classes("a photo of a {}\n\n  a {} thing ")
    assert t.config.classes[0].description == "a photo of a red\na red thing"


def test_zero_shot_threshold_and_flip_settings(ws):
    """The zero-shot threshold is range-checked like reject_below; flip is a plain switch."""
    t = ws.create("t", embedder="fake:color")
    t.update_settings(reject_below_zero_shot=0.25, flip=True)
    assert (t.config.reject_below_zero_shot, t.config.flip) == (0.25, True)
    with pytest.raises(ValueError, match="reject_below_zero_shot must be between"):
        t.update_settings(reject_below_zero_shot=2)
    assert t.summary()["revision"] == t.config.revision
