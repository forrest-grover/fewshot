"""Exporting a task to a zip and importing it (fewshot.archive): round trips, checks on what an
archive may contain, clean-up after failures, and what an import changes."""

import io
import json
import threading
import time
import zipfile

import pytest

import fewshot.api
from fewshot import archive, store
from fewshot.archive import export_task, import_task
from fewshot.embedders import DEFAULT_SPEC
from fewshot.images import image_id
from fewshot.store import Workspace
from fewshot.text import MAX_PROMPTS
from tests.conftest import BLUE, RED, host_client, png, red_blue_task, red_green_task, red_task, two_each, zip_of


# ---- export and import ---------------------------------------------------
def _export(ws, name="t"):
    buf = io.BytesIO()
    n, _ = export_task(ws, name, buf)
    buf.seek(0)
    return buf, n


def test_export_and_import(ws, tmp_path):
    """A task exported to a zip imports into another workspace with its settings, labels and images."""
    t = red_green_task(ws)
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


def test_export_holds_the_task_lock(ws, monkeypatch):
    """A rename from another process waits for an export in progress, so the zip is complete."""
    red_task(ws, n=4)
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
    assert export_task(ws, "t", buf) == (4, 0)
    th.join(5)
    assert renamed.is_set()
    with zipfile.ZipFile(buf) as z:
        assert len([n for n in z.namelist() if n.startswith("images/")]) == 4


@pytest.mark.parametrize("embedder", [None, "fake:color", "fake:other"])
def test_import_with_another_embedder_clears_fitted_settings(ws, embedder):
    """Importing under another embedder drops the old model's temperature and thresholds."""
    t = red_blue_task(ws)
    t.update_settings(temperature=2.0)
    t.set_class_thresholds({"red": 0.8})
    buf = io.BytesIO()
    export_task(ws, "t", buf)
    buf.seek(0)
    cfg = import_task(ws, buf, name="copy", embedder=embedder, embedder_ok=lambda _spec: None).config
    fitted = (cfg.temperature, cfg.classes[0].reject_below)
    assert fitted == ((None, None) if embedder == "fake:other" else (2.0, 0.8))


def test_import_task_checks_the_embedder_whoever_calls_it(ws, run, tmp_path):
    """archive.import_task checks the embedder itself, so the CLI can't skip it."""
    ws.embedder_ok = lambda spec: None  # made when it still loaded
    t = ws.create("t", embedder="nobackend:model")
    buf = io.BytesIO()
    export_task(ws, t.name, buf)
    (tmp_path / "t.zip").write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="nobackend"):
        import_task(ws, io.BytesIO(buf.getvalue()), name="copy")
    r = run("import-task", tmp_path / "t.zip", "--name", "copy")
    assert r.exit_code == 1
    assert "nobackend" in r.output
    assert run("import-task", tmp_path / "t.zip", "--name", "copy", "--embedder", "fake:color").exit_code == 0


def _task_zip(examples, files, **cfg):
    """An export of task 't' (class 'red') with ``examples`` in its task.json and ``files``."""
    return zip_of({"task.json": json.dumps({"name": "t", "classes": [{"name": "red"}], "examples": examples, **cfg}),
                   **files})


def _one(data, ext):
    eid = image_id(data)
    return _task_zip([{"id": eid, "label": "red", "file": f"images/{eid}.{ext}"}], {f"images/{eid}.{ext}": data})


def test_import_checks_the_embedder_first(ws):
    """embedder_ok sees the spec before anything is written, and can refuse it."""
    seen = []

    def refuse(spec):
        seen.append(spec)
        raise PermissionError("not allowed")

    with pytest.raises(PermissionError):
        import_task(ws, _one(png(RED), "png"), embedder_ok=refuse)
    assert seen == [DEFAULT_SPEC]
    assert not list((ws.root / "tasks").iterdir())


def test_api_import_applies_the_model_allowlist(tmp_path, monkeypatch):
    """Importing over the API can't bring in a model the server wouldn't otherwise load."""
    c = host_client(tmp_path, monkeypatch, hosts="*")
    loads: list[str] = []
    monkeypatch.setattr(fewshot.api, "check_embedder", loads.append)
    data = png(RED)
    eid = image_id(data)
    zip_ = _task_zip([{"id": eid, "label": "red", "file": f"images/{eid}.png"}], {f"images/{eid}.png": data},
                embedder="timm:hf-hub:attacker/model")
    r = c.post("/api/tasks/import", files={"file": ("t.zip", zip_.getvalue(), "application/zip")})
    assert r.status_code == 403
    assert "--allow-any-model" in r.json()["detail"]
    assert c.get("/api/tasks").json() == []
    assert not loads


# ---- what an archive may hold --------------------------------------------
def test_import_rejects_bad_zips(ws):
    """Bad archives are refused with a ValueError and leave no task folder behind."""
    img = png(RED)
    eid = image_id(img)
    good = {"name": "t", "classes": [{"name": "red"}],
            "examples": [{"id": eid, "label": "red", "file": f"images/{eid}.png"}]}
    cases = {
        "Not a task export: File is not a zip": io.BytesIO(b"nope"),
        "no task.json": zip_of({"x.txt": "hi"}),
        "task.json is invalid": zip_of({"task.json": "{}"}),
        "is missing": zip_of({"task.json": json.dumps(good)}),
        "doesn't match its id": zip_of({"task.json": json.dumps(good), f"images/{eid}.png": png(BLUE)}),
        "reject_below must be": zip_of({"task.json": json.dumps({**good, "reject_below": 3})}),
    }
    for msg, src in cases.items():
        with pytest.raises(ValueError, match=msg):
            import_task(ws, src)
    assert not list((ws.root / "tasks").iterdir())


def test_import_fixes_up_labels_and_duplicates(ws):
    """Examples whose class is missing get it added; repeated ids are kept once."""
    img = png(RED)
    eid = image_id(img)
    ex = {"id": eid, "label": "red", "file": f"images/{eid}.png"}
    t = import_task(ws, zip_of({"task.json": json.dumps({"name": "t", "examples": [ex, ex]}),
                              f"images/{eid}.png": img}))
    assert t.class_names() == ["red"]
    assert len(t.config.examples) == 1


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


def test_import_refuses_oversized_members(ws, monkeypatch):
    """Members unpacking beyond the limit are refused."""
    img = png(RED)
    eid = image_id(img)
    zip_ = zip_of({"task.json": json.dumps({"name": "t", "examples": [
        {"id": eid, "label": "red", "file": f"images/{eid}.png"}]}), f"images/{eid}.png": img})
    monkeypatch.setattr(archive, "MAX_IMPORT_IMAGE", 10)
    with pytest.raises(ValueError, match="larger than 0 MB"):
        import_task(ws, zip_)


def test_import_refuses_files_that_are_not_images(ws):
    """Every imported file must decode as an image with the extension of its real format."""
    with pytest.raises(ValueError, match="Not a readable image"):
        import_task(ws, _one(b"<script>alert(1)</script>", "html"))
    with pytest.raises(ValueError, match=r"is really a \.png image"):
        import_task(ws, _one(png(RED), "jpg"))
    assert import_task(ws, _one(png(RED), "png")).name == "t"


def test_import_stores_only_the_bytes_it_checked(ws, monkeypatch):
    """An image whose bytes differ when read again to be stored is refused, and nothing is left."""
    real = archive._read_member  # pylint: disable=protected-access
    reads = {"n": 0}

    def changing(z, info, limit=None):
        data = real(z, info, limit)
        if info.filename.startswith("images/"):
            reads["n"] += 1
            if reads["n"] > 1:  # the read that stores it, after the one that checked it
                return png(BLUE)
        return data

    monkeypatch.setattr(archive, "_read_member", changing)
    with pytest.raises(ValueError, match="changed while importing"):
        import_task(ws, _one(png(RED), "png"))
    assert not list((ws.root / "tasks").iterdir())


def test_import_total_size_limit(ws, monkeypatch):
    """The images together may not unpack beyond MAX_IMPORT_TOTAL."""
    monkeypatch.setattr(archive, "MAX_IMPORT_TOTAL", 10)
    with pytest.raises(ValueError, match="unpack to more than"):
        import_task(ws, _one(png(RED), "png"))
    assert not list((ws.root / "tasks").iterdir())


def _many(n, comment=b""):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(n):
            z.writestr(f"{i}", b"")
        z.comment = comment
    return buf.getvalue()


def test_import_counts_the_files_before_zipfile_reads_them(ws, monkeypatch):
    """A zip listing more than MAX_IMPORT_FILES files is refused before zipfile builds an object
    for each (about 6 times the zip's size in memory), Zip64 or not, whatever count it declares."""
    monkeypatch.setattr(archive, "MAX_IMPORT_FILES", 3)
    real, zips = zipfile.ZipFile, (_many(4), _many(4, comment=b"a comment"))
    monkeypatch.setattr(zipfile, "ZipFile", lambda *a, **kw: pytest.fail("zipfile read a zip with too many files"))
    for data in zips:
        with pytest.raises(ValueError, match="holds more than 3 files"):
            import_task(ws, io.BytesIO(data))
    lying = bytearray(zips[0])
    lying[-12:-10] = (1).to_bytes(2, "little")  # declares 1 file of 4
    with pytest.raises(ValueError, match="holds more than 3"):
        import_task(ws, io.BytesIO(bytes(lying)))
    monkeypatch.setattr(zipfile, "ZipFile", real)
    with pytest.raises(ValueError, match=r"no task\.json"):  # 3 is allowed
        import_task(ws, io.BytesIO(_many(3)))
    with pytest.raises(ValueError, match=r"no task\.json"):  # nothing at all
        import_task(ws, io.BytesIO(_many(0)))
    huge = bytearray(_many(3))
    huge[-10:-6] = (10**6).to_bytes(4, "little")  # a directory larger than the zip
    for bad in (b"x" * 100, bytes(huge)):
        with pytest.raises(ValueError, match="Not a task export"):
            import_task(ws, io.BytesIO(bad))
    damaged = bytearray(_many(3))
    damaged[damaged.index(b"PK\x01\x02") + 1] = 0  # zipfile reports the broken directory
    with pytest.raises(ValueError, match="Not a task export"):
        import_task(ws, io.BytesIO(bytes(damaged)))


def test_import_counts_the_files_of_a_zip64_archive(ws, monkeypatch):
    """Over 65,535 files a zip keeps its directory's size and place in Zip64 records."""
    data = _many(0x10000 + 1)
    assert b"PK\x06\x06" in data
    monkeypatch.setattr(archive, "MAX_IMPORT_FILES", 0x10000)
    with pytest.raises(ValueError, match="holds more than 65,536 files"):
        import_task(ws, io.BytesIO(data))
    broken = data.replace(b"PK\x06\x06", b"PK\x00\x00")  # the record the locator points at isn't one
    with pytest.raises(ValueError, match="Not a task export"):
        import_task(ws, io.BytesIO(broken))


def test_import_checks_class_thresholds(ws, tmp_path):
    """An export whose task.json gives a class a threshold outside -1..1 is refused."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    t.set_class_thresholds({"red": 0.5})
    zpath = tmp_path / "t.zip"
    with zpath.open("wb") as f:
        export_task(ws, "t", f)
    edited = tmp_path / "bad.zip"
    with zipfile.ZipFile(zpath) as src, zipfile.ZipFile(edited, "w") as dst:
        for item in src.infolist():
            data = src.read(item)
            if item.filename == "task.json":
                data = data.replace(b'"reject_below": 0.5', b'"reject_below": 5.0')
                assert b"5.0" in data
            dst.writestr(item, data)
    with pytest.raises(ValueError, match=r"threshold of 'red' must be between -1 and 1, got 5\.0"):
        import_task(ws, edited, name="t2")


def test_an_import_with_too_long_a_description_is_refused(ws):
    """Imports check descriptions like edits do."""
    t = two_each(ws)
    buf = io.BytesIO()
    export_task(ws, t.name, buf)
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as z, zipfile.ZipFile(out, "w") as w:
        for info in z.infolist():
            data = z.read(info)
            if info.filename == "task.json":
                data = data.replace(b'"description": ""', b'"description": "' + b"a\\n" * (MAX_PROMPTS + 1) + b'"', 1)
            w.writestr(info, data)
    with pytest.raises(ValueError, match="prompts"):
        import_task(ws, io.BytesIO(out.getvalue()), name="copy")


# ---- failures ------------------------------------------------------------
def test_import_cleans_up_on_failure(ws, monkeypatch):
    """If writing the new task fails, its folder is removed."""
    red_green_task(ws)
    buf, _ = _export(ws)

    def broken(*_args, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(store.Task, "_store_image", broken)
    with pytest.raises(OSError, match="disk full"):
        import_task(ws, buf, name="copy")
    assert not (ws.root / "tasks" / "copy").exists()


def test_failed_import_leaves_nothing(ws, monkeypatch):
    """An import that fails part-way removes its staging folder and frees the name."""
    red_task(ws)
    buf = io.BytesIO()
    export_task(ws, "t", buf)
    buf.seek(0)

    def broken(*_args, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(store.Task, "_save", broken)
    with pytest.raises(OSError, match="disk full"):
        import_task(ws, buf, name="copy")
    assert [p.name for p in (ws.root / "tasks").iterdir()] == ["t"]
