"""Workspace and Task: on-disk layout, validation and mutations, renaming, tags, notes and
descriptions."""

import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from fewshot import embedders, history, models, store
from fewshot.images import image_id, load_image
from fewshot.models import Example, TaskConfig
from fewshot.store import Workspace
from fewshot.text import MAX_DESCRIPTION, MAX_PROMPTS, check_class_name, check_tag, check_template, prompts
from tests.conftest import GREEN, RED, png, red_green_task, two_each

pytestmark = pytest.mark.posix_fs  # also run on macOS: see .github/workflows/ci.yml


def jpeg(size=(4, 2), orientation=None, mode="RGB"):
    """JPEG bytes, optionally tagged with an EXIF orientation."""
    im = Image.new(mode, size, (10, 20, 30) if mode == "RGB" else 128)
    exif = im.getexif()
    if orientation:
        exif[0x0112] = orientation
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def saved(task):
    """The task's config as currently written on disk."""
    return json.loads((task.path / "task.json").read_text())


# ---- module helpers ---------------------------------------------------
def test_default_root_uses_env_or_local_folder(monkeypatch, tmp_path):
    """$FEWSHOT_HOME wins; otherwise ./fewshot_data, always absolute."""
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path / "home"))
    assert store.default_root() == (tmp_path / "home").resolve()
    monkeypatch.delenv("FEWSHOT_HOME")
    monkeypatch.chdir(tmp_path)
    assert store.default_root() == (tmp_path / "fewshot_data").resolve()


def test_image_id_is_content_hash():
    """Same bytes give the same 32-hex (128-bit) id; different bytes differ."""
    a = image_id(b"abc")
    assert a == "ba7816bf8f01cfea414140de5dae2223"
    assert image_id(b"abc") == a
    assert image_id(b"abd") != a


def test_load_image_converts_to_upright_rgb():
    """Greyscale becomes RGB and EXIF rotation is applied."""
    grey = load_image(jpeg(mode="L"))
    assert grey.mode == "RGB"
    assert grey.size == (4, 2)
    rotated = load_image(jpeg(orientation=6))
    assert rotated.size == (2, 4)


@pytest.mark.parametrize("data", [b"", b"not an image", png(RED)[:30]])
def test_load_image_rejects_unreadable_bytes(data):
    """Garbage and truncated files raise ValueError, not a PIL error."""
    with pytest.raises(ValueError, match="Not a readable image"):
        load_image(data)


# ---- workspace --------------------------------------------------------
def test_workspace_default_root(monkeypatch, tmp_path):
    """Without a root the workspace lives in $FEWSHOT_HOME and creates tasks/."""
    monkeypatch.setenv("FEWSHOT_HOME", str(tmp_path / "home"))
    w = Workspace()
    assert w.root == (tmp_path / "home").resolve()
    assert (w.root / "tasks").is_dir()
    assert w.cache_path == w.root / "cache.sqlite"


def test_create_writes_task_folder(ws):
    """A new task has an images folder and a task.json at revision 1."""
    t = ws.create("pets", embedder="fake:color", classifier="knn", text_weight=0.5)
    assert t.name == "pets"
    assert t.path == ws.root / "tasks" / "pets"
    assert (t.path / "images").is_dir()
    assert not (t.path / "task.json.tmp").exists()
    on_disk = saved(t)
    assert on_disk["revision"] == 1
    assert on_disk["classifier"] == "knn"
    assert on_disk["text_weight"] == 0.5
    assert ws.get("pets") is t


@pytest.mark.parametrize("name", ["", "-x", "a b", "../x", "x" * 65, "é", "foo\n"])
def test_create_rejects_bad_names(ws, name):
    """Names must be safe folder names."""
    with pytest.raises(ValueError, match="Task names use"):
        ws.create(name)


def test_create_rejects_duplicates_and_unknown_classifier(ws):
    """Existing names and unknown classifiers fail without leaving a folder behind."""
    ws.create("a")
    with pytest.raises(ValueError, match="already exists"):
        ws.create("a")
    with pytest.raises(ValueError, match="Unknown classifier 'svm'"):
        ws.create("b", classifier="svm")
    assert not (ws.root / "tasks" / "b").exists()


def test_the_library_checks_a_chosen_embedder(ws, tmp_path):
    """A spec given to create, or a new one to update_settings, must load (whoever calls), so a
    typo fails there and not at the first prediction; the check is the workspace's to choose."""
    with pytest.raises(ValueError, match="Could not load embedder 'nobackend:typo'"):
        ws.create("x", embedder="nobackend:typo")
    assert not list((ws.root / "tasks").iterdir())
    t = ws.create("t", embedder="fake:color")
    with pytest.raises(ValueError, match="Could not load embedder 'bogus:zzz'"):
        t.update_settings(embedder="bogus:zzz")
    assert ws.get("t").config.embedder == "fake:color"
    ws.create("d")  # no embedder given: nothing to check (or load)
    checked: list[str] = []
    own = Workspace(tmp_path / "own", embedder_ok=checked.append)
    own.create("a", embedder="any:thing").update_settings(embedder="other:x", classifier="knn")
    Workspace(own.root, embedder_ok=checked.append).get("a").update_settings(embedder="other:y")
    assert checked == ["any:thing", "other:x", "other:y"]


def test_get_list_and_reload(ws):
    """get() caches, loads from disk in a fresh workspace, and rejects bad or missing names."""
    ws.create("b")
    ws.create("a")
    (ws.root / "tasks" / "not-a-task").mkdir()  # folder without task.json
    (ws.root / "tasks" / "stray.txt").write_text("x")
    assert [t.name for t in ws.list_tasks()] == ["a", "b"]
    fresh = Workspace(ws.root)
    loaded = fresh.get("a")
    assert loaded is not ws.get("a")
    assert loaded.config == ws.get("a").config
    assert fresh.get("a") is loaded
    for name in ["missing", "not-a-task", "../tasks/a"]:
        with pytest.raises(KeyError, match="No task"):
            fresh.get(name)


def test_delete_removes_folder(ws):
    """Deleting a task removes its folder; it can then no longer be loaded."""
    t = ws.create("gone")
    ws.delete("gone")
    assert not t.path.exists()
    assert ws.list_tasks() == []
    with pytest.raises(KeyError):
        ws.get("gone")


# ---- settings ---------------------------------------------------------
def test_update_settings(ws):
    """Allowed settings persist and bump the revision; others are rejected unchanged."""
    t = ws.create("t")
    t.update_settings(classifier="logreg", classifier_params={"C": 2.0}, reject_below=0.5)
    on_disk = saved(t)
    assert on_disk["classifier"] == "logreg"
    assert on_disk["classifier_params"] == {"C": 2.0}
    assert on_disk["reject_below"] == 0.5
    assert on_disk["revision"] == 2
    with pytest.raises(ValueError, match=r"Unknown settings: \['name'\]"):
        t.update_settings(name="other")
    with pytest.raises(ValueError, match="Unknown classifier 'svm'"):
        t.update_settings(classifier="svm")
    with pytest.raises(ValueError, match="text_weight: Input should be a valid number"):
        t.update_settings(text_weight="abc")
    with pytest.raises(ValueError, match="reject_below must be between -1 and 1"):
        t.update_settings(reject_below=-5)
    with pytest.raises(ValueError, match="Unknown parameter 'k' for logreg"):
        t.update_settings(classifier_params={"k": 3})
    assert t.config.classifier == "logreg"
    assert t.config.text_weight == 1.0
    assert t.config.revision == 2
    assert [x.name for x in ws.list_tasks()] == ["t"]  # still loadable


# ---- classes ----------------------------------------------------------
def test_upsert_class(ws):
    """Upsert strips names, creates once, and only overwrites an explicit description."""
    t = ws.create("t")
    c = t.upsert_class("  cat ", "a cat")
    assert c.name == "cat"
    assert c.description == "a cat"
    assert t.upsert_class("cat") == c
    assert t.upsert_class("cat", "").description == ""
    assert t.config.classes[0].description == ""
    assert t.class_names() == ["cat"]
    t.upsert_class("dog")
    assert [x["name"] for x in saved(t)["classes"]] == ["cat", "dog"]
    with pytest.raises(ValueError, match="Class name is empty"):
        t.upsert_class("   ")


def test_describe_classes(ws):
    """Templates fill empty descriptions (or all with overwrite) and report the count."""
    t = ws.create("t")
    t.upsert_class("golden_retriever")
    t.upsert_class("cat", "kept")
    with pytest.raises(ValueError, match="must contain"):
        t.describe_classes("a photo")
    assert t.describe_classes("a photo of a {}") == 1
    assert [c.description for c in t.config.classes] == ["a photo of a golden retriever", "kept"]
    assert t.describe_classes("{}!", overwrite=True) == 2
    assert [c["description"] for c in saved(t)["classes"]] == ["golden retriever!", "cat!"]
    # Only the literal {} is substituted; other braces are kept, and every {} is filled.
    t.describe_classes("{} {name} {}", overwrite=True)
    assert t.config.classes[1].description == "cat {name} cat"


def test_rename_class(ws):
    """Renaming relabels only that class's examples and rejects clashes or empty names."""
    t = ws.create("t")
    r = t.add_example(png(RED), "red")
    g = t.add_example(png(GREEN), "green")
    t.rename_class("red", " rouge ")
    assert t.class_names() == ["rouge", "green"]
    assert t.example(r.id).label == "rouge"
    assert t.example(g.id).label == "green"
    with pytest.raises(ValueError, match="already exists"):
        t.rename_class("rouge", "green")
    with pytest.raises(ValueError, match="Class name is empty"):
        t.rename_class("rouge", "  ")
    with pytest.raises(KeyError, match="No class 'red'"):
        t.rename_class("red", "x")
    assert [e["label"] for e in saved(t)["examples"]] == ["rouge", "green"]


def test_delete_class_removes_examples_and_files(ws, monkeypatch):
    """Deleting a class deletes its examples and (with no undo history kept) image files, and
    returns how many."""
    monkeypatch.setattr(history, "HISTORY_KEPT", 0)
    t = ws.create("t")
    r1 = t.add_example(png(RED), "red")
    t.add_example(png(RED, noise=5), "red")
    g = t.add_example(png(GREEN), "green")
    red_file = t.image_path(r1.id)
    (t.path / t.config.examples[1].file).unlink()  # an already-missing file is fine
    assert t.delete_class("red") == 2
    assert not red_file.exists()
    assert t.image_path(g.id).exists()
    assert t.class_names() == ["green"]
    assert [e.id for e in t.config.examples] == [g.id]
    with pytest.raises(KeyError, match="No class 'red'"):
        t.delete_class("red")


# ---- examples ---------------------------------------------------------
def test_add_example_stores_file_and_creates_class(ws):
    """Images are stored under their content id with the right extension."""
    t = ws.create("t")
    t.upsert_class("red")
    p = t.add_example(png(RED), "red", "r.png")
    j = t.add_example(jpeg(), " new ", "x.jpg")
    assert p.file == f"images/{image_id(png(RED))}.png"
    assert p.source == "r.png"
    assert j.file.endswith(".jpg")
    assert j.label == "new"
    assert (t.path / j.file).read_bytes() == jpeg()
    assert t.class_names() == ["red", "new"]
    assert [e["id"] for e in saved(t)["examples"]] == [p.id, j.id]
    again = t.add_example(png(RED), "new", "copy.png")
    assert again.id == p.id
    assert again.label == "new"
    assert again.source == "r.png"
    assert t.example(p.id) == again
    assert len(t.config.examples) == 2


def test_add_example_recreates_missing_images_folder(ws):
    """A task whose images/ folder vanished still accepts examples."""
    t = ws.create("t")
    (t.path / "images").rmdir()
    e = t.add_example(png(RED), "red")
    assert t.image_path(e.id).is_file()


def test_add_example_rejects_empty_label(ws):
    """A blank label is an error and nothing is written."""
    t = ws.create("t")
    with pytest.raises(ValueError, match="Class name is empty"):
        t.add_example(png(RED), "  ")
    assert t.config.examples == []
    assert not list((t.path / "images").iterdir())


def test_add_examples_from_dir(ws, tmp_path):
    """Only image files directly inside class folders are imported."""
    src = tmp_path / "src"
    (src / "red" / "nested").mkdir(parents=True)
    (src / "green").mkdir()
    (src / "red" / "a.PNG").write_bytes(png(RED))
    (src / "red" / "notes.txt").write_text("skip")
    (src / "red" / "nested" / "b.png").write_bytes(png(RED, noise=9))
    (src / "green" / "c.png").write_bytes(png(GREEN))
    (src / "loose.png").write_bytes(png(GREEN, noise=9))
    t = ws.create("t")
    assert t.add_examples_from_dir(src) == 2
    assert t.class_names() == ["green", "red"]
    assert [(e.label, e.source) for e in t.config.examples] == [("green", "c.png"), ("red", "a.PNG")]


def test_example_lookup_relabel_and_delete(ws, monkeypatch):
    """Examples can be found, moved between classes and deleted with their file."""
    monkeypatch.setattr(history, "HISTORY_KEPT", 0)
    t = ws.create("t")
    e = t.add_example(png(RED), "red")
    t.upsert_class("blue")
    with pytest.raises(KeyError, match="No example 'nope'"):
        t.example("nope")
    assert t.relabel(e.id, "blue").label == "blue"
    assert t.relabel(e.id, " green ").label == "green"
    assert t.class_names() == ["red", "blue", "green"]
    assert saved(t)["examples"][0]["label"] == "green"
    path = t.image_path(e.id)
    t.delete_example(e.id)
    assert not path.exists()
    assert t.config.examples == []
    assert saved(t)["examples"] == []
    with pytest.raises(KeyError):
        t.delete_example(e.id)


def test_summary_counts_every_class(ws):
    """Summary lists all classes, including empty ones."""
    t = ws.create("t", embedder="fake:color")
    t.add_example(png(RED), "red")
    t.add_example(png(RED, noise=5), "red")
    t.upsert_class("empty")
    s = t.summary()
    assert s["counts"] == {"red": 2, "empty": 0}
    assert (s["name"], s["embedder"], s["classifier"], s["n_classes"], s["n_examples"]) == (
        "t", "fake:color", "gda", 2, 2)
    assert s["created"] == t.config.created


def test_task_config_defaults():
    """A bare config uses the default embedder and the GDA classifier."""
    cfg = TaskConfig(name="x")
    assert cfg.embedder == embedders.DEFAULT_SPEC
    assert cfg.classifier == "gda"
    assert cfg.revision == 0


def test_create_race_is_a_clean_error(ws, monkeypatch):
    """If another create wins between the name check and mkdir, the loser gets ValueError."""
    real_mkdir = Path.mkdir

    def mkdir_after_rival(self, *args, **kwargs):
        if self.name == "t" and not self.exists():
            real_mkdir(self)  # a concurrent create got there first
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir_after_rival)
    with pytest.raises(ValueError, match="Task 't' already exists"):
        ws.create("t")


def test_load_image_reports_decompression_bombs(monkeypatch):
    """Pillow's DecompressionBombError (not an OSError) becomes ValueError."""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="Not a readable image: Image size"):
        load_image(png(RED, size=8))


def test_load_image_reports_truncated_files():
    """Truncated data that Pillow recognises is reported with Pillow's message."""
    with pytest.raises(ValueError, match="Not a readable image: image file is truncated"):
        load_image(png(RED, size=64, noise=50)[:-40])


# ---- validation of names and task.json -------------------------------
@pytest.mark.parametrize("name", [".", "..", "a/b", "a\\b", "tab\there", "x" * 201])
def test_bad_class_names_are_refused(ws, name):
    """Names that can't be a URL path segment are refused when creating, renaming or relabelling."""
    t = ws.create("t")
    t.add_example(png(RED), "ok")
    with pytest.raises(ValueError, match="Class name"):
        t.upsert_class(name)
    with pytest.raises(ValueError, match="Class name"):
        t.rename_class("ok", name)
    with pytest.raises(ValueError, match="Class name"):
        t.relabel(t.config.examples[0].id, name)
    with pytest.raises(ValueError, match="Class name"):
        t.add_example(png(RED, noise=15), name)
    assert t.class_names() == ["ok"]
    assert check_class_name("  a.b c ") == "a.b c"


def test_text_weight_is_not_a_classifier_param(ws):
    """text_weight inside classifier_params would collide with the task setting at fit time."""
    t = ws.create("t")
    with pytest.raises(ValueError, match="text_weight is a task setting"):
        t.update_settings(classifier_params={"text_weight": 2.0})
    assert t.config.classifier_params == {}


def _write_task_json(ws, name, edit):
    path = ws.root / "tasks" / name / "task.json"
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("file", ["../../victim.txt", "/etc/passwd", "images/../../x.png", "images/abc.png"])
def test_example_files_outside_images_are_refused(ws, file):
    """A task.json pointing an example outside images/ can't be loaded, so nothing follows it."""
    t = ws.create("t")
    t.add_example(png(RED), "red")

    def tamper(data):
        data["examples"][0]["file"] = file

    _write_task_json(ws, "t", tamper)
    with pytest.raises(KeyError, match=r"Task 't' can't be loaded: .*examples\.0\.file"):
        Workspace(ws.root).get("t")
    for _ in range(2):  # the cached copy is dropped on refresh, then re-reading fails the same way
        with pytest.raises(KeyError, match="can't be loaded"):
            ws.get("t")
    assert ws.list_tasks() == []


def test_corrupt_or_misnamed_tasks_are_skipped(ws, caplog):
    """One broken task.json doesn't break listing the others; a task.json naming another task
    (a folder renamed by hand, or an interrupted rename) is taken under its folder's name."""
    ws.create("good")
    ws.create("bad")
    ws.create("moved")
    (ws.root / "tasks" / "bad" / "task.json").write_text("{not json")
    _write_task_json(ws, "moved", lambda d: d.update(name="other"))
    fresh = Workspace(ws.root)
    assert [t.name for t in fresh.list_tasks()] == ["good", "moved"]
    assert "Skipping task folder" in caplog.text
    with pytest.raises(KeyError, match=r"Task 'bad' can't be loaded: .*task\.json is invalid"):
        fresh.get("bad")
    moved = fresh.get("moved")
    assert moved.name == "moved"
    assert "names task 'other'; using its folder name 'moved'" in caplog.text
    moved.upsert_class("x")  # the next save writes the folder's name
    assert json.loads((moved.path / "task.json").read_text())["name"] == "moved"


def test_legacy_16_char_ids_still_match(ws):
    """Tasks saved with 16-char ids load, and re-adding the same image relabels it instead of duplicating."""
    t = ws.create("t")
    t.add_example(png(RED), "red")

    def shorten(data):
        e = data["examples"][0]
        e["id"] = e["id"][:16]
        e["file"] = f"images/{e['id']}.png"

    old = t.path / t.config.examples[0].file
    old.rename(t.path / f"images/{t.config.examples[0].id[:16]}.png")
    _write_task_json(ws, "t", shorten)
    fresh = Workspace(ws.root).get("t")
    assert len(fresh.config.examples[0].id) == 16
    again = fresh.add_example(png(RED), "blue")
    assert again.id == fresh.config.examples[0].id
    assert [e.label for e in fresh.config.examples] == ["blue"]
    assert fresh.image_path(again.id).exists()
    assert "Already an example of blue" in str(fresh.add_to_inbox([(png(RED), "")])[0])


def test_add_examples_in_chunks(ws, monkeypatch):
    """add_examples saves once per chunk and reports each item's example or error in order."""
    t = ws.create("t")
    saves = []
    real = type(t)._save  # pylint: disable=protected-access
    def counted(self, **kw):
        saves.append(kw.get("record", True))
        real(self, **kw)

    monkeypatch.setattr(type(t), "_save", counted)
    a, b = png(RED, noise=5, seed=1), png(GREEN, noise=5, seed=2)
    items = iter([(a, "red", "a"), (b"junk", "red", "j"), (a, "pink", "a again"), (b, " ", "b"), (b, "green", "b")])
    out = t.add_examples(items, chunk=2)
    assert [type(r).__name__ for r in out] == ["Example", "ValueError", "Example", "ValueError", "Example"]
    assert out[2].id == out[0].id  # the same image again was relabelled, not duplicated
    assert [(e.label, e.source) for e in t.config.examples] == [("pink", "a"), ("green", "b")]
    assert t.class_names() == ["red", "pink", "green"]
    assert saves == [True, False, False]  # one per chunk added to; only the first is an undo step
    assert isinstance(t.add_examples([(b"junk", "x", "")])[0], ValueError)
    assert len(saves) == 3  # nothing added: no save


def test_chunked_add_is_one_undo_step_when_the_first_chunk_adds_nothing(ws):
    """The first chunk that saves keeps the version before the call, even if earlier chunks failed:
    one undo takes back the add and nothing before it."""
    t = ws.create("t")
    t.upsert_class("keep")
    t.add_examples([(b"junk", "a", ""), (b"junk2", "a", ""), (png(RED), "a", ""), (png(GREEN), "a", "")], chunk=2)
    t.undo()
    assert (t.class_names(), t.config.examples) == (["keep"], [])


def test_add_examples_from_dir_raises_after_adding_the_rest(ws, tmp_path):
    """An unreadable image in an imported folder is raised once the readable ones are in."""
    src = tmp_path / "src"
    (src / "red").mkdir(parents=True)
    (src / "red" / "a.png").write_bytes(png(RED))
    (src / "red" / "b.png").write_bytes(b"junk")
    t = ws.create("t")
    with pytest.raises(ValueError, match="Not a readable image"):
        t.add_examples_from_dir(src)
    assert [e.source for e in t.config.examples] == ["a.png"]


# ---- rename --------------------------------------------------------------
def test_rename_task(ws):
    """Renaming moves the folder and keeps examples, history and links; the old name is gone."""
    t = red_green_task(ws)
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


def test_concurrent_renames(ws, monkeypatch):
    """Of two renames of one task that both looked it up first, only the first succeeds; a cached
    task under a name it no longer has is never returned for that name."""
    t = red_green_task(ws)
    real_get = ws.get
    monkeypatch.setattr(ws, "get", lambda name: t if name == "t" else real_get(name))  # a stale lookup
    ws.rename("t", "u")
    with pytest.raises(KeyError, match="No task 't'"):
        ws.rename("t", "v")
    monkeypatch.setattr(ws, "get", real_get)
    assert not (ws.root / "tasks" / "v").exists()
    ws._tasks["v"] = t  # pylint: disable=protected-access  # as if cached under a name it has lost
    with pytest.raises(KeyError, match="No task 'v'"):
        ws.get("v")
    assert ws.get("u") is t


def test_rename_errors(ws, monkeypatch):
    """A bad or taken new name is a ValueError; a failed save puts the folder back."""
    red_green_task(ws)
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


# ---- tags, notes, descriptions and lookups -------------------------------
def test_tags_and_notes(ws):
    """Tags can be replaced, added and removed (kept unique, in order); notes set and cleared."""
    t = red_green_task(ws)
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
        t.annotate([a], note="x" * (models.MAX_NOTE + 1))
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
    t = red_green_task(ws)
    eid = t.config.examples[0].id
    assert t.resolve_ids([eid[:6], eid]) == [eid, eid]
    with pytest.raises(KeyError, match="'abc' is too short: give at least 4 characters"):
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
    t = red_green_task(ws)
    t.describe_classes("a photo of a {}\n\n  a {} thing ")
    assert t.config.classes[0].description == "a photo of a red\na red thing"


def test_descriptions_are_capped(ws):
    """A description can't be longer than MAX_DESCRIPTION or hold more than MAX_PROMPTS prompts."""
    t = two_each(ws)
    with pytest.raises(ValueError, match="longer than"):
        t.upsert_class("red", "x" * (MAX_DESCRIPTION + 1))
    with pytest.raises(ValueError, match="prompts"):
        t.edit_class("red", description="a\n" * (MAX_PROMPTS + 1))
    with pytest.raises(ValueError, match="prompts"):
        t.describe_classes("a {}\n" * (MAX_PROMPTS + 1))
    t.upsert_class("red", "a\n" * MAX_PROMPTS)  # at the limit is fine
