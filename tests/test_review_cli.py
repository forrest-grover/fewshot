"""Fixes from the interface review: one change per edit, folder walks, imports, unreadable tasks,
cache pruning, settings rules and CLI conventions."""

import csv
import io
import os
import sqlite3
import time
import zipfile

import numpy as np
import pytest

from fewshot import cache as cache_module
from fewshot import cli, services
from fewshot.archive import export_task, import_task
from fewshot.cache import EmbeddingCache, prune_unused
from fewshot.folders import class_folders, image_files
from fewshot.store import Workspace
from fewshot.text import MAX_DESCRIPTION, MAX_PROMPTS
from tests.conftest import png
from tests.test_cli import COLORS, photos_fixture, root_fixture, run_fixture, task_fixture, write_images  # noqa: F401  # pylint: disable=unused-import  # fixtures

RED, BLUE = COLORS["red"], COLORS["blue"]


def _two(ws):
    """A task with two red and two blue examples."""
    t = ws.create("t", embedder="fake:color")
    t.add_examples([(png(RED, noise=10, seed=i), "red", f"r{i}.png") for i in range(2)]
                   + [(png(BLUE, noise=10, seed=i), "blue", f"b{i}.png") for i in range(2)])
    return t


# ---- one request, one change -----------------------------------------------
def test_edit_class_renames_and_describes_in_one_undo_step(ws):
    """Renaming a class and describing it is one change: one undo takes back both."""
    t = _two(ws)
    rev = t.config.revision
    t.edit_class("red", "crimson", "a photo of something crimson")
    assert t.config.revision == rev + 1
    assert (t.config.classes[0].name, t.config.classes[0].description) == ("crimson", "a photo of something crimson")
    assert {e.label for e in t.config.examples} == {"crimson", "blue"}
    t.undo()
    assert (t.config.classes[0].name, t.config.classes[0].description) == ("red", "")
    with pytest.raises(ValueError, match="already exists"):
        t.edit_class("red", "blue")
    with pytest.raises(KeyError):
        t.edit_class("nope", description="x")
    t.edit_class("red", "crimson")
    assert t.undo_label() == "rename class red to crimson"
    with pytest.raises(ValueError, match="already exists"):
        t.rename_class("crimson", "crimson")


def test_edit_examples_relabels_and_tags_in_one_undo_step(ws):
    """Moving and tagging examples is one change: one undo takes back both."""
    t = _two(ws)
    ids = [e.id for e in t.config.examples if e.label == "red"]
    rev = t.config.revision
    assert t.edit_examples(ids, label="blue", add_tags=["x"], note="n") == 2
    assert t.config.revision == rev + 1
    assert all((e.label, e.tags, e.note) == ("blue", ("x",), "n") for e in t.config.examples if e.id in ids)
    t.undo()
    assert all((e.label, e.tags) == ("red", ()) for e in t.config.examples if e.id in ids)
    t.edit_examples(ids, label="green")
    assert t.undo_label() == "move 2 examples to green"
    with pytest.raises(ValueError, match="contain '/'"):
        t.edit_examples(ids, label="a/b")


# ---- descriptions are capped -----------------------------------------------
def test_descriptions_are_capped(ws):
    """A description can't be longer than MAX_DESCRIPTION or hold more than MAX_PROMPTS prompts."""
    t = _two(ws)
    with pytest.raises(ValueError, match="longer than"):
        t.upsert_class("red", "x" * (MAX_DESCRIPTION + 1))
    with pytest.raises(ValueError, match="prompts"):
        t.edit_class("red", description="a\n" * (MAX_PROMPTS + 1))
    with pytest.raises(ValueError, match="prompts"):
        t.describe_classes("a {}\n" * (MAX_PROMPTS + 1))
    t.upsert_class("red", "a\n" * MAX_PROMPTS)  # at the limit is fine


def test_an_import_with_too_long_a_description_is_refused(ws):
    """Imports check descriptions like edits do."""
    t = _two(ws)
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


# ---- importing folders -----------------------------------------------------
def _messy(folder):
    """A dataset with one real class among hidden and system folders and files."""
    for label in ("cat", ".ipynb_checkpoints", "__MACOSX", "@eaDir"):
        write_images(folder / label, RED, 2)
    (folder / "cat" / "._0.png").write_bytes(b"AppleDouble junk")
    (folder / "cat" / ".hidden.png").write_bytes(png(BLUE))
    write_images(folder / "cat" / ".ipynb_checkpoints", RED, 1)


def test_folder_walks_skip_hidden_and_system_entries(tmp_path):
    """The folder helpers leave out hidden and system entries and honour shots."""
    _messy(tmp_path)
    assert list(class_folders(tmp_path)) == ["cat"]
    assert [f.name for f in class_folders(tmp_path)["cat"]] == ["0.png", "1.png"]
    assert [f.name for f in class_folders(tmp_path, shots=1)["cat"]] == ["0.png"]
    assert [f.name for f in image_files(tmp_path / "cat", recursive=True)] == ["0.png", "1.png"]


def test_import_skips_system_folders_in_the_library_and_cli(ws, run, tmp_path):
    """Neither the library nor `fewshot import` makes classes of __MACOSX and the like."""
    _messy(tmp_path / "data")
    t = ws.create("lib", embedder="fake:color")
    assert t.add_examples_from_dir(tmp_path / "data", shots=1) == 1
    assert t.class_names() == ["cat"]
    assert run("create", "cli", "--embedder", "fake:color").exit_code == 0
    assert run("import", "cli", tmp_path / "data").exit_code == 0
    assert run("show", "cli", "--json").output.count('"name": "cat"') == 1
    assert ".ipynb" not in run("show", "cli").output


def test_add_examples_from_dir_reports_an_unreadable_file(ws, tmp_path):
    """The library import names the file it couldn't read, after adding the others."""
    write_images(tmp_path / "cat", RED, 1)
    (tmp_path / "cat" / "1.png").write_bytes(b"not an image")
    t = ws.create("t", embedder="fake:color")
    with pytest.raises(ValueError, match=r"1\.png"):
        t.add_examples_from_dir(tmp_path)
    assert len(t.config.examples) == 1


# ---- import-task checks the embedder ----------------------------------------
def test_import_task_checks_the_embedder_whoever_calls_it(ws, run, tmp_path):
    """archive.import_task checks the embedder itself, so the CLI can't skip it."""
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


# ---- unreadable tasks ------------------------------------------------------
@pytest.mark.skipif(os.geteuid() == 0, reason="root can read anything")
def test_an_unreadable_task_is_skipped_not_a_crash(run, root):
    """A task.json that can't be read is skipped by list and refused by show and prune, cleanly."""
    ws = Workspace(root)
    _two(ws)
    ws.create("other", embedder="fake:color")
    (root / "tasks" / "t" / "task.json").chmod(0)
    try:
        assert [t.name for t in Workspace(root).list_tasks()] == ["other"]
        r = run("list")
        assert (r.exit_code, r.output.split()[0]) == (0, "other")
        r = run("show", "t")
        assert r.exit_code == 1
        assert "can't be loaded" in r.output
        assert "Permission denied" in r.output
        r = run("cache", "prune")
        assert (r.exit_code, "Can't read task(s) t" in r.output) == (1, True)
        fresh = Workspace(root)
        fresh.get("other")
        with pytest.raises(KeyError, match="Permission denied"):
            fresh.get("t")
    finally:
        (root / "tasks" / "t" / "task.json").chmod(0o644)
    cached = Workspace(root)
    cached.get("t")
    (root / "tasks" / "t" / "task.json").chmod(0)
    os.utime(root / "tasks" / "t" / "task.json", (1, 1))  # changed: the next get re-reads it
    try:
        with pytest.raises(KeyError, match="Permission denied"):
            cached.get("t")
    finally:
        (root / "tasks" / "t" / "task.json").chmod(0o644)


def test_cli_reports_file_system_errors_without_a_traceback(monkeypatch, capsys):
    """An OSError escaping a command is an error message and exit status 1."""
    def broken():
        """Fail like an unreadable workspace."""
        raise PermissionError(13, "Permission denied", "/data/tasks")

    monkeypatch.setattr(cli, "app", broken)
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1
    assert "Error: /data/tasks: Permission denied" in capsys.readouterr().err
    monkeypatch.setattr(cli, "app", lambda: (_ for _ in ()).throw(OSError("disk on fire")))
    with pytest.raises(SystemExit):
        cli.main()
    assert "Error: disk on fire" in capsys.readouterr().err


def test_read_only_commands_make_no_workspace(run, root):
    """list and show with a mistyped --root don't create it; create does."""
    r = run("list")
    assert (r.exit_code, r.output) == (0, "")
    assert run("show", "x").exit_code == 1
    assert not root.exists()
    assert run("create", "x", "--embedder", "fake:color").exit_code == 0
    assert root.exists()


# ---- cache prune --------------------------------------------------------------
def test_prune_cache_holds_the_task_locks(ws, monkeypatch):
    """Pruning decides what to keep and deletes the rest with every task locked."""
    t = _two(ws)
    seen = []
    real = EmbeddingCache.prune

    def prune(self, keep, dry_run=False):
        """Record whether the task lock is held."""
        seen.append(t._lock._is_owned())  # pylint: disable=protected-access
        return real(self, keep, dry_run)

    monkeypatch.setattr(EmbeddingCache, "prune", prune)
    assert prune_unused(ws, ) == (0, 0)
    assert seen == [True]


def test_prune_cache_vacuums_only_when_asked(ws, monkeypatch):
    """VACUUM, which blocks other writers, runs only with vacuum=True."""
    _two(ws)
    cache = EmbeddingCache(ws.cache_path)
    cache.put_many("fake:color", {"gone": np.array([1.0, 0.0, 0.0], np.float32)})
    vacuums: list[int] = []
    monkeypatch.setattr(EmbeddingCache, "vacuum", lambda self: vacuums.append(1))
    assert prune_unused(ws, dry_run=True) == (1, 0)
    assert prune_unused(ws, ) == (1, 0)
    assert not vacuums
    cache.put_many("fake:color", {"gone": np.array([1.0, 0.0, 0.0], np.float32)})
    monkeypatch.undo()
    assert prune_unused(ws, vacuum=True) == (1, 0)
    with sqlite3.connect(ws.cache_path) as db:
        assert db.execute("SELECT COUNT(*) FROM emb WHERE key='gone'").fetchone() == (0,)


def test_cli_cache_prune_vacuum_option(run, task, monkeypatch):  # pylint: disable=unused-argument
    """`cache prune --vacuum` asks for a vacuum; plain prune doesn't."""
    calls: list[bool] = []

    def prune(_ws, dry_run, vacuum):  # pylint: disable=unused-argument
        """Record whether a vacuum was asked for."""
        calls.append(vacuum)
        return 0, 0

    monkeypatch.setattr(cache_module, "prune_unused", prune)
    run("cache", "prune")
    run("cache", "prune", "--vacuum")
    assert calls == [False, True]


# ---- settings rules ------------------------------------------------------------
def test_settings_rules_live_in_the_library(ws):
    """Param merging, a new classifier's fresh params and temperature 1 are library rules."""
    t = _two(ws)
    t.update_settings(classifier="knn", classifier_params={"k": 3})
    assert t.config.classifier_params == {"k": 3}
    t.update_settings(merge_params={"scale": 20})
    assert t.config.classifier_params == {"k": 3, "scale": 20}  # merged
    t.update_settings(classifier="prototype")
    assert t.config.classifier_params == {}  # a new classifier starts afresh
    t.update_settings(classifier="knn", merge_params={"k": 2})
    assert t.config.classifier_params == {"k": 2}
    t.update_settings(temperature=2.0)
    t.update_settings(temperature=1)
    assert t.config.temperature is None  # 1 means uncalibrated, from any interface


def test_change_settings_checks_clears_and_embedders(ws):
    """change_settings refuses a value both set and cleared and checks only a new embedder."""
    t = _two(ws)
    t.update_settings(reject_below=0.5)
    with pytest.raises(ValueError, match="not both"):
        services.change_settings(t, {"reject_below": 0.4}, {"reject_below": "clear_reject"})
    services.change_settings(t, {}, {"reject_below": "clear_reject"})
    assert t.config.reject_below is None
    checked: list[str] = []
    services.change_settings(t, {"embedder": "fake:color"}, embedder_ok=checked.append)
    assert not checked  # unchanged: not checked again
    with pytest.raises(ValueError, match="nobackend"):
        services.change_settings(t, {"embedder": "nobackend:x"})


def test_cli_temperature_one_and_class_add_keeps_description(run, task):
    """`class add` keeps a description; --temperature 1 clears it."""
    assert run("class", "describe", task, "a photo of a {} thing").exit_code == 0
    assert run("class", "add", task, "red").exit_code == 0
    assert "a photo of a red thing" in run("show", task).output  # not wiped by `class add`
    assert run("config", task, "--temperature", "2").exit_code == 0
    assert "temperature: 2.0" in run("config", task, "--temperature", "2").output
    assert "temperature: 1" in run("config", task, "--temperature", "1").output


def test_check_specs():
    """Specs are stripped and deduplicated, and their number checked."""
    assert services.check_specs([" a ", "b", "a", ""]) == ["a", "b"]
    with pytest.raises(ValueError, match="at least one"):
        services.check_specs([" "])
    with pytest.raises(ValueError, match="between 1 and 1"):
        services.check_specs(["a", "b"], limit=1)


# ---- CLI conventions --------------------------------------------------------------
def test_usage_errors_exit_2(run, task):
    """Options that don't go together exit 2, like Typer's own usage errors."""
    assert run("predict", task, "--json", "--csv", "x.png").exit_code == 2
    r = run("eval", task, "--json", "--csv")
    assert r.exit_code == 2
    assert "at most one" in r.output
    assert run("class", "threshold", task, "red").exit_code == 2


def test_csv_booleans_are_lowercase(run, task, tmp_path):
    """CSV output writes booleans as true/false."""
    q = tmp_path / "q.png"
    q.write_bytes(png(RED, seed=3))
    rows = list(csv.DictReader(io.StringIO(run("predict", task, q, "--csv").output)))
    assert rows[0]["rejected"] == "false"
    rows = list(csv.DictReader(io.StringIO(run("eval", task, "--csv").output)))
    assert {r["correct"] for r in rows} <= {"true", "false"}


def test_watch_backs_off_from_a_file_it_cannot_move(tmp_path, monkeypatch):
    """Failures double the wait before the next try, up to a maximum."""
    retry = cli.Backoff(1.0)
    f = tmp_path / "a.png"
    assert not retry.waiting(f)
    assert [retry.failed(f) for _ in range(3)] == [1.0, 2.0, 4.0]
    assert retry.waiting(f)
    for _ in range(20):
        retry.failed(f)
    assert retry.failed(f) == cli.WATCH_MAX_BACKOFF
    retry.keep([])
    assert not retry.waiting(f)
    retry.failed(f)
    retry.succeeded(f)
    assert not retry.waiting(f)
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    retry.failed(f)
    clock[0] = 1.5
    assert not retry.waiting(f)  # due again


def test_watch_retries_a_failed_move_later(run, task, tmp_path, monkeypatch):
    """watch doesn't retry a file it couldn't move at every look."""
    inbox = tmp_path / "in"
    write_images(inbox, RED, 1)
    tries = []

    def place(f, folder, copy):  # pylint: disable=unused-argument
        """Fail to move anything."""
        tries.append(f.name)
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "_place", place)
    sleeps = []

    def sleep(_s):
        """Stop after a few looks."""
        sleeps.append(1)
        if len(sleeps) > 6:
            raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", sleep)
    r = run("watch", task, inbox, "--out", tmp_path / "out", "--interval", "100")
    assert "trying again in 100 s" in r.output
    assert len(tries) == 1  # waits out its back-off instead of retrying every look


def test_rich_is_a_declared_dependency():
    """The CLI's progress bars need rich, so it is a dependency."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as f:
        assert '"rich"' in f.read()
