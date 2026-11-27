"""Tests for how the command line helps its users: help text, messages, streams, aliases, typos,
partial failures, repeated runs and machine-readable output."""

import csv
import io
import json
import os
import sys
import zipfile
from unittest import mock

import pytest
from typer.testing import CliRunner

import fewshot.__main__ as entry
from fewshot import cli, cli_common, cli_eval, services
from fewshot.evaluate import TEMPERATURE_MARGIN
from fewshot.store import Workspace
from tests.conftest import BLUE, RED, PlainResult, png, served, write_images


@pytest.fixture(name="photos")
def photos_fixture(tmp_path):
    """FOLDER/<class>/<image> with 3 red and 3 blue images."""
    folder = tmp_path / "photos"
    write_images(folder / "red", RED, 3)
    write_images(folder / "blue", BLUE, 3, start=10)
    return folder


@pytest.fixture(name="task")
def task_fixture(run, photos):
    """Task 't' with the photos imported."""
    assert run("create", "t", "--embedder", "fake:color").exit_code == 0
    assert run("import", "t", photos).exit_code == 0
    return "t"


def load(root, name="t"):
    """Read a task fresh from disk."""
    return Workspace(root).get(name)


def terminal(stream="stdin"):
    """Within the block, the CLI takes ``stream`` for a terminal."""
    return mock.patch(f"fewshot.cli_common.{stream}_is_terminal", return_value=True)


def plain(*args, env=None):
    """Run the CLI without a workspace option."""
    return PlainResult(CliRunner().invoke(cli.app, [str(a) for a in args], env=env))


def commands(group=None, path=()):
    """Every (path, click command) of the CLI, groups' commands included."""
    import typer.main  # pylint: disable=import-outside-toplevel

    group = group or typer.main.get_command(cli.app)
    for name, cmd in group.commands.items():
        if hasattr(cmd, "commands"):
            yield from commands(cmd, (*path, name))
        elif not cmd.hidden:
            yield (*path, name), cmd


# ---- help ---------------------------------------------------------------------
def test_dash_h_and_version():
    """-h works everywhere, as --help does (and on a command taking negative numbers), and
    --version names the versions results depend on."""
    for args in ([], ["class"], ["add"], ["class", "threshold"]):
        r = plain(*args, "-h")
        assert (r.exit_code, "Usage: " in r.output) == (0, True), args
    r = plain("--version")
    assert r.exit_code == 0
    assert r.output.startswith("fewshot ")
    assert "(torch " in r.output
    assert ", open_clip " in r.output
    assert ", python " in r.output
    assert "--version" in plain("--help").output


def test_root_help_groups_commands_and_documents_the_environment():
    """The command list is in sections, shows each summary in full, and the help ends with the
    environment variables and exit statuses."""
    raw = plain("--help").output
    out = " ".join(raw.split())
    sections = [line for line in raw.splitlines() if line.endswith(":") and not line.startswith(" ")]
    assert sections == ["Options:", "Tasks:", "Labelling:", "Predicting:", "Evaluating:", "Maintenance:", "Server:"]
    assert "Copy a task under a new name, to try other settings without touching it." in out
    for var in ("FEWSHOT_HOME", "FEWSHOT_DEVICE", "FEWSHOT_MAX_MODELS", "HF_HUB_OFFLINE", "FEWSHOT_QUIET"):
        assert var in out
    assert "130 interrupted" in out
    assert "\n  ls " not in raw  # aliases aren't listed twice


def test_every_command_has_examples_and_described_arguments():
    """Each command's help ends with examples, starts with one short sentence, and describes
    every argument and option."""
    for path, cmd in commands():
        assert "Examples:" in (cmd.epilog or ""), path
        summary = cmd.help.strip().split("\n\n")[0]
        assert len(summary) <= 80, (path, summary)
        assert summary.endswith("."), (path, summary)
        for p in cmd.params:
            assert p.help or p.hidden, (path, p.name)
        r = plain(*path, "-h")
        assert (r.exit_code, "Examples:" in r.output) == (0, True), path


def test_param_help_lists_each_classifiers_parameters():
    """--param's help names the parameters of each classifier, not one that gda lacks."""
    out = " ".join(plain("config", "-h", env={"COLUMNS": "300"}).output.split())
    assert "gda: none" in out
    assert "knn: k, scale" in out
    assert "logreg: C" in out


# ---- global options -----------------------------------------------------------
def test_status_goes_to_stderr_and_quiet_hides_it(run, task, root):
    """Confirmations go to stderr, so stdout holds only results; -q (or FEWSHOT_QUIET, or -q
    before the command) hides them but not errors."""
    r = run("class", "add", task, "green")
    assert (r.stdout, r.stderr.strip()) == ("", "Added class green to t")
    assert run("class", "add", task, "yellow", "-q").output == ""
    r = PlainResult(CliRunner().invoke(cli.app, ["-q", "class", "add", task, "pink", "--root", str(root)]))
    assert r.output == ""
    r = PlainResult(CliRunner().invoke(cli.app, ["class", "add", task, "cyan", "--root", str(root)],
                                       env={"FEWSHOT_QUIET": "1"}))
    assert r.output == ""
    r = run("class", "rm", task, "nope", "-q")
    assert "Error: No class 'nope'" in r.output
    assert load(root).class_names() == ["blue", "red", "green", "yellow", "pink", "cyan"]


def test_debug_shows_the_traceback(run, root):
    """--debug prints the traceback of the error behind a failure, and OSError tracebacks from main."""
    r = run("create", "x", "--embedder", "fake:nope", "--debug")
    assert r.exit_code == 1
    assert "Traceback" in r.output
    assert "Traceback" not in run("create", "x", "--embedder", "fake:nope").output
    del root


def test_main_with_debug_raises_file_system_errors(monkeypatch):
    """Without --debug main reports an OSError as a message; with it, the error propagates."""
    monkeypatch.setattr(cli, "app", lambda: (_ for _ in ()).throw(OSError("disk on fire")))
    monkeypatch.setattr(cli_common.FLAGS, "debug", True)
    with pytest.raises(OSError, match="disk on fire"):
        cli.main()


def test_device_option_sets_the_device(monkeypatch):
    """--device, before or after the command, sets FEWSHOT_DEVICE while the command runs (and only
    then); the help names the variable."""
    seen = []
    monkeypatch.setattr(cli_common, "report_loading", lambda r: seen.append(os.environ.get("FEWSHOT_DEVICE")))
    monkeypatch.delenv("FEWSHOT_DEVICE", raising=False)
    for args in (["--device", "cpu", "list"], ["list", "--device", "cpu"]):
        seen.clear()
        assert plain(*args).exit_code == 0
        assert seen[0] == "cpu"
        assert "FEWSHOT_DEVICE" not in os.environ
    seen.clear()
    assert plain("--device", "meta", "list", "--device", "cpu").exit_code == 0
    assert seen[0] == "cpu"  # after the command wins
    monkeypatch.setenv("FEWSHOT_DEVICE", "meta")
    assert plain("list", "--device", "cpu").exit_code == 0
    assert os.environ["FEWSHOT_DEVICE"] == "meta"  # put back
    assert "FEWSHOT_DEVICE" in plain("-h").output


def test_a_bad_device_is_reported_up_front_under_its_own_name(run, task, monkeypatch, tmp_path):
    """A bad --device fails before the command does anything, naming --device; a bad
    FEWSHOT_DEVICE fails any command that may use a model, even with every embedding cached,
    naming FEWSHOT_DEVICE. Neither points at `fewshot models`."""
    q = tmp_path / "q.png"
    q.write_bytes(png(RED, seed=9))
    assert run("predict", task, q).exit_code == 0  # every embedding cached now
    for args, said in ((["--device", "cpux", "list"], "Error: --device='cpux' is not a torch device"),
                       (["predict", task, q, "--device", "cuda:7"], "Error: --device='cuda:7': this machine has")):
        r = run(*args)
        assert (r.exit_code, said in r.output, "fewshot models" in r.output) == (2, True, False), r.output
    monkeypatch.setenv("FEWSHOT_DEVICE", "cpux")
    r = run("predict", task, q)
    assert (r.exit_code, "Error: FEWSHOT_DEVICE='cpux' is not a torch device" in r.output) == (2, True)
    assert "fewshot models" not in r.output
    assert run("list").exit_code == 0  # no model involved: not checked


def test_root_goes_before_or_after_the_command(root, task, monkeypatch):
    """--root works before the command as after it; after it wins."""
    assert plain("--root", root, "show", task).exit_code == 0
    assert plain("--root", root / "nowhere", "show", task, "--root", root).exit_code == 0
    assert "No task 't'" in plain("--root", root, "show", task, "--root", root / "nowhere").output
    _, server, _ = served(monkeypatch, before=["--root", root])
    assert server.ws.root == root.resolve()


def test_ctrl_c(run, task, tmp_path, monkeypatch):
    """Ctrl-C ends a command with status 130 and says what that left, without a traceback."""
    q = tmp_path / "q.png"
    q.write_bytes(png(RED, seed=9))

    def interrupt(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr("fewshot.services.predict", interrupt)
    r = run("predict", task, q)
    assert (r.exit_code, r.output.strip()) == (130, "Interrupted; no task was changed")
    monkeypatch.setattr("fewshot.services.add_examples", interrupt)
    r = run("add", task, "red", q)
    assert r.exit_code == 130
    assert "Interrupted; images stored before it are kept (`fewshot undo t` takes them back)" in r.output


def test_ctrl_c_outside_a_command_and_while_loading(monkeypatch, capsys):
    """main and the entry point (which loads the CLI) report Ctrl-C the same way."""
    monkeypatch.setattr(cli, "app", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 130
    with pytest.raises(SystemExit) as e:
        entry.main()  # loads fewshot.cli, then runs it
    assert e.value.code == 130

    def interrupted(_name):
        raise KeyboardInterrupt

    monkeypatch.setattr("fewshot.__main__.importlib.import_module", interrupted)
    with pytest.raises(SystemExit) as e:
        entry.main()
    assert e.value.code == 130
    assert capsys.readouterr().err.count("Interrupted") == 3


# ---- the workspace ------------------------------------------------------------
def test_workspace_creation_and_an_empty_list_are_announced(run, root):
    """Making the workspace says where; listing an empty one says how to start, on stderr."""
    r = run("list")
    assert (r.exit_code, r.stdout) == (0, "")
    assert f"No tasks in {root}; create one with `fewshot create NAME`" in r.stderr
    assert run("list", "--json").stdout.strip() == "[]"
    r = run("create", "t", "--embedder", "fake:color")
    assert f"Created workspace {root} (set FEWSHOT_HOME or --root to use another)" in r.stderr
    assert "Next: `fewshot import t FOLDER`" in r.stderr
    assert "Created workspace" not in run("create", "u", "--embedder", "fake:color").output


def test_unknown_task_says_where_and_what_there_is(run, task, root):
    """A missing task's error names the workspace, its tasks, the closest name and the fix."""
    r = run("show", "T")
    assert r.exit_code == 1
    assert f"No task 'T' in {root} (tasks: t); did you mean 't'?" in r.output
    assert "run `fewshot list` or set --root/FEWSHOT_HOME" in r.output
    r = PlainResult(CliRunner().invoke(cli.app, ["show", task, "--root", str(root / "elsewhere")]))
    assert "(no tasks)" in r.output


def test_list_and_show_are_readable(run, task, root):
    """list has a header on a terminal and warnings on stderr; show aligns classes and prints
    settings as on/off/none."""
    with terminal("stdout"):
        r = run("list")
    assert r.stdout.splitlines()[0].split() == ["TASK", "CLASSES", "EXAMPLES", "INBOX", "ACCURACY", "CLASSIFIER",
                                                "EMBEDDER"]
    run("class", "add", task, "a_much_longer_name")
    r = run("show", task)
    assert "flip: off   names: on   temperature: none (1)" in r.stdout
    lines = [line for line in r.stdout.splitlines() if line.strip().split()[0] in {"red", "a_much_longer_name"}]
    assert len({line.index(line.split()[1], len(line.split()[0]) + 2) for line in lines}) == 1  # counts aligned
    assert "t: Fewer than 2 examples in a_much_longer_name" in run("list").stderr
    del root


# ---- classes ----------------------------------------------------------------------
def test_class_commands_and_aliases(run, task, root):
    """class ls lists classes (also as `list`, and as JSON); class commands confirm what they did;
    rm and delete say how to undo; unknown classes suggest the closest."""
    rows = json.loads(run("class", "ls", task, "--json").stdout)
    assert rows == [{"class": "blue", "count": 3, "description": "", "threshold": None},
                    {"class": "red", "count": 3, "description": "", "threshold": None}]
    assert run("class", "list", task).stdout.split() == ["blue", "3", "-", "red", "3", "-"]
    assert "Set the description of red" in run("class", "add", task, "red", "--desc", "a red thing").stderr
    assert run("class", "add", task, "red").output == ""  # nothing changed
    assert "Set 1 descriptions" in run("class", "describe", task).stderr
    assert "all were set; --overwrite replaces them" in run("class", "describe", task).stderr
    assert "Set the threshold of red to 0.5" in run("class", "threshold", task, "red", "0.5").stderr
    assert "Cleared the threshold of red" in run("class", "threshold", task, "red", "--clear").stderr
    r = run("class", "threshold", task, "Red", "0.5")
    assert (r.exit_code, "No class 'Red' in t; did you mean 'red'?" in r.output) == (1, True)
    assert "see `fewshot class ls t`" in r.output
    assert run("class", "rename", task, "bleu", "navy").exit_code == 1
    r = run("class", "delete", task, "blue")
    assert "Removed class blue and 3 examples (undo with `fewshot undo t`)" in r.stderr
    assert run("class", "rm", task, "blue").exit_code == 1
    assert run("class", "ls", task).stdout.splitlines()[0].startswith("red")
    assert load(root).class_names() == ["red"]


def test_a_class_name_close_to_another_is_taken_for_a_typo(run, task, root, tmp_path):
    """add, example relabel and inbox accept refuse a new class whose name is close to an existing
    one unless --new-class is given; a distinct new name is created, and said so."""
    q = tmp_path / "q.png"
    q.write_bytes(png((200, 200, 20), seed=1))
    r = run("add", task, "Red", q)
    assert r.exit_code == 1
    assert "No class 'Red' in t; did you mean 'red'? Pass --new-class to create it" in r.output
    assert run("add", task, "blu", q).exit_code == 1
    assert run("add", task, "bluest", q).exit_code == 1  # two edits
    assert load(root).class_names() == ["blue", "red"]
    r = run("add", task, "Red", q, "--new-class")
    assert (r.exit_code, "Created class Red in t" in r.stderr) == (0, True)
    r = run("add", task, "yellow", q)
    assert "Created class yellow in t" in r.stderr
    assert "Added 1 to yellow (1 moved from Red); t now has 7 examples" in r.stderr
    eid = load(root).find_examples("yellow")[0].id
    assert run("example", "relabel", task, "yelow", eid[:8]).exit_code == 1
    r = run("example", "relabel", task, "gold", eid[:8])
    assert "Created class gold in t" in r.stderr
    assert "Moved 1 examples to gold" in r.stderr
    run("example", "rm", task, eid[:8])
    run("inbox", "add", task, q)
    item = load(root).config.inbox[0].id
    assert run("inbox", "accept", task, item[:8], "--label", "Gold").exit_code == 1
    r = run("inbox", "accept", task, item[:8], "--label", "amber")
    assert "Created class amber in t" in r.stderr


def test_near_miss():
    """A near miss is the same name but for case, or 1 edit away for short names and 2 for longer ones."""
    names = ["cat", "Abyssinian", "Basset_Hound"]
    assert cli_common.near_miss("CAT", names) == "cat"
    assert cli_common.near_miss("cats", names) == "cat"
    assert cli_common.near_miss("dog", names) is None
    assert cli_common.near_miss("abysinian", names) == "Abyssinian"
    assert cli_common.near_miss("Basset_Hund", names) == "Basset_Hound"
    assert cli_common.near_miss("Bulldog", names) is None
    assert cli_common.did_you_mean("x", names) == ""


# ---- adding images ------------------------------------------------------------------
def test_adding_again_changes_nothing(run, task, root, photos):
    """Re-adding or re-importing images that are there already adds no undo step and says so;
    --strict turns the skip into a failure."""
    rev = load(root).config.revision
    r = run("import", task, photos)
    assert (r.exit_code, "Nothing new: all 6 images are already in t" in r.stderr) == (0, True)
    r = run("add", task, "red", photos / "red")
    assert (r.exit_code, "Nothing new: all 3 images are already examples of red in t" in r.stderr) == (0, True)
    assert load(root).config.revision == rev
    assert run("add", task, "red", photos / "red", "--strict").exit_code == 1
    r = run("import", task, photos, "--strict")
    assert (r.exit_code, "Skipped 6 images that were there already (--strict)" in r.output) == (1, True)
    assert run("undo", task).stderr.startswith("t: undid add 6 examples to blue, red")


def test_import_points_at_import_task_for_a_zip(run, task, tmp_path):
    """Importing an export's zip with `import` says to use import-task."""
    z = tmp_path / "t.zip"
    z.write_bytes(b"PK")
    r = run("import", task, z)
    assert f"{z} looks like an export; use `fewshot import-task {z}`" in r.output


def test_predict_reports_missing_files_as_rows(run, task, tmp_path):
    """A missing path doesn't stop predict: it gets an error row, the others are printed, and the
    command exits 1 at the end."""
    q = tmp_path / "q.png"
    q.write_bytes(png(RED, seed=9))
    r = run("predict", task, q, tmp_path / "missing.png", "--json")
    assert r.exit_code == 1
    rows = json.loads(r.stdout)
    assert [(x["label"], x["error"] is not None) for x in rows] == [("red", False), (None, True)]
    assert "1 of 2 images could not be read" in r.stderr


def test_reading_paths_from_a_terminal_says_so(run, task, tmp_path):
    """--from-file - on a terminal says it is waiting for paths."""
    q = tmp_path / "q.png"
    q.write_bytes(png(RED, seed=9))
    with terminal():
        r = run("predict", task, "--from-file", "-", stdin=f"{q}\n")
    assert r.exit_code == 0
    assert "Reading image paths from stdin, one per line; Ctrl-D to finish" in r.stderr


def test_progress_without_a_terminal_prints_lines(monkeypatch, capsys):
    """Without a terminal, a long run prints its progress every 10%, under the stage's name."""
    monkeypatch.setattr(cli_common, "PLAIN_AFTER", 0.0)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    with cli_common.progress("Working") as update:
        update(0, None)
        for i in range(1, 11):
            update(i, 10, "evaluating")
        update(5, 10, "embedding")
        update(6, 10, "embedding")
        update(1, 1, "2/3 embedding")
    err = capsys.readouterr().err.splitlines()
    assert err[0] == "Cross-validating fold 1/10 (10%)"
    assert err[-3:] == ["Embedding 5/10 (50%)", "Embedding 6/10 (60%)", "Embedding (model 2/3) 1/1 (100%)"]
    assert len(err) == 13


# ---- errors name the fix ---------------------------------------------------------------
def test_errors_say_what_to_do(run, task, root, tmp_path):
    """Errors that have a next step name it."""
    run("create", "empty", "--embedder", "fake:color")
    q = tmp_path / "q.png"
    q.write_bytes(png(RED, seed=9))
    assert "add examples with `fewshot add empty CLASS IMAGE...`" in run("predict", "empty", q).output
    assert "add some with `fewshot add empty CLASS IMAGE...`" in run("eval", "empty").output
    assert "see `fewshot models`" in run("config", task, "--embedder", "timm:").output
    out = tmp_path / "t.zip"
    run("export", task, out)
    assert "already exists; pass --name NEW" in run("import-task", out).output
    del root


def test_delete_without_a_terminal(run, task, root):
    """delete without --yes and without a terminal refuses, naming --yes."""
    r = run("delete", task)
    assert (r.exit_code, r.output.strip()) == (1, "Error: Refusing to delete task t without confirmation; pass --yes")
    with terminal():
        r = run("rm", task, stdin="n\n")
    assert (r.exit_code, "Not deleted" in r.output) == (1, True)
    assert "Deleted task t" in run("rm", task, "-y").stderr
    assert not (root / "tasks" / task).exists()


# ---- evaluation -------------------------------------------------------------------------
def test_eval_json_names_sources(run, task):
    """eval's predictions and mistakes carry each example's file name."""
    r = json.loads(run("eval", task, "--json").stdout)
    assert {p["source"] for p in r["predictions"]} == {f"{i}.png" for i in (0, 1, 2, 10, 11, 12)}


@pytest.mark.parametrize(("ece", "after", "said"), [
    (0.2, 0.05, "--temperature 2 would make it 0.050"),
    (0.016, 0.02, "Calibration error 0.016: already well calibrated; no temperature change needed"),
    (0.2, 0.198, "Calibration error 0.200; no --temperature would lower it"),
])
def test_a_temperature_is_suggested_only_when_it_helps(run, task, monkeypatch, ece, after, said):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """eval suggests a temperature only when the evaluation says it helps (lowers the calibration
    error by a margin), and says why not otherwise."""
    real = services.evaluate

    def evaluate(*a, **k):
        r = real(*a, **k)
        r["calibration"] = {"temperature": 2, "ece": ece, "ece_calibrated": after, "reliability": [],
                            "reliability_calibrated": [], "temperature_helps": after < ece - TEMPERATURE_MARGIN}
        return r

    monkeypatch.setattr(services, "evaluate", evaluate)
    assert said in run("eval", task).stdout


def test_eval_mistakes_and_dupes_show_sources(run, root, tmp_path):
    """eval's mistakes and dupes show ids at one length with the file names."""
    run("create", "m", "--embedder", "fake:color")
    for i in range(3):
        (tmp_path / f"r{i}.png").write_bytes(png(RED, noise=5, seed=i))
        (tmp_path / f"b{i}.png").write_bytes(png(BLUE, noise=5, seed=i))
    run("add", "m", "red", *(tmp_path / f"r{i}.png" for i in range(3)), tmp_path / "b0.png")
    run("add", "m", "blue", tmp_path / "b1.png", tmp_path / "b2.png")
    out = run("eval", "m").stdout
    assert "blue -> red" not in out
    assert "red -> blue" in out
    assert "b0.png" in out
    eid = next(e.id for e in load(root, "m").config.examples if e.source == "b0.png")
    assert f"  {eid[:12]}  red -> blue" in out
    pairs = json.loads(run("dupes", "m", "--threshold", "0.5", "--json").stdout)
    assert all(len(p["sources"]) == 2 and p["sources"][0].endswith(".png") for p in pairs)
    r = run("dupes", "m", "--threshold", "0.5")
    assert len(r.stdout.splitlines()) == len(pairs)
    assert f"{len(pairs)} pairs at similarity >= 0.5" in r.stderr


@pytest.fixture(name="bench_data")
def bench_data_fixture(tmp_path):
    """Three classes of 4 images."""
    folder = tmp_path / "bench"
    for i, rgb in enumerate((RED, BLUE, (30, 220, 30))):
        write_images(folder / f"c{i}", rgb, 4, start=10 * i)
    return folder


def test_bench_machine_readable_output(run, bench_data):
    """bench prints one record per run as JSON, JSON lines or CSV, with its seed; --shot still works."""
    args = [bench_data, "-e", "fake:color", "--classifier", "prototype", "--classifier", "knn", "--way", "3",
            "--query", "2", "--episodes", "3", "--shots", "1", "--seed", "7"]
    runs = json.loads(run("bench", *args, "--json").stdout)
    assert [(r["classifier"], r["shot"], r["seed"]) for r in runs] == [("prototype", 1, 7), ("knn", 1, 7)]
    assert set(runs[0]) == set(cli_eval.BENCH_FIELDS)
    lines = [json.loads(line) for line in run("bench", *args, "--jsonl").stdout.splitlines()]
    assert lines == runs
    rows = list(csv.DictReader(io.StringIO(run("bench", *args, "--csv").stdout)))
    assert [r["classifier"] for r in rows] == ["prototype", "knn"]
    text = run("bench", *args).stdout
    assert "acc 1.000 ± 0.000 (95% CI)  seed 7" in text
    old = run("bench", bench_data, "-e", "fake:color", "--way", "3", "--query", "2", "--episodes", "2", "--shot", "2")
    assert "3-way 2-shot" in old.stdout
    assert "at most one of --json, --jsonl and --csv" in run("bench", *args, "--json", "--csv").output


def test_bench_with_too_few_images_says_what_to_lower(run, bench_data):
    """Too few classes with enough images: the error names the options to lower."""
    r = run("bench", bench_data, "-e", "fake:color", "--query", "10")
    assert "lower --way (0 = all), --shots or --query" in r.output


# ---- maintenance ------------------------------------------------------------------------
def test_cache_and_gc_json_and_dry_run(run, task, root):
    """cache info, cache prune and gc print JSON; gc --dry-run removes nothing."""
    info = json.loads(run("cache", "info", "--json").stdout)
    assert info["path"] == str(root / "cache.sqlite")
    assert info["models"] == []
    run("eval", task)
    info = json.loads(run("cache", "info", "--json").stdout)
    assert info["models"][0]["vectors"] >= 6
    assert info["bytes"] > 0
    pruned = json.loads(run("cache", "prune", "--dry-run", "--json").stdout)
    assert pruned.pop("cache_bytes") > 0
    assert pruned == {"dry_run": True, "vectors": 0, "stored_images": 0}
    stray = root / "tasks" / ".tmp-crashed"
    stray.mkdir()
    os.utime(stray, (0, 0))
    unused = load(root).path / "images" / "unused.png"
    unused.write_bytes(b"x")
    assert json.loads(run("gc", "--dry-run", "--json").stdout) == {"dry_run": True, "folders": 1, "images": 1,
                                                                   "blobs": 0}
    assert stray.exists()
    assert unused.exists()
    assert "Would remove 1 leftover folders, 1 unused image files" in run("gc", "--dry-run").stderr
    assert json.loads(run("gc", "--json").stdout)["folders"] == 1
    assert not stray.exists()
    assert not unused.exists()
    empty = PlainResult(CliRunner().invoke(cli.app, ["cache", "prune", "--json", "--root", str(root / "none")]))
    assert json.loads(empty.stdout)["vectors"] == 0


def test_gc_dry_run_counts_unused_stored_images(ws):
    """A dry run counts the stored images no task uses without deleting them."""
    stray = ws.blobs / "ab" / "abcdef.png"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"x")
    os.utime(stray, (0, 0))
    assert ws.gc(dry_run=True)["blobs"] == 1
    assert stray.exists()
    assert ws.gc()["blobs"] == 1
    assert not stray.exists()


# ---- export and import through pipes -------------------------------------------------------
def test_export_to_stdout_and_import_from_stdin(run, task, root):
    """`export TASK -` writes the zip to stdout (not to a terminal), and `import-task -` reads one."""
    runner = CliRunner()
    r = runner.invoke(cli.app, ["export", task, "-", "--root", str(root)])
    assert r.exit_code == 0, r.output
    data = r.stdout_bytes
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert "task.json" in z.namelist()
    assert "Exported t (6 examples) to stdout" in r.stderr
    r = runner.invoke(cli.app, ["import-task", "-", "--name", "u", "--root", str(root)], input=data)
    assert r.exit_code == 0, r.output
    assert len(load(root, "u").config.examples) == 6
    with terminal("stdout"):
        r = run("export", task, "-")
    assert "Refusing to write a zip to the terminal" in r.output


# ---- aliases ------------------------------------------------------------------------------
def test_aliases(run, task, root, tmp_path):
    """list/ls and delete/rm work in every group, `example add` is `add`, and old spellings stay."""
    assert run("ls").stdout == run("list").stdout
    assert run("example", "list", task).stdout == run("example", "ls", task).stdout
    q = tmp_path / "q.png"
    q.write_bytes(png((200, 200, 20)))
    assert "Added 1 to yellow" in run("example", "add", task, "yellow", q).stderr
    eid = load(root).find_examples("yellow")[0].id
    assert run("example", "delete", task, eid[:8]).exit_code == 0
    run("inbox", "add", task, q)
    assert run("inbox", "list", task).stdout == run("inbox", "ls", task).stdout
    assert "Note cleared" in run("example", "note", task, load(root).config.examples[0].id[:8], "").stderr


def test_watch_stopped_goes_to_stderr(run, task, tmp_path, monkeypatch):
    """Stopping watch with Ctrl-C says "Stopped" on stderr and exits 0."""
    (tmp_path / "in").mkdir()

    def stop(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr("fewshot.cli_predict.time.sleep", stop)
    r = run("watch", task, tmp_path / "in", "--out", tmp_path / "out")
    assert (r.exit_code, r.stdout, r.stderr.strip()) == (0, "", "Stopped")


def test_a_class_deleted_meanwhile_is_an_error(run, task, monkeypatch):
    """A class that goes away between the check and the change is reported, not a traceback."""

    def gone(*_a, **_k):
        raise KeyError("No class 'red'")

    monkeypatch.setattr("fewshot.store.Task.set_class_thresholds", gone)
    monkeypatch.setattr("fewshot.store.Task.delete_class", gone)
    assert "Error: No class 'red'" in run("class", "threshold", task, "red", "0.5").output
    assert "Error: No class 'red'" in run("class", "rm", task, "red").output


def test_quiet_hides_warnings_and_progress(run, task, monkeypatch, capsys):
    """-q hides warnings too, and the same percentage isn't printed twice."""
    run("class", "add", task, "lonely")
    assert run("list", "-q").stderr == ""
    monkeypatch.setattr(cli_common, "PLAIN_AFTER", 0.0)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    with cli_common.progress("Working") as update:
        update(1, 100)
        update(2, 100)
    assert capsys.readouterr().err == "Working 1/100 (0%)\n"


def test_serve_turns_load_reporting_off(monkeypatch):
    """The server loads models on several threads, so the CLI's load reporting (which flips the
    Hub's offline switch for the process) is off while it runs."""
    from fewshot import embedders  # pylint: disable=import-outside-toplevel

    seen = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.append(list(embedders._reporter)))  # pylint: disable=protected-access
    assert CliRunner().invoke(cli.app, ["serve"]).exit_code == 0
    assert seen == [[]]
