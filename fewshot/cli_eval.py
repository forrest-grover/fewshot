"""`fewshot eval`, `compare`, `dupes` and `bench`: measuring a task, its models and its labels."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from . import services
from .cli_common import (
    CLASSIFIER_LIST,
    ID_LEN,
    JSON_OPT,
    ROOT_OPT,
    TASK_ARG,
    App,
    classifier_names,
    echo_json,
    examples,
    explained,
    fail,
    fail_on,
    get_task,
    make_pipeline,
    one_format,
    progress,
    say,
    usage,
    workspace,
    write_csv,
)
from .cli_common import FLAGS as _FLAGS
from .embedders import DEFAULT_SPEC, PRESETS, spec_supports_text

if TYPE_CHECKING:
    from .pipeline import Pipeline

commands = App()
WELL_CALIBRATED = 0.05  # a calibration error under which confidence is called well calibrated


@commands.command("eval", rich_help_panel="Evaluating", epilog=examples(
    "fewshot eval pets",
    "fewshot eval pets --csv > held-out.csv        # every example's held-out prediction",
    "fewshot eval pets --apply-class-thresholds    # also set each class's suggested threshold"))
def eval_cmd(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    name: str = TASK_ARG,
    folds: int | None = typer.Option(
        None, min=2, help="Number of folds [default: leave-one-out up to 100 examples (300 for prototype and "
                   "knn), then 10, then 5 above 300]"),
    as_json: bool = JSON_OPT,
    as_csv: bool = typer.Option(False, "--csv", help="Print CSV: each example's held-out prediction"),
    apply_class_thresholds: bool = typer.Option(
        False, "--apply-class-thresholds", help="Set each class's reject threshold to the one suggested"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Cross-validate a task on its own labelled examples.

    Each example is predicted by a classifier fitted without it. Prints accuracy, macro-F1,
    per-class recall and precision, the most confident mistakes, a suggested --reject-below,
    per-class thresholds and a calibration temperature (when it would help). The result is shown
    by `fewshot list`.
    """
    fmt = one_format(json=as_json, csv=as_csv)
    t = get_task(root, name)
    _FLAGS.interrupt_note = "no task was changed"
    try:
        with progress("Evaluating") as update:
            r = services.evaluate(make_pipeline(workspace(root)), t, folds,
                                  lambda stage, done, total: update(done, total, stage), apply_class_thresholds)
    except ValueError as e:
        fail_on(e, name)
    except KeyError as e:  # a class was renamed or deleted meanwhile
        fail(f"Thresholds not set: {e.args[0]}")
    if apply_class_thresholds and r["suggested_class_thresholds"]:
        say(f"Set the thresholds of {len(r['suggested_class_thresholds'])} classes")
    if fmt == "json":
        echo_json(r)
    elif fmt == "csv":
        write_csv(["id", "source", "true", "pred", "correct", "confidence", "similarity"],
                  [[p["id"], p["source"], p["true"], p["pred"], p["correct"], p["confidence"], p["similarity"]]
                   for p in r["predictions"]])
    else:
        _print_eval(r, apply_class_thresholds)


def _print_eval(r: dict[str, Any], applied: bool) -> None:
    acc = r["accuracy"]
    if acc is None:
        typer.echo("Nothing scored")
    else:
        typer.echo(f"{r['method']}: accuracy {acc:.3f} on {r['n_scored']}/{r['n_examples']} examples, "
                   f"macro-F1 {r['macro_f1']:.3f}")
        topk = "  ".join(f"top-{k} {v:.3f}" for k, v in r["top_k_accuracy"].items())
        if topk:
            typer.echo(f"  {topk}")
    if r["suggested_reject_below"] is not None:
        typer.echo(f"Suggested --reject-below {r['suggested_reject_below']} (keeps 95% of correct predictions)")
    if cal := r["calibration"]:
        if cal["temperature_helps"]:
            typer.echo(f"Calibration error {cal['ece']:.3f}; --temperature {cal['temperature']} would make it "
                       f"{cal['ece_calibrated']:.3f}")
        elif cal["ece"] < WELL_CALIBRATED:
            typer.echo(f"Calibration error {cal['ece']:.3f}: already well calibrated; no temperature change needed")
        else:
            typer.echo(f"Calibration error {cal['ece']:.3f}; no --temperature would lower it")
    suggested = r["suggested_class_thresholds"]
    width = max((len(c) for c in r["per_class"]), default=0)
    for c, m in r["per_class"].items():
        rec = "-" if m["recall"] is None else f"{m['recall']:.2f}"
        prec = "-" if m["precision"] is None else f"{m['precision']:.2f}"
        thr = f"  suggested threshold {suggested[c]}" if c in suggested else ""
        typer.echo(f"  {c:{width}}  n={m['support']:<4} recall={rec:5} precision={prec:5}{thr}")
    if suggested and not applied:
        say("(--apply-class-thresholds sets the suggested per-class thresholds)")
    if r["mistakes"]:
        typer.echo(f"{len(r['mistakes'])} mistakes; most confident:")
        for m in r["mistakes"][:10]:
            typer.echo(f"  {m['id'][:ID_LEN]}  {m['true']} -> {m['pred']} ({m['confidence']:.2f})  {m['source']}")


@commands.command(rich_help_panel="Evaluating", epilog=examples(
    "fewshot compare pets -e timm:vit_small_patch14_dinov2.lvd142m",
    "fewshot compare pets --presets --json"))
def compare(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # one per CLI option
    name: str = TASK_ARG,
    embedder: list[str] = typer.Option([], "--embedder", "-e", help="An embedder to try (repeatable); the task's "
                                                                   "own is always included"),
    presets: bool = typer.Option(False, "--presets", help="Also try every preset (see `fewshot models`)"),
    folds: int | None = typer.Option(None, min=2, help="Number of folds [default: as for eval]"),
    as_json: bool = JSON_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """Cross-validate the task with other embedders, to choose a model.

    Each embedder is evaluated on the task's own labelled examples with its other settings; nothing
    changes until you switch with `fewshot config TASK --embedder ...`. Exits 1 if any embedder
    failed (the others are still reported).
    """
    from .evaluate import compare_embedders  # pylint: disable=import-outside-toplevel  # torch-backed

    t = get_task(root, name)
    specs = services.check_specs([t.config.embedder, *embedder, *([p.spec for p in PRESETS] if presets else [])])
    if len(specs) < 2:
        usage("Give another --embedder to compare with, or --presets")
    _FLAGS.interrupt_note = "no task was changed"
    try:
        with progress("Comparing") as update:
            r = compare_embedders(make_pipeline(workspace(root)), t, specs, folds,
                                  progress=lambda stage, done, total: update(done, total, stage))
    except ValueError as e:
        fail_on(e, name)
    if as_json:
        echo_json(r)
        _fail_if_any_failed(r)
        return
    for x in r["results"]:
        mark = ("*" if x["embedder"] == r["best"] else " ") + ("<" if x["embedder"] == r["current"] else " ")
        if x.get("error"):
            typer.echo(f"{mark} {'failed':>8}  {x['embedder']}: {x['error']}")
        elif x["accuracy"] is None:
            typer.echo(f"{mark} {'-':>8}  {x['embedder']}: nothing scored")
        else:
            typer.echo(f"{mark} {x['accuracy']:8.3f}  macro-F1 {x['macro_f1']:.3f}  {x['seconds']:7.1f}s  "
                       f"{x['embedder']}")
    say(f"* most accurate, < current ({r['n_examples']} examples)")
    if r["best"] and r["best"] != r["current"]:
        say(f"To switch: fewshot config {name} --embedder {r['best']}")
    _fail_if_any_failed(r)


def _fail_if_any_failed(r: dict[str, Any]) -> None:
    """Exit 1 if any embedder of a comparison failed."""
    if failed := [x["embedder"] for x in r["results"] if x.get("error")]:
        fail("Failed: " + ", ".join(failed))


@commands.command(rich_help_panel="Labelling", epilog=examples("fewshot dupes pets",
                                                               "fewshot dupes pets --threshold 0.9 --json"))
def dupes(
    name: str = TASK_ARG,
    threshold: float = typer.Option(0.95, min=0.5, max=1.0, help="Cosine similarity at or above which "
                                                                  "two examples count as near-duplicates"),
    limit: int = typer.Option(200, min=1, help="Show at most this many pairs"),
    as_json: bool = JSON_OPT,
    root: Path | None = ROOT_OPT,
) -> None:
    """List near-duplicate examples, most similar first.

    Pairs with different labels are marked: they are often labelling mistakes. Identical files are
    stored once already, so these are re-encodes, crops and the like.
    """
    t = get_task(root, name)
    try:
        pairs = make_pipeline(workspace(root)).near_duplicates(t, threshold, limit)
    except ValueError as e:  # the model can't be loaded, say
        fail_on(e, name)
    if as_json:
        echo_json(pairs)
        return
    for d in pairs:
        mark = "" if d["same_label"] else "  DIFFERENT LABELS"
        typer.echo(f"{d['similarity']:.3f}  {d['a'][:ID_LEN]} ({d['labels'][0]}) {d['sources'][0]}  "
                   f"{d['b'][:ID_LEN]} ({d['labels'][1]}) {d['sources'][1]}{mark}")
    say(f"{len(pairs)} pairs at similarity >= {threshold}")


# ---- bench ---------------------------------------------------------------------
BENCH_FIELDS = ["embedder", "classifier", "way", "shot", "query", "episodes", "seed", "text_template", "flip",
                "accuracy", "std", "ci95"]


def _bench_runs(specs: list[str], classifiers: list[str], shots: list[int], text_template: str | None
                ) -> Iterator[tuple[str, str, int, str | None]]:
    """Yield (embedder, classifier, shot, text_template) combinations; text is used only by CLIP embedders,
    and 0-shot runs need it, so they are skipped otherwise."""
    for e in specs:
        tt = text_template if spec_supports_text(e) else None
        for c in classifiers:
            for s in shots:
                if s > 0 or tt:
                    yield e, c, s, tt


def _bench_one(pipe: Pipeline, folder: Path, run: tuple[str, str, int, str | None],
               opts: dict[str, Any]) -> dict[str, Any]:
    # evaluate imports the torch-backed pipeline; importing it lazily keeps `fewshot --help` fast.
    from .evaluate import benchmark  # pylint: disable=import-outside-toplevel

    e, c, s, tt = run
    try:
        r = benchmark(pipe, folder, e, c, shot=s, text_template=tt, progress=say, **opts)
    except ValueError as err:
        msg = explained(err)
        fail(f"{msg}; lower --way (0 = all), --shots or --query" if msg.startswith("Only ") else msg)
    return {k: r[k] for k in BENCH_FIELDS}


def _bench_row(r: dict[str, Any]) -> str:
    text = " +text" if r["text_template"] else "      "
    return (f"{r['embedder']:42} {r['classifier']:9} {r['way']}-way {r['shot']}-shot{text}  "
            f"acc {r['accuracy']:.3f} ± {r['ci95']:.3f} (95% CI)  seed {r['seed']}")


@commands.command(rich_help_panel="Evaluating", epilog=examples(
    "fewshot bench ./photos --shots 1 --shots 5",
    "fewshot bench ./photos -e timm:vit_small_patch14_dinov2.lvd142m -e openclip:PE-Core-B-16@meta --json",
    "fewshot bench ./photos --way 0 --query 50 --episodes 3 --shots 1 --shots 16 --csv > results.csv"))
def bench(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals  # one per CLI option
    folder: Path = typer.Argument(..., metavar="FOLDER", help="A folder of class folders: FOLDER/<class>/<image>",
                                  show_default=False),
    embedder: list[str] = typer.Option(None, "--embedder", "-e",
                                       help=f"Embedder spec(s) to compare [default: {DEFAULT_SPEC}]"),
    classifier: list[str] = typer.Option(["gda"], callback=classifier_names,
                                         help=f"Classifier(s) to compare: {CLASSIFIER_LIST}"),
    way: int = typer.Option(5, min=0, help="Classes per episode; 0 for every class with enough images"),
    shots: list[int] = typer.Option(None, "--shots", min=0, help="Labelled examples per class (repeatable) "
                                                                  "[default: 1 and 5]"),
    shot: list[int] = typer.Option(None, "--shot", min=0, hidden=True),  # the old spelling of --shots
    query: int = typer.Option(15, min=1, help="Images per class to classify in each episode"),
    episodes: int = typer.Option(200, min=1, help="Random episodes per run, averaged"),
    seed: int = typer.Option(0, help="Seed of the random episodes; the same seed gives the same episodes"),
    text_template: list[str] = typer.Option(
        None, help="e.g. 'a photo of a {}' to use class names as text (CLIP only). Repeat to average "
                   "several prompts."),
    flip: bool = typer.Option(False, "--flip", help="Average each image with its mirror image"),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON list: one object per run"),
    as_jsonl: bool = typer.Option(False, "--jsonl", help="Print one JSON object per run as it finishes"),
    as_csv: bool = typer.Option(False, "--csv", help="Print CSV: one row per run"),
    root: Path | None = ROOT_OPT,
) -> None:
    """Episodic N-way K-shot benchmark over a folder of class folders.

    Each run (embedder x classifier x shots) samples random episodes of --way classes with --shots
    labelled and --query test images each, and reports the mean accuracy with its 95% confidence
    interval (1.96 standard errors over episodes). Results are reproducible for a given --seed.
    Embeddings go into the workspace's cache, so a second run is fast; `fewshot cache prune`
    removes them once no task uses them.
    """
    fmt = one_format(json=as_json, jsonl=as_jsonl, csv=as_csv)
    if not folder.is_dir():
        fail(f"Not a directory: {folder}")
    _FLAGS.interrupt_note = "no task was changed"
    pipe = make_pipeline(workspace(root, create=True))
    opts = {"way": way, "query": query, "episodes": episodes, "flip": flip, "seed": seed}
    template = "\n".join(text_template) if text_template else None
    results = []
    for n, run in enumerate(_bench_runs(embedder or [DEFAULT_SPEC], classifier, [*(shots or []), *(shot or [])]
                                        or [1, 5], template)):
        r = _bench_one(pipe, folder, run, opts)
        results.append(r)
        if fmt == "jsonl":
            typer.echo(json.dumps(r))
        elif fmt == "csv":
            write_csv(BENCH_FIELDS, [[r[k] for k in BENCH_FIELDS]], with_header=n == 0)
        elif fmt == "text":
            typer.echo(_bench_row(r))
    if fmt == "json":
        echo_json(results)
