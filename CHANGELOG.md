# Changelog

## Unreleased

### Command line

Help and discovery
- `-h` works everywhere, as `--help` does, and `fewshot --version` prints the versions of fewshot, torch, open_clip and Python, for reporting results.
- `fewshot -h` lists commands by topic (Tasks, Labelling, Predicting, Evaluating, Maintenance, Server), each with its whole one-line summary, and ends with the environment variables and exit statuses. Every command's help describes its arguments and ends with examples.
- Aliases: `ls`/`list` and `rm`/`delete` in every group (and at the top level), `example add` (the same as `add`), `bench --shots` (`--shot` still works), and `-e` for `--embedder` in `create`, `config`, `bench` and `import-task`.
- New `fewshot class ls TASK [--json]`.
- New global options: `-q`/`--quiet` (`FEWSHOT_QUIET`), `--debug` (`FEWSHOT_DEBUG`) and `--device` (`FEWSHOT_DEVICE`).

Output
- **Changed:** confirmations, hints, warnings and counts are printed on stderr, not stdout, so stdout holds only results (`dupes | wc -l` counts pairs; `list` with no tasks prints nothing). Scripts that read lines such as "Imported 40 images" or "Exported ..." from stdout must read stderr.
- `add` and `import` say what changed: how many were added, moved from which classes and skipped as already there, and the task's new size; `undo` and `redo` say how the number of examples and classes changed. `create` and a first `import` say what to do next.
- `list` has a header on a terminal, explains the `*` of an outdated evaluation, and prints warnings on stderr; `show` aligns its columns and prints `on`/`off`/`none`.
- `bench` takes `--json`, `--jsonl` and `--csv` (one record per run: embedder, classifier, way, shot, query, episodes, seed, text template, flip, accuracy, std, ci95) and `--seed`, and labels its interval as the 95% CI. `cache info`, `cache prune` and `gc` take `--json`. `eval --json`/`--csv` and `dupes` give each example's file name (`source`, `sources`), and `eval --json` says whether a temperature would help (`calibration.temperature_helps`).
- `eval` suggests a `--temperature` only when it would lower the calibration error.
- Ids are shown at one length (12 characters) everywhere.
- The first model load is announced with its device; weights are loaded from the Hugging Face cache without contacting the Hub, and a download is announced. Library warnings are hidden unless `--debug`. Without a terminal, long runs print their progress every 10%.

Behaviour
- **Changed:** `add` (and `inbox accept --label`, `example relabel`) refuses a new class whose name is close to an existing class's (`abysinian` for `Abyssinian`) as a likely typo; `--new-class` creates it anyway. A new class is announced.
- **Changed:** a missing file given to `predict`, `add` or `inbox add` no longer stops the command: `predict` gives it an error row, the others skip it, and the command exits 1 at the end.
- **Changed:** `inbox add` no longer exits 1 when images were only skipped as labelled or waiting already; `--strict` (also on `add` and `import`) makes any skip a failure.
- Adding or importing images that are all there already no longer creates a revision or undo step (in the library and the API too).
- `delete` without `--yes` and without a terminal fails at once, naming `--yes`; its prompt says it can't be undone. `class rm` and `example rm` say how to undo. `gc --dry-run` reports without removing.
- `export TASK -` writes the zip to stdout (refusing a terminal) and `import-task -` reads one from stdin.
- Errors name the next step: the workspace and closest task for a missing task, the closest class for a missing class, `fewshot models` for a bad embedder, the options to lower for a `bench` dataset that is too small, `--name` for an imported task that exists, `import-task` for a zip given to `import`. An unknown `--param` key is a usage error (status 2), and `--param`'s help lists each classifier's parameters.
- Creating the workspace is announced, and `list` says where it looked when it finds no tasks.
- Ctrl-C prints "Interrupted" (and what was kept), with no traceback, and exits 130.

### Library
- `evaluate.benchmark` also returns `seed` and `std`, and `Workspace.gc` takes `dry_run`.
