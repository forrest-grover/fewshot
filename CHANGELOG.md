# Changelog

## Unreleased

### Command line

Help and discovery
- `-h` works everywhere, as `--help` does, and `fewshot --version` prints the versions of fewshot, torch, open_clip and Python, for reporting results.
- `fewshot -h` lists commands by topic (Tasks, Labelling, Predicting, Evaluating, Maintenance, Server), each with its whole one-line summary, and ends with the environment variables and exit statuses. Every command's help describes its arguments and ends with examples.
- Aliases: `ls`/`list` and `rm`/`delete` in every group (and at the top level), `example add` (the same as `add`), `bench --shots` (`--shot` still works), and `-e` for `--embedder` in `create`, `config`, `bench` and `import-task`.
- New `fewshot class ls TASK [--json]`; its JSON has the same keys as `show --json`'s classes and the API's (`name`, `description`, `count`, `reject_below`).
- New global options: `-q`/`--quiet` (`FEWSHOT_QUIET`), `--debug` (`FEWSHOT_DEBUG`) and `--device` (`FEWSHOT_DEVICE`). These and `--root` go before or after the command name.
- A bad `--device`, or one this machine doesn't have, is a usage error before the command runs, naming `--device` (or `FEWSHOT_DEVICE` when that set it).

Output
- **Changed:** confirmations, hints, warnings and counts are printed on stderr, not stdout, so stdout holds only results (`dupes | wc -l` counts pairs; `list` with no tasks prints nothing). Scripts that read lines such as "Imported 40 images" or "Exported ..." from stdout must read stderr.
- `add` and `import` say what changed: how many were added, moved from which classes and skipped as already there, and the task's new size; `undo` and `redo` say how the number of examples and classes changed. `create` and a first `import` say what to do next.
- `list` has a header on a terminal, explains the `*` of an outdated evaluation, and prints warnings on stderr; `show` aligns its columns and prints `on`/`off`/`none`.
- `bench` takes `--json`, `--jsonl` and `--csv` (one record per run: embedder, classifier, way, shot, query, episodes, seed, text template, flip, accuracy, std, ci95) and `--seed`, and labels its interval as the 95% CI. `cache info`, `cache prune` and `gc` take `--json`. `eval --json`/`--csv` and `dupes` give each example's file name (`source`, `sources`).
- `eval` suggests a `--temperature` only when it would lower the calibration error by more than 0.005 (see the API's `temperature_helps`).
- Ids are shown at one length (12 characters) everywhere.
- The first model load is announced with its device; weights are loaded from the Hugging Face cache without contacting the Hub, and a download is announced. Library warnings are hidden unless `--debug`. Without a terminal, long runs print their progress every 10%.

Behaviour
- **Changed:** `add` (and `inbox accept --label`, `example relabel`) refuses a new class whose name is close to an existing class's (`abysinian` for `Abyssinian`) as a likely typo; `--new-class` creates it anyway. A new class is announced.
- **Changed:** `import` refuses, importing nothing, a folder with class folders whose names are close to existing classes (`beagle/` for `Beagle`), naming them; `--new-class` imports them as new classes. The classes an import creates are named.
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
- An evaluation's `calibration` has `temperature_helps`: whether its temperature lowers the calibration error by more than `evaluate.TEMPERATURE_MARGIN` (0.005).
- `embedders.check_device(device, name)` checks a device; a bad `FEWSHOT_DEVICE` is reported as such by `get_embedder`, not as a model that could not be loaded.

### API and web UI
- **Changed:** the evaluation report's `calibration` has a new field, `temperature_helps`. The Evaluate tab offers a temperature only when it is true, as the CLI does.

Errors
- **Changed:** every error, the framework's 404 and 405 included, is an RFC 9457 problem (`application/problem+json`): `{type, title, status, detail, instance, code}` plus members such as `task`, `label`, `similar_to` and `pointer`. `detail` stays a one-line string; `code` is stable (see the README's code table). The one break is the 422 body: its list of problems moved from `detail` to `errors` (each entry has `pointer` or `parameter`, `detail` and `code`, and still pydantic's `loc` and `msg`, deprecated, for one version), and `detail` is now a summary sentence. A 422 lists every problem at once.
- **Changed:** statuses follow one policy (README, Errors): 409 for conflicts (creating, renaming, duplicating or importing onto a task name that is taken, renaming a class onto another, undo or redo with nothing to do), which were 400; 422, which was 400, for a field out of its range or choices (`folds` below 2, `text_weight` outside 0–100, a reject threshold outside -1–1, a temperature outside 0.01–100, an unknown classifier, an empty `class-thresholds` body); 415 for an upload route sent something other than a multipart form, and for a JSON route sent a body that isn't `application/json` (a form, or `curl -d` without a content type; it was a 422 saying the body wasn't an object); 412 for a failed `If-Match`. `/openapi.json` declares the error responses of every route, as `application/problem+json`.
- A body that isn't valid JSON is a 422 saying where parsing stopped (`the body isn't valid JSON (at character 8): Expecting value`), with an empty `pointer`.
- A problem's `type` is `urn:fewshot:problem:<code>`, a name rather than a link.
- A missing task's 404 names the closest tasks (`similar_to`) and how to create one; a missing token (`NOT_SIGNED_IN`) is told apart from a wrong one (`WRONG_TOKEN`). An unexpected error is a 500 `INTERNAL` that is logged and says nothing more.

Requests
- **Changed:** a label close to an existing class's (the same but for case, or one or two letters off) is refused with 422 `CLASS_NAME_NEAR_MISS` on every route that labels images (uploads, `.../examples/urls`, relabelling one or several examples, accepting inbox images), as the CLI refuses it; send `new_class=true` to create it. Scripts that relied on the API creating such names must send it.
- **Changed:** unknown query parameters (`?topk=`) are a 422, as unknown body fields are; `GET .../examples?label=` with a class the task doesn't have is a 404 `CLASS_NOT_FOUND`, not an empty page.
- `PATCH /api/tasks/{name}` is a JSON Merge Patch: `null` clears `reject_below`, `reject_below_zero_shot` or `temperature` (it was ignored), and null for another field is a 422. `clear_reject`, `clear_reject_zero_shot` and `clear_temperature` still work, are deprecated in the schema, and are answered with `Deprecation: true`.
- Responses carrying a task send `ETag: "<revision>"`, and changes to a task honour `If-Match` (412 `PRECONDITION_FAILED` with the current `revision` if it moved on), compared under the task's write lock with the change, so of several changes sent at once with the same `If-Match` only one is made; without `If-Match` nothing changes. The schema declares `If-Match` on every route that changes a task, and the `ETag` of every response that carries one.

Responses
- **Changed:** uploads say what was deduplicated: `added` lists only the examples (or inbox images) new to the task, `unchanged` those there already, `moved` those moved from another class (they were in `added` before), and `created_classes` the class an upload created. An upload that stores nothing new answers 200, not 201; 400 `NO_IMAGES_ADDED` is only for an upload that could add nothing (no readable image, or for the inbox, only images that are examples already), and lists each file and why in `errors` (`[{file, error}]`, as a successful upload's `errors`).
- Accepting inbox images and relabelling one or several examples answer the task with `created_classes`, the classes the change created (as uploads do).
- **Changed:** a prediction for an unreadable image has `top_label`, `confidence` and `similarity` null (they were `""`, 0.0 and 0.0), in the API and in `fewshot predict --json`.
- **Changed:** `POST .../evaluate/jobs` and `.../compare/jobs` refuse a task without examples at once (400 `NO_EXAMPLES`) instead of starting a job that fails. A job's 202 carries `Location` and `Retry-After`, a running job's `GET` carries `Retry-After`, `status` is an enum in the schema, and a failed job has `problem` (with its `code`) beside `error`. 429 `TOO_MANY_JOBS` carries `Retry-After`.
- Evaluation reports give each held-out prediction's `source`, and duplicate pairs their `sources`, as the CLI's JSON does. `/api/health` reports `version` and `api_version` (`1`), every response carries `X-Fewshot-Version`, and the OpenAPI schema has the package version, tags, a summary per route and examples. The README states the compatibility policy.
- `fewshot list --json` and `GET /api/tasks` warn about a task with only one class that has examples, whose evaluation accuracy is always 1.0. Evaluating such a task is allowed (its suggested reject threshold is what one-class detection needs), and the evaluation report says so in a new field, `warnings`, which `fewshot eval` prints and the Evaluate tab shows.

### Python client
- `Client()` reads `FEWSHOT_URL` (default `http://127.0.0.1:8000`) and `FEWSHOT_TOKEN`.
- Errors are a hierarchy under `FewshotError`: `ApiError` (with `status`, `status_code`, `code`, `detail`, `body`, `fields`; `str()` is `CODE: detail`) and its subclasses by status (`BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `ConflictError`, `PreconditionFailedError`, `ValidationError` with `.errors`, `RateLimitError` with `.retry_after`, `ServerError`); `APIConnectionError` and `APITimeoutError`; `JobFailedError`, `JobCancelledError` and `JobResultExpiredError` (`JobError`s, with `job_id`). `PermissionDeniedError` (403) is not an `AuthenticationError` (401): it is about the server's settings, not the token. **Changed:** a failed connection was `ApiError` with status 0, and a failed, cancelled or expired evaluation `ApiError` with the made-up statuses 400, 409 and 410; they are no longer `ApiError`s (catch `FewshotError`). A file that can't be read given to `request(files=...)` raises `FewshotError` (it was `ApiError` 0).
- `start_evaluate` and `start_compare` return a `Job` (`.id`, `.status()`, `.wait(timeout=, on_progress=)`, `.cancel()`, `.result()`), `client.job(id)` comes back to one, and `evaluate` and the new `compare` wait for one (`evaluate(task, folds, poll)` keeps its positional arguments; the new ones are keyword-only), with `timeout=` (`APITimeoutError` with the job's id) and a Ctrl-C message naming the job.
- `inbox()` takes `offset` and `limit`; `iter_inbox`, `examples` and `iter_examples` page through the inbox and the examples.
- Images may be `bytes`, binary file objects and PIL images too; another type raises `TypeError` naming its index before anything is sent. A `predict` result for a file that can't be read has every key, as the server's.
- Upload results combine `added`, `unchanged`, `moved`, `created_classes` and `errors` over the batches (`moved` was dropped); a batch the server refuses adds each of its files' reasons to `errors`. `accept_inbox` returns `TaskEdited`, the task with `created_classes`. `add_examples` and `accept_inbox` take `new_class=`, and the methods that change a task `if_match=`.
- Methods return `TypedDict`s mirroring the API's schemas (`Prediction`, `TaskDetail`, `AddResult`, `InboxPage`, `EvalReport`, `JobInfo`...); they are plain dicts at runtime.

### Web UI
- A drop made while a cancelled upload's last batch is still being sent (or after switching tasks and back meanwhile) is its own undo step: it joined the cancelled upload's, so undoing it also undid the images the cancel said got in.
- Clearing a setting sends `null`. An "Add as" of an image that is an example already counts it as added, and a refused change reloads the task on 412 too.
