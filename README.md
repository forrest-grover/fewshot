# fewshot

[![CI](https://github.com/forrest-grover/fewshot/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/forrest-grover/fewshot/actions/workflows/ci.yml) [![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/forrest-grover/fewshot/badges/coverage.json)](https://github.com/forrest-grover/fewshot/actions/workflows/ci.yml)

Few-shot image classification from a handful of labeled examples per class. Images are embedded with a pretrained vision model (CLIP or DINOv2) and classified against the class examples, so there is no training run: adding, removing, or relabeling an example changes predictions immediately.

The same engine is exposed four ways: a Python library, a CLI, a REST API, and a web UI.

## Setup

```bash
conda create -p ./.env python=3.12 nodejs=22 -c conda-forge
./.env/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx needs cu128+
./.env/bin/pip install -e ".[dev]"
cd web && npm install && npm run build && cd ..
./.env/bin/fewshot serve            # UI at http://127.0.0.1:8000, API docs at /docs
```

Data lives in `$FEWSHOT_HOME` (default `./fewshot_data`): one folder per task with its images and `task.json`, plus a shared SQLite embedding cache. The server and CLI can use the same workspace at once: each change to a task locks it, starts from the latest saved `task.json`, and replaces the file atomically (the lock needs Linux or macOS), and the cache runs in WAL mode so one process's writes don't fail the other's. `fewshot cache info` shows the cache's size per model; `fewshot cache prune` deletes vectors no task's examples use, such as those of deleted examples and `bench` datasets. At most two built-in models stay loaded at once (`FEWSHOT_MAX_MODELS`); loading another unloads the least recently used. The web UI scales images down to at most 1024 px on the longest side (as JPEG) before uploading them as examples or for prediction; the CLI and API store whatever they are given. In the UI you can drag thumbnails between classes, tick several (Shift+click for a range) to move or delete them together, and paste images: on the Examples tab into the class under the pointer, on the Predict tab to classify them. Number keys 1–9 move ticked examples to that class, or add the highlighted prediction as that class (Enter accepts the prediction). Prediction and evaluation results stay while you switch tabs. The home page is a dashboard of all tasks: size, inbox, the last evaluation's accuracy (marked * when the task changed since) and warnings such as classes with too few examples or strong imbalance; `fewshot list` shows the same. On the Examples tab you can search file names, tags and notes, sort, and collapse classes; clicking a thumbnail opens it full size, where ←/→ step through the class, 1–9 move the example and Delete removes it. Press ? for every keyboard shortcut. A task can be duplicated (Settings, `fewshot copy`, `POST .../duplicate`), with or without its examples, to try other settings safely. For UI development, run `fewshot serve` and `npm run dev` in `web/`; Vite proxies `/api` to port 8000.

Every change to a task can be undone (`fewshot undo`, the Undo button or Ctrl+Z in the UI, `POST .../undo`) and redone until the next change (`fewshot redo`, Ctrl+Shift+Z, `POST .../redo`); each step is labelled, e.g. "delete 3 examples". The last 20 versions of `task.json` are kept in the task's `history/` folder, and a deleted example's image stays until no kept version names it, and `cache prune` keeps the vectors of images undo or redo may bring back. A damaged `cache.sqlite` is moved aside and a new cache started. Images are stored once per workspace, in `blobs/`, and hard-linked into each task that uses them; `fewshot cache prune` also removes stored images no task uses, and `fewshot gc` cleans up after crashes: half-made or half-deleted task folders, image files no version of their task uses, and unused stored images. Creating, importing and deleting a task move whole folders into place, so a crash never leaves half a task, and a task whose folder was renamed (by hand, or by a rename that was interrupted) opens under the folder's name. One undo takes back a whole upload or import, however large. Exports lock the task while they run. Examples can carry tags and a note (`fewshot example tag/note`, or in the UI the ✎ button on a thumbnail, or tick several and "Tag…", which open a small editor with suggestions from the task's tags), and the Examples tab can show only one tag. After an undo or redo the UI says what it was and offers the reverse. On a phone, the task list folds into a menu. A task can be renamed, and exported as a zip of its settings, labels and images and imported into another workspace (`fewshot export` / `fewshot import-task`, or Settings → Export and New task → Import).

Models run on CUDA if available, then Apple's GPU (MPS), then the CPU; `FEWSHOT_DEVICE` (e.g. `cpu`, `cuda:1`) picks one. If the GPU runs out of memory, the batch is split in halves; later batches stay smaller until 20 have fitted, then grow back. A model that is still embedding is never unloaded to make room for another. The SigLIP 2 preset needs `transformers` for its tokenizer (`pip install -e ".[siglip]"`).

## Inbox: labelling with the model's help

Each task has an inbox for images that aren't labelled yet (the Inbox tab, `fewshot inbox add TASK dir/`, `POST .../inbox`). The model suggests a class for each, and the inbox can be ordered so that the images it is least sure about come first (labelling those teaches it the most), the most confident come first (to accept them in bulk), or the least familiar come first (new kinds of image). On the Inbox tab, Enter accepts the highlighted suggestion, 1–9 label it as that class and Delete dismisses it; "Accept N suggestions" labels everything above a confidence. From the command line: `fewshot inbox ls TASK --sort uncertain`, `fewshot inbox accept TASK --min-confidence 0.9`, `fewshot inbox accept TASK ID... --label cat`, `fewshot inbox dismiss TASK ID...`. Labelling moves images to the examples; every step can be undone, and the inbox is part of exports.

## How classification works

Each task has an embedder, a classifier, and classes. A class has example images, an optional text description, or both.

| Classifier | Image score |
|---|---|
| `gda` (default) | Gaussian discriminant analysis: class means with a shared covariance shrunk towards a sphere, on centred features; its weight against the text and its confidence scale are fitted per task |
| `prototype` | cosine similarity to the class's mean embedding |
| `knn` | similarity-weighted vote of the k nearest examples |
| `logreg` | logistic regression trained on the examples |

`gda` shrinks its covariance by D/(D + dof) (D: embedding size, dof: images minus classes), so with one image per class it is a nearest-mean classifier and with many it uses how each class actually spreads. With descriptions, the image term's weight against `text_weight × 100 × cos` is chosen by leave-one-out cross-entropy on the examples, pulled towards 0.2·k^0.6 for k images per class (fully at k = 1, less as k grows), since how much to trust the images over the text differs a lot between tasks: flowers want five times the weight pets do. A last scale, fitted the same way with each example also taken out of the covariance, makes confidence match accuracy. Over six datasets (Pets, DTD and Flowers to develop it; Aircraft, EuroSAT and Food-101 held out) with CLIP ViT-B/32 and SigLIP 2, it beats `prototype` at every shot count; on the held-out datasets with descriptions, 1/2/4/8/16 shots go from 61.6/64.7/66.8/67.7/68.2% to 61.6/65.7/70.6/75.1/78.1%, and image-only from 45.1/55.0/63.6/68.9/71.7% to 46.6/58.4/67.5/73.8/77.6% (all classes of each dataset at once, 3 seeds). Fitting 101 classes × 16 images takes about 0.2 s.

With a CLIP embedder, class descriptions add `text_weight × 100 × cos(image, description)` to the image log-probabilities. A class with examples but no description is described by its name ("a photo of a <name>.") when the names match the examples: the name texts must classify the examples better than chance would, by a one-sided binomial test at p < 0.05, so codes such as `K-104` or `class_3` are ignored and the task stays image-only. On the six benchmark datasets with PE Core B/16 and no descriptions, this takes 1 example per class from 56.1% to 75.7% and 16 from 87.3% to 88.4%, while meaningless names cost nothing. Turn it off with `fewshot config TASK --no-names` or in Settings. A class can then work from its description alone, and a task with no images at all is plain CLIP zero-shot. Combining text and images at the logit level, rather than averaging their vectors, is what makes adding images improve accuracy instead of diluting the text signal. Text-only classes still get an image-side anchor by shifting their text vector across CLIP's modality gap.

`reject_below` returns "unknown" when an image's best similarity falls below a threshold. Use similarity for this, not confidence: softmax confidence can be 1.00 for an image of a class the task doesn't contain. The Evaluate tab (and `fewshot eval`) suggests a value that keeps 95% of correct held-out predictions, and shows a histogram of held-out similarities where you can pick another. Zero-shot tasks compare images with text, which scores far lower (about 0.2–0.35 for CLIP), so they use a separate `reject_below_zero_shot`. With a threshold set, a prediction's `labels` lists every class whose own similarity reaches it, best first (multi-label). A task with a single class predicts that class for everything, so with a threshold it tells members from outsiders (one-class or anomaly detection).

Classes differ in how tightly their images cluster, so a class can have its own threshold, which overrides `reject_below` for predictions of that class and decides whether that class is listed in another prediction's `labels` (even one whose own top class was rejected). Evaluation suggests one per class with at least 5 correct held-out predictions (again keeping 95% of them); apply them from the Evaluate tab's per-class table, with `fewshot eval --apply-class-thresholds`, or one at a time with `fewshot class threshold TASK CLASS 0.82` (`--clear` goes back to the task's). Through the API: `PUT .../class-thresholds` with `{"thresholds": {"cat": 0.82, "dog": null}}`. Zero-shot predictions keep using `reject_below_zero_shot`.

A class description can hold several prompts, one per line (`fewshot class describe` and the UI's template accept several); their embeddings are averaged. The `flip` setting averages each image's embedding with its mirror image's, which usually helps a little when left and right don't matter. Predictions can list the labelled examples most like each image as evidence (`?evidence=3`, `predict --evidence 3`; the Predict tab shows them), which helps spot a mislabelled example behind a wrong answer. Confidence can be calibrated: evaluation fits a temperature so that a confidence of 0.9 means right about 90% of the time and reports the calibration error before and after (with at least 50 scored examples); apply it from the Evaluate tab or with `config --temperature`. Predictions don't change, only their confidence. Changing the classifier, its parameters, the text weight or names-as-text clears it, since it calibrated the old confidences. With CLIP and the prototype classifier, uncalibrated confidence is usually far too low. Evaluation reports accuracy, macro-F1, top-2/3/5 accuracy, per-class recall and precision, a confusion matrix with row totals (click a cell in the UI to list its examples, right or wrong) and every held-out prediction (`fewshot eval --csv`, or Export in the UI). Leave-one-out for the prototype and kNN classifiers is computed in closed form instead of refitting per example, for tasks without descriptions. "Find near-duplicate examples" on the Evaluate tab (`fewshot dupes`) lists pairs of examples that look almost the same; pairs with different labels are usually labelling mistakes, and the UI offers to give both the same label.

## Measured accuracy

Accuracy over every class of six datasets at once (Oxford Pets, DTD textures, Oxford Flowers, FGVC Aircraft, EuroSAT, Food-101; 10 to 102 classes), with k example images per class, 3 random draws, up to 50 test images per class, the `gda` classifier, and a description per class such as "a photo of a {}, a type of pet." (averaged over the six datasets):

| Embedder | 0 | 1 | 2 | 4 | 8 | 16 examples per class |
|---|---|---|---|---|---|---|
| PE Core L/14, 336 px (most accurate preset) | 80.3% | 83.3% | 85.9% | 88.2% | 90.6% | 92.3% |
| SigLIP 2 L/16, 256 px | 77.7% | 81.1% | 83.8% | 86.8% | 89.3% | 91.0% |
| PE Core B/16 (default) | 75.5% | 78.3% | 80.9% | 83.3% | 86.3% | 88.6% |
| SigLIP 2 B/16, 256 px | 73.0% | 76.7% | 79.6% | 83.2% | 86.6% | 88.5% |
| CLIP ViT-B/32 LAION-2B (previous default) | 60.5% | 66.7% | 70.0% | 73.9% | 77.5% | 79.8% |

Images only, without descriptions:

| Embedder | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| PE Core L/14, 336 px | 64.3% | 78.9% | 85.6% | 89.5% | 91.7% |
| PE Core B/16 | 56.1% | 70.0% | 78.9% | 84.3% | 87.3% |
| DINOv2 ViT-L/14 | 60.0% | 67.9% | 74.8% | 80.7% | 85.2% |
| DINOv3 ViT-B/16 | 50.9% | 63.9% | 73.4% | 80.3% | 84.1% |
| DINOv2 ViT-S/14 | 54.6% | 65.3% | 72.5% | 79.2% | 83.5% |
| CLIP ViT-B/32 | 48.0% | 60.4% | 69.3% | 75.6% | 78.9% |

On these datasets the text-capable models beat DINOv2 and DINOv3 even on images alone, so prefer PE Core whether or not your classes have meaningful names; DINOv2 may still suit images unlike web photos (try both with "Compare models"). Descriptions help most at few examples (zero-shot PE Core B/16 is 75.5%); naming the kind of thing ("…, a type of flower") is worth about 2 points over "a photo of a {}" at 0–1 examples, while generic prompt ensembles did not help. Averaging each image with its mirror image (`flip`) added under a point, and combining two models' predictions (PE Core L with SigLIP 2 L) about one point at twice the embedding cost. Embedding speed on an RTX 5070: about 300 images/s for the B/16 models, 70 for PE Core L/14-336. Run the same protocol on your own folder of `<class>/<image>` with `fewshot bench <folder> --way 0 --query 50 --episodes 3 --shot 1 --shot 16 --embedder ... [--text-template ...]` (`--way 0` uses every class; the default is 5-way episodes).

To choose a model for your own task, compare embedders on its labelled examples: "Compare models" on the Evaluate tab, `fewshot compare TASK -e timm:vit_small_patch14_dinov2.lvd142m` (or `--presets` for all of them), or `POST .../compare/jobs` with `{"embedders": [...]}` (a background job, like evaluation). Each model is cross-validated with the task's other settings, and the results give accuracy, macro-F1 and the time taken (images already in the embedding cache are fast, so the first run of a model is slower). A model that can't be loaded is reported and the others still run. Switch with "Use this model" or `fewshot config TASK --embedder ...`; nothing changes until you do. Switching clears the temperature and reject thresholds (the task's and each class's), which were fitted to the old model's scores; evaluate again for new suggestions, or undo the switch to get them back. Changing only thresholds keeps the last evaluation current, since evaluation doesn't depend on them.

## CLI

```bash
fewshot create pets --embedder openclip:PE-Core-B-16@meta
fewshot import pets ./photos --shots 5        # ./photos/<class>/<image>
fewshot add pets Beagle a.jpg b.jpg
fewshot class describe pets "a photo of a {}, a type of pet"
fewshot eval pets                             # leave-one-out cross-validation; --csv for every prediction
fewshot eval pets --apply-class-thresholds    # also set each class's suggested reject threshold
fewshot compare pets --presets                # cross-validate the task with every preset embedder
fewshot predict pets new/ https://example.com/cat.jpg --json   # also --csv
fewshot predict pets --from-file list.txt --jsonl  # a JSON line per image, printed 256 images at a time
fewshot predict pets new/ --revision 41       # with an earlier version of the task (one kept for undo)
fewshot watch pets ./incoming --out ./sorted  # sort new images into ./sorted/<class>/ as they arrive
fewshot config pets --classifier knn --param k=3 --reject-below 0.7 --flip
fewshot example ls pets --label Beagle        # also: rm, relabel, tag --add/--remove, note
fewshot dupes pets                            # near-duplicate examples
fewshot undo pets
fewshot rename pets dogs-and-cats
fewshot export pets pets.zip && fewshot import-task pets.zip --name pets-copy
fewshot list --json                           # also show --json, models --json
fewshot bench ./photos --embedder timm:vit_small_patch14_dinov2.lvd142m --shot 1 --shot 5
fewshot cache prune --dry-run
fewshot serve --host 0.0.0.0 --token "$(openssl rand -hex 16)"
```

`serve` loads the models existing tasks use at startup (`--no-warm` skips this). With `--token`, the API and UI require that token; the UI asks for it once and keeps it in a cookie. Serving on anything but localhost without a token prints a warning, since anyone who can reach the port can read and change every task. By default API callers can only choose preset embedders, registered ones, and ones a task already uses, so they can't make the server download arbitrary models; `--allow-any-model` lifts that. The server only answers requests addressed to an IP address or to the host names it expects (localhost, plus the machine's own name when serving on another address; add others with `--allowed-host`, or `--allowed-host '*'` for any), and refuses writes that a browser marks as coming from another site, so web pages can't reach it through DNS rebinding or cross-site requests. Example images are always served as images (`nosniff`, sandboxed). Images over 50 megapixels are refused before they are decoded (a JPEG counts at the reduced scale it is decoded at), and images are shrunk to 2048 pixels before embedding. Transparent areas are filled with white; vectors cached before this change (made with black) are reused for those same images. Imported zips are checked like uploads: every file must be an image whose extension matches its format, the total unpacked size is limited, and the task's embedder must pass the same model allowlist. With `--allow-urls`, API callers can send image URLs for the server to download (public addresses only, with each connection made to the address that was checked, redirects followed only to http(s) URLs, and a time limit on each whole download; the UI then shows a URL field on the Predict tab). The CLI accepts URLs in `add` and `predict` either way.

## Python

```python
from fewshot import Workspace, Pipeline
ws = Workspace("./fewshot_data")
task = ws.create("defects", embedder="timm:vit_small_patch14_dinov2.lvd142m")
task.add_example(open("ok.jpg", "rb").read(), "ok")
task.add_example(open("crack.jpg", "rb").read(), "crack")
pred = Pipeline(ws).predict(task, ["new.jpg"])[0]   # .label, .confidence, .similarity, .scores
```

To call a running server from another program, `fewshot.client.Client` wraps the REST API with only the standard library (images are paths or `(filename, bytes)` pairs, sent 256 per request and read a batch at a time, so any number works; errors raise `ApiError` with the HTTP status and the server's message):

```python
from fewshot.client import Client
fs = Client("http://gpu-box:8000", token="...")
fs.add_examples("pets", "Beagle", ["a.jpg", "b.jpg"])
for p in fs.predict("pets", ["new.jpg"], top_k=3):
    print(p["file"], p["label"], p["confidence"], p["revision"])
```

`task.undo()`, `task.annotate(ids, add_tags=[...], note=...)`, `ws.rename(old, new)` and `fewshot.archive.export_task` / `import_task` do what the matching commands do. To add many images, `task.add_examples(items)` takes `(bytes, label, source)` tuples (a generator is fine) and saves `task.json` once per 256 images instead of once per image.

To use your own model, subclass `fewshot.embedders.Embedder` (implement `dim` and `_embed_image_batch`, returning L2-normalised float32 rows) and register it under any spec string; `unregister_embedder` removes it again. Output with the wrong shape or non-finite values raises `ValueError`; rows that aren't unit length are normalised. To keep CPU preprocessing outside the per-model lock, also override `_prepare_images` (runs unlocked) and `_embed_prepared` (runs locked).

```python
from fewshot import register_embedder
register_embedder("mine:v1", MyEmbedder())
task = ws.create("parts", embedder="mine:v1")
```

## REST API

`GET/POST /api/tasks`, `GET/PATCH/DELETE /api/tasks/{name}`, `POST /api/tasks/{name}/classes`, `PATCH/DELETE .../classes/{label}`, `POST .../describe`, `POST .../examples` (multipart `label` + `files`), `PATCH/DELETE .../examples/{id}`, `PATCH .../examples` (`{"ids": [...], "label": ...}` moves several) and `POST .../examples/delete` (`{"ids": [...]}`), both changing nothing if any id is unknown, `GET .../examples/{id}/image` (`?size=N` returns a JPEG scaled down to fit N×N pixels, 16–2048), `POST .../predict` (multipart `files`, optional `?top_k=N`, `?evidence=N` and `?revision=N` to use an earlier version still kept for undo; each prediction carries the `revision` that made it, and an unreadable file gets an `error` field instead of a label), `POST .../evaluate`, `GET /api/health` (also `/healthz`). Also: `GET .../examples?offset=&limit=&label=&tag=` (one page of examples), `PATCH .../examples/{id}` with `tags` and `note`, `PATCH .../examples` with `add_tags`/`remove_tags`, `POST .../undo`, `POST .../rename` (`{"name": ...}`), `GET .../export` (zip) and `POST /api/tasks/import` (multipart `file`, optional `name`), `GET .../duplicates?threshold=0.95`, the inbox (`POST .../inbox` with files, `POST .../inbox/urls`, `GET .../inbox?sort=uncertain|confident|novel|added`, `POST .../inbox/accept` with `{"labels": {id: class}}`, `POST .../inbox/dismiss`), and, on a server started with `--allow-urls`, `POST .../predict/urls` and `POST .../examples/urls` (`{"urls": [...]}`, plus `label` to add). OpenAPI docs are served at `/docs` and the schema at `/openapi.json`; each operation id is the handler's name (`predict`, `add_examples`, `list_tasks`...), so generated clients get readable method names. Request bodies reject unknown fields (422). Each uploaded image is limited to 50 MB (`FEWSHOT_MAX_UPLOAD_MB`), a whole request to 1024 MB (`FEWSHOT_MAX_REQUEST_MB`, refused with 413 before the body is stored), and a request to 1000 files.

Evaluation of a large task can take a while, so it can also run as a background job: `POST .../evaluate/jobs` returns a job; `GET /api/jobs/{id}` reports `status` (`running`, `done`, `failed`, `cancelled`), `stage` (`embedding` counts images, `evaluating` counts folds), `done`/`total`, and the report as `result` once done; `DELETE /api/jobs/{id}` cancels it. The web UI uses this to show progress and a Cancel button.

When the server was started with a token, send it as `Authorization: Bearer <token>`, or `POST /api/login` with `{"token": ...}` to get a cookie. After 10 wrong tokens within a minute from one address, `/api/login` answers 429 for the rest of that minute; over HTTPS the cookie is `Secure`. At most 4 background jobs run at once (429 beyond that), and starting an evaluation that is already running returns that job. URL downloads share a budget per request: the request size limit in total and 2 minutes in all, 4 at a time. `serve --max-upload-mb/--max-request-mb` set the size limits. `POST /api/tasks/import` also takes an `embedder` field to import with another model; the export's `X-Fewshot-Skipped` header counts examples left out because their image was missing. `/api/health` and `/api/login` need no token.

## Deploying

`fewshot predict` streams: with `--jsonl` or `--csv` each batch of 256 images is printed as soon as it is classified, and a `--from-file` list (`-` for stdin) is read as it arrives, so a long one can be piped through (a listed file that can't be read gets an error row instead of stopping the run), and every prediction records the task `revision` that made it. `fewshot watch TASK FOLDER --out OUT` classifies images as they land in FOLDER (once a file's size has stopped changing) and moves each into `OUT/<class>/`, `OUT/_unknown/` (rejected, or below `--min-confidence`) or `OUT/_error/` (an unreadable file is retried for 30 seconds after it last changed, in case it is still arriving), appending a JSON line per image to `OUT/predictions.jsonl`; files are never overwritten, `--copy` leaves the originals, `--once` handles what is there and stops, and the task is re-read each time, so labelling more examples improves the sorting while it runs (a task without examples yet is waited for).

The Dockerfile builds the web UI and a server image (CPU PyTorch by default):

```bash
export FEWSHOT_TOKEN=$(openssl rand -hex 16)
docker compose up -d                      # http://localhost:8000; sign in with the token
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t fewshot:gpu .   # NVIDIA
docker run --gpus all -p 8000:8000 -e FEWSHOT_TOKEN -v fewshot-data:/data -v fewshot-models:/models fewshot:gpu
```

Tasks, images and the embedding cache live in the `/data` volume and downloaded model weights in `/models`, so both survive rebuilds. The container serves on all interfaces, so set `FEWSHOT_TOKEN`, and set `FEWSHOT_ALLOWED_HOSTS` (comma-separated) to the host names clients use: by default the server answers requests by IP address and for localhost and the container's own name only. It runs as an unprivileged user and its health check polls `/healthz`. Put HTTPS in front of it with a reverse proxy; over HTTPS the login cookie is `Secure`.

## Tests and checks

`scripts/check.sh` runs everything CI runs (`.github/workflows/ci.yml`):

| Check | Requirement |
|---|---|
| `ruff check .` | no findings |
| `mypy` | `--strict` on `fewshot`; relaxed annotations for `tests` |
| `pylint fewshot tests` | 10.00/10 |
| `pytest --cov` | 100% line and branch coverage (`fail_under = 100`) |
| `npm run typecheck` / `lint` / `build` in `web/` | `tsc`, ESLint (typescript-eslint strict type-checked) and Stylelint with zero warnings |

The Python tests take about 10 seconds and download no models: they use fake embedders registered with `register_embedder`, and mock open_clip, timm and CUDA.
