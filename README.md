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

Data lives in `$FEWSHOT_HOME` (default `./fewshot_data`): one folder per task with its images and `task.json`, plus a shared SQLite embedding cache. The server and CLI can use the same workspace at once: each change to a task locks it, starts from the latest saved `task.json`, and replaces the file atomically (the lock needs Linux or macOS), and the cache runs in WAL mode so one process's writes don't fail the other's. `fewshot cache info` shows the cache's size per model; `fewshot cache prune` deletes vectors no task's examples use, such as those of deleted examples and `bench` datasets. At most two built-in models stay loaded at once (`FEWSHOT_MAX_MODELS`); loading another unloads the least recently used. The web UI scales images down to at most 1024 px on the longest side (as JPEG) before uploading them as examples or for prediction; the CLI and API store whatever they are given. In the UI you can drag thumbnails between classes, tick several (Shift+click for a range) to move or delete them together, and paste images: on the Examples tab into the class under the pointer, on the Predict tab to classify them. Number keys 1–9 move ticked examples to that class, or add the highlighted prediction as that class (Enter accepts the prediction). Prediction and evaluation results stay while you switch tabs. For UI development, run `fewshot serve` and `npm run dev` in `web/`; Vite proxies `/api` to port 8000.

## How classification works

Each task has an embedder, a classifier, and classes. A class has example images, an optional text description, or both.

| Classifier | Image score |
|---|---|
| `prototype` (default) | cosine similarity to the class's mean embedding |
| `knn` | similarity-weighted vote of the k nearest examples |
| `logreg` | logistic regression trained on the examples |

With a CLIP embedder, class descriptions add `text_weight × 100 × cos(image, description)` to the image log-probabilities. A class can then work from its description alone, and a task with no images at all is plain CLIP zero-shot. Combining text and images at the logit level, rather than averaging their vectors, is what makes adding images improve accuracy instead of diluting the text signal. Text-only classes still get an image-side anchor by shifting their text vector across CLIP's modality gap.

`reject_below` returns "unknown" when an image's best similarity falls below a threshold. Use similarity for this, not confidence: softmax confidence can be 1.00 for an image of a class the task doesn't contain.

## Measured accuracy

Oxford-IIIT Pets, 5-way episodes, 300 episodes each, ±95% CI under about 0.01:

| Embedder | 0-shot | 1-shot | 5-shot |
|---|---|---|---|
| CLIP ViT-B/32, images only | – | 84.5% | 96.3% |
| CLIP ViT-B/32, images + "a photo of a {}, a type of pet" | 97.6% | 97.6–98.0% | 98.4–98.6% |
| DINOv2 ViT-S/14, images only | – | 94.7% | 98.3% |

Use CLIP with descriptions when class names are meaningful words. Use DINOv2 when they aren't (part numbers, defect codes, individual identities), since it is stronger on images alone. Reproduce these numbers with `fewshot bench <folder> --embedder ... --shot 1 --shot 5 [--text-template ...]`.

## CLI

```bash
fewshot create pets --embedder openclip:ViT-B-32@laion2b_s34b_b79k
fewshot import pets ./photos --shots 5        # ./photos/<class>/<image>
fewshot add pets Beagle a.jpg b.jpg
fewshot class describe pets "a photo of a {}, a type of pet"
fewshot eval pets                             # leave-one-out cross-validation
fewshot predict pets new/ --json
fewshot config pets --classifier knn --param k=3 --reject-below 0.7
fewshot bench ./photos --embedder timm:vit_small_patch14_dinov2.lvd142m --shot 1 --shot 5
fewshot cache prune --dry-run
fewshot serve --host 0.0.0.0 --token "$(openssl rand -hex 16)"
```

`serve` loads the models existing tasks use at startup (`--no-warm` skips this). With `--token`, the API and UI require that token; the UI asks for it once and keeps it in a cookie. Serving on anything but localhost without a token prints a warning, since anyone who can reach the port can read and change every task. By default API callers can only choose preset embedders, registered ones, and ones a task already uses, so they can't make the server download arbitrary models; `--allow-any-model` lifts that.

## Python

```python
from fewshot import Workspace, Pipeline
ws = Workspace("./fewshot_data")
task = ws.create("defects", embedder="timm:vit_small_patch14_dinov2.lvd142m")
task.add_example(open("ok.jpg", "rb").read(), "ok")
task.add_example(open("crack.jpg", "rb").read(), "crack")
pred = Pipeline(ws).predict(task, ["new.jpg"])[0]   # .label, .confidence, .similarity, .scores
```

To add many images, `task.add_examples(items)` takes `(bytes, label, source)` tuples (a generator is fine) and saves `task.json` once per 256 images instead of once per image.

To use your own model, subclass `fewshot.embedders.Embedder` (implement `dim` and `_embed_image_batch`, returning L2-normalised float32 rows) and register it under any spec string; `unregister_embedder` removes it again. Output with the wrong shape or non-finite values raises `ValueError`; rows that aren't unit length are normalised. To keep CPU preprocessing outside the per-model lock, also override `_prepare_images` (runs unlocked) and `_embed_prepared` (runs locked).

```python
from fewshot import register_embedder
register_embedder("mine:v1", MyEmbedder())
task = ws.create("parts", embedder="mine:v1")
```

## REST API

`GET/POST /api/tasks`, `GET/PATCH/DELETE /api/tasks/{name}`, `POST /api/tasks/{name}/classes`, `PATCH/DELETE .../classes/{label}`, `POST .../describe`, `POST .../examples` (multipart `label` + `files`), `PATCH/DELETE .../examples/{id}`, `PATCH .../examples` (`{"ids": [...], "label": ...}` moves several) and `POST .../examples/delete` (`{"ids": [...]}`), both changing nothing if any id is unknown, `GET .../examples/{id}/image` (`?size=N` returns a JPEG scaled down to fit N×N pixels, 16–2048), `POST .../predict` (multipart `files`; an unreadable file gets an `error` field instead of a label), `POST .../evaluate`, `GET /api/health` (also `/healthz`). OpenAPI docs are served at `/docs`. Request bodies reject unknown fields (422). Each uploaded image is limited to 50 MB (`FEWSHOT_MAX_UPLOAD_MB`), a whole request to 1024 MB (`FEWSHOT_MAX_REQUEST_MB`, refused with 413 before the body is stored), and a request to 1000 files.

Evaluation of a large task can take a while, so it can also run as a background job: `POST .../evaluate/jobs` returns a job; `GET /api/jobs/{id}` reports `status` (`running`, `done`, `failed`, `cancelled`), `stage` (`embedding` counts images, `evaluating` counts folds), `done`/`total`, and the report as `result` once done; `DELETE /api/jobs/{id}` cancels it. The web UI uses this to show progress and a Cancel button.

When the server was started with a token, send it as `Authorization: Bearer <token>`, or `POST /api/login` with `{"token": ...}` to get a cookie. `/api/health` and `/api/login` need no token.

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
