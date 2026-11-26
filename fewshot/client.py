"""A small client for a fewshot server's REST API, using only the standard library.

    from fewshot.client import Client

    fs = Client("http://localhost:8000", token="...")
    fs.add_examples("pets", "Beagle", ["a.jpg", "b.jpg"])
    for p in fs.predict("pets", ["new.jpg"]):
        print(p["file"], p["label"], p["confidence"])

Images are file paths or ``(filename, bytes)`` pairs. Every method returns the API's JSON; an error
response raises ``ApiError`` with the HTTP status and the server's ``detail``, and so does a failed
connection (status 0). Many images are sent in batches of at most ``BATCH`` images (the server
takes at most 1000 files per request) and at most half the server's request size limit, reading
only one batch of files at a time; the results are combined.

The methods cover the common work (tasks, adding images, the inbox, predicting, evaluating);
``request`` reaches every other endpoint (see the server's /docs).
"""

from __future__ import annotations

import http.client
import json
import mimetypes
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

Image = str | Path | tuple[str, bytes]
BATCH = 256  # images per upload request, at most
BATCH_BYTES = 256 * 1024 * 1024  # bytes per upload request, at most, if the server doesn't say its limit


def _size(img: Image) -> int:
    """How many bytes ``img`` will add to a request (0 for a file that can't be read: it fails when sent)."""
    if isinstance(img, tuple):
        return len(img[1])
    try:
        return Path(img).stat().st_size
    except OSError:
        return 0


def _batches(images: Iterable[Image], max_bytes: int) -> Iterator[list[Image]]:
    """``images`` in lists of at most ``BATCH`` images and ``max_bytes`` (an image larger than that
    goes alone, for the server to judge)."""
    batch: list[Image] = []
    size = 0
    for img in images:
        n = _size(img)
        if batch and (len(batch) >= BATCH or size + n > max_bytes):
            yield batch
            batch, size = [], 0
        batch.append(img)
        size += n
    if batch:
        yield batch


class ApiError(Exception):
    """An error response from the server."""

    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class Client:
    """Calls the API of the fewshot server at ``url``, sending ``token`` (if the server needs one)."""

    def __init__(self, url: str = "http://127.0.0.1:8000", token: str | None = None, timeout: float = 300) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._batch_bytes: int | None = None  # half the server's request limit, once asked

    # ---- transport -----------------------------------------------------
    def request(  # pylint: disable=too-many-arguments,too-many-locals  # keyword-only body options
            self, method: str, path: str, *, params: dict[str, Any] | None = None, json_body: Any = None,
            fields: dict[str, str] | None = None, files: Sequence[Image] = ()) -> Any:
        """Send a request to ``path`` (under ``/api``) and return the decoded JSON (None if empty).

        ``params`` become the query string (None values are left out); ``json_body`` is sent as JSON;
        ``fields`` and ``files`` as a multipart form, the files under the name ``files``.
        """
        query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
        elif fields is not None or files:
            data, headers["Content-Type"] = _multipart(fields or {}, files)
        req = Request(f"{self.url}/api{path}{'?' + query if query else ''}", data=data, headers=headers,
                      method=method)
        try:
            with urlopen(req, timeout=self.timeout) as r:
                body = r.read()
        except HTTPError as e:
            raise _api_error(e) from None
        except (URLError, OSError, http.client.HTTPException) as e:
            reason = getattr(e, "reason", None) or e
            hint = " (the request may be larger than the server accepts)" if data and len(data) > 1024 * 1024 else ""
            raise ApiError(0, f"Connection to {self.url} failed: {reason}{hint}") from None
        return json.loads(body) if body else None

    def batch_bytes(self) -> int:
        """The most bytes of images sent per request: half the server's request size limit (as its
        health check reports it), or ``BATCH_BYTES`` if it doesn't say."""
        if self._batch_bytes is None:
            limit = self.health().get("max_request_bytes")
            self._batch_bytes = limit // 2 if isinstance(limit, int) and limit > 0 else BATCH_BYTES
        return self._batch_bytes

    # ---- tasks ---------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """Server status: loaded models, running jobs, whether a token is required."""
        return self.request("GET", "/health")  # type: ignore[no-any-return]

    def tasks(self) -> list[dict[str, Any]]:
        """A summary of every task."""
        return self.request("GET", "/tasks")  # type: ignore[no-any-return]

    def task(self, name: str) -> dict[str, Any]:
        """One task's details: classes, examples, settings, revision."""
        return self.request("GET", f"/tasks/{_seg(name)}")  # type: ignore[no-any-return]

    def create_task(self, name: str, **settings: Any) -> dict[str, Any]:
        """Create a task; ``settings`` are the other fields of the create body (``embedder``, ...)."""
        return self.request("POST", "/tasks", json_body={"name": name, **settings})  # type: ignore[no-any-return]

    def delete_task(self, name: str) -> None:
        """Delete a task and its images."""
        self.request("DELETE", f"/tasks/{_seg(name)}")

    # ---- data ----------------------------------------------------------
    def add_examples(self, task: str, label: str, images: Iterable[Image]) -> dict[str, Any]:
        """Add labelled examples (the class is created if needed). Images that can't be added are
        listed in ``errors``, with the reason for a whole batch the server refused (``file`` None);
        raises ``ApiError`` (400) if no image at all could be added."""
        return self._upload(f"/tasks/{_seg(task)}/examples", images, {"label": label})

    def add_to_inbox(self, task: str, images: Iterable[Image]) -> dict[str, Any]:
        """Add unlabelled images to the task's inbox; errors are reported as by ``add_examples``,
        and ``ApiError`` (400) is raised if no image could be added (e.g. all are examples already)."""
        return self._upload(f"/tasks/{_seg(task)}/inbox", images, {})

    def _upload(self, path: str, images: Iterable[Image], fields: dict[str, str]) -> dict[str, Any]:
        """POST ``images`` a batch at a time, all one undo step; ``added`` and ``errors`` are
        combined, ``task`` is the last. A batch refused with 400 (nothing in it added) adds its
        reason to ``errors``; if every batch was refused, the first refusal is raised."""
        out: dict[str, Any] = {"added": [], "errors": [], "task": None}
        refused: ApiError | None = None
        fields = {**fields, "undo_group": uuid.uuid4().hex}  # the batches are one undo step
        for batch in _batches(images, self.batch_bytes()):
            try:
                r = self.request("POST", path, fields=fields, files=batch)
            except ApiError as e:
                if e.status != 400:
                    raise
                refused = refused or e
                out["errors"] = out["errors"] + [{"file": None, "error": str(e.detail)}]
                continue
            out = {"added": out["added"] + r["added"], "errors": out["errors"] + r["errors"], "task": r["task"]}
        if out["task"] is None:
            raise refused or ValueError("No images given")
        return out

    def inbox(self, task: str, sort: str = "uncertain") -> dict[str, Any]:
        """The inbox with the model's suggestion for each image (``sort``: uncertain, confident, novel, added)."""
        return self.request("GET", f"/tasks/{_seg(task)}/inbox", params={"sort": sort})  # type: ignore[no-any-return]

    def accept_inbox(self, task: str, labels: dict[str, str]) -> dict[str, Any]:
        """Label inbox images (``{id: class}``), moving them to the task's examples."""
        return self.request("POST", f"/tasks/{_seg(task)}/inbox/accept",  # type: ignore[no-any-return]
                            json_body={"labels": labels})

    # ---- inference -----------------------------------------------------
    def predict(self, task: str, images: Iterable[Image], *, top_k: int | None = None, evidence: int = 0,
                revision: int | None = None) -> list[dict[str, Any]]:
        """Classify images; each result has ``label``, ``confidence``, ``scores``, ``revision`` (or ``error``).

        ``revision`` predicts with an earlier version of the task that the server still keeps.
        """
        params = {"top_k": top_k, "evidence": evidence or None, "revision": revision}
        out: list[dict[str, Any]] = []
        for batch in _batches(images, self.batch_bytes()):
            out += self.request("POST", f"/tasks/{_seg(task)}/predict", params=params, files=batch)
        return out

    def evaluate(self, task: str, folds: int | None = None, poll: float = 1.0) -> dict[str, Any]:
        """Cross-validate the task and return the report (accuracy, per-class metrics, mistakes...).

        Runs as a background job on the server, polled every ``poll`` seconds, so a long evaluation
        isn't cut off by the request timeout. A failed or cancelled job raises ``ApiError``."""
        job = self.request("POST", f"/tasks/{_seg(task)}/evaluate/jobs", params={"folds": folds})
        while job["status"] in {"running", "cancelling"}:
            time.sleep(poll)
            job = self.request("GET", f"/jobs/{_seg(job['id'])}")
        if job["status"] != "done":
            raise ApiError(409 if job["status"] == "cancelled" else 400, job["error"] or f"Evaluation {job['status']}")
        return job["result"]  # type: ignore[no-any-return]


def _api_error(e: HTTPError) -> ApiError:
    """``e`` as an ``ApiError`` carrying the server's ``detail`` (or the body as text if it has none)."""
    raw = e.read()
    try:
        detail = json.loads(raw).get("detail", raw.decode(errors="replace"))
    except (ValueError, AttributeError):
        detail = raw.decode(errors="replace")
    return ApiError(e.code, detail)


def _seg(s: str) -> str:
    """``s`` quoted as one path segment."""
    return quote(s, safe="")


def _multipart(fields: dict[str, str], files: Sequence[Image]) -> tuple[bytes, str]:
    """A multipart/form-data body with ``fields`` and each image as a ``files`` part, and its content type."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for img in files:
        name, data = (img[0], img[1]) if isinstance(img, tuple) else (Path(img).name, Path(img).read_bytes())
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        safe = name.replace('"', "%22").replace("\r", "").replace("\n", "")
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{safe}"\r\n'
                     f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
