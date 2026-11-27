"""A small client for a fewshot server's REST API, using only the standard library.

    from fewshot.client import Client

    fs = Client("http://localhost:8000", token="...")
    fs.add_examples("pets", "Beagle", ["a.jpg", "b.jpg"])
    for p in fs.predict("pets", ["new.jpg"]):
        print(p["file"], p["label"], p["confidence"])

Images are file paths or ``(filename, bytes)`` pairs. Every method returns the API's JSON; an error
response raises ``ApiError`` with the HTTP status and the server's ``detail``, and so does a failed
connection (status 0), a response that isn't JSON, and a redirect: redirects aren't followed, so the
token is never sent to another server and a POST is never repeated as a GET. Many images are
sent in batches of at most ``BATCH`` images (the server takes at most 1000 files per request) and
at most half the server's request size limit, reading only one batch of files at a time; the
results are combined.

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
from urllib.request import HTTPRedirectHandler, Request, build_opener

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


class _NoRedirects(HTTPRedirectHandler):
    """Refuses every redirect: urllib would send the Authorization header on to whatever server
    it names, and turn a POST into a GET without its body."""

    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any,  # pylint: disable=too-many-arguments,too-many-positional-arguments  # urllib's signature
                         newurl: str) -> None:
        fp.close()
        raise ApiError(code, f"The server redirected to {newurl}; give that URL instead (redirects aren't "
                             "followed, so the token isn't sent elsewhere)")


_OPENER = build_opener(_NoRedirects)


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
        ``fields`` and ``files`` as a multipart form, the files under the name ``files`` (a file
        that can't be read raises ``ApiError`` with status 0, and nothing is sent).
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
            loaded, unread = _read(files)
            if unread:
                raise ApiError(0, f"Can't read {unread[0][1]}: {unread[0][2]}")
            data, headers["Content-Type"] = _multipart(fields or {}, loaded)
        req = Request(f"{self.url}/api{path}{'?' + query if query else ''}", data=data, headers=headers,
                      method=method)
        try:
            with _OPENER.open(req, timeout=self.timeout) as r:
                status, body = r.status, r.read()
        except HTTPError as e:
            raise _api_error(e) from None
        except (URLError, OSError, http.client.HTTPException) as e:
            reason = getattr(e, "reason", None) or e
            hint = " (the request may be larger than the server accepts)" if data and len(data) > 1024 * 1024 else ""
            raise ApiError(0, f"Connection to {self.url} failed: {reason}{hint}") from None
        try:
            return json.loads(body) if body else None
        except ValueError:  # an HTML page from a proxy, say
            start = body[:200].decode(errors="replace")
            raise ApiError(status, f"The response from {self.url} isn't JSON: {start!r}") from None

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
        combined, ``task`` is the last. A file that can't be read is listed in ``errors``. A batch
        refused with 400 or 413 (nothing in it added) adds its reason to ``errors``; if every
        batch was refused, the first refusal is raised (``ApiError`` 400 if no file could be read)."""
        out: dict[str, Any] = {"added": [], "errors": [], "task": None}
        refused: ApiError | None = None
        fields = {**fields, "undo_group": uuid.uuid4().hex}  # the batches are one undo step
        for batch in _batches(images, self.batch_bytes()):
            loaded, unread = _read(batch)
            out["errors"] = out["errors"] + [{"file": name, "error": why} for _, name, why in unread]
            if not loaded:
                continue
            try:
                r = self.request("POST", path, fields=fields, files=loaded)
            except ApiError as e:
                if e.status not in {400, 413}:
                    raise
                refused = refused or e
                out["errors"] = out["errors"] + [{"file": None, "error": str(e.detail)}]
                continue
            out = {"added": out["added"] + r["added"], "errors": out["errors"] + r["errors"], "task": r["task"]}
        if out["task"] is None:
            if refused is None and out["errors"]:
                refused = ApiError(400, f"No image could be read: {out['errors'][0]['file']}: "
                                        f"{out['errors'][0]['error']}")
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
        """Classify images; each result has ``label``, ``confidence``, ``scores``, ``revision`` (or
        ``error``, as has a file that can't be read), in the order given.

        ``revision`` predicts with an earlier version of the task that the server still keeps.
        """
        params = {"top_k": top_k, "evidence": evidence or None, "revision": revision}
        out: list[dict[str, Any]] = []
        for batch in _batches(images, self.batch_bytes()):
            loaded, unread = _read(batch)
            preds = iter(self.request("POST", f"/tasks/{_seg(task)}/predict", params=params, files=loaded)
                         if loaded else [])
            errors = {i: {"file": name, "error": why} for i, name, why in unread}
            out += [errors[i] if i in errors else next(preds) for i in range(len(batch))]
        return out

    def evaluate(self, task: str, folds: int | None = None, poll: float = 1.0) -> dict[str, Any]:
        """Cross-validate the task and return the report (accuracy, per-class metrics, mistakes...).

        Runs as a background job on the server, polled every ``poll`` seconds, so a long evaluation
        isn't cut off by the request timeout. A failed or cancelled job raises ``ApiError``, as
        does (410) one whose report the server dropped before it was fetched."""
        job = self.request("POST", f"/tasks/{_seg(task)}/evaluate/jobs", params={"folds": folds})
        while job["status"] in {"running", "cancelling"}:
            time.sleep(poll)
            job = self.request("GET", f"/jobs/{_seg(job['id'])}")
        if job["status"] != "done":
            raise ApiError(409 if job["status"] == "cancelled" else 400, job["error"] or f"Evaluation {job['status']}")
        if job["result"] is None:  # finished, but the server has dropped the report since
            raise ApiError(410, "The evaluation finished, but the server no longer keeps its report (it keeps "
                                "only the most recent ones); evaluate again")
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


def _read(batch: Sequence[Image]) -> tuple[list[tuple[str, bytes]], list[tuple[int, str, str]]]:
    """The images of ``batch`` that can be read, as ``(name, bytes)``, and ``(position, file, reason)``
    for each file that can't be."""
    loaded: list[tuple[str, bytes]] = []
    unread: list[tuple[int, str, str]] = []
    for i, img in enumerate(batch):
        if isinstance(img, tuple):
            loaded.append((img[0], img[1]))
            continue
        try:
            loaded.append((Path(img).name, Path(img).read_bytes()))
        except OSError as e:
            unread.append((i, _text(str(img)), e.strerror or str(e)))
    return loaded, unread


def _text(s: str) -> str:
    """``s`` as valid Unicode: a file name that isn't UTF-8 (its bytes escaped as lone surrogates)
    has each such byte replaced by U+FFFD."""
    try:
        return s.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    except UnicodeEncodeError:  # a surrogate not from a file name's bytes
        return s.encode("utf-8", "replace").decode("utf-8")


def _multipart(fields: dict[str, str], files: Sequence[tuple[str, bytes]]) -> tuple[bytes, str]:
    """A multipart/form-data body with ``fields`` and each ``(name, bytes)`` image as a ``files``
    part, and its content type."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{_text(v)}\r\n'.encode())
    for name, data in files:
        name = _text(name)
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        safe = name.replace('"', "%22").replace("\r", "").replace("\n", "")
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{safe}"\r\n'
                     f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
