"""The REST API's state: one ``Server`` per app, holding its settings, workspace, pipeline,
background jobs and access control, and ``Srv``, the route dependency that hands it to a route.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, TypeVar

from fastapi import Depends, Request
from fastapi.responses import Response

from . import errors
from .auth import Sessions, WrongTokens
from .embedders import DEFAULT_SPEC, PRESETS, check_embedder, get_embedder, is_registered, max_models
from .errors import ApiProblem, from_error
from .exceptions import TaskNotFound, message
from .jobs import Jobs
from .pipeline import Pipeline
from .responses import json_response
from .server_settings import ServerSettings
from .store import Task, Workspace
from .text import closest

log = logging.getLogger(__name__)

T = TypeVar("T")


class Server:  # pylint: disable=too-many-instance-attributes  # the server's parts, one each
    """Everything one app serves: its settings, workspace, pipeline, jobs and access control."""

    def __init__(self, settings: ServerSettings, workspace: Workspace | None = None,
                 allow_no_token: bool = False) -> None:
        self.settings = settings
        self.allow_no_token = allow_no_token  # see access.guard
        self.ws = workspace or Workspace(settings.home)
        self.pipe = Pipeline(self.ws)
        self.jobs = Jobs()
        self.wrong_tokens = WrongTokens()
        self.sessions = Sessions(settings.token) if settings.token else None
        self.warned_proxy = False  # see access._note_untrusted_proxy
        self.warned_exposed = False  # see access.guard
        self._heavy: ThreadPoolExecutor | None = None  # made when first needed, after each shutdown too
        self._heavy_lock = threading.Lock()

    @property
    def heavy(self) -> ThreadPoolExecutor:
        """The threads for heavy work (``settings.workers`` of them)."""
        with self._heavy_lock:
            if self._heavy is None:
                self._heavy = ThreadPoolExecutor(self.settings.workers, thread_name_prefix="fewshot-work")
            return self._heavy

    def stop_heavy(self) -> None:
        """Stop the heavy-work threads, dropping work not started yet (the app is shutting down); the
        next heavy request, if the app is started again, gets new ones."""
        with self._heavy_lock:
            pool, self._heavy = self._heavy, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        """``fn(*args)`` on the heavy-work threads, so it waits there, not in Starlette's pool."""
        ctx = contextvars.copy_context()  # the request's (see ``preconditions``), as Starlette's pool does
        return await asyncio.get_running_loop().run_in_executor(self.heavy, ctx.run, functools.partial(fn, *args))

    async def render(self, model: Any, fn: Callable[..., Any], *args: Any, status: int = 200) -> Response:
        """``fn(*args)`` on the heavy-work threads, as JSON validated as ``model`` (see ``_json``)."""
        return await self.run(lambda: json_response(model, fn(*args), status))

    def task(self, name: str) -> Task:
        """Task ``name``; 404 if there is none, naming the closest ones."""
        try:
            return self.ws.get(name)
        except TaskNotFound as e:
            near = closest(name, [p.name for p in self.ws.task_folders()])
            hint = f"; did you mean {' or '.join(map(repr, near))}? Or create it" if near else "; create it"
            raise ApiProblem(404, errors.TASK_NOT_FOUND, f"No task {name!r}{hint} with POST /api/tasks "
                             "(fewshot.client: create_task)", task=name, similar_to=near) from e
        except KeyError as e:
            raise from_error(e) from e

    def warm_models(self) -> list[str]:
        """Load the embedders the most tasks use (up to the loaded-model limit), or the default one
        if there are no tasks, so the first request doesn't wait for a model; returns those loaded."""
        counts = Counter(t.config.embedder for t in self.ws.list_tasks())
        specs = [s for s, _ in counts.most_common(max_models())] or [DEFAULT_SPEC]
        loaded = []
        for spec in specs:
            try:
                get_embedder(spec)
            except Exception:  # pylint: disable=broad-exception-caught  # a bad model mustn't stop the server
                log.exception("Could not preload embedder %r", spec)
            else:
                loaded.append(spec)
        return loaded

    def allowed_embedder(self, spec: str) -> None:
        """Refuse an empty spec (400), or (403) one that isn't a preset, registered, or used by a
        task already, unless the server allows any model. Specs in use are checked last, as that
        reads every task."""
        if not spec.strip():
            raise ApiProblem(400, errors.INVALID_EMBEDDER, "Embedder spec is empty")
        if not (self.settings.allow_any_model or spec in {p.spec for p in PRESETS} or is_registered(spec)
                or any(t.config.embedder == spec for t in self.ws.list_tasks())):
            raise ApiProblem(403, errors.EMBEDDER_NOT_ALLOWED, f"Embedder {spec!r} is not a preset (see GET "
                             "/api/models). To use other models, start the server with --allow-any-model "
                             "(FEWSHOT_ALLOW_ANY_MODEL=1).")

    def check_embedder(self, spec: str) -> None:
        """``allowed_embedder``, then load the embedder so a bad spec fails now (400), not at predict time."""
        self.allowed_embedder(spec)
        try:
            check_embedder(spec)
        except ValueError as e:
            raise ApiProblem(400, errors.INVALID_EMBEDDER, message(e)) from e


async def _server(request: Request) -> Server:
    server: Server = request.app.state.server
    return server


Srv = Annotated[Server, Depends(_server)]  # async, so resolving it takes no thread
