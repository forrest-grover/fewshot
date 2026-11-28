"""Changes made only if a task is still at a given revision (the REST API's If-Match), checked
atomically with the change.

``expect(name, if_match)`` holds for the current context: a request, and the threads it runs its
work in (``contextvars`` are copied into them). Each change to task ``name`` made in that context
checks, under the task's write lock, that the task is at a revision ``if_match`` names (or ``*``),
and raises ``RevisionChanged`` if not, before changing anything; once one of them has saved
(``settle``), later ones aren't checked, since that save moved the revision on. Until then every
change is checked: a request made of several changes (a large upload, stored in chunks) whose
first ones change nothing must still be refused if the task changed meanwhile. Outside such a
context (the command line, the library) nothing is checked.
"""

from __future__ import annotations

from contextvars import ContextVar

from .exceptions import RevisionChanged

# task name -> the If-Match header sent, until a change to that task is saved
_EXPECTED: ContextVar[dict[str, str] | None] = ContextVar("fewshot_if_match", default=None)


def tags(if_match: str) -> list[str]:
    """The revisions an If-Match header names (weak or strong), ``*`` for any."""
    return [g.strip().removeprefix("W/").strip('"') for g in if_match.split(",")]


def require(name: str, if_match: str, revision: int) -> None:
    """Raise ``RevisionChanged`` unless ``if_match`` names ``revision`` (or is ``*``)."""
    named = tags(if_match)
    if "*" not in named and str(revision) not in named:
        raise RevisionChanged(f"Task {name!r} is at revision {revision}, not {if_match}: it changed since; get it "
                              f"again (GET /api/tasks/{name}) and redo the change if it still applies",
                              task=name, revision=revision)


def expect(name: str, if_match: str) -> None:
    """Have the changes to task ``name`` in this context check ``if_match`` until one saves (see the module)."""
    _EXPECTED.set({**(_EXPECTED.get() or {}), name: if_match})


def check(name: str, revision: int) -> None:
    """Called as a change to task ``name``, now at ``revision``, starts (holding its write lock):
    ``require`` what ``expect`` asked for this task in this context, if no change has settled it."""
    pending = _EXPECTED.get()
    if pending and name in pending:
        require(name, pending[name], revision)


def settle(name: str) -> None:
    """Called once a change to task ``name`` (as it was called when ``check`` ran) has saved: its
    expectation is met, and later changes in this context aren't checked."""
    pending = _EXPECTED.get()
    if pending:
        pending.pop(name, None)
