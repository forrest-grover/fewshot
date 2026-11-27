"""The errors the library raises for a missing thing or a refused request, each with a stable
``code``, so the REST API can answer each with its own status and code and the command line can
still catch them as the ``KeyError`` (missing) or ``ValueError`` (refused) they subclass.

``fields`` holds what an interface may report beside the message: the ``task``, ``label``,
``example_id`` or ``similar_to`` (closest existing names) concerned. Nothing here imports more
than the standard library.
"""

from __future__ import annotations

from typing import Any


class Missing(KeyError):
    """Something named doesn't exist. ``str()`` is the message, not its repr as for a KeyError."""

    code = "NOT_FOUND"

    def __init__(self, text: str, **fields: Any) -> None:
        super().__init__(text)
        self.fields = fields

    def __str__(self) -> str:
        return str(self.args[0])


class Refused(ValueError):
    """A request that can't be done as asked."""

    code = "INVALID_REQUEST"

    def __init__(self, text: str, **fields: Any) -> None:
        super().__init__(text)
        self.fields = fields


class Conflict(Refused):
    """A request that conflicts with the current state (a name taken, nothing to undo)."""

    code = "CONFLICT"


class TaskNotFound(Missing):
    """No task of that name."""

    code = "TASK_NOT_FOUND"


class TaskUnreadable(Missing):
    """A task whose files can't be read."""

    code = "TASK_UNREADABLE"


class ClassNotFound(Missing):
    """No class of that name in the task."""

    code = "CLASS_NOT_FOUND"


class ExampleNotFound(Missing):
    """No example (or inbox image) with that id."""

    code = "EXAMPLE_NOT_FOUND"


class RevisionNotKept(Missing):
    """An earlier version of a task that is no longer kept."""

    code = "REVISION_NOT_FOUND"


class TaskExists(Conflict):
    """A task of that name exists already."""

    code = "TASK_EXISTS"


class ClassExists(Conflict):
    """A class of that name exists already."""

    code = "CLASS_EXISTS"


class NothingToUndo(Conflict):
    """Undo with no change to take back."""

    code = "NOTHING_TO_UNDO"


class NothingToRedo(Conflict):
    """Redo with no undone change to make again."""

    code = "NOTHING_TO_REDO"


class ClassNameNearMiss(Refused):
    """A new class name so close to an existing class's (``similar_to``) that it is probably a
    typo of it; ``label`` is the name given."""

    code = "CLASS_NAME_NEAR_MISS"


class UnknownClassifier(Refused):
    """A classifier name that isn't registered."""

    code = "UNKNOWN_CLASSIFIER"


class NoExamples(Refused):
    """A task with no examples, asked to do what needs them (evaluate)."""

    code = "NO_EXAMPLES"


class RevisionChanged(Exception):
    """A change asked for only at a revision the task has moved on from (the API's If-Match; see
    ``preconditions``). Not a ``ValueError``, so nothing that reports refused items swallows it."""

    code = "PRECONDITION_FAILED"

    def __init__(self, text: str, **fields: Any) -> None:
        super().__init__(text)
        self.fields = fields


def http_status(e: BaseException) -> int:
    """The HTTP status that answers ``e``: 404 for a missing thing, 409 for a conflict, 422 for a
    likely typo of a class name, 412 for a revision moved on from, 400 for another refusal, else 500."""
    if isinstance(e, RevisionChanged):
        return 412
    if isinstance(e, KeyError):
        return 404
    if isinstance(e, Conflict):
        return 409
    if isinstance(e, ClassNameNearMiss):
        return 422
    return 400 if isinstance(e, ValueError) else 500


def error_code(e: BaseException) -> str:
    """The stable code of ``e``: its own, or that of the kind of error it is."""
    code = getattr(e, "code", None)
    if isinstance(code, str):
        return code
    return {404: Missing.code, 400: Refused.code}.get(http_status(e), "INTERNAL")


def message(e: BaseException) -> str:
    """The message of ``e`` on one line: for a KeyError its first argument, not its repr."""
    text = str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)
    return " ".join(text.split())
