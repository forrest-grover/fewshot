"""A task's inbox: unlabelled images kept with the task until someone labels or dismisses them.

Inbox images are stored like examples (``images/<id>.<ext>``, shared through the workspace store)
and listed in ``task.json``, so undo, redo, export and the cross-process lock cover them too.
Labelling one moves it to the examples; the model's suggestions for the inbox come from
``Pipeline.inbox_suggestions``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, TypeVar

from .images import image_ext, image_id
from .text import check_class_name

if TYPE_CHECKING:
    from .store import InboxItem, TaskConfig

INBOX_CHUNK = 256  # images stored per save while adding


T = TypeVar("T")


def chunked(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """``items`` in lists of ``size`` (the last may be shorter), holding one list at a time."""
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def listed(names: Iterable[str]) -> str:
    """The distinct ``names`` in order, at most three: "red, blue, green, ..."."""
    unique = list(dict.fromkeys(names))
    return ", ".join(unique[:3]) + (", ..." if len(unique) > 3 else "")


def plural(count: int, what: str = "image") -> str:
    """ "1 image", "3 images"."""
    return f"{count} {what}{'' if count == 1 else 's'}"


def _decoded(data: bytes, source: str) -> tuple[bytes, str, str, str] | ValueError:
    """``(data, id, extension, source)`` for a readable image, else the error."""
    try:
        return data, image_id(data), image_ext(data), source
    except ValueError as e:
        return e


class TaskInbox:
    """The inbox half of ``store.Task``, which provides the attributes and methods declared here."""

    config: TaskConfig

    def _writing(self) -> contextlib.AbstractContextManager[None]:  # pragma: no cover  # Task's
        raise NotImplementedError

    def _save(self, gone: Iterable[str] = (), record: bool = True, label: str = "",
              keep_redo: bool = False) -> None:  # pragma: no cover  # Task's
        raise NotImplementedError

    def _store_image(self, file: str, data: bytes) -> bool:  # pragma: no cover  # Task's
        raise NotImplementedError

    def _save_new(self, created: list[str], record: bool, label: str) -> None:  # pragma: no cover  # Task's
        raise NotImplementedError

    def _drop_file(self, file: str) -> None:  # pragma: no cover  # Task's
        raise NotImplementedError

    def _ensure_class(self, label: str) -> None:  # pragma: no cover  # Task's
        raise NotImplementedError

    def add_to_inbox(self, items: Iterable[tuple[bytes, str]],
                     chunk: int = INBOX_CHUNK) -> list[InboxItem | ValueError]:
        """Store ``(image bytes, source)`` items in the inbox, one save per ``chunk`` and one undo
        step for the whole call. Returns per item its ``InboxItem`` (the existing one if it is
        already waiting) or the ``ValueError`` that kept it out: unreadable, or already labelled."""
        out: list[InboxItem | ValueError] = []
        recorded = False  # once a chunk has saved, the version before the call is kept for undo
        for batch in chunked(items, chunk):
            part, saved = self._inbox_batch(batch, record=not recorded)
            out += part
            recorded = recorded or saved
        return out

    def _inbox_batch(self, batch: list[tuple[bytes, str]],
                     record: bool) -> tuple[list[InboxItem | ValueError], bool]:
        """Add one chunk; returns its results and whether it saved (it added an image)."""
        from .store import InboxItem  # pylint: disable=import-outside-toplevel,cyclic-import  # store imports this

        checked = [_decoded(data, source) for data, source in batch]  # before locking, like examples
        out: list[InboxItem | ValueError] = []
        created: list[str] = []  # files this call wrote (an image kept for undo may be there already)
        new = 0
        with self._writing():
            labelled = {e.id: e.label for e in self.config.examples}
            waiting = {x.id: x for x in self.config.inbox}
            for c in checked:
                if isinstance(c, ValueError):
                    out.append(c)
                elif c[1] in labelled:
                    out.append(ValueError(f"Already an example of {labelled[c[1]]}"))
                elif c[1] in waiting:
                    out.append(waiting[c[1]])
                else:
                    item = waiting[c[1]] = InboxItem(id=c[1], file=f"images/{c[1]}.{c[2]}", source=c[3])
                    if self._store_image(item.file, c[0]):
                        created.append(item.file)
                    self.config.inbox.append(item)
                    out.append(item)
                    new += 1
            if new:
                self._save_new(created, record, f"add {plural(new)} to the inbox")
        return out, new > 0

    def inbox_item(self, iid: str) -> InboxItem:
        """The inbox image with id ``iid``; ``KeyError`` if it isn't waiting there."""
        for x in self.config.inbox:
            if x.id == iid:
                return x
        raise KeyError(f"No inbox image {iid!r}")

    def accept(self, labels: Mapping[str, str]) -> int:
        """Label inbox images (``{id: class}``), moving them to the examples in one save and undo
        step; classes are created as needed. Raises ``KeyError`` for an id not in the inbox and
        ``ValueError`` for a bad class name, changing nothing; returns how many moved."""
        from .store import Example  # pylint: disable=import-outside-toplevel,cyclic-import  # store imports this

        clean = {iid: check_class_name(label) for iid, label in labels.items()}
        with self._writing():
            items = [self.inbox_item(iid) for iid in clean]
            for item in items:
                self._ensure_class(clean[item.id])
                self.config.examples.append(Example(id=item.id, file=item.file, source=item.source,
                                                    label=clean[item.id]))
            self.config.inbox = [x for x in self.config.inbox if x.id not in clean]
            self._save(label=f"label {plural(len(items))} from the inbox as {listed(sorted(clean.values()))}")
            return len(items)

    def dismiss(self, ids: Iterable[str]) -> int:
        """Remove images from the inbox without labelling them (undo brings them back). Raises
        ``KeyError`` for an id not in the inbox, changing nothing; returns how many went."""
        wanted = set(ids)
        with self._writing():
            gone = [self.inbox_item(iid).file for iid in wanted]
            self.config.inbox = [x for x in self.config.inbox if x.id not in wanted]
            self._save(gone=gone, label=f"dismiss {plural(len(gone))} from the inbox")
            return len(gone)
