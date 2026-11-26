"""A task's inbox: unlabelled images kept with the task until someone labels or dismisses them.

Inbox images are stored like examples (``images/<id>.<ext>``, shared through the workspace store)
and listed in ``task.json``, so undo, redo, export and the cross-process lock cover them too.
Labelling one moves it to the examples; the model's suggestions for the inbox come from
``Pipeline.inbox_suggestions``.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING

from .images import image_ext, image_id
from .models import Example, InboxItem
from .text import check_class_name

if TYPE_CHECKING:
    from .models import TaskConfig

INBOX_CHUNK = 256  # images stored per save while adding




def chunked[T](items: Iterable[T], size: int) -> Iterator[list[T]]:
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

    def _save_in_group(self, created: list[str], group: tuple[str, str], count: int, names: list[str],  # pylint: disable=too-many-arguments,too-many-positional-arguments
                       describe: Callable[[int, list[str]], str]) -> None:  # pragma: no cover  # TaskHistory's
        raise NotImplementedError

    def _drop_file(self, file: str) -> None:  # pragma: no cover  # Task's
        raise NotImplementedError

    def _ensure_class(self, label: str) -> None:  # pragma: no cover  # Task's
        raise NotImplementedError

    def add_to_inbox(self, items: Iterable[tuple[bytes, str]], chunk: int = INBOX_CHUNK,
                     undo_group: str | None = None) -> list[InboxItem | ValueError]:
        """Store ``(image bytes, source)`` items in the inbox, one save per ``chunk`` and one undo
        step for the whole call, or for consecutive calls with the same ``undo_group`` (as
        ``Task.add_examples``). Returns per item its ``InboxItem`` (the existing one if it is
        already waiting) or the ``ValueError`` that kept it out: unreadable, or already labelled."""
        group = ("inbox", undo_group or f"call:{uuid.uuid4().hex}")
        out: list[InboxItem | ValueError] = []
        for batch in chunked(items, chunk):
            out += self._inbox_batch(batch, group)
        return out

    def _inbox_batch(self, batch: list[tuple[bytes, str]], group: tuple[str, str]) -> list[InboxItem | ValueError]:
        """Add one chunk, in undo group ``group``."""
        checked = [_decoded(data, source) for data, source in batch]  # before locking, like examples
        out: list[InboxItem | ValueError] = []
        created: list[str] = []  # files this call wrote (an image kept for undo may be there already)
        new = 0
        with self._writing():
            # 16 characters: an example from a task made before 32-character ids, same content
            labelled = {e.id: e.label for e in self.config.examples}
            short = {e.id: e.label for e in self.config.examples if len(e.id) == 16}
            waiting = {x.id: x for x in self.config.inbox}
            for c in checked:
                if isinstance(c, ValueError):
                    out.append(c)
                elif c[1] in labelled or c[1][:16] in short:
                    out.append(ValueError(f"Already an example of {labelled.get(c[1]) or short[c[1][:16]]}"))
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
                self._save_in_group(created, group, new, [], lambda n, _: f"add {plural(n)} to the inbox")
        return out

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
