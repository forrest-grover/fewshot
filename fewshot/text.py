"""Checks and helpers for the text in a task: class names, tags and description prompts."""

from __future__ import annotations

import difflib
from collections.abc import Sequence

MAX_CLASS_NAME = 200
MAX_TAG = 50
# A description's prompts are each embedded (and averaged): cap how many, and how long.
MAX_DESCRIPTION = 10_000  # characters
MAX_PROMPTS = 64  # non-blank lines


def check_class_name(name: str) -> str:
    """Return ``name`` stripped; raise ``ValueError`` if it can't be a class name.

    Class names appear as URL path segments, so ``/``, ``\\``, ``.`` and ``..`` are refused (a
    client would resolve ``.../classes/..`` to the task itself), as are control characters and
    text that isn't valid Unicode (a command-line argument that isn't UTF-8).
    """
    name = name.strip()
    if not name:
        raise ValueError("Class name is empty")
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"Class name {name!r} isn't valid UTF-8") from None
    if name in {".", ".."} or any(ch in "/\\" or ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise ValueError(f"Class name {name!r} can't be '.' or '..' or contain '/', '\\' or control characters")
    if len(name) > MAX_CLASS_NAME:
        raise ValueError(f"Class name is longer than {MAX_CLASS_NAME} characters")
    return name


def check_tag(tag: str) -> str:
    """Return ``tag`` stripped; raise ``ValueError`` if it is empty, too long, or holds a comma or
    control character."""
    tag = tag.strip()
    if not tag:
        raise ValueError("Tag is empty")
    if len(tag) > MAX_TAG:
        raise ValueError(f"Tag is longer than {MAX_TAG} characters")
    if any(ch == "," or ord(ch) < 32 or ord(ch) == 127 for ch in tag):
        raise ValueError(f"Tag {tag!r} can't contain ',' or control characters")
    return tag


def prompts(description: str) -> list[str]:
    """The text prompts in a class description: one per non-blank line. Several prompts are
    averaged into one text embedding (prompt ensembling)."""
    return [line.strip() for line in description.splitlines() if line.strip()]


def fill_template(template: str, name: str) -> str:
    """Text for class ``name`` from a template such as 'a photo of a {}' (underscores become spaces).

    Only the literal ``{}`` is substituted; any other braces are kept as written. A template of
    several lines gives several prompts (see ``prompts``).
    """
    return template.replace("{}", name.replace("_", " "))


def check_description(description: str) -> str:
    """Return ``description``; raise ``ValueError`` if it is longer than ``MAX_DESCRIPTION``
    characters or holds more than ``MAX_PROMPTS`` prompts (each is a text embedding to compute)."""
    if len(description) > MAX_DESCRIPTION:
        raise ValueError(f"Description is longer than {MAX_DESCRIPTION} characters")
    if len(prompts(description)) > MAX_PROMPTS:
        raise ValueError(f"Description has more than {MAX_PROMPTS} prompts (non-blank lines)")
    return description


def check_template(template: str) -> None:
    """Raise ``ValueError`` unless every non-blank line of ``template`` contains ``{}``, or if it
    is too long for a description (see ``check_description``)."""
    check_description(template)
    lines = prompts(template)
    if not lines or any("{}" not in line for line in lines):
        raise ValueError("Template must contain '{}' (on every line, for several prompts)")


def closest(name: str, names: Sequence[str]) -> list[str]:
    """The names among ``names`` closest to ``name``, best first: those the same but for case,
    else up to 3 similar ones (``[]`` if none is)."""
    same_case = [n for n in names if n.lower() == name.lower()]
    return same_case or difflib.get_close_matches(name, names, n=3, cutoff=0.6)


def _edits(a: str, b: str) -> int:
    """The edit (Levenshtein) distance between ``a`` and ``b``."""
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def near_miss(name: str, names: Sequence[str]) -> str | None:
    """An existing class name that ``name`` (not one of them) is probably a typo of: the same but
    for case, or within 2 edits (1 for names of 4 characters or fewer)."""
    allowed = 1 if len(name) <= 4 else 2
    for n in names:
        if n.lower() == name.lower() or _edits(n.lower(), name.lower()) <= allowed:
            return n
    return None
