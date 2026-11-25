"""Checks and helpers for the text in a task: class names, tags and description prompts."""

from __future__ import annotations

MAX_CLASS_NAME = 200
MAX_TAG = 50


def check_class_name(name: str) -> str:
    """Return ``name`` stripped; raise ``ValueError`` if it can't be a class name.

    Class names appear as URL path segments, so ``/``, ``\\``, ``.`` and ``..`` are refused (a
    client would resolve ``.../classes/..`` to the task itself), as are control characters.
    """
    name = name.strip()
    if not name:
        raise ValueError("Class name is empty")
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


def check_template(template: str) -> None:
    """Raise ``ValueError`` unless every non-blank line of ``template`` contains ``{}``."""
    lines = prompts(template)
    if not lines or any("{}" not in line for line in lines):
        raise ValueError("Template must contain '{}' (on every line, for several prompts)")
