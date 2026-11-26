"""Walking folders of images the way every importer does: ``<folder>/<class>/<image>``.

Hidden entries (``.ipynb_checkpoints``, ``.DS_Store``, macOS ``._*`` files) and the folders
operating systems and archivers add (``__MACOSX``, ``@eaDir``, ``$RECYCLE.BIN``...) are skipped,
so they never become classes or images.
"""

from __future__ import annotations

from pathlib import Path

from .images import IMAGE_EXTS

# Folders made by operating systems, NAS boxes and archivers, never by a person sorting images.
SYSTEM_NAMES = {"__MACOSX", "@eaDir", "$RECYCLE.BIN", "System Volume Information", "lost+found", "Thumbs.db",
                "desktop.ini", "__pycache__"}


def skipped(path: Path) -> bool:
    """Whether ``path`` is hidden or a system file or folder, which imports leave out."""
    return path.name.startswith(".") or path.name in SYSTEM_NAMES


def image_files(folder: Path, recursive: bool = False) -> list[Path]:
    """Image files in ``folder`` (and with ``recursive`` its subfolders), sorted; hidden and system
    entries, and anything that isn't a file with an image extension, are left out."""
    if not recursive:
        found = list(folder.iterdir())
    else:
        found = [p for p in folder.rglob("*") if not any(skipped(Path(part)) for part in p.relative_to(folder).parts)]
    return sorted(f for f in found if f.suffix.lower() in IMAGE_EXTS and not skipped(f) and f.is_file())


def class_folders(folder: Path, shots: int | None = None) -> dict[str, list[Path]]:
    """Each class folder of ``folder`` (``<folder>/<class>/<image>``) mapped to its sorted image
    files, at most ``shots`` of them; classes without images are left out."""
    out = {}
    for d in sorted(p for p in folder.iterdir() if p.is_dir() and not skipped(p)):
        files = image_files(d)[:shots]
        if files:
            out[d.name] = files
    return out


def read_or_empty(path: Path) -> bytes:
    """The bytes of ``path``, or empty bytes if it can't be read: adding them then reports an
    unreadable image, as for any other bad file."""
    try:
        return path.read_bytes()
    except OSError:
        return b""
