"""Image bytes: content ids, decoding, format detection and thumbnails."""

from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def image_id(data: bytes) -> str:
    """Content id of an image: the first 32 hex chars (128 bits) of its SHA-256.

    128 bits make a deliberate collision (one image passing for another) infeasible; older tasks
    hold 16-char ids, which ``Task.add_example`` still recognises.
    """
    return hashlib.sha256(data).hexdigest()[:32]


def load_image(data: bytes, draft: int | None = None) -> Image.Image:
    """Decode image bytes to an upright RGB image; raises ``ValueError`` if unreadable.

    With ``draft``, a JPEG may be decoded at a reduced scale that is still at least ``draft``
    pixels on each side (much faster for large photos); other formats decode in full.
    """
    try:
        im = Image.open(io.BytesIO(data))
        if draft:
            im.draft("RGB", (draft, draft))
        im.load()
    except UnidentifiedImageError as e:
        raise ValueError("Not a readable image: unrecognised file format") from e
    except Image.DecompressionBombError as e:  # not an OSError subclass
        raise ValueError(f"Not a readable image: {e}") from e
    except OSError as e:
        raise ValueError(f"Not a readable image: {e}") from e
    return ImageOps.exif_transpose(im).convert("RGB")


def thumbnail(data: bytes, size: int) -> bytes:
    """JPEG of the image scaled down to fit in ``size`` x ``size`` pixels (never enlarged);
    raises ``ValueError`` if unreadable."""
    im = load_image(data, draft=size)
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=85)
    return out.getvalue()


def image_ext(data: bytes) -> str:
    """File extension for image bytes from their actual format (e.g. "jpg", "png"), after decoding
    them in full; raises ``ValueError`` if they aren't a readable image."""
    load_image(data)
    fmt = (Image.open(io.BytesIO(data)).format or "png").lower()
    return {"jpeg": "jpg"}.get(fmt, fmt)
