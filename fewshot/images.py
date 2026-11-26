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


VERIFY_SIDE = 2048  # JPEGs are checked decoded at about this size, so huge photos pass
MAX_PIXELS = 50_000_000  # largest image decoded (after any JPEG draft scaling): about 150 MB as RGB


def load_image(data: bytes, draft: int | None = None, max_side: int | None = None) -> Image.Image:
    """Decode image bytes to an upright RGB image; raises ``ValueError`` if unreadable or over
    ``MAX_PIXELS`` (a small file can hold a huge, highly compressible image).

    With ``draft``, a JPEG may be decoded at a reduced scale that is still at least ``draft``
    pixels on each side (much faster for large photos); other formats decode in full. With
    ``max_side``, the result is scaled down to fit in ``max_side`` x ``max_side``.
    """
    try:
        im = Image.open(io.BytesIO(data))
        if draft:
            im.draft("RGB", (draft, draft))
        if im.width * im.height > MAX_PIXELS:
            raise ValueError(f"Not a readable image: {im.width}x{im.height} is over the "
                             f"{MAX_PIXELS / 1e6:g} megapixel limit")
        im.load()
    except UnidentifiedImageError as e:
        raise ValueError("Not a readable image: unrecognised file format") from e
    except Image.DecompressionBombError as e:  # not an OSError subclass
        raise ValueError(f"Not a readable image: {e}") from e
    except OSError as e:
        raise ValueError(f"Not a readable image: {e}") from e
    out = to_rgb(ImageOps.exif_transpose(im))
    if max_side and max(out.size) > max_side:
        out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def to_rgb(im: Image.Image) -> Image.Image:
    """``im`` as RGB, with any transparency composited onto white: converting directly would show
    whatever colour transparent pixels happen to hold, usually black."""
    if "A" in im.getbands() or "a" in im.getbands() or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
    return im.convert("RGB")


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
    them (a JPEG at a reduced scale); raises ``ValueError`` if they aren't a readable image."""
    load_image(data, draft=VERIFY_SIDE)
    fmt = (Image.open(io.BytesIO(data)).format or "png").lower()
    return {"jpeg": "jpg"}.get(fmt, fmt)
