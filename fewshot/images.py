"""Image bytes: content ids, decoding, format detection and thumbnails."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
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
        out = to_rgb(ImageOps.exif_transpose(im), white_level(im, data))
    except ValueError:
        raise
    except UnidentifiedImageError as e:
        raise ValueError("Not a readable image: unrecognised file format") from e
    except Exception as e:  # pylint: disable=broad-exception-caught  # damaged files raise all sorts
        # (OSError, DecompressionBombError, SyntaxError for a broken PNG chunk, TypeError or
        # IndexError for a damaged TIFF, struct.error, ...): each is one unreadable file, not a failed request
        raise ValueError(f"Not a readable image: {e or type(e).__name__}") from e
    if max_side and max(out.size) > max_side:
        out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def to_rgb(im: Image.Image, white: float | None = None) -> Image.Image:
    """``im`` as RGB, with any transparency composited onto white: converting directly would show
    whatever colour transparent pixels happen to hold, usually black. That covers an alpha band,
    a palette's transparent entries and a single transparent colour (PNG's tRNS chunk on
    grayscale or RGB). Grayscale with more than 8 bits per pixel is rescaled first: ``white`` is
    the pixel value shown as white, if known (see ``white_level``), else see ``_to_8bit``."""
    if "A" in im.getbands() or "a" in im.getbands() or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
    clear = _keyed_out(im)
    out = (_to_8bit(im, white) if im.mode in HIGH_DEPTH_MODES else im).convert("RGB")
    if clear is not None:
        out.paste((255, 255, 255), mask=Image.fromarray(clear))
    return out


def _keyed_out(im: Image.Image) -> np.ndarray | None:
    """Mask (255 where transparent) of the pixels of ``im`` in its one transparent colour, if it
    has one (PNG's tRNS chunk on a grayscale or RGB image), else None."""
    t = im.info.get("transparency")
    if t is None or im.mode == "P" or not isinstance(t, (int, tuple)):
        return None
    a = np.asarray(im)
    hit: np.ndarray = (a == np.array(t)).all(axis=-1) if a.ndim == 3 else a == t
    return hit.astype(np.uint8) * 255


PNG_HEAD = 1 << 20  # bytes of a PNG file read for its sBIT chunk (which comes before the pixels)
HIGH_DEPTH_MODES = {"I;16", "I;16B", "I;16L", "I;16N", "I", "F"}
STRETCH_PERCENTILE = 99.9  # without a known white level, this percentile of the pixels is white
STRETCH_SAMPLE = 1_000_000  # the percentile is taken over about this many evenly spaced pixels
SCALE_CHUNK = 1 << 18  # pixels rescaled at once by ``_to_8bit``


def white_level(im: Image.Image, data: bytes | None = None) -> float | None:
    """The pixel value that is white in ``im``, a grayscale image of more than 8 bits, if its file
    says: a TIFF's MaxSampleValue tag (which writers of 10- to 14-bit data stored in 16 bits may
    set), else its BitsPerSample if under 16; a PNG (``data``, the file's bytes) with an sBIT
    chunk, which Pillow doesn't read, says its samples were scaled to the full 16 bits, as the PNG
    standard requires. None if the file doesn't say, or ``im`` isn't such an image.

    Without ``data``, a PNG opened from a file has the start of that file read for the chunk; one
    opened from bytes (the file is gone) is taken as saying nothing."""
    if im.mode not in HIGH_DEPTH_MODES or im.mode == "F":
        return None
    name = getattr(im, "filename", None)
    if data is None and im.format == "PNG" and isinstance(name, str) and name:
        try:
            with open(name, "rb") as f:
                data = f.read(PNG_HEAD)
        except OSError:
            pass
    if data is not None and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return 65535.0 if _has_sbit(data) else None
    tags = getattr(im, "tag_v2", None)  # MaxSampleValue, BitsPerSample:
    peak, bits = (_tag(tags, 281), _tag(tags, 258)) if tags is not None else (None, None)
    if peak:
        return float(peak)
    return float(2**bits - 1) if bits and bits < 16 else None


def _tag(tags: Any, tag: int) -> int | None:
    """A TIFF tag's (first) value, if it is a positive integer."""
    v = tags.get(tag)
    v = v[0] if isinstance(v, tuple) and v else v
    return v if isinstance(v, int) and v > 0 else None


def _has_sbit(data: bytes) -> bool:
    """Whether a PNG has an sBIT chunk (which precedes its pixels)."""
    at = 8
    while at + 8 <= len(data):
        kind = data[at + 4 : at + 8]
        if kind == b"sBIT":
            return True
        if kind in (b"IDAT", b"IEND"):
            return False
        at += 12 + int.from_bytes(data[at : at + 4], "big")
    return False


def _to_8bit(im: Image.Image, white: float | None = None) -> Image.Image:
    """16-bit, 32-bit integer or float grayscale as 8-bit ("L"). Pillow's own conversion clips at
    255, so a 16-bit image comes out almost all white.

    Integers are scaled so that 0 stays black and ``white`` (the file's own white level, see
    ``white_level``) is white. Most files don't record one: 10-, 12- or 14-bit data from medical
    and industrial cameras is usually stored in 16 bits as it is, so dividing by 257 as a browser
    does would leave it nearly black. Then the image's ``STRETCH_PERCENTILE``-th percentile is
    white, as an automatic exposure (or ImageJ's auto contrast) would show it: every image is well
    exposed, as embedders trained on photos expect, and a few hot pixels or a slightly brighter
    scene change the result only slightly. Absolute brightness isn't kept across images, though.
    Floats from 0 to 2 are taken as the usual 0 to 1 range, any overshoot clipped; other floats
    are treated as integers are. Images with negative values have no known black either: they are
    stretched from their ``100 - STRETCH_PERCENTILE``-th to their ``STRETCH_PERCENTILE``-th
    percentile."""
    a = np.asarray(im)  # as stored (2 or 4 bytes a pixel): a 50 MP image as floats would be 400 MB a copy
    values = a[np.isfinite(a)] if im.mode == "F" else a.ravel()
    values = values if values.size else np.zeros(1)
    lo, hi = float(values.min()), float(values.max())
    if im.mode == "F" and lo >= 0 and hi <= 2:
        lo, hi = 0.0, 1.0
    elif white and im.mode != "F":
        lo, hi = 0.0, white
    else:
        sample = values[:: max(1, values.size // STRETCH_SAMPLE)].astype(np.float64)
        lo = float(np.percentile(sample, 100 - STRETCH_PERCENTILE)) if lo < 0 else 0.0
        top = float(np.percentile(sample, STRETCH_PERCENTILE))
        hi = top if top > lo else hi  # (almost) one level: the brightest pixel is white
    out = np.zeros(a.shape, np.uint8)
    if hi > lo:
        rows = max(1, SCALE_CHUNK // max(1, a.shape[1]))
        for r in range(0, a.shape[0], rows):  # a few rows at a time, so the floats stay small
            part = np.nan_to_num(a[r : r + rows].astype(np.float64), nan=lo, posinf=hi, neginf=lo)
            out[r : r + rows] = np.rint((part - lo) * (255 / (hi - lo))).clip(0, 255)
    return Image.fromarray(out)


# Appended to the cache key of an image whose conversion to RGB changed since keys were first
# used (see ``conversion_key``), so vectors cached from the old conversion aren't reused.
CONVERSION_KEY = ":px2"  # grayscale of more than 8 bits, or a single transparent colour
ALPHA_KEY = ":px3"  # an alpha band or transparent palette entries: composited on white, once black
CONVERSION_KEYS = (CONVERSION_KEY, ALPHA_KEY)
_NEVER_CHANGED = {".jpg", ".jpeg"}  # formats Pillow never decodes to a mode that changed


def conversion_key(source: bytes | str | Path | Image.Image) -> str:
    """The suffix (one of ``CONVERSION_KEYS``) for the cache key of ``source`` if it is an image
    whose conversion to RGB has changed, else "": ``CONVERSION_KEY`` for grayscale of more than
    8 bits (first rescaled at all, then as ``_to_8bit`` now does) or one with a single transparent
    colour (``_keyed_out``); ``ALPHA_KEY`` for one with an alpha band or a palette with transparent
    entries, whose transparent pixels became white instead of whatever colour they hold (usually
    black). Only the file's header is read; a file or bytes that can't be opened get "" (decoding
    them will fail anyway)."""
    if isinstance(source, Image.Image):
        return _changed(source)
    if isinstance(source, (str, Path)) and Path(source).suffix.lower() in _NEVER_CHANGED:
        return ""
    try:
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as im:
            return _changed(im)
    except Exception:  # pylint: disable=broad-exception-caught  # damaged: decoding will say so
        return ""


def _changed(im: Image.Image) -> str:
    """``conversion_key`` of the opened image ``im``."""
    alpha = "A" in im.getbands() or "a" in im.getbands()
    if alpha or (im.mode == "P" and "transparency" in im.info):
        return ALPHA_KEY
    return CONVERSION_KEY if im.mode in HIGH_DEPTH_MODES or "transparency" in im.info else ""


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
