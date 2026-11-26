"""Decoding images (fewshot.images): pixel limits, shrinking before embedding, transparency and
high bit depths."""

import io
import zlib

import numpy as np
import pytest
from PIL import Image

from fewshot import images, pipeline
from fewshot.images import load_image, to_rgb
from tests.conftest import RED


def _jpeg_of(size):
    buf = io.BytesIO()
    Image.new("RGB", size, RED).save(buf, "JPEG")
    return buf.getvalue()


def _png_of(mode, size):
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, "PNG")
    return buf.getvalue()


def test_pixel_limit(monkeypatch):
    """An image over MAX_PIXELS is refused before decoding; a JPEG counts at its draft scale."""
    monkeypatch.setattr(images, "MAX_PIXELS", 10_000)
    with pytest.raises(ValueError, match=r"200x200 is over the 0\.01 megapixel limit"):
        images.load_image(_png_of("1", (200, 200)))
    with pytest.raises(ValueError, match="megapixel limit"):
        images.image_ext(_png_of("1", (200, 200)))
    monkeypatch.setattr(images, "VERIFY_SIDE", 50)
    assert images.image_ext(_jpeg_of((400, 400))) == "jpg"  # verified at 1/8 scale: 50x50


def test_images_are_shrunk_before_embedding(monkeypatch):
    """Decoded images, and PIL images passed in, are scaled to fit EMBED_MAX_SIDE."""
    assert images.load_image(_png_of("RGB", (300, 150)), max_side=100).size == (100, 50)
    assert images.load_image(_png_of("RGB", (30, 15)), max_side=100).size == (30, 15)
    monkeypatch.setattr(pipeline, "EMBED_MAX_SIDE", 64)
    assert pipeline._to_pil(_png_of("RGB", (256, 128))).size == (64, 32)  # pylint: disable=protected-access
    assert pipeline._to_pil(Image.new("L", (128, 256))).size == (32, 64)  # pylint: disable=protected-access
    assert pipeline._to_pil(Image.new("L", (16, 16))).mode == "RGB"  # pylint: disable=protected-access


def test_transparent_pixels_become_white():
    """RGBA, LA and palette transparency are composited onto white; opaque images are unchanged;
    PIL images passed to predict are turned upright too."""
    for im in (Image.new("RGBA", (4, 4), (0, 0, 0, 0)), Image.new("LA", (4, 4), (0, 0))):
        assert to_rgb(im).getpixel((0, 0)) == (255, 255, 255)
    buf = io.BytesIO()
    Image.new("P", (4, 4), 0).save(buf, "PNG", transparency=0)
    assert load_image(buf.getvalue()).getpixel((0, 0)) == (255, 255, 255)
    assert to_rgb(Image.new("RGB", (4, 4), RED)).getpixel((0, 0)) == RED
    tall = Image.new("RGB", (4, 8), RED)
    exif = tall.getexif()
    exif[0x0112] = 6  # rotated 90 degrees
    buf = io.BytesIO()
    tall.save(buf, "JPEG", exif=exif.tobytes())
    assert pipeline._to_pil(Image.open(io.BytesIO(buf.getvalue()))).size == (8, 4)  # pylint: disable=protected-access


def _saved(a, fmt, **kw):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, fmt, **kw)
    return buf.getvalue()


def _with_sbit(png, bits):
    """``png`` with an sBIT chunk (which Pillow doesn't write) after its IHDR."""
    body = b"sBIT" + bytes([bits])
    chunk = (1).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")
    return png[:33] + chunk + png[33:]  # signature (8) + IHDR (25)


def _gray(data):
    return np.asarray(load_image(data))[..., 0].astype(float)


def test_high_bit_depth_grayscale_is_rescaled():
    """16-bit PNG and TIFF gradients load as gradients, not clipped to white at 255; 12-bit data
    stored in 16 bits fills the 8-bit range; 32-bit and float images are scaled to their range."""
    full = np.linspace(0, 65535, 256 * 16).reshape(16, 256).astype(np.uint16)
    for fmt in ("PNG", "TIFF"):
        g = _gray(_saved(full, fmt))
        assert (g == 255).mean() < 0.01
        assert g[0, 0] == 0
        assert g[-1, -1] == 255
        assert abs(g.mean() - 127.5) < 1
    twelve = _gray(_saved((full >> 4).astype(np.uint16), "PNG"))
    assert twelve.max() == 255
    assert abs(twelve.mean() - 127.5) < 1
    big = np.asarray(to_rgb(Image.fromarray(np.array([[-100, 100_000]], np.int32))))[..., 0]
    assert big.tolist() == [[0, 255]]
    eight = np.asarray(to_rgb(Image.fromarray(np.array([[0, 200]], np.uint16))))[0, :, 0]
    assert eight.tolist() == [0, 255]  # no known depth: the brightest pixel is white
    assert np.asarray(to_rgb(Image.fromarray(np.zeros((2, 2), np.uint16)))).max() == 0
    sparse = np.zeros((40, 40), np.uint16)
    sparse[0, 0] = 9
    assert _gray(_saved(sparse, "PNG"))[0, 0] == 255  # under 0.1% lit: the brightest is white


def test_bit_depth_is_not_guessed_from_the_brightest_pixel():
    """One hot pixel doesn't change how an image is scaled (the 99.9th percentile is white), and
    neither does crossing a power of two: a scene from a 12-bit camera with its brightest pixel at
    1023 and one at 1024 look alike."""
    base = np.random.default_rng(0).integers(0, 1000, (64, 64)).astype(np.uint16)
    means = []
    for peak in (1023, 1024, 4095, 65535):
        a = base.copy()
        a[0, 0] = peak
        means.append(_gray(_saved(a, "PNG"))[1:, 1:].mean())
    assert max(means) - min(means) < 0.5
    assert abs(means[0] - 127.5) < 3


def test_white_level_from_metadata():
    """A file that says which value is white is scaled by it, so its absolute brightness is kept:
    a TIFF's MaxSampleValue or BitsPerSample, or a PNG's sBIT, which per the PNG standard means
    the samples fill all 16 bits."""
    dark = np.full((4, 4), 3000, np.uint16)
    assert (_gray(_saved(dark, "TIFF", tiffinfo={281: 4095})) == round(3000 * 255 / 4095)).all()
    assert (_gray(_saved(dark, "PNG")) == 255).all()  # no metadata: auto exposure
    assert (_gray(_with_sbit(_saved(dark, "PNG"), 12)) == round(3000 / 257)).all()
    twelve = Image.fromarray(dark)
    twelve.tag_v2 = {258: (12,)}  # as Pillow reads a 12-bit TIFF
    assert images.white_level(twelve) == 4095
    twelve.tag_v2 = {258: (16,)}
    assert images.white_level(twelve) is None
    assert images.white_level(Image.new("L", (1, 1))) is None
    assert images.white_level(Image.fromarray(dark), _png_of("L", (1, 1))) is None  # no sBIT before IDAT
    assert images.white_level(Image.fromarray(dark), b"\x89PNG\r\n\x1a\n") is None  # truncated


def test_float_images():
    """Floats from 0 to 2 are the 0..1 range, overshoot clipped (not stretched); a constant one
    isn't black; others are scaled as integers are, or stretched if they go negative."""
    unit = Image.fromarray(np.array([[0, 0.5, 1]], np.float32))
    assert np.asarray(to_rgb(unit))[0, :, 0].tolist() == [0, 128, 255]
    over = Image.fromarray(np.array([[0.5, 1.5, np.nan]], np.float32))
    assert np.asarray(to_rgb(over))[0, :, 0].tolist() == [128, 255, 0]
    assert np.asarray(to_rgb(Image.fromarray(np.full((2, 2), np.nan, np.float32)))).max() == 0
    for v in (0.5, 2.0, 200.0):
        assert np.asarray(to_rgb(Image.fromarray(np.full((3, 3), v, np.float32)))).min() > 0
    wide = np.random.default_rng(0).uniform(0, 255, (32, 32)).astype(np.float32)
    assert abs(_gray(_saved(wide, "TIFF")).mean() - wide.mean()) < 1
    signed = np.linspace(-1, 1, 1000, dtype=np.float32).reshape(10, 100)
    g = np.asarray(to_rgb(Image.fromarray(signed)))[..., 0]
    assert g[0, 0] == 0
    assert g[-1, -1] == 255
    assert abs(g.mean() - 127.5) < 1
    assert np.asarray(to_rgb(Image.fromarray(np.full((2, 2), -3.0, np.float32)))).max() == 0


def test_single_transparent_colour_becomes_white():
    """PNG's tRNS on grayscale (8- and 16-bit) and RGB makes that colour white, as for palettes."""
    gray = np.array([[0, 10, 20]], np.uint8)
    assert _gray(_saved(gray, "PNG", transparency=10)).tolist() == [[0, 255, 20]]
    deep = np.array([[0, 30000, 60000]], np.uint16)
    assert _gray(_saved(deep, "PNG", transparency=0)).tolist()[0][0] == 255
    rgb = io.BytesIO()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(rgb, "PNG", transparency=(1, 2, 3))
    assert load_image(rgb.getvalue()).getpixel((0, 0)) == (255, 255, 255)
    assert load_image(_saved(gray, "PNG")).getpixel((1, 0)) == (10, 10, 10)


def test_conversion_key():
    """Images whose conversion to RGB changed get CONVERSION_KEY, judged from the header alone."""
    deep = _saved(np.zeros((2, 2), np.uint16), "PNG")
    keyed = _saved(np.zeros((2, 2), np.uint8), "PNG", transparency=0)
    assert images.conversion_key(deep) == images.conversion_key(keyed) == images.CONVERSION_KEY
    assert images.conversion_key(Image.new("F", (1, 1))) == images.CONVERSION_KEY
    for plain in (_png_of("L", (2, 2)), _png_of("RGBA", (2, 2)), _jpeg_of((2, 2)), b"not an image"):
        assert images.conversion_key(plain) == ""
    assert images.conversion_key(Image.new("RGB", (1, 1))) == ""


def test_conversion_key_of_files(tmp_path):
    """Files are judged from their header; JPEG and WebP files, which can't change, aren't opened."""
    (tmp_path / "a.png").write_bytes(_saved(np.zeros((2, 2), np.uint16), "PNG"))
    (tmp_path / "b.jpg").write_bytes(b"not even a JPEG")
    assert images.conversion_key(tmp_path / "a.png") == images.CONVERSION_KEY
    assert images.conversion_key(str(tmp_path / "b.jpg")) == ""
    assert images.conversion_key(tmp_path / "missing.png") == ""
