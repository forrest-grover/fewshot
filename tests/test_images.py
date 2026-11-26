"""Decoding images (fewshot.images): pixel limits, shrinking before embedding, transparency and
high bit depths."""

import io

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


def _saved(a, fmt):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, fmt)
    return buf.getvalue()


def test_high_bit_depth_grayscale_is_rescaled():
    """16-bit PNG and TIFF gradients load as gradients, not clipped to white at 255; 12-bit data
    stored in 16 bits fills the 8-bit range; 32-bit and float images are scaled to their range."""
    full = np.linspace(0, 65535, 256 * 16).reshape(16, 256).astype(np.uint16)
    for fmt in ("PNG", "TIFF"):
        g = np.asarray(load_image(_saved(full, fmt)))[..., 0].astype(float)
        assert (g == 255).mean() < 0.01
        assert g[0, 0] == 0
        assert g[-1, -1] == 255
        assert abs(g.mean() - 127.5) < 1
    twelve = np.asarray(load_image(_saved((full >> 4).astype(np.uint16), "PNG")))[..., 0]
    assert twelve.max() == 255
    assert abs(twelve.mean() - 127.5) < 1
    dark = np.asarray(load_image(_saved(np.full((4, 4), 3000, np.uint16), "PNG")))  # 3000 of 12 bits
    assert (dark == round(3000 * 255 / 4095)).all()
    big = np.asarray(to_rgb(Image.fromarray(np.array([[-100, 100_000]], np.int32))))[..., 0]
    assert big.tolist() == [[0, 255]]
    unit = Image.fromarray(np.array([[0, 0.5, 1]], np.float32))
    assert np.asarray(to_rgb(unit))[0, :, 0].tolist() == [0, 128, 255]
    hdr = Image.fromarray(np.array([[2.0, 2.0]], np.float32))  # no range: one level, black
    assert np.asarray(to_rgb(hdr)).max() == 0
    eight = np.asarray(to_rgb(Image.fromarray(np.array([[0, 200]], np.uint16))))[0, :, 0]
    assert eight.tolist() == [0, 200]  # 8-bit values stored in 16 bits are kept
