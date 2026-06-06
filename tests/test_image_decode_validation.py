"""Unit tests for app.utils.image_uri.validate_decodable.

Pure-PIL tests (no model handler / GPU), so they run anywhere. They verify
that a corrupt image is rejected with ImageDecodeError (a ValueError) — which
the inference routers map to HTTP 400, so consumers treat a corrupt image as a
permanent (non-retryable) client error rather than a retryable 5xx.
"""

import io

import pytest
from PIL import Image

from app.utils.image_uri import ImageDecodeError, validate_decodable


def _valid_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), (120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, "jpeg", quality=90)
    return buf.getvalue()


def _corrupt_jpeg_bytes() -> bytes:
    """A JPEG with a valid header but a broken entropy-coded scan stream.

    Splicing 0xFF 0x99 (an unsupported marker) into the scan data reproduces the
    real-world failure class ("Unsupported marker type 0x99" /
    "broken data stream when reading image file") that fails during full load(),
    not at header parse.
    """
    data = bytearray(_valid_jpeg_bytes())
    at = max(20, len(data) - 40)
    for i in range(at, min(at + 32, len(data) - 1), 2):
        data[i] = 0xFF
        data[i + 1] = 0x99
    return bytes(data)


def test_valid_image_passes():
    # Should not raise.
    validate_decodable(_valid_jpeg_bytes())


def test_corrupt_image_raises_image_decode_error():
    with pytest.raises(ImageDecodeError):
        validate_decodable(_corrupt_jpeg_bytes())


def test_image_decode_error_is_value_error():
    # Routers catch ValueError -> HTTP 400; ImageDecodeError must subclass it
    # so an undecodable image is reported as a 4xx, not an unhandled 5xx.
    assert issubclass(ImageDecodeError, ValueError)


def test_non_image_bytes_raise_image_decode_error():
    with pytest.raises(ImageDecodeError):
        validate_decodable(b"this is definitely not an image")


def test_empty_bytes_raise_image_decode_error():
    with pytest.raises(ImageDecodeError):
        validate_decodable(b"")


def test_truncated_jpeg_raises_image_decode_error():
    # Keep only the first 200 bytes (header + partial scan) — load() must fail.
    truncated = _valid_jpeg_bytes()[:200]
    with pytest.raises(ImageDecodeError):
        validate_decodable(truncated)


def test_decompression_bomb_raises_image_decode_error():
    # A decompression bomb raises PIL's DecompressionBombError, which is NOT an
    # OSError; verify it is still mapped to ImageDecodeError. Force the guard by
    # lowering Pillow's pixel limit so an ordinary image trips it.
    from PIL import Image

    valid = _valid_jpeg_bytes()  # 64x64 = 4096 px
    saved = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = 1  # 2*1 < 4096 -> DecompressionBombError on load
        with pytest.raises(ImageDecodeError):
            validate_decodable(valid)
    finally:
        Image.MAX_IMAGE_PIXELS = saved
