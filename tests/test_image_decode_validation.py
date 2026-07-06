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


def _mp4_bytes() -> bytes:
    """Minimal ISO BMFF (mp4/mov) prefix: size + 'ftyp' box at offset 4.

    Reproduces the production failure class where Wildbook dispatches a video
    MediaAsset (e.g. sharkbook .mp4) to the image-only pipeline.
    """
    return bytes.fromhex("00000018667479706d70343200000000") + b"\x00" * 64


def test_video_bytes_raise_image_decode_error_with_video_hint():
    with pytest.raises(ImageDecodeError, match="video"):
        validate_decodable(_mp4_bytes())


def test_webm_bytes_raise_image_decode_error_with_video_hint():
    ebml = b"\x1a\x45\xdf\xa3" + b"\x00" * 64  # Matroska/WebM EBML magic
    with pytest.raises(ImageDecodeError, match="video"):
        validate_decodable(ebml)


def test_avi_bytes_raise_image_decode_error_with_video_hint():
    avi = b"RIFF" + b"\x24\x00\x00\x00" + b"AVI " + b"\x00" * 64
    with pytest.raises(ImageDecodeError, match="video"):
        validate_decodable(avi)


def _gif_bytes() -> bytes:
    """A valid GIF: PIL loads it fine, but cv2.imdecode returns None, which
    crashes every cv2-backed model (EfficientNet/DenseNet/LightNet) with the
    same cvtColor !_src.empty() 500 the video fix targets."""
    buf = io.BytesIO()
    Image.new("P", (32, 32)).save(buf, "gif")
    return buf.getvalue()


def test_gif_pil_loadable_but_cv2_undecodable_raises_image_decode_error():
    with pytest.raises(ImageDecodeError):
        validate_decodable(_gif_bytes())


def test_ico_pil_loadable_but_cv2_undecodable_raises_image_decode_error():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, "ico")
    with pytest.raises(ImageDecodeError):
        validate_decodable(buf.getvalue())


def test_avif_brand_is_not_reported_as_video():
    # AVIF is ISO-BMFF (has an ftyp box) but is a still image; if Pillow can't
    # decode it the 400 must not mislabel it a video.
    avif = bytes.fromhex("0000001c667479706176696600000000") + b"\x00" * 64
    with pytest.raises(ImageDecodeError) as exc_info:
        validate_decodable(avif)
    assert "video" not in str(exc_info.value)


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
