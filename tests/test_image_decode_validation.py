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


def test_pillow_out_of_memory_is_not_a_decode_error(monkeypatch):
    """Pillow reports a codec allocation failure as OSError(-9, 'out of memory
    error'). That is a server-side resource failure: it must escape (-> 500,
    retried by Wildbook), never be reported as a permanently bad image."""
    from app.utils import image_uri

    def exploding_open(*a, **k):
        raise OSError("out of memory error")
    monkeypatch.setattr(image_uri.Image, "open", exploding_open)
    with pytest.raises(OSError) as exc_info:
        validate_decodable(_valid_jpeg_bytes())
    assert not isinstance(exc_info.value, ImageDecodeError)


def test_unknown_bmff_brand_is_not_reported_as_video():
    # 'heim' is a registered HEIF still-image brand the sniff does not list.
    # The video check is affirmative, so an unknown brand must fall through
    # to the generic message rather than be called a video.
    heim = bytes.fromhex("0000001c6674797068656d6d00000000") + b"\x00" * 64
    with pytest.raises(ImageDecodeError) as exc_info:
        validate_decodable(heim)
    assert "video" not in str(exc_info.value)


def test_non_image_message_has_no_object_address():
    with pytest.raises(ImageDecodeError) as exc_info:
        validate_decodable(b"this is definitely not an image")
    assert "BytesIO" not in str(exc_info.value)
    assert "0x" not in str(exc_info.value)


def test_png_over_opencv_width_limit_is_decode_error():
    # 1048577 px wide is under IMAGE_MAX_PIXELS but over OpenCV's per-side
    # limit. libpng rejects it inside imdecode, which returns None rather
    # than raising, so it is a 400 like any other cv2-undecodable image.
    buf = io.BytesIO()
    Image.new("L", ((1 << 20) + 1, 1)).save(buf, "png")
    with pytest.raises(ImageDecodeError, match="OpenCV"):
        validate_decodable(buf.getvalue())


def test_pillow_core_is_closed_before_opencv_decode(monkeypatch):
    """Both decoders produce a full-size pixel buffer; PIL's must be gone
    before cv2 allocates its own. Image.close() is what destroys the core (a
    `with` block only closes the file pointer), so assert close() precedes
    imdecode()."""
    from app.utils import image_uri
    events = []
    real_open = image_uri.Image.open

    def spy_open(*a, **k):
        img = real_open(*a, **k)
        real_close = img.close

        def close():
            events.append("close")
            real_close()
        img.close = close
        return img

    real_imdecode = image_uri.cv2.imdecode

    def spy_imdecode(*a, **k):
        events.append("imdecode")
        return real_imdecode(*a, **k)

    monkeypatch.setattr(image_uri.Image, "open", spy_open)
    monkeypatch.setattr(image_uri.cv2, "imdecode", spy_imdecode)
    validate_decodable(_valid_jpeg_bytes())
    assert events == ["close", "imdecode"]


def _png_with_corrupt_mid_stream_chunk() -> bytes:
    """Header and first IDAT intact, so Image.open() and check_image_header()
    pass; the SECOND IDAT's chunk id is zeroed. Pillow's load() then raises
    SyntaxError("broken PNG file"), not OSError -- the one decode failure the
    original except-tuple did not cover."""
    import struct
    import numpy as np
    rng = np.random.default_rng(0)  # noise compresses badly -> several IDATs
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(buf, "png")
    data = bytearray(buf.getvalue())
    idat_ids, pos = [], 8
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        if bytes(data[pos + 4:pos + 8]) == b"IDAT":
            idat_ids.append(pos + 4)
        pos += 12 + length
    assert len(idat_ids) >= 2, "test image did not produce multiple IDAT chunks"
    data[idat_ids[1]:idat_ids[1] + 4] = b"\x00\x00\x00\x00"
    return bytes(data)


def test_png_corrupt_after_first_idat_raises_image_decode_error():
    data = _png_with_corrupt_mid_stream_chunk()
    Image.open(io.BytesIO(data))  # header parses: this is a load()-stage failure
    with pytest.raises(ImageDecodeError, match="SyntaxError"):
        validate_decodable(data)


def test_bmp_over_opencv_width_limit_is_decode_error():
    # Unlike libpng, OpenCV's BMP reader has no width limit of its own, so
    # imdecode reaches validateInputImageSize(), which RAISES cv2.error
    # rather than returning None. Still a permanently bad input -> 400.
    buf = io.BytesIO()
    Image.new("L", ((1 << 20) + 1, 1)).save(buf, "bmp")
    with pytest.raises(ImageDecodeError, match="OpenCV"):
        validate_decodable(buf.getvalue())


def test_opencv_allocation_failure_is_not_a_decode_error(monkeypatch):
    from app.utils import image_uri
    oom = image_uri.cv2.error(
        "OpenCV(4.10.0) alloc.cpp:73: error: (-4:Insufficient memory) "
        "Failed to allocate 3145728 bytes in function 'OutOfMemoryError'")

    def exploding_imdecode(*a, **k):
        raise oom
    monkeypatch.setattr(image_uri.cv2, "imdecode", exploding_imdecode)
    with pytest.raises(image_uri.cv2.error) as exc_info:
        validate_decodable(_valid_jpeg_bytes())
    assert exc_info.value is oom
