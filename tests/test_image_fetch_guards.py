"""Guards: encoded-length cap for data URIs, stat-first local files,
header-parsed pixel cap."""
import asyncio
import base64
import io
import os
import struct
import zlib

import pytest
from PIL import Image

from app.utils import image_uri

# Valid 1x1 red PNG
ONE_PX_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _png_with_dimensions(width, height):
    """Craft a PNG whose IHDR declares width x height (header-only parse)."""
    img = Image.new("RGB", (1, 1))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = bytearray(buf.getvalue())
    # IHDR starts at byte 16: 4-byte width, 4-byte height (big-endian)
    data[16:24] = struct.pack(">II", width, height)
    # fix the IHDR CRC (bytes 29-33 cover chunk type+data at 12..29)
    data[29:33] = struct.pack(">I", zlib.crc32(bytes(data[12:29])))
    return bytes(data)


def test_check_image_header_accepts_small_png():
    image_uri.init_image_fetch()
    image_uri.check_image_header(ONE_PX_PNG)  # must not raise


def test_check_image_header_rejects_bomb():
    image_uri.init_image_fetch()
    bomb = _png_with_dimensions(20000, 20000)  # 400 MP > 150 MP cap
    with pytest.raises(image_uri.ImageTooLargeError):
        image_uri.check_image_header(bomb)


def test_check_image_header_rejects_non_image():
    image_uri.init_image_fetch()
    with pytest.raises(ValueError):
        image_uri.check_image_header(b"\x00\x00\x00\x18ftypmp42 not an image")


def test_data_uri_encoded_length_cap(monkeypatch):
    monkeypatch.setenv("IMAGE_FETCH_MAX_BYTES", "8")
    image_uri.init_image_fetch()
    big = "data:image/png;base64," + base64.b64encode(b"x" * 64).decode()
    with pytest.raises(image_uri.ImageTooLargeError):
        asyncio.run(image_uri.resolve_image_uri(big))


def test_local_path_rejects_fifo(tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError):
        asyncio.run(image_uri.resolve_image_uri(str(fifo)))


def test_local_path_rejects_oversize(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_FETCH_MAX_BYTES", "4")
    f = tmp_path / "big.jpg"
    f.write_bytes(b"x" * 64)
    with pytest.raises(image_uri.ImageTooLargeError):
        asyncio.run(image_uri.resolve_image_uri(str(f)))


def test_local_path_reads_regular_file(tmp_path):
    f = tmp_path / "ok.png"
    f.write_bytes(ONE_PX_PNG)
    assert asyncio.run(image_uri.resolve_image_uri(str(f))) == ONE_PX_PNG


def test_pil_bomb_guard_is_finite():
    # process-wide backstop for decode paths that bypass check_image_header
    assert Image.MAX_IMAGE_PIXELS is not None
    assert Image.MAX_IMAGE_PIXELS == 150000000
