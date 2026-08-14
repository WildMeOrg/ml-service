"""Utilities for resolving image URIs to bytes."""

import asyncio
import base64
import io
import logging
import os
import stat as stat_module
from pathlib import Path
from typing import Optional, Tuple

import httpx
from fastapi.concurrency import run_in_threadpool
from PIL import Image

# PIL's own decompression-bomb guard is replaced by the explicit
# IMAGE_MAX_PIXELS check in check_image_header(), which runs before any
# decode. Bytes never reach a decoder without passing that check.
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


class ImageTooLargeError(ValueError):
    """Image exceeds IMAGE_FETCH_MAX_BYTES or IMAGE_MAX_PIXELS."""


class EmptyImageError(ValueError):
    """Upstream returned a successful but empty body."""


def is_data_uri(uri: str) -> bool:
    return uri.startswith('data:')


def sanitize_uri_for_logging(uri: str) -> str:
    """Return a safe-to-log version of an image URI (truncates data URIs)."""
    if is_data_uri(uri):
        return uri[:40] + '...[truncated]'
    return uri


def sanitize_uri_for_response(uri: str) -> str:
    """Return a safe-to-return version of an image URI (strips data URI payload)."""
    if is_data_uri(uri):
        # Return just the MIME header, not the megabytes of base64
        comma = uri.find(',')
        if comma > 0:
            return uri[:comma] + ',[base64 data omitted]'
        return 'data:[base64 data omitted]'
    return uri


def decode_data_uri(uri: str) -> bytes:
    """Decode a data URI to raw bytes. Raises ValueError on invalid input."""
    if ',' not in uri:
        raise ValueError("Data URI missing comma separator")
    header, encoded = uri.split(',', 1)
    if ';base64' not in header:
        raise ValueError("Only base64-encoded data URIs are supported")
    _ensure_initialized()
    # 4/3 base64 expansion + header slack: reject before materializing
    if len(encoded) > _settings["max_image_bytes"] * 4 // 3 + 8:
        raise ImageTooLargeError("Encoded data URI exceeds size cap")
    return base64.b64decode(encoded, validate=True)


def check_image_header(data: bytes) -> None:
    """Reject bytes whose image header is unparseable or declares more
    pixels than IMAGE_MAX_PIXELS. Parses the header only — no pixel decode."""
    _ensure_initialized()
    max_pixels = _settings["max_pixels"]
    try:
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
    except Exception as e:
        raise ValueError(f"Unrecognized image format: {e}")
    if width * height > max_pixels:
        raise ImageTooLargeError(
            f"Image dimensions {width}x{height} exceed {max_pixels} pixel cap")


async def _fetch_url_bytes(uri: str) -> bytes:
    """Stream a URL through the shared client, enforcing the byte cap."""
    max_bytes = _settings["max_image_bytes"]
    async with _client.stream("GET", uri) as response:
        response.raise_for_status()
        declared = response.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > max_bytes:
            raise ImageTooLargeError(
                f"Declared content-length {declared} exceeds cap {max_bytes}")
        chunks = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > max_bytes:
                raise ImageTooLargeError(
                    f"Body exceeded cap {max_bytes} bytes mid-stream")
            chunks.append(chunk)
        if size == 0:
            raise EmptyImageError("Upstream returned an empty body")
        return b"".join(chunks)


async def resolve_image_uri(uri: str) -> bytes:
    """Resolve an image URI (URL, data URI, or local path) to raw bytes.

    Raises:
        ValueError: If the URI is invalid or the file is not found.
        httpx.HTTPStatusError: If URL fetch fails.
    """
    if is_data_uri(uri):
        return decode_data_uri(uri)
    elif uri.startswith(('http://', 'https://')):
        _ensure_initialized()
        return await asyncio.wait_for(
            _fetch_url_bytes(uri), timeout=_settings["total_deadline_s"])
    else:
        _ensure_initialized()
        file_path = Path(uri)
        try:
            st = os.stat(file_path)
        except OSError:
            raise ValueError(f"File not found: {uri}")
        if not stat_module.S_ISREG(st.st_mode):
            raise ValueError(f"Not a regular file: {uri}")
        if st.st_size > _settings["max_image_bytes"]:
            raise ImageTooLargeError(
                f"File size {st.st_size} exceeds cap {_settings['max_image_bytes']}")
        return await run_in_threadpool(file_path.read_bytes)


# --- Shared fetch state -----------------------------------------------------
# One httpx client + one admission semaphore per worker process. Created
# lazily (first request) or eagerly (app startup); asyncio primitives bind
# to the running event loop, so tests reset between cases (see conftest).

_client: Optional["httpx.AsyncClient"] = None
_admission: Optional[asyncio.Semaphore] = None
_settings: Optional[dict] = None


def load_fetch_settings() -> dict:
    return {
        "connect_timeout_s": float(os.getenv("IMAGE_FETCH_CONNECT_TIMEOUT_S", "10")),
        "read_timeout_s": float(os.getenv("IMAGE_FETCH_READ_TIMEOUT_S", "30")),
        "total_deadline_s": float(os.getenv("IMAGE_FETCH_TOTAL_DEADLINE_S", "60")),
        "admission_limit": int(os.getenv("IMAGE_ADMISSION_LIMIT", "8")),
        "admission_wait_s": float(os.getenv("IMAGE_ADMISSION_WAIT_S", "20")),
        "max_image_bytes": int(os.getenv("IMAGE_FETCH_MAX_BYTES", "52428800")),
        "max_pixels": int(os.getenv("IMAGE_MAX_PIXELS", "150000000")),
    }


async def _log_redirect_hop(response: httpx.Response) -> None:
    if response.is_redirect:
        logger.info(
            "Image fetch redirect: %s -> %s",
            response.request.url.host,
            response.headers.get("location", "?"),
        )


def init_image_fetch(transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
    """Create the shared client + admission semaphore. Idempotent."""
    global _client, _admission, _settings
    if _client is not None:
        return
    _settings = load_fetch_settings()
    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=_settings["connect_timeout_s"],
            read=_settings["read_timeout_s"],
            write=10.0,
            pool=None,  # admission semaphore keeps requests <= max_connections
        ),
        limits=httpx.Limits(
            max_connections=_settings["admission_limit"],
            max_keepalive_connections=4,
        ),
        follow_redirects=True,
        max_redirects=3,
        event_hooks={"response": [_log_redirect_hop]},
        transport=transport,
    )
    _admission = asyncio.Semaphore(_settings["admission_limit"])


async def shutdown_image_fetch() -> None:
    global _client, _admission, _settings
    if _client is not None:
        await _client.aclose()
    _client = None
    _admission = None
    _settings = None


def reset_fetch_state_for_tests() -> None:
    """Synchronous reset for test isolation (drops, doesn't close, the client)."""
    global _client, _admission, _settings
    _client = None
    _admission = None
    _settings = None


def _ensure_initialized() -> None:
    if _client is None:
        init_image_fetch()
