"""Utilities for resolving image URIs to bytes."""

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


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
    return base64.b64decode(encoded, validate=True)


async def resolve_image_uri(uri: str) -> bytes:
    """Resolve an image URI (URL, data URI, or local path) to raw bytes.

    Raises:
        ValueError: If the URI is invalid or the file is not found.
        httpx.HTTPStatusError: If URL fetch fails.
    """
    if is_data_uri(uri):
        return decode_data_uri(uri)
    elif uri.startswith(('http://', 'https://')):
        async with httpx.AsyncClient() as client:
            response = await client.get(uri)
            response.raise_for_status()
            return response.content
    else:
        file_path = Path(uri)
        if not file_path.exists():
            raise ValueError(f"File not found: {uri}")
        with open(file_path, "rb") as f:
            return f.read()


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
