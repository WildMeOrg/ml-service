"""Utilities for resolving image URIs to bytes."""

import asyncio
import base64
import io
import logging
import os
import re
import stat as stat_module
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import NoReturn, Optional, Tuple

import httpx
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image

# check_image_header() enforces IMAGE_MAX_PIXELS with a friendly 400 before
# any decode on the fetch_image_for_request path. Setting PIL's own guard to
# the same cap (instead of disabling it) keeps a hard backstop on decode
# paths that do not go through that helper (e.g. the wbia-compat router).
Image.MAX_IMAGE_PIXELS = int(os.getenv("IMAGE_MAX_PIXELS", "150000000"))

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
        with warnings.catch_warnings():
            # PIL's own guard (set to the same cap, see module top) only
            # warns at 1x and raises at 2x; suppress the warning here so our
            # ImageTooLargeError stays the single user-visible error for
            # both the 1x-2x and >2x cases.
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                width, height = im.size
    except Image.DecompressionBombError as e:
        raise ImageTooLargeError(f"Image header exceeds pixel cap: {e}")
    except Exception as e:
        raise ValueError(f"Unrecognized image format: {e}")
    if width * height > max_pixels:
        raise ImageTooLargeError(
            f"Image dimensions {width}x{height} exceed {max_pixels} pixel cap")


def encode_bare_fragment(uri: str) -> str:
    """Percent-encode a bare '#' inside a URL's PATH so it stays in the path.

    '#' opens a fragment, which is client-side only and never sent to the
    server, so a filename containing an unencoded '#' truncates the request
    silently: Flukebook's `.../3c8f4264-.../2000_174419#R8123A.jpg` was
    requested as `.../2000_174419` and 404'd (2026-08-28). Wildbook builds
    these URLs without encoding the filename.

    Only the path is rewritten -- never the authority, and never after a real
    query begins:

      * `https://h/dir/2000_174419#R8123A.jpg` -> path `/dir/2000_174419%23R8123A.jpg`
      * `https://h/a#b.jpg?sig=x`              -> path `/a%23b.jpg`, query `sig=x`
      * `https://h/img.jpg?sig=x#frag`         -> unchanged; a '#' after the
        query is a genuine fragment and stays one
      * `https://evil#@internal.example/a.jpg` -> unchanged. Here '#' ends the
        authority, so there is no path to fix. A blanket replace would turn
        this into userinfo `evil%23` on host `internal.example` -- sending a
        caller-supplied fetch to a DIFFERENT HOST than the URL names.

    Idempotent: an already-encoded '%23' contains no '#'.

    Trade-off: on a path-bearing URL a genuine trailing fragment is now read
    as part of the filename and will 404. A fragment is meaningless for a
    binary image fetch and Wildbook has no path that appends one, so this
    reading is right for the producers we serve. No non-heuristic rule can
    separate the two intents; the real fix belongs where the URL is built.

    '?' has the same class of problem and is deliberately NOT encoded: query
    strings are legitimate and common (signed URLs), so blanket-encoding
    would break them.

    Contract: absolute http(s) URLs only. A scheme-relative input ('//h/a#b')
    is returned unchanged rather than guessed at -- both call sites gate on
    an 'http://' or 'https://' prefix, so one cannot reach here.
    """
    if '#' not in uri:
        return uri
    scheme_sep = uri.find('://')
    if scheme_sep == -1:
        return uri
    authority_start = scheme_sep + 3
    delimiter = re.search(r'[/?#]', uri[authority_start:])
    if delimiter is None or delimiter.group() != '/':
        # No path component: the '#' terminates the authority (or none is
        # present). Nothing here is a filename, so leave it entirely alone.
        return uri
    path_start = authority_start + delimiter.start()
    head, rest = uri[:path_start], uri[path_start:]
    query_at = rest.find('?')
    if query_at == -1:
        fixed = rest.replace('#', '%23')
    else:
        fixed = rest[:query_at].replace('#', '%23') + rest[query_at:]
    if fixed == rest:
        return uri
    encoded = head + fixed
    logger.info("Encoded bare '#' in image URI path: %s",
                sanitize_uri_for_logging(encoded))
    return encoded


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
            _fetch_url_bytes(encode_bare_fragment(uri)),
            timeout=_settings["total_deadline_s"])
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
    # pool=None on the client is safe only while the admission semaphore
    # bounds concurrent requests to the pool size
    assert _settings["admission_limit"] >= 1
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


@asynccontextmanager
async def admission_slot():
    """Bounded admission: one slot per in-flight image request, held from
    fetch start through inference end. 503 (retryable) if the wait exceeds
    IMAGE_ADMISSION_WAIT_S."""
    _ensure_initialized()
    try:
        await asyncio.wait_for(
            _admission.acquire(), timeout=_settings["admission_wait_s"])
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Service saturated: no admission slot available")
    try:
        yield
    finally:
        _admission.release()


def _fail(status_code: int, uri: str, exc: Exception, started: float) -> NoReturn:
    logger.warning(
        "Image fetch failed (%d, %s, %.0f ms): %s",
        status_code, type(exc).__name__, (time.monotonic() - started) * 1000,
        sanitize_uri_for_logging(uri))
    raise HTTPException(
        status_code=status_code,
        detail=f"Image fetch failed for {sanitize_uri_for_response(uri)}: "
               f"{type(exc).__name__}")


async def fetch_image_for_request(uri: str) -> bytes:
    """Router entry point: resolve + header-check an image URI, mapping
    every failure to the retry-ladder-correct HTTPException."""
    started = time.monotonic()
    try:
        data = await resolve_image_uri(uri)
        check_image_header(data)
        logger.debug(
            "Image fetched: %d bytes in %.0f ms from %s",
            len(data), (time.monotonic() - started) * 1000,
            sanitize_uri_for_logging(uri))
        return data
    except ValueError as e:               # incl. ImageTooLarge/Empty/bad header
        _fail(400, uri, e, started)
    except asyncio.TimeoutError as e:     # total deadline
        _fail(504, uri, e, started)
    except httpx.TimeoutException as e:   # connect/read/write/pool timeout
        _fail(504, uri, e, started)
    except httpx.TooManyRedirects as e:
        _fail(400, uri, e, started)
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code in (408, 429) or code >= 500:
            _fail(502, uri, e, started)
        else:
            _fail(400, uri, e, started)
    except httpx.UnsupportedProtocol as e:
        _fail(400, uri, e, started)
    except httpx.InvalidURL as e:
        _fail(400, uri, e, started)
    except httpx.RequestError as e:       # DNS, refused, reset, TLS, decoding
        _fail(502, uri, e, started)
