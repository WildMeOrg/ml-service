"""Utilities for resolving image URIs to bytes."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Tuple

import httpx
from PIL import Image, UnidentifiedImageError


class ImageDecodeError(ValueError):
    """Raised when image bytes cannot be decoded into a usable image.

    Subclasses ValueError so callers that already map ValueError from image
    resolution to an HTTP 400 treat an undecodable image the same way — a
    client/input error, not a server (5xx) error. This matters because
    consumers (e.g. Wildbook) retry 5xx responses as transient, but a corrupt
    image is a permanent failure: it must be reported as a 4xx so it is marked
    terminal rather than retried indefinitely.
    """


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


def validate_decodable(image_bytes: bytes) -> None:
    """Confirm image_bytes decode into a usable image, else raise ImageDecodeError.

    A header-only check (Image.verify) is insufficient: corrupt JPEGs often have
    a valid header but a broken entropy-coded scan stream that only fails during
    a full pixel load (e.g. "broken data stream when reading image file" /
    "Unsupported marker type 0xNN"). We therefore fully load() the image.

    Catches:
        - UnidentifiedImageError: not a recognizable image at all.
        - OSError: broken/truncated scan stream surfaced during load().
        - Image.DecompressionBombError: pathologically large image rejected by
          Pillow's bomb guard. It is not an OSError, so it must be listed
          explicitly or it would escape to the routers' generic 500 handler.
          Like the others it is a permanent, non-retryable bad input.

    Raises:
        ImageDecodeError (a ValueError): if the bytes cannot be decoded.
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ImageDecodeError(f"unprocessable image: cannot decode ({e})")
