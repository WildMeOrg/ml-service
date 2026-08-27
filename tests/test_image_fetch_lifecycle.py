"""Lifecycle and settings tests for the shared image-fetch state."""
import asyncio
import httpx
import pytest

from app.utils import image_uri


def test_load_fetch_settings_defaults():
    s = image_uri.load_fetch_settings()
    assert s["connect_timeout_s"] == 10.0
    assert s["read_timeout_s"] == 30.0
    assert s["total_deadline_s"] == 60.0
    assert s["admission_limit"] == 8
    assert s["admission_wait_s"] == 20.0
    assert s["max_image_bytes"] == 52428800
    assert s["max_pixels"] == 150000000


def test_load_fetch_settings_env_override(monkeypatch):
    monkeypatch.setenv("IMAGE_FETCH_READ_TIMEOUT_S", "5")
    monkeypatch.setenv("IMAGE_ADMISSION_LIMIT", "2")
    s = image_uri.load_fetch_settings()
    assert s["read_timeout_s"] == 5.0
    assert s["admission_limit"] == 2


def test_init_creates_client_and_admission():
    image_uri.init_image_fetch()
    assert isinstance(image_uri._client, httpx.AsyncClient)
    assert isinstance(image_uri._admission, asyncio.Semaphore)
    # invariant asserted at init: pool max_connections == admission_limit
    assert image_uri._settings["admission_limit"] == 8


def test_init_accepts_mock_transport():
    transport = httpx.MockTransport(lambda req: httpx.Response(200))
    image_uri.init_image_fetch(transport=transport)
    assert image_uri._client is not None


def test_reset_clears_state():
    image_uri.init_image_fetch()
    image_uri.reset_fetch_state_for_tests()
    assert image_uri._client is None
    assert image_uri._admission is None
