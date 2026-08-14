"""Shared fixtures. The fetch state (httpx client + admission semaphore)
binds to the event loop that first uses it; TestClient creates a fresh loop
per instance, so state must be reset between tests to avoid cross-loop
reuse of asyncio primitives."""
import pytest

from app.utils import image_uri


@pytest.fixture(autouse=True)
def _reset_image_fetch_state():
    image_uri.reset_fetch_state_for_tests()
    yield
    image_uri.reset_fetch_state_for_tests()
