"""/explain/ must use the shared hardened image-fetch path.

Regression tests for the Flukebook incident of 2026-08-28. explain_router
was the only router PR #39 did not migrate: it kept an inline
`httpx.AsyncClient()` (httpx's 5s default timeout, no env knob) wrapped in a
bare `except Exception` that mapped *every* failure -- read timeout, upstream
5xx, undecodable body -- to 400. Wildbook does not retry 4xx, so a transient
upstream stall was reported to the caller as a permanent client error.

The observed signature was `POST /explain/ 400` seconds after a sibling
`POST /extract/ 504 ... 60001 ms` on a URL that had served 200 in under a
second moments earlier -- the fetch was not slow, it was starved of event-loop
time by PairX running synchronously in the handler.
"""
import json

import cv2
import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.models.miewid import MiewidModel
from app.routers import explain_router
from app.utils import image_uri


def _png_bytes():
    """A real, decodable 8x8 PNG."""
    ok, buf = cv2.imencode(".png", np.full((8, 8, 3), 127, np.uint8))
    assert ok
    return buf.tobytes()


def _miewid():
    m = MagicMock(spec=MiewidModel)
    m.model = MagicMock()
    return m


def _client(responder):
    """Test app whose shared fetch client is backed by `responder`."""
    image_uri.init_image_fetch(transport=httpx.MockTransport(responder))
    app = FastAPI()
    app.include_router(explain_router.router)
    handler = MagicMock()
    handler.get_model.side_effect = lambda mid: {"miewid-msv4.1": _miewid()}.get(mid)
    handler.list_models.return_value = {"miewid-msv4.1": {}}
    app.state.model_handler = handler
    app.state.device = "cpu"
    return TestClient(app, raise_server_exceptions=False)


BODY = {
    "image1_uris": ["https://wb.example/a.jpg"],
    "bb1": [[0, 0, 8, 8]],
    "image2_uris": ["https://wb.example/b.jpg"],
    "bb2": [[0, 0, 8, 8]],
    "model_id": "miewid-msv4.1",
}


def test_upstream_read_timeout_is_504_not_400():
    """A slow Wildbook is retryable (504), not a caller error (400).

    This is the incident: the same asset served 200 a second earlier, so
    reporting 400 told Wildbook to give up on a request that would succeed.
    """
    def responder(request):
        raise httpx.ReadTimeout("upstream stalled", request=request)

    r = _client(responder).post("/explain/", json=BODY)
    assert r.status_code == 504, r.text


def test_upstream_5xx_is_502_not_400():
    """A Wildbook 503 is an upstream failure, not a malformed request."""
    def responder(request):
        return httpx.Response(503)

    r = _client(responder).post("/explain/", json=BODY)
    assert r.status_code == 502, r.text


def test_undecodable_body_is_400_without_opencv_internals():
    """A 200 carrying an HTML error page is a clean 400, not a cv2 assertion.

    Previously `cv2.imdecode` returned None and `cvtColor` raised
    '(-215:Assertion failed) !_src.empty()', which leaked into the response.
    """
    def responder(request):
        return httpx.Response(200, content=b"<html>upstream error</html>")

    r = _client(responder).post("/explain/", json=BODY)
    assert r.status_code == 400, r.text
    # The hardened path names the URI that failed; the old inline fetch
    # echoed only the exception, so this also proves which path ran.
    assert "wb.example" in r.text, r.text
    assert "Assertion failed" not in r.text
    assert "cvtColor" not in r.text


def test_fetch_failure_does_not_consume_an_inference_slot(monkeypatch):
    """A request that dies at fetch must never hold the PairX semaphore."""
    class SpySemaphore:
        acquired = False

        async def __aenter__(self):
            SpySemaphore.acquired = True

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(explain_router, "explain_semaphore", SpySemaphore())

    def responder(request):
        raise httpx.ReadTimeout("upstream stalled", request=request)

    r = _client(responder).post("/explain/", json=BODY)
    assert r.status_code == 504, r.text
    assert SpySemaphore.acquired is False


def _counting(responder):
    """Wrap a responder so the test can assert how many fetches happened."""
    calls = []

    def wrapped(request):
        calls.append(str(request.url))
        return responder(request)

    return wrapped, calls


def test_oversized_batch_is_rejected_before_any_fetch():
    """A batch over MAX_BATCH_SIZE must 400 without touching the network.

    Every image fetched takes a slot from the process-wide admission gate
    that /extract, /predict, /classify and /pipeline also queue on, so a
    request already known to be invalid must not consume any.
    """
    responder, calls = _counting(lambda r: httpx.Response(200, content=_png_bytes()))
    n = explain_router.MAX_BATCH_SIZE + 1
    body = dict(BODY,
                image1_uris=[f"https://wb.example/a{i}.jpg" for i in range(n)],
                bb1=[[0, 0, 8, 8]] * n,
                image2_uris=[f"https://wb.example/b{i}.jpg" for i in range(n)],
                bb2=[[0, 0, 8, 8]] * n)

    r = _client(responder).post("/explain/", json=body)
    assert r.status_code == 400, r.text
    assert calls == [], f"fetched {len(calls)} images before rejecting the batch"


def test_mismatched_pair_counts_rejected_before_any_fetch():
    """Unequal image1/image2 counts are a permanent caller error."""
    responder, calls = _counting(lambda r: httpx.Response(200, content=_png_bytes()))
    body = dict(BODY,
                image1_uris=["https://wb.example/a1.jpg", "https://wb.example/a2.jpg"],
                bb1=[[0, 0, 8, 8]] * 2,
                image2_uris=[f"https://wb.example/b{i}.jpg" for i in range(3)],
                bb2=[[0, 0, 8, 8]] * 3)

    r = _client(responder).post("/explain/", json=body)
    assert r.status_code == 400, r.text
    assert calls == [], f"fetched {len(calls)} images before rejecting the pairing"


def test_non_finite_bbox_is_400_not_500():
    """NaN survives Pydantic and every `x < 0` test, then breaks int().

    It must stay a caller error. As a 500 it would be retried by Wildbook
    forever against a request that can never succeed.
    """
    responder, _ = _counting(lambda r: httpx.Response(200, content=_png_bytes()))
    body = dict(BODY, bb1=[[0, 0, float("nan"), 8]])
    # json.dumps emits a bare NaN, which json.loads (and so FastAPI) accepts;
    # the test client's own encoder refuses it, hence the raw body.
    raw = json.dumps(body)
    assert "NaN" in raw

    r = _client(responder).post(
        "/explain/", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text


def test_opencv_undecodable_body_is_400():
    """Bytes PIL accepts but OpenCV cannot decode hit the `image is None` guard.

    Distinct from the HTML case, which check_image_header() rejects earlier --
    this is the only test that reaches cv2.imdecode returning None.
    """
    def responder(request):
        return httpx.Response(200, content=_png_bytes())

    client = _client(responder)
    with patch.object(explain_router.cv2, "imdecode", return_value=None):
        r = client.post("/explain/", json=BODY)
    assert r.status_code == 400, r.text
    assert "Assertion failed" not in r.text
