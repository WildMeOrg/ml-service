# tests/test_router_fetch_isolation.py
"""A request that fails at fetch must never touch the inference semaphore,
and mapped statuses must pass through the router unchanged."""
import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.routers import pipeline_router
from app.utils import image_uri


class SpySemaphore:
    def __init__(self):
        self.acquired = False

    async def __aenter__(self):
        self.acquired = True

    async def __aexit__(self, *args):
        return False


def _client(monkeypatch, spy):
    monkeypatch.setattr(pipeline_router, "pipeline_semaphore", spy)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    handler = MagicMock()
    handler.get_model.return_value = MagicMock()
    handler.get_model_info.return_value = {"config": {}}
    handler.list_models.return_value = {}
    app.state.model_handler = handler
    return TestClient(app)


PAYLOAD = {
    "predict_model_id": "p", "classify_model_id": "c",
    "extract_model_id": "e", "image_uri": "https://wb.example/img.jpg",
}


@pytest.mark.parametrize("code", [400, 502, 503, 504])
def test_fetch_failure_status_passes_through_and_skips_inference(
        monkeypatch, code):
    spy = SpySemaphore()

    async def failing_fetch(uri):
        raise HTTPException(status_code=code, detail="mapped upstream failure")
    monkeypatch.setattr(pipeline_router, "fetch_image_for_request", failing_fetch)
    # model-type validation must pass so we reach the fetch
    monkeypatch.setattr(pipeline_router, "isinstance",
                        lambda obj, cls: True, raising=False)

    client = _client(monkeypatch, spy)
    r = client.post("/pipeline/", json=PAYLOAD)
    assert r.status_code == code
    assert spy.acquired is False
