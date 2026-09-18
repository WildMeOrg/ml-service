"""Tests for the /readyz readiness probe and its /ping alias.

/health is a liveness check: it reports "degraded" while models are still
loading but still answers 200, so routing load-balancer traffic on it admits
requests the service cannot yet serve. /readyz withholds traffic until startup
has loaded every configured model.

/ping exists because RunPod's load-balancer edge health-probes GET /ping
unconditionally and ignores the documented HEALTH_CHECK_PATH setting. Without
the alias every worker reports healthy while every request fails with
400 "timed out waiting for worker".

app.main parses sys.argv at import time, so argv is neutralised before import.
"""
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.main"])
    from app import main
    # No lifespan: TestClient used without its context manager skips startup,
    # which is what lets these tests exercise the not-yet-ready state.
    yield TestClient(main.app, raise_server_exceptions=False), main


class _Handler:
    def __init__(self, models):
        self.models = models


def test_readyz_503_before_models_load(client):
    http, main = client
    if hasattr(main.app.state, "model_handler"):
        delattr(main.app.state, "model_handler")

    r = http.get("/readyz")

    assert r.status_code == 503, "readiness must fail while startup is still loading models"
    assert r.json()["detail"]["ready"] is False


def test_readyz_503_when_handler_has_no_models(client):
    """A handler present but empty means startup failed partway; not ready."""
    http, main = client
    main.app.state.model_handler = _Handler({})

    assert http.get("/readyz").status_code == 503


def test_readyz_200_once_models_loaded(client):
    http, main = client
    main.app.state.model_handler = _Handler({"msv3": object(), "miewid-msv4.1": object()})

    r = http.get("/readyz")

    assert r.status_code == 200
    assert r.json() == {"ready": True, "models_loaded": 2}


def test_ping_is_registered(client):
    """RunPod's edge probes /ping and nothing else; the route must exist."""
    http, main = client

    assert "/ping" in [r.path for r in main.app.routes], \
        "removing /ping makes every RunPod request 400 'timed out waiting for worker'"


def test_ping_mirrors_readyz(client):
    http, main = client
    main.app.state.model_handler = _Handler({"msv3": object()})
    assert http.get("/ping").status_code == http.get("/readyz").status_code == 200

    main.app.state.model_handler = _Handler({})
    assert http.get("/ping").status_code == http.get("/readyz").status_code == 503


def test_health_stays_available_while_not_ready(client):
    """Liveness must not start failing just because readiness does."""
    http, main = client
    main.app.state.model_handler = _Handler({})

    assert http.get("/readyz").status_code == 503
    assert http.get("/health").status_code == 200, \
        "/health is the liveness probe; autoheal restarts the container when it fails"
