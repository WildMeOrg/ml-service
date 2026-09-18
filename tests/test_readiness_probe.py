"""Tests for the /readyz readiness probe and its /ping alias.

/health is the liveness check. When DEVICE is exactly "cuda" it shells out to
nvidia-smi on every call, and a handler holding an empty registry still gets
status "healthy" with models_loaded 0 -- so it cannot gate routing. /readyz
fails closed instead: 200 only for a published, non-empty handler. /ping aliases it because RunPod's load-balancer edge
health-probes GET /ping unconditionally and ignores HEALTH_CHECK_PATH -- without
the alias the edge gets a 404, every worker reports healthy, and every request
fails with 400 "timed out waiting for worker".

app.main parses sys.argv at import time, so argv is neutralised before import.
TestClient is used without its context manager, which by design does not run the
lifespan -- that is what lets these tests drive the not-yet-ready state directly.
app.state is module-global, so each test restores whatever it found.
"""
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.main"])
    from app import main

    sentinel = object()
    previous = getattr(main.app.state, "model_handler", sentinel)
    previous_device = getattr(main.app.state, "device", sentinel)
    # /health runs nvidia-smi when device is cuda; keep these tests off the GPU.
    main.app.state.device = "cpu"
    try:
        yield TestClient(main.app, raise_server_exceptions=False), main
    finally:
        for name, value in (("model_handler", previous), ("device", previous_device)):
            if value is sentinel:
                try:
                    delattr(main.app.state, name)
                except (AttributeError, KeyError):
                    pass
            else:
                setattr(main.app.state, name, value)


class _Handler:
    def __init__(self, models):
        self.models = models


def _clear_handler(main):
    """Starlette's State.__delattr__ raises KeyError when the key is absent."""
    try:
        delattr(main.app.state, "model_handler")
    except (AttributeError, KeyError):
        pass


def test_readyz_503_when_handler_absent(client):
    """Startup publishes app.state.model_handler only after every model loads."""
    http, main = client
    _clear_handler(main)

    r = http.get("/readyz")

    assert r.status_code == 503
    assert r.json()["detail"]["ready"] is False


def test_readyz_503_when_registry_is_empty(client):
    """A worker that came up with no models must not be routed traffic."""
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
    assert http.get("/ping").json() == http.get("/readyz").json()

    main.app.state.model_handler = _Handler({})
    assert http.get("/ping").status_code == http.get("/readyz").status_code == 503


def test_health_is_not_a_substitute_for_readyz(client):
    """The reason a load balancer must not probe /health: it admits a model-less worker."""
    http, main = client
    main.app.state.model_handler = _Handler({})

    assert http.get("/readyz").status_code == 503
    health = http.get("/health")
    assert health.status_code == 200, \
        "/health is the liveness probe; autoheal restarts the container when it fails"
    assert health.json()["checks"]["models_loaded"] == 0, \
        "/health reports zero models and still answers 200 -- hence the separate probe"


def test_startup_publishes_a_handler_that_satisfies_readyz(client, monkeypatch):
    """Tie the probe to the real startup path rather than only to synthetic state."""
    http, main = client
    _clear_handler(main)

    loaded = {}

    class FakeHandler:
        def __init__(self):
            self.models = loaded

        def load_model(self, model_id, model_type, device, **params):
            loaded[model_id] = model_type

    monkeypatch.setattr(main, "ModelHandler", FakeHandler)
    monkeypatch.setattr(main.image_uri, "init_image_fetch", lambda: None)
    monkeypatch.setattr(main.explain_router, "init_explain_settings", lambda: None)

    assert http.get("/readyz").status_code == 503

    import asyncio
    asyncio.run(main.startup_event())

    r = http.get("/readyz")
    assert r.status_code == 200, "readiness must flip once startup completes"
    assert r.json()["models_loaded"] == len(loaded) > 0


def test_startup_failure_leaves_readyz_failing(client, monkeypatch):
    """If a model cannot load, startup re-raises and nothing is published."""
    http, main = client
    _clear_handler(main)

    class ExplodingHandler:
        def __init__(self):
            self.models = {}

        def load_model(self, *a, **k):
            raise RuntimeError("weights missing")

    monkeypatch.setattr(main, "ModelHandler", ExplodingHandler)
    monkeypatch.setattr(main.image_uri, "init_image_fetch", lambda: None)
    monkeypatch.setattr(main.explain_router, "init_explain_settings", lambda: None)

    import asyncio
    with pytest.raises(RuntimeError):
        asyncio.run(main.startup_event())

    assert http.get("/readyz").status_code == 503, \
        "a failed startup must never leave the worker advertising readiness"
