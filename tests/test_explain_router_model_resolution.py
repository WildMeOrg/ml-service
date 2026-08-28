"""Model resolution for the /explain/ (PairX) endpoint.

Regression tests for the kaiju outage of 2026-08-27: the router validated
`model_id` against a hardcoded allowlist (["miewid-msv3", "miewid-msv4.1"])
rather than against the models actually loaded, then dereferenced
`handler.get_model(...).model` without a None check. On a deployment whose
registry held only `miewid-msv4_v3`, every request failed -- allowlisted
names 500'd because they were not loaded, and the real name was rejected 400
because it was not allowlisted.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.miewid import MiewidModel
from app.models.yolo_ultralytics import YOLOUltralyticsModel
from app.routers import explain_router


def _make_client(models: dict):
    """Build a test app whose registry contains exactly `models`."""
    app = FastAPI()
    app.include_router(explain_router.router)
    handler = MagicMock()
    handler.get_model.side_effect = lambda mid: models.get(mid)
    handler.list_models.return_value = {k: {} for k in models}
    app.state.model_handler = handler
    app.state.device = "cpu"
    return TestClient(app, raise_server_exceptions=False)


def _body(model_id, **over):
    b = {
        "image1_uris": ["https://example.invalid/a.jpg"],
        "bb1": [[0, 0, 10, 10]],
        "image2_uris": ["https://example.invalid/b.jpg"],
        "bb2": [[0, 0, 10, 10]],
        "model_id": model_id,
    }
    b.update(over)
    return b


def _miewid():
    m = MagicMock(spec=MiewidModel)
    m.model = MagicMock()
    return m


def test_unregistered_model_returns_404_not_500():
    """An unloaded model must be a 404, never an AttributeError -> 500.

    Wildbook retries 5xx, so a permanently-misconfigured model id would
    otherwise retry forever against an error that can never resolve.
    """
    client = _make_client({"miewid-msv4_v3": _miewid()})
    r = client.post("/explain/", json=_body("miewid-msv4.1"))
    assert r.status_code == 404, r.text
    assert "miewid-msv4_v3" in r.text, "404 must name the available models"


def test_registered_miewid_outside_legacy_allowlist_is_accepted():
    """The registry, not a hardcoded list, decides what pairx accepts."""
    client = _make_client({"miewid-msv4_v3": _miewid()})
    with patch.object(explain_router, "process_image") as pi, \
         patch.object(explain_router, "run_pairx") as rp:
        pi.return_value = (np.zeros((4, 4, 3), np.uint8), MagicMock())
        rp.return_value = [np.zeros((4, 4, 3), np.uint8)]
        r = client.post("/explain/", json=_body("miewid-msv4_v3"))
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 1


def test_non_miewid_model_returns_400():
    """pairx needs a MiewID backbone; a loaded detector must be rejected."""
    client = _make_client({"yolo-det": MagicMock(spec=YOLOUltralyticsModel)})
    r = client.post("/explain/", json=_body("yolo-det"))
    assert r.status_code == 400, r.text
    assert "miewid" in r.text.lower()


def test_model_is_validated_before_images_are_fetched():
    """Bad model id must not cost an image download.

    kaiju's uplink drops ~10% of packets; fetching two images before
    discovering the model does not exist wastes a scarce resource and
    holds an extraction slot.
    """
    client = _make_client({"miewid-msv4_v3": _miewid()})
    with patch.object(explain_router, "process_image") as pi:
        r = client.post("/explain/", json=_body("does-not-exist"))
    assert r.status_code == 404, r.text
    pi.assert_not_called()


def test_model_id_lookup_is_case_insensitive_with_validation():
    """Validation lowercased the id but the lookup did not -- a latent 500.

    `MiewID-MSV4_V3` passed the old allowlist check (which lowercased) and
    then missed in the registry (which did not), producing the same crash.
    """
    client = _make_client({"miewid-msv4_v3": _miewid()})
    r = client.post("/explain/", json=_body("MiewID-MSV4_V3"))
    assert r.status_code != 500, r.text


def test_batch_path_threads_resolved_instance_to_fetch_and_pairx():
    """The multi-pair batch branch must carry the instance, not the id.

    Codex review flagged that the resolver alone is not enough coverage:
    this pins that `process_image` receives the MiewidModel instance and
    that `run_pairx` receives its underlying `.model`.
    """
    miew = _miewid()
    client = _make_client({"miewid-msv4_v3": miew})
    body = _body(
        "miewid-msv4_v3",
        image1_uris=["https://example.invalid/a1.jpg", "https://example.invalid/a2.jpg"],
        bb1=[[0, 0, 10, 10], [0, 0, 10, 10]],
        image2_uris=["https://example.invalid/b1.jpg", "https://example.invalid/b2.jpg"],
        bb2=[[0, 0, 10, 10], [0, 0, 10, 10]],
    )
    with patch.object(explain_router, "process_image") as pi, \
         patch.object(explain_router, "run_pairx") as rp:
        pi.return_value = (np.zeros((4, 4, 3), np.uint8), MagicMock())
        rp.return_value = [np.zeros((4, 4, 3), np.uint8)] * 2
        r = client.post("/explain/", json=body)

    assert r.status_code == 200, r.text
    assert pi.call_count == 4, "two pairs => four image fetches"
    for call in pi.call_args_list:
        assert call.args[4] is miew, "process_image must get the instance, not the id"
    assert rp.call_args.args[4] is miew.model, "run_pairx must get the underlying .model"


def test_omitted_model_id_uses_default_and_404s_when_default_not_loaded():
    """The default `miewid-msv4.1` is absent on kaiju: 404, and no fetch.

    Documents the remaining deployment decision -- a caller that omits
    model_id gets a clear, non-retryable error naming what is loaded,
    rather than a 500 or a silent fallback.
    """
    client = _make_client({"miewid-msv4_v3": _miewid()})
    payload = _body("ignored")
    del payload["model_id"]
    with patch.object(explain_router, "process_image") as pi:
        r = client.post("/explain/", json=payload)
    assert r.status_code == 404, r.text
    assert "miewid-msv4_v3" in r.text
    pi.assert_not_called()


def _reinit_settings(monkeypatch, value=None):
    """Re-snapshot the startup config the way lifespan init would."""
    if value is None:
        monkeypatch.delenv("EXPLAIN_DEFAULT_MODEL_ID", raising=False)
    else:
        monkeypatch.setenv("EXPLAIN_DEFAULT_MODEL_ID", value)
    monkeypatch.setattr(explain_router, "_default_model_id", None)
    explain_router.init_explain_settings()


def test_omitted_model_id_uses_configured_default(monkeypatch):
    """EXPLAIN_DEFAULT_MODEL_ID lets a deployment name its own model."""
    _reinit_settings(monkeypatch, "miewid-msv4_v3")
    client = _make_client({"miewid-msv4_v3": _miewid()})
    payload = _body("ignored")
    del payload["model_id"]
    with patch.object(explain_router, "process_image") as pi, \
         patch.object(explain_router, "run_pairx") as rp:
        pi.return_value = (np.zeros((4, 4, 3), np.uint8), MagicMock())
        rp.return_value = [np.zeros((4, 4, 3), np.uint8)]
        r = client.post("/explain/", json=payload)
    assert r.status_code == 200, r.text


def test_explicit_model_id_overrides_configured_default(monkeypatch):
    """An explicitly sent id always wins over the configured fallback."""
    _reinit_settings(monkeypatch, "miewid-msv4_v3")
    client = _make_client({"miewid-msv4_v3": _miewid()})
    r = client.post("/explain/", json=_body("not-loaded"))
    assert r.status_code == 404, r.text
    assert "not-loaded" in r.text, "the explicitly sent id must be the one reported"


def test_default_is_historic_value_when_env_unset(monkeypatch):
    """Unset env keeps the shipped default, so existing deployments are unaffected."""
    _reinit_settings(monkeypatch)
    assert explain_router.default_explain_model_id() == "miewid-msv4.1"


def test_explicit_null_model_id_is_rejected_by_validation(monkeypatch):
    """JSON null must stay a 422, not become "use the deployment default".

    The field is `str` with a default_factory, not Optional[str]: making it
    nullable would turn a client bug into silently running a different model.
    """
    _reinit_settings(monkeypatch, "miewid-msv4_v3")
    client = _make_client({"miewid-msv4_v3": _miewid()})
    payload = _body("ignored")
    payload["model_id"] = None
    r = client.post("/explain/", json=payload)
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_explicitly_blank_model_id_is_400_not_silently_defaulted(monkeypatch, blank):
    """A blank id is a caller error, never a request for the default.

    Before the configurable default existed, "" failed the allowlist with a
    400. Substituting the deployment default would run inference the caller
    never asked for.
    """
    _reinit_settings(monkeypatch, "miewid-msv4_v3")
    client = _make_client({"miewid-msv4_v3": _miewid()})
    with patch.object(explain_router, "process_image") as pi:
        r = client.post("/explain/", json=_body(blank))
    assert r.status_code == 400, r.text
    assert "blank" in r.text.lower()
    pi.assert_not_called()


def test_default_is_snapshotted_at_startup_not_read_per_request(monkeypatch):
    """Changing the env after init must not change request behaviour.

    Matches load_fetch_settings() in image_uri.py: config is read once at
    lifespan init so every request in a process sees the same value.
    """
    _reinit_settings(monkeypatch, "miewid-msv4_v3")
    monkeypatch.setenv("EXPLAIN_DEFAULT_MODEL_ID", "changed-after-startup")
    assert explain_router.default_explain_model_id() == "miewid-msv4_v3"
