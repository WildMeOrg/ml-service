"""/pipeline/ must accept a config with no classifier.

Single-species deployments (e.g. Bombina variegata: a dedicated lightnet
detector + MiewID, no labeler) legitimately have nothing for a classifier
to decide. Wildbook's buildPipelinePayload omits `classify_model_id`
whenever the IA.json `_mlservice_conf` entry lacks it, so requiring the
field rejects a valid caller with a 422 before any work is done.
"""
from unittest.mock import MagicMock
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routers import pipeline_router

VALID_PNG_DATA_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _make_client(predict_model, extract_model, orientation_model=None):
    """App whose handler knows a detector and an extractor but NO classifier."""
    app = FastAPI()
    app.include_router(pipeline_router.router)

    models = {"p": predict_model, "e": extract_model}
    infos = {"p": {"config": {}}, "e": {"config": {"version": 4.1}}}
    if orientation_model is not None:
        models["o"] = orientation_model
        infos["o"] = {"config": {}}

    handler = MagicMock()
    handler.get_model.side_effect = models.get
    handler.get_model_info.side_effect = infos.get
    handler.list_models.return_value = {k: {} for k in models}
    app.state.model_handler = handler
    return TestClient(app)


def _detector():
    from app.models.lightnet_model import LightNetModel
    pm = MagicMock(spec=LightNetModel)
    # LightNet's own output shape: no class_ids, no thetas.
    pm.predict.return_value = {
        "bboxes": [[10, 20, 100, 120]],
        "scores": [0.91],
        "class_names": ["yellow_bellied_toad"],
        "num_detections": 1,
    }
    return pm


def _extractor():
    from app.models.miewid import MiewidModel
    em = MagicMock(spec=MiewidModel)
    em.extract_embeddings.return_value = np.array([[0.1] * 2152])
    return em


def test_pipeline_without_classify_model_id_succeeds():
    """Omitting classify_model_id runs detect+extract instead of 422ing."""
    client = _make_client(_detector(), _extractor())

    resp = client.post("/pipeline/", json={
        "predict_model_id": "p",
        "extract_model_id": "e",
        "image_uri": VALID_PNG_DATA_URI,
    })

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert len(body["results"]) == 1
    r = body["results"][0]
    # No classifier ran, so nothing classifier-derived is emitted...
    assert r["classification"] is None
    assert "iaClass" not in r
    assert "viewpoint" not in r
    # ...but the detector's own class survives: it is what Wildbook's
    # MlServiceProcessor falls back to when resolving iaClass.
    assert r["detection_class"] == "yellow_bellied_toad"
    assert r["embedding"] == [0.1] * 2152


def test_pipeline_without_classifier_does_not_load_a_classify_model():
    """The classify slot must not be resolved when no id was supplied.

    Guards against a None id reaching handler.get_model() and matching some
    fallback entry.
    """
    client = _make_client(_detector(), _extractor())

    resp = client.post("/pipeline/", json={
        "predict_model_id": "p",
        "extract_model_id": "e",
        "image_uri": VALID_PNG_DATA_URI,
    })

    assert resp.status_code == 200, resp.text
    handler = client.app.state.model_handler
    requested = [c.args[0] for c in handler.get_model.call_args_list]
    assert None not in requested
    assert set(requested) <= {"p", "e"}


def test_pipeline_without_classifier_still_reports_orientation():
    """Orientation output must not be mis-read once classify leaves the gather.

    Results are unpacked positionally; dropping the classify task shifts
    every index, so a fixed results[2] would read past the end (or hand
    the extractor's embedding to the orientation slot).
    """
    from app.models.densenet_orientation import DenseNetOrientationModel

    om = MagicMock(spec=DenseNetOrientationModel)
    om.predict.return_value = {
        "predictions": [{"label": "up", "probability": 0.77, "index": 0}],
    }
    client = _make_client(_detector(), _extractor(), orientation_model=om)

    resp = client.post("/pipeline/", json={
        "predict_model_id": "p",
        "extract_model_id": "e",
        "orientation_model_id": "o",
        "image_uri": VALID_PNG_DATA_URI,
    })

    assert resp.status_code == 200, resp.text
    r = resp.json()["results"][0]
    assert r["orientation"] == {
        "label": "up", "probability": 0.77, "class_id": 0,
    }
    assert r["classification"] is None
    assert r["embedding"] == [0.1] * 2152


def test_pipeline_explicit_null_classify_model_id_is_accepted():
    """A caller that emits null rather than omitting the key is equivalent."""
    client = _make_client(_detector(), _extractor())

    resp = client.post("/pipeline/", json={
        "predict_model_id": "p",
        "classify_model_id": None,
        "extract_model_id": "e",
        "image_uri": VALID_PNG_DATA_URI,
    })

    assert resp.status_code == 200, resp.text
    assert resp.json()["results"][0]["classification"] is None


def test_pipeline_unknown_classify_model_id_still_404s():
    """Optional must not become 'silently ignored': a named-but-missing
    classifier is still a caller error."""
    client = _make_client(_detector(), _extractor())

    resp = client.post("/pipeline/", json={
        "predict_model_id": "p",
        "classify_model_id": "nope",
        "extract_model_id": "e",
        "image_uri": VALID_PNG_DATA_URI,
    })

    assert resp.status_code == 404, resp.text
