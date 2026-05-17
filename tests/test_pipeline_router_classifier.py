"""Layer-3 integration tests for the new classify-slot model type."""
from unittest.mock import MagicMock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routers import pipeline_router


def _make_app_with_models(predict_model, classify_model, extract_model):
    """Build a minimal FastAPI app with monkey-patched ModelHandler."""
    app = FastAPI()
    app.include_router(pipeline_router.router)

    handler = MagicMock()
    handler.get_model.side_effect = lambda mid: {
        "p": predict_model, "c": classify_model, "e": extract_model
    }.get(mid)
    handler.get_model_info.side_effect = lambda mid: {
        "p": {"config": {}}, "c": {"config": {}},
        "e": {"config": {"version": 4.1}},
    }.get(mid)
    handler.list_models.return_value = {"p": {}, "c": {}, "e": {}}
    app.state.model_handler = handler
    return TestClient(app)


def test_pipeline_classify_densenet_classifier_emits_top_level_iaclass_and_viewpoint():
    from app.models.densenet_classifier import DenseNetClassifierModel
    from app.models.miewid import MiewidModel
    from app.models.yolo_ultralytics import YOLOUltralyticsModel
    import numpy as np

    pm = MagicMock(spec=YOLOUltralyticsModel)
    pm.predict.return_value = {
        "predictions": [{
            "bbox": [0, 0, 10, 10],
            "theta": 0.0,
            "score": 0.9,
            "class": "detection",
            "class_id": 0
        }]
    }
    cm = MagicMock(spec=DenseNetClassifierModel)
    cm.predict.return_value = {
        "class": "salamander_fire_adult:up",
        "probability": 0.95,
        "class_id": 0,
        "predictions": [{
            "label": "salamander_fire_adult:up", "probability": 0.95,
            "index": 0,
            "species": "salamander_fire_adult", "viewpoint": "up",
        }, {
            "label": "salamander_fire_juvenile:left", "probability": 0.03,
            "index": 1,
            "species": "salamander_fire_juvenile", "viewpoint": "left",
        }],
    }
    em = MagicMock(spec=MiewidModel)
    em.extract_embeddings.return_value = np.array([[0.1] * 2152])

    client = _make_app_with_models(pm, cm, em)
    resp = client.post("/pipeline/", json={
        "predict_model_id": "p",
        "classify_model_id": "c",
        "extract_model_id": "e",
        "image_uri": "data:image/png;base64,iVBORw0KGgo=",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert len(body["results"]) == 1
    r = body["results"][0]
    assert r["iaClass"] == "salamander_fire_adult"
    assert r["viewpoint"] == "up"
    # raw classification kept too
    assert r["classification"]["class"] == "salamander_fire_adult:up"


def test_pipeline_classify_densenet_classifier_pure_viewpoint_omits_iaclass():
    from app.models.densenet_classifier import DenseNetClassifierModel
    from app.models.miewid import MiewidModel
    from app.models.yolo_ultralytics import YOLOUltralyticsModel
    import numpy as np

    pm = MagicMock(spec=YOLOUltralyticsModel)
    pm.predict.return_value = {
        "predictions": [{"bbox": [0, 0, 10, 10], "theta": 0.0,
                         "score": 0.9, "class": "detection", "class_id": 0}]
    }
    cm = MagicMock(spec=DenseNetClassifierModel)
    cm.predict.return_value = {
        "class": "up", "probability": 0.8, "class_id": 0,
        "predictions": [
            {"label": "up", "probability": 0.8, "index": 0,
             "species": None, "viewpoint": "up"},
        ],
    }
    em = MagicMock(spec=MiewidModel)
    em.extract_embeddings.return_value = np.array([[0.1] * 2152])

    client = _make_app_with_models(pm, cm, em)
    resp = client.post("/pipeline/", json={
        "predict_model_id": "p", "classify_model_id": "c",
        "extract_model_id": "e",
        "image_uri": "data:image/png;base64,iVBORw0KGgo=",
    })
    body = resp.json()
    r = body["results"][0]
    assert r["viewpoint"] == "up"
    assert "iaClass" not in r


def test_pipeline_classify_densenet_orientation_rejected_with_400():
    from app.models.densenet_orientation import DenseNetOrientationModel
    from app.models.miewid import MiewidModel
    from app.models.yolo_ultralytics import YOLOUltralyticsModel

    pm = MagicMock(spec=YOLOUltralyticsModel)
    cm = MagicMock(spec=DenseNetOrientationModel)
    em = MagicMock(spec=MiewidModel)

    client = _make_app_with_models(pm, cm, em)
    resp = client.post("/pipeline/", json={
        "predict_model_id": "p", "classify_model_id": "c",
        "extract_model_id": "e", "image_uri": "data:image/png;base64,iVBORw0KGgo=",
    })
    assert resp.status_code == 400
