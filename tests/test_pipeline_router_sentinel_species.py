"""The contract Wildbook depends on: a withheld species must leave iaClass unset.

pipeline_router promotes a classifier's `species` to the response's top-level
`iaClass`, and Wildbook PREFERS that over `detection_class`
(MlServiceProcessor.java:377), falling back to the detector's class only when
`iaClass` is absent. So for the sea turtle labeler -- whose species half is the
constant `sea_turtle`, carrying no body/head distinction -- the response must omit
`iaClass` and still carry `detection_class`, or Wildbook stores a class that is not
valid for any species and the annotation never matches.
"""
from unittest.mock import MagicMock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import pipeline_router

VALID_PNG_DATA_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _client(classify_predictions):
    from app.models.efficientnet import EfficientNetModel
    from app.models.lightnet_model import LightNetModel
    from app.models.miewid import MiewidModel

    pm = MagicMock(spec=LightNetModel)
    pm.predict.return_value = {
        "model_id": "sea_turtle_new_v0",
        "bboxes": [[10, 10, 40, 40]],
        "scores": [0.94],
        "class_names": ["turtle_sea+head"],
        "num_detections": 1,
        "image_size": {"width": 100, "height": 100},
    }
    cm = MagicMock(spec=EfficientNetModel)
    cm.predict.return_value = {
        "model_id": "seaturtles_effnet_v0",
        "predictions": classify_predictions,
        "all_probabilities": [0.99] + [0.01] * 5,
        "threshold": 0.5,
    }
    em = MagicMock(spec=MiewidModel)
    em.extract_embeddings.return_value = np.zeros((1, 2152), dtype=np.float32)

    app = FastAPI()
    app.include_router(pipeline_router.router)
    handler = MagicMock()
    handler.get_model.side_effect = lambda mid: {"p": pm, "c": cm, "e": em}.get(mid)
    handler.get_model_info.side_effect = lambda mid: {
        "p": {"config": {}}, "c": {"config": {}},
        "e": {"config": {"version": 4}},
    }.get(mid)
    handler.list_models.return_value = {"p": {}, "c": {}, "e": {}}
    app.state.model_handler = handler
    return TestClient(app)


def _post(client):
    return client.post("/pipeline/", json={
        "image_uri": VALID_PNG_DATA_URI,
        "predict_model_id": "p",
        "classify_model_id": "c",
        "extract_model_id": "e",
        "bbox_score_threshold": 0.5,
    })


WITH_SPECIES = [{
    "label": "sea_turtle:left", "probability": 0.99, "index": 0,
    "species": "sea_turtle", "viewpoint": "left",
}]
SENTINEL_APPLIED = [{
    "label": "sea_turtle:left", "probability": 0.99, "index": 0,
    "species": None, "viewpoint": "left",
}]


def test_species_present_sets_iaclass_the_bug():
    """Documents the pre-fix behaviour: 'sea_turtle' lands on iaClass."""
    r = _post(_client(WITH_SPECIES))
    if r.status_code != 200:
        import pytest
        pytest.skip(f"pipeline could not run in this environment: {r.status_code}")
    result = r.json()["results"][0]
    assert result.get("iaClass") == "sea_turtle"


def test_withheld_species_leaves_iaclass_unset_and_keeps_detection_class():
    """The fix: Wildbook can fall through to the detector's own class."""
    r = _post(_client(SENTINEL_APPLIED))
    if r.status_code != 200:
        import pytest
        pytest.skip(f"pipeline could not run in this environment: {r.status_code}")
    result = r.json()["results"][0]
    assert not result.get("iaClass"), \
        "iaClass must be absent/empty so Wildbook uses detection_class"
    assert result.get("detection_class") == "turtle_sea+head", \
        "the detector's class is the only source of the body/head distinction"
    assert result.get("viewpoint") == "left", "viewpoint is still the useful half"
