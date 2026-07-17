"""Integration tests for the wbia-orientation theta path through /pipeline/.

Whale sharks lost rotation in June 2026 because theta came only from the
detector, and whaleshark_v0 is a lightnet model that never emits it — so theta
defaulted to 0.0 and MiewID embedded tilted animals. These tests pin the wiring
that repairs it, and specifically the properties whose absence would recreate the
bug in a new place:

  - theta comes from the regressor, NOT the detector's 0.0
  - orientation runs BEFORE classify/extract (theta must precede the crop)
  - a failure is REQUEST-level: 500, no results, consumers never invoked
  - effective_bbox reaches classify, extract, AND the emitted (persisted) bbox
"""
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import pipeline_router


def _client(predict_model, classify_model, extract_model, orientation_model=None):
    app = FastAPI()
    app.include_router(pipeline_router.router)
    handler = MagicMock()
    handler.get_model.side_effect = lambda mid: {
        "p": predict_model, "c": classify_model, "e": extract_model,
        "o": orientation_model,
    }.get(mid)
    handler.get_model_info.side_effect = lambda mid: {
        "p": {"config": {}}, "c": {"config": {}},
        "e": {"config": {"version": 4.1}}, "o": {"config": {}},
    }.get(mid)
    handler.list_models.return_value = {"p": {}, "c": {}, "e": {}, "o": {}}
    app.state.model_handler = handler
    return TestClient(app)


def _models(n_bboxes=1):
    from app.models.densenet_classifier import DenseNetClassifierModel
    from app.models.miewid import MiewidModel
    from app.models.yolo_ultralytics import YOLOUltralyticsModel

    pm = MagicMock(spec=YOLOUltralyticsModel)
    pm.predict.return_value = {"predictions": [
        {"bbox": [10 * k, 10 * k, 50, 50], "theta": 0.0, "score": 0.9,
         "class": "whaleshark", "class_id": 0}
        for k in range(1, n_bboxes + 1)
    ]}
    cm = MagicMock(spec=DenseNetClassifierModel)
    cm.predict.return_value = {"predictions": [
        {"label": "whaleshark:left", "probability": 0.9, "index": 0,
         "species": "whaleshark", "viewpoint": "left"}]}
    em = MagicMock(spec=MiewidModel)
    em.extract_embeddings.return_value = np.zeros((1, 2152))
    return pm, cm, em


def _orientation(thetas, effective_bboxes=None):
    from app.models.wbia_orientation import WbiaOrientationModel
    om = MagicMock(spec=WbiaOrientationModel)
    effs = effective_bboxes or [[10, 10, 50, 50]] * len(thetas)
    om.predict_batch.return_value = [
        {"model_id": "o", "theta": t, "coords_normalized": [0.5] * 5,
         "effective_bbox": e}
        for t, e in zip(thetas, effs)
    ]
    return om


PAYLOAD = {"image_uri": "data:image/png;base64,iVBORw0KGgo=",
           "predict_model_id": "p", "classify_model_id": "c",
           "extract_model_id": "e"}


def _payload(**kw):
    return {**PAYLOAD, **kw}


# The data URI below is decoded by resolve_image_uri but never really rendered:
# every model here is mocked, so the bytes only need to travel. Same convention as
# tests/test_pipeline_router_classifier.py.


# ------------------------------------------------------------------ happy path

def test_theta_comes_from_the_regressor_not_the_detector():
    """The whole point: lightnet reports theta=0.0, the regressor reports the
    real angle, and the regressor must win."""
    pm, cm, em = _models()
    om = _orientation([1.234])
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    assert res["theta"] == pytest.approx(1.234)      # detector said 0.0
    assert res["theta_source"] == "orientation"


def test_theta_source_says_detector_when_no_orientation_configured():
    pm, cm, em = _models()
    r = _client(pm, cm, em).post("/pipeline/", json=_payload())
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["theta_source"] == "detector"


def test_orientation_runs_before_classify_and_extract():
    """theta must be known before the crop is embedded, so orientation cannot sit
    in the parallel gather with its consumers."""
    pm, cm, em = _models()
    om = _orientation([0.5])
    order = []
    om.predict_batch.side_effect = lambda **kw: (
        order.append("orientation"),
        [{"model_id": "o", "theta": 0.5, "coords_normalized": [0.5] * 5,
          "effective_bbox": [10, 10, 50, 50]}])[1]
    cm.predict.side_effect = lambda **kw: (order.append("classify"), {"predictions": []})[1]
    em.extract_embeddings.side_effect = lambda **kw: (
        order.append("extract"), np.zeros((1, 2152)))[1]
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 200, r.text
    assert order[0] == "orientation", f"orientation must run first, got {order}"


def test_orientation_is_batched_once_for_all_bboxes():
    """One predict_batch per image (3 TTA forwards), not one call per bbox."""
    pm, cm, em = _models(n_bboxes=3)
    om = _orientation([0.1, 0.2, 0.3],
                      [[10, 10, 50, 50], [20, 20, 50, 50], [30, 30, 50, 50]])
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 200, r.text
    assert om.predict_batch.call_count == 1
    assert [x["theta"] for x in r.json()["results"]] == pytest.approx([0.1, 0.2, 0.3])


# ----------------------------------------------------------- effective_bbox

def test_effective_bbox_is_emitted_as_the_persisted_bbox():
    """Wildbook persists the result's bbox alongside theta
    (MlServiceProcessor.featureParams). Emitting the detector's original while
    theta describes effective_bbox would STORE a theta/region mismatch."""
    pm, cm, em = _models()
    om = _orientation([0.5], [[0, 0, 640, 480]])     # e.g. degenerate -> full frame
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["bbox"] == [0, 0, 640, 480]   # not [10,10,50,50]


def test_effective_bbox_is_what_classify_and_extract_receive():
    """All three consumers must see the identical region."""
    pm, cm, em = _models()
    om = _orientation([0.5], [[0, 0, 640, 480]])
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 200, r.text
    assert cm.predict.call_args.kwargs["bbox"] == [0, 0, 640, 480]
    assert list(em.extract_embeddings.call_args.kwargs["bbox"]) == [0, 0, 640, 480]


def test_orientation_theta_is_passed_to_classify_and_extract():
    pm, cm, em = _models()
    om = _orientation([1.234])
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 200, r.text
    assert cm.predict.call_args.kwargs["theta"] == pytest.approx(1.234)
    assert em.extract_embeddings.call_args.kwargs["theta"] == pytest.approx(1.234)


# --------------------------------------------------------------- fail-closed

def test_orientation_failure_fails_the_whole_request():
    """A soft-fail here would fall back to the detector's 0.0 and embed an
    unrotated crop — the exact regression being repaired."""
    from app.models.wbia_orientation import OrientationInferenceError
    pm, cm, em = _models()
    om = _orientation([0.5])
    om.predict_batch.side_effect = OrientationInferenceError("non-finite theta")
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 500
    assert "theta" in r.text.lower()


def test_orientation_failure_means_classify_and_extract_never_run():
    from app.models.wbia_orientation import OrientationInferenceError
    pm, cm, em = _models()
    om = _orientation([0.5])
    om.predict_batch.side_effect = OrientationInferenceError("boom")
    _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    cm.predict.assert_not_called()
    em.extract_embeddings.assert_not_called()


def test_one_bad_bbox_fails_the_whole_request_not_just_that_bbox():
    """Partial success would silently drop a detection; a 5xx body cannot carry
    successful siblings anyway."""
    from app.models.wbia_orientation import OrientationInferenceError
    pm, cm, em = _models(n_bboxes=2)
    om = _orientation([0.1, 0.2])
    om.predict_batch.side_effect = OrientationInferenceError("bbox 1 is NaN")
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 500
    assert r.json().get("results") is None


def test_misaligned_orientation_result_count_fails_the_request():
    """A short/long list would attach one bbox's theta to another's crop."""
    pm, cm, em = _models(n_bboxes=3)
    om = _orientation([0.1])          # 1 result for 3 bboxes
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 500
    assert "1 result" in r.text or "result(s)" in r.text


# ------------------------------------------------------- legacy label path

def test_densenet_orientation_label_path_still_works():
    """The legacy viewpoint-label orientation type must keep functioning."""
    from app.models.densenet_orientation import DenseNetOrientationModel
    pm, cm, em = _models()
    om = MagicMock(spec=DenseNetOrientationModel)
    om.predict.return_value = {"predictions": [
        {"label": "left", "probability": 0.9, "index": 0}]}
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    assert res["theta_source"] == "detector"      # a label model supplies no theta
    assert res["orientation"]["label"] == "left"


def test_unknown_orientation_model_type_is_rejected():
    from app.models.miewid import MiewidModel
    pm, cm, em = _models()
    om = MagicMock(spec=MiewidModel)              # not an orientation model at all
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 400
