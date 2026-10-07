"""/pipeline/ guards for the yolo `obb_theta` option (Codex review round 1).

(2) The pipeline feeds the detector's emitted bbox to a wbia-orientation
regressor BEFORE it replaces bbox/theta, and a DenseNet orientation classifier
crops the same emitted box. `min_rotation` swaps w/h, so the orientation model
would see a different window than it was validated on. Until that combination
is tested end to end it is refused, from config and from a request override.

(1) A config that spells the key as `null` is copied into predict() by the
router and must round-trip as "raw".
"""
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models import yolo_ultralytics as yu
from app.models.yolo_ultralytics import YOLOUltralyticsModel
from app.routers import pipeline_router

VALID_PNG_DATA_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class _Tensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def tolist(self):
        return self._arr.tolist()


class _OBB:
    def __init__(self, rows):
        self.xywhr = _Tensor(rows)
        self.conf = _Tensor([0.9] * len(rows))
        self.cls = _Tensor([1] * len(rows))


class _Results:
    def __init__(self, rows):
        self.obb = _OBB(rows)
        self.boxes = None
        self.names = {0: "cat_face", 1: "dog_face"}


KAIJU_ROW = (92.0, 110.0, 213.0, 182.0, 1.5497775077819824)


class _DummyYOLO:
    def __init__(self, path):
        pass

    def to(self, device):
        return self

    def predict(self, *a, **k):
        return [_Results([KAIJU_ROW])]


def _real_detector(monkeypatch, **cfg):
    monkeypatch.setattr(yu, "YOLO", _DummyYOLO)
    monkeypatch.setattr(yu, "decode_image_rgb", lambda b: np.zeros((8, 8, 3), dtype=np.uint8))
    pm = YOLOUltralyticsModel()
    pm.load("x.pt", "cpu", **cfg)
    return pm


def _client(pm, p_config, orientation_model=None):
    from app.models.densenet_classifier import DenseNetClassifierModel
    from app.models.miewid import MiewidModel
    cm = MagicMock(spec=DenseNetClassifierModel)
    cm.predict.return_value = {"predictions": [
        {"label": "dog_face:front", "probability": 0.9, "index": 0,
         "species": "dog_face", "viewpoint": "front"}]}
    em = MagicMock(spec=MiewidModel)
    em.extract_embeddings.return_value = np.zeros((1, 2152))

    app = FastAPI()
    app.include_router(pipeline_router.router)
    handler = MagicMock()
    handler.get_model.side_effect = lambda mid: {
        "p": pm, "c": cm, "e": em, "o": orientation_model}.get(mid)
    handler.get_model_info.side_effect = lambda mid: {
        "p": {"config": {"model_path": "x.pt", "device": "cpu", **p_config}},
        "c": {"config": {}}, "e": {"config": {"version": 4.1}}, "o": {"config": {}},
    }.get(mid)
    handler.list_models.return_value = {"p": {}, "c": {}, "e": {}, "o": {}}
    app.state.model_handler = handler
    return TestClient(app), em


def _payload(**kw):
    return {"image_uri": VALID_PNG_DATA_URI, "predict_model_id": "p",
            "classify_model_id": "c", "extract_model_id": "e", **kw}


def _wbia_orienter():
    from app.models.wbia_orientation import WbiaOrientationModel
    om = MagicMock(spec=WbiaOrientationModel)
    om.predict_batch.return_value = [{
        "model_id": "o", "theta": 1.6, "theta_oriented": 0.03,
        "oriented_bbox": [1.0, 3.5, 182.0, 213.0], "coords_normalized": [0.5] * 5,
        "effective_bbox": [1, 3, 182, 213]}]
    return om


def _densenet_orienter():
    from app.models.densenet_orientation import DenseNetOrientationModel
    om = MagicMock(spec=DenseNetOrientationModel)
    om.predict.return_value = {"predictions": [{"label": "front", "probability": 0.9}]}
    return om


# (1) null in config round-trips through the router as raw
def test_null_obb_theta_in_config_round_trips_as_raw(monkeypatch):
    pm = _real_detector(monkeypatch, obb_theta=None)
    client, em = _client(pm, {"obb_theta": None})
    r = client.post("/pipeline/", json=_payload(bbox_score_threshold=0.1))
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    assert (res["bbox"][2], res["bbox"][3]) == (213, 182)          # raw representation kept
    assert res["theta"] == pytest.approx(1.5497775077819824)


def test_min_rotation_from_config_reaches_the_persisted_box(monkeypatch):
    pm = _real_detector(monkeypatch, obb_theta="min_rotation")
    client, em = _client(pm, {"obb_theta": "min_rotation"})
    r = client.post("/pipeline/", json=_payload(bbox_score_threshold=0.1))
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    assert (res["bbox"][2], res["bbox"][3]) == (182, 213)
    assert abs(res["theta"]) < 0.03
    # and the extractor was handed the same normalised box + theta
    kw = em.extract_embeddings.call_args.kwargs
    assert tuple(kw["bbox"][2:]) == (182, 213) and abs(kw["theta"]) < 0.03


# (2) min_rotation + an orientation model is refused before any inference
@pytest.mark.parametrize("orienter", [_wbia_orienter, _densenet_orienter])
def test_min_rotation_config_with_orientation_model_is_rejected(monkeypatch, orienter):
    pm = _real_detector(monkeypatch, obb_theta="min_rotation")
    om = orienter()
    client, em = _client(pm, {"obb_theta": "min_rotation"}, orientation_model=om)
    r = client.post("/pipeline/", json=_payload(orientation_model_id="o", bbox_score_threshold=0.1))
    assert r.status_code == 400, r.text
    assert "obb_theta" in r.text and "orientation" in r.text
    assert not em.extract_embeddings.called
    # the regressor exposes predict_batch, the DenseNet classifier only predict;
    # whichever exists must not have been invoked
    for meth in ("predict_batch", "predict"):
        if hasattr(type(om), meth) or meth in dir(om):
            try:
                assert not getattr(om, meth).called
            except AttributeError:
                pass


def test_min_rotation_request_override_with_orientation_model_is_rejected(monkeypatch):
    pm = _real_detector(monkeypatch)                      # config says raw ...
    om = _wbia_orienter()
    client, em = _client(pm, {}, orientation_model=om)
    r = client.post("/pipeline/", json=_payload(orientation_model_id="o", bbox_score_threshold=0.1,
                                       predict_model_params={"obb_theta": "min_rotation"}))
    assert r.status_code == 400, r.text   # ... but the request asked for min_rotation
    assert not em.extract_embeddings.called


def test_raw_config_with_orientation_model_still_works(monkeypatch):
    pm = _real_detector(monkeypatch)
    om = _wbia_orienter()
    client, em = _client(pm, {}, orientation_model=om)
    r = client.post("/pipeline/", json=_payload(orientation_model_id="o", bbox_score_threshold=0.1))
    assert r.status_code == 200, r.text
    assert om.predict_batch.called


# Codex round 2: a request override of null must resolve the SAME way in the
# guard and in the handler (raw), so the orientation model receives the raw
# (unswapped) box and the persisted box is raw too.
@pytest.mark.parametrize("orienter", [_wbia_orienter, _densenet_orienter])
def test_null_override_on_min_rotation_config_gives_orienter_the_raw_box(monkeypatch, orienter):
    pm = _real_detector(monkeypatch, obb_theta="min_rotation")
    om = orienter()
    client, em = _client(pm, {"obb_theta": "min_rotation"}, orientation_model=om)
    r = client.post("/pipeline/", json=_payload(orientation_model_id="o", bbox_score_threshold=0.1,
                                                predict_model_params={"obb_theta": None}))
    assert r.status_code == 200, r.text
    if om.predict_batch.called if hasattr(om, "predict_batch") else False:
        seen = om.predict_batch.call_args.kwargs["bboxes"][0]
    else:
        seen = om.predict.call_args.kwargs["bbox"]
    # raw representation: w=213 > h=182 (ints after the pipeline's integerization)
    assert (int(seen[2]), int(seen[3])) == (213, 182), seen
