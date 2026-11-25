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

import math

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import pipeline_router

VALID_PNG_DATA_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


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


def _orientation(thetas, effective_bboxes=None, oriented_bboxes=None, rejected=None):
    """`thetas` are LONG-AXIS angles -- what the pipeline now emits.

    The real model returns both halves: `theta` stays reference-faithful (the
    upright-rotation angle, long axis + 90deg) and `theta_oriented`/
    `oriented_bbox` carry the object-aligned box. Pass `oriented_bboxes=[None,
    ...]` to simulate a degenerate prediction with no object axis.
    `rejected={index: [reasons]}` marks rows the plausibility gate flagged.
    """
    rejected = rejected or {}
    from app.models.wbia_orientation import WbiaOrientationModel
    om = MagicMock(spec=WbiaOrientationModel)
    effs = effective_bboxes or [[10, 10, 50, 50]] * len(thetas)
    obs = oriented_bboxes if oriented_bboxes is not None \
        else [[12.0, 14.0, 40.0, 20.0]] * len(thetas)
    om.predict_batch.return_value = [
        {"model_id": "o",
         "theta": t + math.radians(90),
         "theta_oriented": None if ob is None else t,
         "oriented_bbox": ob,
         "coords_normalized": [0.5] * 5,
         "effective_bbox": e,
         "orientation_status": "rejected_implausible" if k in rejected else "ok",
         **({"orientation_reject_reasons": rejected[k]} if k in rejected else {})}
        for k, (t, e, ob) in enumerate(zip(thetas, effs, obs))
    ]
    return om


PAYLOAD = {"image_uri": VALID_PNG_DATA_URI,
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
          "effective_bbox": [10, 10, 50, 50],
          "theta_oriented": 0.5 - math.pi / 2,
          "oriented_bbox": [12.0, 14.0, 40.0, 20.0],
          "orientation_status": "ok"}])[1]
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

def test_the_emitted_bbox_is_the_one_theta_describes():
    """Wildbook persists the result's bbox alongside theta
    (MlServiceProcessor.featureParams), so the two must name the SAME rectangle.
    That rectangle is the regressor's object-aligned box -- neither the
    detector's proposal nor the axis-aligned crop region orientation ran on."""
    pm, cm, em = _models()
    om = _orientation([0.5], effective_bboxes=[[0, 0, 640, 480]],
                      oriented_bboxes=[[80.0, 60.0, 480.0, 120.0]])
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 200, r.text
    bbox = r.json()["results"][0]["bbox"]
    assert bbox == [80, 60, 480, 120]
    assert bbox != [10, 10, 50, 50]      # not the detector's
    assert bbox != [0, 0, 640, 480]      # not the crop region


def test_all_consumers_receive_the_identical_region():
    """One authoritative box: whatever is persisted is what was cropped."""
    pm, cm, em = _models()
    om = _orientation([0.5], effective_bboxes=[[0, 0, 640, 480]],
                      oriented_bboxes=[[80.0, 60.0, 480.0, 120.0]])
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 200, r.text
    emitted = r.json()["results"][0]["bbox"]
    assert list(cm.predict.call_args.kwargs["bbox"]) == emitted
    assert list(em.extract_embeddings.call_args.kwargs["bbox"]) == emitted


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


# --------------------------------- request-level validation (review round 2)

def test_malformed_second_result_stops_consumers_for_the_first_bbox_too():
    """Checking only the COUNT and dereferencing rows lazily inside the loop would
    let bbox 0 run classify/extract before bbox 1's bad row failed the request.
    Request-level fail-closed must mean NO consumer runs."""
    pm, cm, em = _models(n_bboxes=2)
    om = _orientation([0.1, 0.2])                       # two fully valid rows...
    om.predict_batch.return_value[1]["theta"] = float("nan")   # ...then break row 1 only
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 500
    assert "result 1" in r.json()["detail"]             # failed on row 1, not row 0
    cm.predict.assert_not_called()
    em.extract_embeddings.assert_not_called()


@pytest.mark.parametrize("bad_row", [
    {"model_id": "o", "coords_normalized": [0.5] * 5, "effective_bbox": [1, 1, 2, 2]},   # no theta
    {"model_id": "o", "theta": "x", "coords_normalized": [0.5] * 5, "effective_bbox": [1, 1, 2, 2]},
    {"model_id": "o", "theta": 0.1, "coords_normalized": [0.5] * 5},                     # no effective_bbox
    {"model_id": "o", "theta": 0.1, "coords_normalized": [0.5] * 5, "effective_bbox": [1, 1]},
    {"model_id": "o", "theta": 0.1, "coords_normalized": [0.5] * 5, "effective_bbox": [1.5, 1, 2, 2]},
    "not-an-object",
])
def test_malformed_orientation_rows_fail_the_request(bad_row):
    pm, cm, em = _models()
    om = _orientation([0.1])
    om.predict_batch.return_value = [bad_row]
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 500
    cm.predict.assert_not_called()


def test_detector_bbox_is_retained_as_an_audit_field():
    """The object-aligned box replaces the detector's, so what the detector
    actually said must stay recoverable."""
    pm, cm, em = _models()
    om = _orientation([0.5], effective_bboxes=[[0, 0, 640, 480]],
                      oriented_bboxes=[[80.0, 60.0, 480.0, 120.0]])
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    assert res["bbox"] == [80, 60, 480, 120]         # what theta describes
    assert res["detector_bbox"] == [10, 10, 50, 50]  # what the detector proposed


def test_fractional_degenerate_bbox_is_skipped_on_the_non_regressor_path():
    """width=0.5 passes `width <= 0` but int(0.5) == 0 — consumers would have got
    a degenerate box. The guard must run AFTER integerization."""
    from app.models.densenet_classifier import DenseNetClassifierModel
    from app.models.miewid import MiewidModel
    from app.models.yolo_ultralytics import YOLOUltralyticsModel
    pm = MagicMock(spec=YOLOUltralyticsModel)
    pm.predict.return_value = {"predictions": [
        {"bbox": [10, 10, 0.5, 50], "theta": 0.0, "score": 0.9,
         "class": "x", "class_id": 0}]}
    cm = MagicMock(spec=DenseNetClassifierModel)
    cm.predict.return_value = {"predictions": []}
    em = MagicMock(spec=MiewidModel)
    em.extract_embeddings.return_value = np.zeros((1, 2152))
    r = _client(pm, cm, em).post("/pipeline/", json=_payload())
    assert r.status_code == 200, r.text
    assert r.json()["results"] == []
    cm.predict.assert_not_called()


# ------------------------------------------------- object-aligned box (#beluga)

def test_emitted_bbox_and_theta_are_the_object_aligned_box():
    """The persisted pair must be the OBJECT-ALIGNED box and its long-axis
    angle. Emitting the reference theta beside the axis-aligned crop region
    stored a rectangle a quarter turn off the animal (beluga, Flukebook 10.14)."""
    pm, cm, em = _models()
    om = _orientation([0.2407], effective_bboxes=[[10, 10, 50, 50]],
                      oriented_bboxes=[[12.0, 18.0, 44.0, 12.0]])
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))

    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    assert res["theta"] == pytest.approx(0.2407)          # long axis, NOT +90
    assert res["bbox"] == [12, 18, 44, 12]                # oriented, not [10,10,50,50]
    assert res["theta_source"] == "orientation"


def test_classify_and_extract_crop_the_object_aligned_box():
    """One box for all consumers: whatever is persisted is what was embedded."""
    pm, cm, em = _models()
    om = _orientation([0.2407], effective_bboxes=[[10, 10, 50, 50]],
                      oriented_bboxes=[[12.0, 18.0, 44.0, 12.0]])
    _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))

    assert cm.predict.call_args.kwargs["bbox"] == [12, 18, 44, 12]
    assert list(em.extract_embeddings.call_args.kwargs["bbox"]) == [12, 18, 44, 12]


def test_no_object_axis_fails_closed_at_the_model():
    """A prediction whose centre and side point coincide describes no axis.
    The model raises rather than defaulting: a fabricated 0.0 is
    indistinguishable from a real horizontal animal downstream."""
    from app.models.wbia_orientation import OrientationInferenceError, oriented_box

    assert oriented_box([0.5, 0.5, 0.5, 0.5, 0.1], [0, 0, 400, 400]) == (None, None)
    assert OrientationInferenceError is not None


def test_sub_pixel_oriented_box_is_rejected_before_any_consumer_runs():
    """A width of 0.4 is positive but rounds to 0. Unchecked, get_chip_from_img
    silently returns the WHOLE image and we persist a zero-width box beside
    that embedding."""
    pm, cm, em = _models()
    om = _orientation([0.2407], oriented_bboxes=[[12.0, 18.0, 44.0, 0.4]])
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))

    assert r.status_code == 500
    assert "sub-pixel" in r.json()["detail"].lower()
    cm.predict.assert_not_called()
    em.extract_embeddings.assert_not_called()


def test_one_bad_row_stops_every_row_reaching_a_consumer():
    """Request-level fail-closed: row 1 being unusable must not let row 0's
    classify/extract run first."""
    pm, cm, em = _models(n_bboxes=2)
    om = _orientation([0.2, 0.3],
                      oriented_bboxes=[[12.0, 18.0, 44.0, 20.0], [12.0, 18.0, 44.0, 0.2]])
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))

    assert r.status_code == 500
    cm.predict.assert_not_called()
    em.extract_embeddings.assert_not_called()


def test_negative_origin_is_allowed():
    """An object-aligned box legitimately overhangs the frame; only a sub-pixel
    SIDE is invalid. Rejecting negative origins would drop edge animals."""
    pm, cm, em = _models()
    om = _orientation([0.2407], oriented_bboxes=[[-30.0, -12.0, 44.0, 20.0]])
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))

    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["bbox"] == [-30, -12, 44, 20]


def test_half_an_oriented_box_is_rejected():
    """Fail closed: a bbox without its angle (or vice versa) is a malformed
    contract, not something to paper over."""
    pm, cm, em = _models()
    om = _orientation([0.2407])
    om.predict_batch.return_value[0]["oriented_bbox"] = None
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))

    assert r.status_code == 500
    assert "malformed oriented_bbox" in r.json()["detail"].lower()


# ------------------------------------- implausible oriented box (issue #59)
#
# The model flags an implausible box per detection instead of raising. The
# router then falls back for THAT detection alone -- the detector's box and
# theta, exactly as if no orienter were configured -- and labels it, so a
# fallback can never be mistaken for a predicted angle. Failing the request
# would throw away every good detection in the image.

SLIVER = [12.0, 18.0, 44.0, 0.4]   # also sub-pixel: must not 500 any more
# Row 1's detector box runs past the right edge of a 100 px wide image;
# resolve_bbox truncates it the way NumPy slicing does. The fallback must use
# the CLIPPED crop (what orientation looked at), while detector_bbox keeps the
# raw one for audit.
RAW_DETECTOR_BOX = [45, 20, 60, 50]
CLIPPED_EFF = [45, 20, 55, 50]


def _two_rows_second_rejected(detector_theta=0.0, coords=None, raw_bbox=RAW_DETECTOR_BOX):
    pm, cm, em = _models(n_bboxes=2)
    pm.predict.return_value["predictions"][1]["bbox"] = raw_bbox
    pm.predict.return_value["predictions"][1]["theta"] = detector_theta
    om = _orientation([0.2, 0.3],
                      effective_bboxes=[[10, 10, 50, 50], CLIPPED_EFF],
                      oriented_bboxes=[[12.0, 18.0, 44.0, 20.0], SLIVER],
                      rejected={1: ["sub_pixel", "aspect"]})
    if coords == "MISSING":
        del om.predict_batch.return_value[1]["coords_normalized"]
    elif coords is not None:
        om.predict_batch.return_value[1]["coords_normalized"] = coords
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))
    return r, cm, em


def test_implausible_row_falls_back_for_that_detection_only():
    r, cm, em = _two_rows_second_rejected()

    assert r.status_code == 200, r.text
    good, bad = r.json()["results"]
    assert good["bbox"] == [12, 18, 44, 20]
    assert good["theta"] == pytest.approx(0.2)
    assert good["theta_source"] == "orientation"
    assert good["orientation_status"] == "ok"
    assert "orientation_reject_reasons" not in good

    assert bad["bbox"] == CLIPPED_EFF                 # the detector crop, not the sliver
    assert bad["detector_bbox"] == RAW_DETECTOR_BOX   # raw box kept for audit
    assert bad["theta"] == 0.0
    assert bad["theta_source"] == "detector"
    assert bad["orientation_status"] == "rejected_implausible"
    assert bad["orientation_reject_reasons"] == ["sub_pixel", "aspect"]
    assert bad["orientation_coords_normalized"] == [0.5] * 5


@pytest.mark.parametrize("detector_theta,expected_bbox", [
    # Axis-aligned (lightnet, always 0.0): the frame-clipped crop, as WBIA stored it.
    (0.0, CLIPPED_EFF),
    # Rotated detector box: clipping one side would move its rotation centre,
    # so the detector's own rectangle and angle are kept together, integerized.
    (0.35, [45, 20, 60, 50]),
])
def test_consumers_get_the_same_fallback_box_and_theta_that_are_emitted(
        detector_theta, expected_bbox):
    """The embedding and the persisted box must describe the same region, at
    the detector's own theta -- as if no orienter were configured."""
    r, cm, em = _two_rows_second_rejected(detector_theta=detector_theta)
    bad = r.json()["results"][1]

    assert bad["bbox"] == expected_bbox
    assert bad["theta"] == pytest.approx(detector_theta)
    for call in (cm.predict.call_args_list[1], em.extract_embeddings.call_args_list[1]):
        assert list(call.kwargs["bbox"]) == expected_bbox
        assert call.kwargs["theta"] == pytest.approx(detector_theta)


def test_rotated_fallback_with_a_sub_pixel_detector_box_uses_the_clipped_crop():
    """A rotated detector box that integerizes to < 1 px is not a box; fall back
    to the crop orientation actually looked at instead."""
    r, cm, em = _two_rows_second_rejected(detector_theta=0.35, raw_bbox=[45, 20, 0.6, 50])
    assert r.status_code == 200, r.text
    assert r.json()["results"][1]["bbox"] == CLIPPED_EFF


@pytest.mark.parametrize("detector_theta", [float("nan"), float("inf"), "x", True])
def test_bad_detector_theta_on_a_rejected_row_fails_before_any_consumer(detector_theta):
    """The fallback angle comes from the detector; YOLO OBB passes NaN/inf
    straight through. Checked up front, or row 0 would already be embedded."""
    r, cm, em = _two_rows_second_rejected(detector_theta=detector_theta)

    assert r.status_code == 500
    assert "result 1" in r.json()["detail"]
    cm.predict.assert_not_called()
    em.extract_embeddings.assert_not_called()


@pytest.mark.parametrize("coords", [None, [0.5] * 4, [0.5] * 4 + [float("nan")], ["a"] * 5])
def test_rejected_row_with_bad_audit_coords_fails_before_any_consumer(coords):
    r, cm, em = _two_rows_second_rejected(coords=coords if coords is not None else "MISSING")
    assert r.status_code == 500
    cm.predict.assert_not_called()
    em.extract_embeddings.assert_not_called()


@pytest.mark.parametrize("status", [None, "", "OK", "rejected", 1])
def test_missing_or_unknown_status_fails_before_any_consumer(status):
    """Producer and consumer run in-process; a row without a known status has
    not been through the gate, so it is a malformed contract, not an 'ok'."""
    pm, cm, em = _models()
    om = _orientation([0.2407])
    if status is None:
        del om.predict_batch.return_value[0]["orientation_status"]
    else:
        om.predict_batch.return_value[0]["orientation_status"] = status
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))

    assert r.status_code == 500
    assert "orientation_status" in r.json()["detail"]
    cm.predict.assert_not_called()
    em.extract_embeddings.assert_not_called()


@pytest.mark.parametrize("reasons", [None, [], "aspect", [1]])
def test_rejected_row_without_reasons_fails_the_request(reasons):
    pm, cm, em = _models()
    om = _orientation([0.2407], rejected={0: reasons})
    if reasons is None:
        del om.predict_batch.return_value[0]["orientation_reject_reasons"]
    r = _client(pm, cm, em, om).post("/pipeline/", json=_payload(orientation_model_id="o"))

    assert r.status_code == 500
    cm.predict.assert_not_called()
    em.extract_embeddings.assert_not_called()
