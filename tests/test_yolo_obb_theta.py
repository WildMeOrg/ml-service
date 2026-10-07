"""OBB theta representation handling in the YOLO handler.

ultralytics 8.3.x (the pinned 8.3.139 included) runs `ops.regularize_rboxes`
on every OBB prediction: theta is forced into [0, pi/2) and the edges are
swapped when it wraps. That is a discontinuity at 0 -- a subject tilted -1
degree comes back as an 89-degree box with w/h swapped, while +1 degree
stays +1 degree. `get_chip_from_img` honours theta, so the two
representations of the same rectangle yield chips a quarter turn apart.

`obb_theta: "min_rotation"` (per model, opt in) normalises each box to the
representation with the smallest |theta|, continuous through 0. The default
path must stay byte-identical to the pre-option behaviour.
"""
import math

import numpy as np
import pytest

from app.models import yolo_ultralytics as yu
from app.models.yolo_ultralytics import YOLOUltralyticsModel


# --------------------------------------------------------------------------
# fakes: the smallest surface of an ultralytics Results object that
# _process_results reads
# --------------------------------------------------------------------------
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
    def __init__(self, xywhr, conf, cls):
        self.xywhr = _Tensor(xywhr)
        self.conf = _Tensor(conf)
        self.cls = _Tensor(cls)


class _Boxes:
    def __init__(self, xywh, conf, cls):
        self.xywh = _Tensor(xywh)
        self.conf = _Tensor(conf)
        self.cls = _Tensor(cls)


class _Results:
    def __init__(self, obb=None, boxes=None, names=None):
        self.obb = obb
        self.boxes = boxes
        self.names = names or {0: "cat_face", 1: "dog_face"}


def _obb_results(rows, conf=None, cls=None):
    n = len(rows)
    return _Results(obb=_OBB(rows, conf or [0.9] * n, cls or [1] * n))


PI2 = math.pi / 2
PI4 = math.pi / 4

# What Kaiju (ultralytics 8.3.139) returned for the PetFace dog 000054/00.png:
# centre (92, 110), edges swapped to w=213 > h=182, theta = -0.021 + pi/2.
KAIJU_ROW = (92.0, 110.0, 213.0, 182.0, 1.5497775077819824)


# --------------------------------------------------------------------------
# default path: characterization guard -- these literal values are what the
# handler produced BEFORE the option existed and must not move
# --------------------------------------------------------------------------
def test_default_path_is_unchanged_for_obb_rows():
    m = YOLOUltralyticsModel()
    out = m._process_results(_obb_results([KAIJU_ROW]), [0.3, 0.3])
    x, y, w, h = out["bboxes"][0]
    assert (x, y, w, h) == pytest.approx(
        (92.0 - 213.0 * 1.3 / 2, 110.0 - 182.0 * 1.3 / 2, 276.9, 236.6))
    assert out["thetas"][0] == pytest.approx(1.5497775077819824)
    assert out["class_names"] == ["dog_face"]


def test_default_path_assigns_long_dilation_to_the_longer_raw_edge():
    # w < h here, so the long factor must land on h -- exactly as before.
    m = YOLOUltralyticsModel()
    out = m._process_results(_obb_results([(50.0, 60.0, 20.0, 40.0, 0.1)]), [0.5, 0.0])
    x, y, w, h = out["bboxes"][0]
    assert (w, h) == pytest.approx((20.0, 60.0))
    assert (x, y) == pytest.approx((50.0 - 10.0, 60.0 - 30.0))
    assert out["thetas"][0] == pytest.approx(0.1)


def test_default_path_explicit_raw_matches_omitted():
    m = YOLOUltralyticsModel()
    rows = [KAIJU_ROW, (50.0, 60.0, 20.0, 40.0, -0.9)]
    assert m._process_results(_obb_results(rows), [0.3, 0.1], obb_theta="raw") == \
        m._process_results(_obb_results(rows), [0.3, 0.1])


# --------------------------------------------------------------------------
# min_rotation
# --------------------------------------------------------------------------
def test_min_rotation_swaps_the_regularized_kaiju_box_back_to_upright():
    m = YOLOUltralyticsModel()
    out = m._process_results(_obb_results([KAIJU_ROW]), [0.0, 0.0], obb_theta="min_rotation")
    x, y, w, h = out["bboxes"][0]
    # same centre, edges swapped back, theta brought down by pi/2
    assert (w, h) == pytest.approx((182.0, 213.0))
    assert (x, y) == pytest.approx((92.0 - 91.0, 110.0 - 106.5))
    assert out["thetas"][0] == pytest.approx(1.5497775077819824 - PI2, abs=1e-9)
    assert abs(out["thetas"][0]) < 0.03


def test_min_rotation_leaves_a_small_positive_angle_alone():
    m = YOLOUltralyticsModel()
    row = (92.0, 110.0, 182.0, 213.0, 0.03)
    out = m._process_results(_obb_results([row]), [0.0, 0.0], obb_theta="min_rotation")
    assert out["bboxes"][0] == pytest.approx([92.0 - 91.0, 110.0 - 106.5, 182.0, 213.0])
    assert out["thetas"][0] == pytest.approx(0.03)


def test_min_rotation_result_is_always_in_half_open_quarter_turn_range():
    m = YOLOUltralyticsModel()
    thetas = [-PI2 + 1e-6, -1.2, -PI4 - 1e-9, -PI4, -0.5, 0.0, 0.5, PI4, PI4 + 1e-9, 1.2, PI2 - 1e-6]
    rows = [(10.0, 10.0, 30.0, 20.0, t) for t in thetas]
    out = m._process_results(_obb_results(rows), [0.0, 0.0], obb_theta="min_rotation")
    for t in out["thetas"]:
        assert -PI4 < t <= PI4 + 1e-12, t


def test_min_rotation_boundaries_match_the_half_open_interval():
    m = YOLOUltralyticsModel()
    rows = [(10.0, 10.0, 30.0, 20.0, PI4),          # exactly +45: untouched
            (10.0, 10.0, 30.0, 20.0, -PI4),         # exactly -45: swapped, becomes +45
            (10.0, 10.0, 30.0, 20.0, PI4 + 1e-6)]   # just over +45: swapped, becomes just over -45
    out = m._process_results(_obb_results(rows), [0.0, 0.0], obb_theta="min_rotation")
    (w0, h0), (w1, h1), (w2, h2) = [tuple(b[2:]) for b in out["bboxes"]]
    assert (w0, h0) == (30.0, 20.0) and out["thetas"][0] == pytest.approx(PI4)
    assert (w1, h1) == (20.0, 30.0) and out["thetas"][1] == pytest.approx(PI4)
    assert (w2, h2) == (20.0, 30.0) and out["thetas"][2] == pytest.approx(-PI4 + 1e-6)


def test_min_rotation_handles_raw_negative_angles_from_newer_ultralytics():
    # 8.4.x returns the raw angle; a -0.9 rad box is closer to upright as (h, w, -0.9 + pi/2)
    m = YOLOUltralyticsModel()
    out = m._process_results(_obb_results([(50.0, 60.0, 20.0, 40.0, -0.9)]), [0.0, 0.0],
                             obb_theta="min_rotation")
    assert tuple(out["bboxes"][0][2:]) == pytest.approx((40.0, 20.0))
    assert out["thetas"][0] == pytest.approx(-0.9 + PI2)


def test_min_rotation_preserves_the_centre_and_dilates_after_normalising():
    # dilation axes must follow the NORMALISED box: after the swap the long
    # edge is h (213), so the long factor lands on h.
    m = YOLOUltralyticsModel()
    out = m._process_results(_obb_results([KAIJU_ROW]), [0.3, 0.0], obb_theta="min_rotation")
    x, y, w, h = out["bboxes"][0]
    assert (w, h) == pytest.approx((182.0, 213.0 * 1.3))
    assert (x + w / 2, y + h / 2) == pytest.approx((92.0, 110.0))


def test_min_rotation_does_not_touch_axis_aligned_boxes():
    m = YOLOUltralyticsModel()
    res = _Results(boxes=_Boxes([(50.0, 60.0, 20.0, 40.0)], [0.8], [0]))
    assert m._process_results(res, [0.2, 0.1], obb_theta="min_rotation") == \
        m._process_results(res, [0.2, 0.1])


def test_unknown_obb_theta_value_is_rejected():
    m = YOLOUltralyticsModel()
    with pytest.raises(ValueError):
        m._process_results(_obb_results([KAIJU_ROW]), [0.0, 0.0], obb_theta="sideways")


# --------------------------------------------------------------------------
# wiring: the config key must reach predict(), not just sit in model_config
# --------------------------------------------------------------------------
class _DummyYOLO:
    def __init__(self, path):
        self.path = path
        self.names = {0: "cat_face", 1: "dog_face"}

    def to(self, device):
        return self

    def predict(self, *a, **k):
        return [_obb_results([KAIJU_ROW])]


@pytest.fixture
def dummy_yolo(monkeypatch):
    monkeypatch.setattr(yu, "YOLO", _DummyYOLO)
    monkeypatch.setattr(yu, "decode_image_rgb", lambda b: np.zeros((8, 8, 3), dtype=np.uint8))


def test_load_stores_obb_theta_and_predict_applies_it(dummy_yolo):
    m = YOLOUltralyticsModel()
    m.load("x.pt", "cpu", obb_theta="min_rotation")
    assert m.get_model_info()["obb_theta"] == "min_rotation"
    out = m.predict(b"not-really-an-image")
    assert tuple(out["bboxes"][0][2:]) == pytest.approx((182.0, 213.0))
    assert abs(out["thetas"][0]) < 0.03


def test_load_default_is_raw_and_predict_passes_through(dummy_yolo):
    m = YOLOUltralyticsModel()
    m.load("x.pt", "cpu")
    assert m.get_model_info()["obb_theta"] == "raw"
    out = m.predict(b"not-really-an-image")
    assert tuple(out["bboxes"][0][2:]) == pytest.approx((213.0, 182.0))
    assert out["thetas"][0] == pytest.approx(1.5497775077819824)


def test_load_rejects_unknown_obb_theta(dummy_yolo):
    m = YOLOUltralyticsModel()
    with pytest.raises(ValueError):
        m.load("x.pt", "cpu", obb_theta="sideways")


# --------------------------------------------------------------------------
# Codex review round 1
# --------------------------------------------------------------------------
# (1) The routers copy model_info['config'] into every predict() call, so a
# config that spells the key as null must behave exactly like an absent key at
# EVERY entry point, while falsy junk ("", False, 0) must still be rejected.
def test_none_resolves_to_raw_in_process_results():
    m = YOLOUltralyticsModel()
    assert m._process_results(_obb_results([KAIJU_ROW]), [0.3, 0.1], obb_theta=None) == \
        m._process_results(_obb_results([KAIJU_ROW]), [0.3, 0.1])


def test_none_resolves_to_raw_in_predict(dummy_yolo):
    m = YOLOUltralyticsModel()
    m.load("x.pt", "cpu", obb_theta=None)
    assert m.get_model_info()["obb_theta"] == "raw"
    out = m.predict(b"img", obb_theta=None)   # what a router forwards from a `null` config
    assert tuple(out["bboxes"][0][2:]) == pytest.approx((213.0, 182.0))


@pytest.mark.parametrize("junk", ["", False, 0, [], {}])
def test_falsy_junk_is_rejected_not_silently_treated_as_raw(dummy_yolo, junk):
    m = YOLOUltralyticsModel()
    with pytest.raises(ValueError):
        m.load("x.pt", "cpu", obb_theta=junk)
    m2 = YOLOUltralyticsModel()
    with pytest.raises(ValueError):
        m2._process_results(_obb_results([KAIJU_ROW]), [0.0, 0.0], obb_theta=junk)


# (3) Exact, complete-output characterization of the default path on float32
# inputs (what ultralytics actually yields), generated from origin/main's
# implementation. `==`, not approx: any drift of the raw path fails here.
def _f32_obb_results():
    rows = [(92.0, 110.0, 213.0, 182.0, 1.5497775077819824),
            (50.0, 60.0, 20.0, 40.0, -0.9), (10.5, 20.25, 30.0, 30.0, 0.0)]
    obb = _OBB(rows, [0.699, 0.5, 0.25], [1, 0, 1])
    for t in (obb.xywhr, obb.conf, obb.cls):
        t._arr = t._arr.astype(np.float32)
    return _Results(obb=obb)


def _f32_aabb_results():
    bx = _Boxes([(50.0, 60.0, 20.0, 40.0), (7.0, 8.0, 9.0, 3.0)], [0.8, 0.4], [0, 1])
    for t in (bx.xywh, bx.conf, bx.cls):
        t._arr = t._arr.astype(np.float32)
    return _Results(boxes=bx)


OBB_F32_EXPECTED = {'bboxes': [[-46.45000000000002, 9.899999999999991, 276.90000000000003, 200.20000000000002], [39.0, 34.0, 22.0, 52.0], [-9.0, 3.75, 39.0, 33.0]], 'thetas': [1.5497775077819824, -0.8999999761581421, 0.0], 'scores': [0.6990000009536743, 0.5, 0.25], 'class_ids': [1, 0, 1], 'class_names': ['dog_face', 'cat_face', 'dog_face']}
AABB_F32_EXPECTED = {'bboxes': [[39.0, 36.0, 22.0, 48.0], [1.6000000000000005, 6.35, 10.799999999999999, 3.3000000000000003]], 'thetas': [0.0, 0.0], 'scores': [0.800000011920929, 0.4000000059604645], 'class_ids': [0, 1], 'class_names': ['cat_face', 'dog_face']}


def test_default_path_exact_output_float32_obb():
    assert YOLOUltralyticsModel()._process_results(_f32_obb_results(), [0.3, 0.1]) == OBB_F32_EXPECTED


def test_default_path_exact_output_float32_axis_aligned():
    assert YOLOUltralyticsModel()._process_results(_f32_aabb_results(), [0.2, 0.1]) == AABB_F32_EXPECTED


def test_raw_explicit_exact_output_float32_obb():
    assert YOLOUltralyticsModel()._process_results(_f32_obb_results(), [0.3, 0.1], obb_theta="raw") == OBB_F32_EXPECTED


# (4) The interval guarantee needs a finite angle; a NaN/inf theta would also
# wreck the chip downstream, so the opt-in path refuses it loudly. Raw stays raw.
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_min_rotation_rejects_nonfinite_theta(bad):
    m = YOLOUltralyticsModel()
    with pytest.raises(ValueError):
        m._process_results(_obb_results([(10.0, 10.0, 30.0, 20.0, bad)]), [0.0, 0.0], obb_theta="min_rotation")


def test_raw_passes_nonfinite_theta_through_unchanged():
    m = YOLOUltralyticsModel()
    out = m._process_results(_obb_results([(10.0, 10.0, 30.0, 20.0, float("nan"))]), [0.0, 0.0])
    assert math.isnan(out["thetas"][0])


# --------------------------------------------------------------------------
# Codex review round 2: the request-level `null` must mean the same thing in
# the handler as in the /pipeline/ guard. Rule: an ABSENT key inherits the
# loaded mode; an EXPLICIT null (config copy or request override) means raw.
# --------------------------------------------------------------------------
def test_explicit_none_override_forces_raw_even_when_loaded_mode_is_min_rotation(dummy_yolo):
    m = YOLOUltralyticsModel()
    m.load("x.pt", "cpu", obb_theta="min_rotation")
    out = m.predict(b"img", obb_theta=None)
    assert tuple(out["bboxes"][0][2:]) == pytest.approx((213.0, 182.0))   # raw representation
    assert out["thetas"][0] == pytest.approx(1.5497775077819824)


def test_absent_key_inherits_loaded_min_rotation(dummy_yolo):
    m = YOLOUltralyticsModel()
    m.load("x.pt", "cpu", obb_theta="min_rotation")
    out = m.predict(b"img")
    assert tuple(out["bboxes"][0][2:]) == pytest.approx((182.0, 213.0))
    assert abs(out["thetas"][0]) < 0.03
