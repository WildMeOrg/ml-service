"""Tests for WbiaOrientationModel — the wbia-plugin-orientation theta port.

This port's failure mode is being SILENTLY wrong by an angle, which downstream is
indistinguishable from correct output — the same class of bug as issue #33. So
these tests pin the reference algorithm's arithmetic exactly, plus every failure
mode surfaced in design review:

  - fail-closed (a 0.0 fallback IS the bug being repaired; 0.0 is also valid)
  - NumPy's non-clamping slice (a wide negative bbox crops the FAR EDGE)
  - integerization rule (int(-0.5)==0 vs floor(-0.5)==-1 -> far edge)
  - no label/probability ever emitted (the #33 confusion)
  - predict_batch ordering (a misaligned result attaches theta to the wrong crop)

Reference: wbia-plugin-orientation. Real-weight fidelity vs the reference runs in
the host preflight, not here — the checkpoints are hundreds of MB.
"""
import io
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


RGB = _png(np.random.RandomState(0).randint(0, 255, (300, 400, 3), dtype=np.uint8))


def _model(coords=(0.5, 0.5, 1.0, 0.5, 0.1), hflip=False, vflip=False):
    """A loaded model whose backbone returns fixed pre-sigmoid logits."""
    from app.models.wbia_orientation import WbiaOrientationModel
    m = WbiaOrientationModel()
    m.model_id, m.device, m.imsize = "t", "cpu", (224, 224)
    m.hflip, m.vflip = hflip, vflip
    # logit(p) so sigmoid(logit(p)) == p, letting tests state coords directly.
    # Clamped off the open interval's ends: logit(0)/logit(1) are +-inf.
    def _logit(c, eps=1e-9):
        c = min(max(c, eps), 1 - eps)
        return math.log(c / (1 - c))
    logits = torch.tensor([[_logit(c) for c in coords]])
    m.model = MagicMock(return_value=logits)
    return m


# ---------------------------------------------------------------- theta math

def test_compute_theta_matches_reference_formula():
    """core/evaluate.py:9-20 — arctan2(yt-yc, xt-xc) + radians(90)."""
    from app.models.wbia_orientation import compute_theta
    # xt-xc = 0.5, yt-yc = 0 -> arctan2(0, 0.5) = 0, +90deg
    assert compute_theta([0.5, 0.5, 1.0, 0.5, 0.1]) == pytest.approx(math.radians(90))


@pytest.mark.parametrize("coords,expected_deg", [
    ([0.5, 0.5, 1.0, 0.5, 0.1], 90.0),    # tip east   -> 90
    ([0.5, 0.5, 0.5, 1.0, 0.1], 180.0),   # tip south  -> 180
    ([0.5, 0.5, 0.0, 0.5, 0.1], 270.0),   # tip west   -> 270
    ([0.5, 0.5, 0.5, 0.0, 0.1], 0.0),     # tip north  -> 0
])
def test_compute_theta_cardinals(coords, expected_deg):
    from app.models.wbia_orientation import compute_theta
    got = math.degrees(compute_theta(coords))
    assert math.cos(math.radians(got - expected_deg)) == pytest.approx(1.0, abs=1e-9)


def test_the_plus_90_degrees_is_not_dropped():
    """Without the +90 this returns 0 — a silent, plausible, wrong angle."""
    from app.models.wbia_orientation import compute_theta
    assert compute_theta([0.5, 0.5, 1.0, 0.5, 0.1]) != pytest.approx(0.0)


# ------------------------------------------------- bbox resolution (Critical)

@pytest.mark.parametrize("bbox,expected", [
    ([10, 10, 50, 50], [10, 10, 50, 50]),        # ordinary
    ([-20, 10, 50, 50], [380, 10, 0, 50]),       # narrow negative -> empty
    ([-20, 10, 500, 50], [380, 10, 20, 50]),     # WIDE negative -> FAR EDGE
    ([380, 10, 50, 50], [380, 10, 20, 50]),      # overrun -> truncated
    ([500, 400, 50, 50], [400, 300, 0, 0]),      # fully outside -> empty
])
def test_resolve_bbox_reproduces_numpy_slicing(bbox, expected):
    """NumPy slicing does NOT clamp: a negative origin resolves from the far
    edge. resolve_bbox must reproduce exactly what the reference's raw slice did,
    or theta describes a region the detector never pointed at."""
    from app.models.wbia_orientation import resolve_bbox
    assert resolve_bbox(bbox, width=400, height=300) == expected


def test_resolve_bbox_agrees_with_actual_numpy_slice():
    """Ground the rule against NumPy itself rather than our belief about it."""
    from app.models.wbia_orientation import resolve_bbox
    img = np.zeros((300, 400, 3))
    for bbox in ([10, 10, 50, 50], [-20, 10, 500, 50], [380, 10, 50, 50],
                 [500, 400, 50, 50], [-20, 10, 50, 50]):
        x, y, w, h = bbox
        actual = img[y:y + h, x:x + w]
        eff = resolve_bbox(bbox, 400, 300)
        assert (eff[2], eff[3]) == (actual.shape[1], actual.shape[0]), bbox


def test_wide_negative_bbox_really_does_hit_the_far_edge():
    """Documents WHY resolve_bbox exists: the crop is real, and wrong."""
    img = np.zeros((300, 400, 3))
    img[:, 380:400] = 1.0                       # mark the far-right edge
    assert img[0:300, -20:480].sum() > 0        # a "left" bbox crops the RIGHT edge


@pytest.mark.parametrize("bad", [
    [float("nan"), 0, 10, 10],
    [0, float("inf"), 10, 10],
    [None, 0, 10, 10],
])
def test_resolve_bbox_rejects_non_finite(bad):
    from app.models.wbia_orientation import resolve_bbox, OrientationInferenceError
    with pytest.raises(OrientationInferenceError, match="non-finite|non-numeric"):
        resolve_bbox(bad, 400, 300)


def test_resolve_bbox_rejects_wrong_arity():
    from app.models.wbia_orientation import resolve_bbox, OrientationInferenceError
    with pytest.raises(OrientationInferenceError, match=r"\[x, y, w, h\]"):
        resolve_bbox([1, 2, 3], 400, 300)


@pytest.mark.parametrize("bbox,expected_x", [
    ([-0.5, 10, 50, 50], 0),      # int(-0.5) == 0   (truncate toward zero)
    ([10.9, 10, 50, 50], 10),     # int(10.9) == 10
    ([-1.9, 10, 50, 50], -1),     # int(-1.9) == -1
])
def test_integerization_truncates_toward_zero(bbox, expected_x):
    """int(), matching pipeline_router:218. floor(-0.5) would give -1, which then
    resolves from the far edge — the rule choice is load-bearing."""
    from app.models.wbia_orientation import resolve_bbox
    eff = resolve_bbox(bbox, 400, 300)
    assert eff[0] == (expected_x if expected_x >= 0 else 400 + expected_x)


# ------------------------------------------------------------ TTA arithmetic

def test_tta_hflip_mirrors_only_x_coords():
    """utils.py:88-102 with image_h_w=[1.0,1.0]: indices 0 and 2 -> 1-x."""
    m = _model(coords=(0.25, 0.4, 0.75, 0.6, 0.1), hflip=True, vflip=False)
    out = m._forward_tta(torch.zeros(1, 3, 224, 224))[0].tolist()
    # mean of original and its x-mirrored self: x-coords -> 0.5, others unchanged
    assert out[0] == pytest.approx((0.25 + 0.75) / 2)
    assert out[2] == pytest.approx((0.75 + 0.25) / 2)
    assert out[1] == pytest.approx(0.4)
    assert out[3] == pytest.approx(0.6)


def test_tta_vflip_mirrors_only_y_coords():
    """utils.py:104-116: indices 1 and 3 -> 1-y."""
    m = _model(coords=(0.25, 0.4, 0.75, 0.6, 0.1), hflip=False, vflip=True)
    out = m._forward_tta(torch.zeros(1, 3, 224, 224))[0].tolist()
    assert out[1] == pytest.approx((0.4 + 0.6) / 2)
    assert out[3] == pytest.approx((0.6 + 0.4) / 2)
    assert out[0] == pytest.approx(0.25)
    assert out[2] == pytest.approx(0.75)


def test_tta_is_a_three_way_mean_when_both_flips_on():
    """orientation_net.py:93 — (out + out_h + out_v) / 3, the deployed default."""
    m = _model(coords=(0.25, 0.4, 0.75, 0.6, 0.1), hflip=True, vflip=True)
    out = m._forward_tta(torch.zeros(1, 3, 224, 224))[0].tolist()
    assert out[0] == pytest.approx((0.25 + 0.75 + 0.25) / 3)   # h mirrors x only
    assert out[1] == pytest.approx((0.40 + 0.40 + 0.60) / 3)   # v mirrors y only
    assert out[4] == pytest.approx(0.1)                         # w never mirrors


def test_tta_disabled_is_a_single_forward():
    m = _model(coords=(0.25, 0.4, 0.75, 0.6, 0.1))
    m.model.reset_mock()
    m._forward_tta(torch.zeros(1, 3, 224, 224))
    assert m.model.call_count == 1


def test_head_is_sigmoid_not_softmax():
    """Softmaxing these coords is issue #33: it yields ~0.2 pseudo-probabilities.
    Sigmoid is why they are independent values in [0,1] — and they need not sum
    to 1, which softmax would force."""
    m = _model(coords=(0.9, 0.9, 0.9, 0.9, 0.9))
    out = m._forward_tta(torch.zeros(1, 3, 224, 224))[0]
    assert out.sum().item() == pytest.approx(4.5, abs=1e-4)
    assert all(0.0 <= v <= 1.0 for v in out.tolist())


# ----------------------------------------------------------- output contract

def test_predict_returns_theta_and_never_a_label():
    """The #33 failure mode: any label/probability could be mistaken downstream
    for a viewpoint classification."""
    m = _model()
    r = m.predict_batch(RGB, [[10, 10, 100, 100]])[0]
    assert set(r) == {"model_id", "theta", "coords_normalized", "effective_bbox"}
    for forbidden in ("label", "probability", "class_id", "class", "predictions"):
        assert forbidden not in r
    assert math.isfinite(r["theta"])
    assert len(r["coords_normalized"]) == 5


def test_predict_batch_is_ordered_one_per_bbox():
    """A misaligned result would attach one bbox's theta to another's crop."""
    m = _model()
    bboxes = [[10, 10, 100, 100], [50, 50, 60, 60], [0, 0, 400, 300]]
    out = m.predict_batch(RGB, bboxes)
    assert len(out) == len(bboxes)
    for r, b in zip(out, bboxes):
        assert r["effective_bbox"] == [b[0], b[1], b[2], b[3]]


def test_degenerate_bbox_falls_back_to_full_frame_in_effective_bbox():
    """animal_wbia.py:25-28 reloads the full image; effective_bbox must SAY so,
    or classify/extract crop a different region than theta describes."""
    m = _model()
    r = m.predict_batch(RGB, [[500, 400, 10, 10]])[0]     # fully outside
    assert r["effective_bbox"] == [0, 0, 400, 300]


def test_none_bbox_means_full_frame():
    m = _model()
    assert m.predict_batch(RGB, [None])[0]["effective_bbox"] == [0, 0, 400, 300]


# --------------------------------------------------------------- fail-closed

def test_non_finite_theta_raises_rather_than_defaulting_to_zero():
    """A 0.0 fallback IS the bug being repaired — and 0.0 is also a legitimate
    prediction, so a sentinel could not be told apart from a real answer."""
    from app.models.wbia_orientation import OrientationInferenceError
    m = _model()
    m.model = MagicMock(return_value=torch.tensor([[float("nan")] * 5]))
    with pytest.raises(OrientationInferenceError, match="non-finite theta"):
        m.predict_batch(RGB, [[10, 10, 100, 100]])


def test_predicted_zero_theta_is_valid_not_an_error():
    """The converse: 0.0 must pass through."""
    m = _model(coords=(0.5, 0.5, 0.5, 0.0, 0.1))   # tip north -> theta 0
    r = m.predict_batch(RGB, [[10, 10, 100, 100]])[0]
    assert math.cos(r["theta"]) == pytest.approx(1.0, abs=1e-9)


def test_predict_rejects_a_pre_rotated_crop():
    """The reference consumes the axis-aligned bbox and derives theta itself;
    a detector-rotated crop would double-rotate."""
    from app.models.wbia_orientation import OrientationInferenceError
    m = _model()
    with pytest.raises(OrientationInferenceError, match="pre-rotated|axis-aligned"):
        m.predict(RGB, bbox=[10, 10, 100, 100], theta=0.5)


def test_unloaded_model_raises():
    from app.models.wbia_orientation import WbiaOrientationModel, OrientationInferenceError
    with pytest.raises(OrientationInferenceError, match="not loaded"):
        WbiaOrientationModel().predict_batch(RGB, [None])


def test_undecodable_bytes_raise():
    from app.models.wbia_orientation import OrientationInferenceError
    m = _model()
    with pytest.raises(OrientationInferenceError, match="could not decode"):
        m.predict_batch(b"not an image", [None])


# ------------------------------------------------- RGB canonicalization

def test_rgb_input_passes_through_untouched():
    """Fidelity depends on (H,W,3) reaching the model exactly as the reference
    would feed it."""
    from app.models.wbia_orientation import _canonicalize_rgb
    arr = np.random.rand(10, 12, 3)
    assert _canonicalize_rgb(arr) is arr


def test_grayscale_is_replicated_to_three_channels():
    """Deliberate deviation: the reference RAISES here (Normalize expects 3ch)."""
    from app.models.wbia_orientation import _canonicalize_rgb
    out = _canonicalize_rgb(np.random.rand(10, 12))
    assert out.shape == (10, 12, 3)
    assert np.array_equal(out[:, :, 0], out[:, :, 2])


def test_rgba_alpha_is_dropped():
    from app.models.wbia_orientation import _canonicalize_rgb
    assert _canonicalize_rgb(np.random.rand(10, 12, 4)).shape == (10, 12, 3)


def test_unsupported_channel_count_raises():
    from app.models.wbia_orientation import _canonicalize_rgb, OrientationInferenceError
    with pytest.raises(OrientationInferenceError, match="Unsupported image shape"):
        _canonicalize_rgb(np.random.rand(10, 12, 7))


def test_grayscale_image_end_to_end():
    m = _model()
    gray = _png(np.random.RandomState(1).randint(0, 255, (300, 400), dtype=np.uint8))
    assert math.isfinite(m.predict_batch(gray, [[10, 10, 100, 100]])[0]["theta"])


# ----------------------------------------------------------------- loading

def test_registered_in_model_registry():
    from app.models.model_handler import MODEL_REGISTRY
    assert MODEL_REGISTRY["wbia-orientation"]["class"] == "WbiaOrientationModel"


def test_load_rejects_unknown_config_keys():
    from app.models.wbia_orientation import WbiaOrientationModel
    with pytest.raises(ValueError, match="unknown config key"):
        WbiaOrientationModel().load(model_id="t", checkpoint_path="/x", label_map={0: "a"})


def test_load_requires_a_checkpoint():
    from app.models.wbia_orientation import WbiaOrientationModel
    with pytest.raises(ValueError, match="checkpoint_path is required"):
        WbiaOrientationModel().load(model_id="t")


def test_load_uses_strict_state_dict():
    """A mismatched checkpoint must fail loudly, not silently mispredict."""
    from app.models.wbia_orientation import WbiaOrientationModel
    with patch("torch.load", return_value={"classifier.weight": torch.zeros(5, 2048)}), \
         patch("app.models.wbia_orientation.get_checkpoint_path", side_effect=lambda p: p), \
         patch("timm.create_model") as tm:
        backbone = MagicMock()
        backbone.load_state_dict.side_effect = RuntimeError("size mismatch")
        tm.return_value = backbone
        with pytest.raises(RuntimeError, match="size mismatch"):
            WbiaOrientationModel().load(model_id="t", checkpoint_path="/x")
        assert backbone.load_state_dict.call_args.kwargs.get("strict") is True
