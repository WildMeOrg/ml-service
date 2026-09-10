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
    # Size the output to the incoming batch: predict_batch stacks all crops into
    # ONE tensor, so a fixed (1,5) would fail the shape contract.
    m.model = MagicMock(side_effect=lambda x, *a, **k: logits.repeat(x.shape[0], 1))
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


def test_none_bbox_is_rejected_not_defaulted_to_full_frame():
    """Defaulting a missing bbox to the full frame would silently change the
    region theta describes — the same class of safe-looking default this model
    type exists to remove."""
    from app.models.wbia_orientation import OrientationInferenceError
    m = _model()
    with pytest.raises(OrientationInferenceError, match="bbox is required"):
        m.predict_batch(RGB, [None])
    with pytest.raises(OrientationInferenceError, match="bbox is required"):
        m.predict(RGB)


# --------------------------------------------------------------- fail-closed

def test_non_finite_theta_raises_rather_than_defaulting_to_zero():
    """A 0.0 fallback IS the bug being repaired — and 0.0 is also a legitimate
    prediction, so a sentinel could not be told apart from a real answer."""
    from app.models.wbia_orientation import OrientationInferenceError
    m = _model()
    m.model = MagicMock(side_effect=lambda x, *a, **k:
                        torch.tensor([[float("nan")] * 5]).repeat(x.shape[0], 1))
    with pytest.raises(OrientationInferenceError, match="non-finite"):
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
        WbiaOrientationModel().predict_batch(RGB, [[0, 0, 10, 10]])


def test_undecodable_bytes_raise():
    from app.models.wbia_orientation import OrientationInferenceError
    m = _model()
    with pytest.raises(OrientationInferenceError, match="could not decode"):
        m.predict_batch(b"not an image", [[0, 0, 10, 10]])


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


# ------------------------------- coverage added after implementation review

def test_nan_in_w_is_rejected_even_though_theta_stays_finite():
    """theta reads indices 0-3, so a NaN in w (index 4) yields a FINITE theta and
    would sail out inside coords_normalized. All five outputs must be checked."""
    from app.models.wbia_orientation import OrientationInferenceError
    m = _model()
    m.model = MagicMock(side_effect=lambda x, *a, **k:
                        torch.tensor([[0.0, 0.0, 0.0, 0.0, float("nan")]]).repeat(x.shape[0], 1))
    with pytest.raises(OrientationInferenceError, match="non-finite model output"):
        m.predict_batch(RGB, [[10, 10, 100, 100]])


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4])
def test_non_finite_at_any_coord_index_is_rejected(idx):
    """NaN, not inf: these are pre-sigmoid logits and sigmoid(inf) == 1.0, which
    is perfectly finite. Only NaN propagates through the head."""
    from app.models.wbia_orientation import OrientationInferenceError
    m = _model()
    vals = [0.0] * 5
    vals[idx] = float("nan")
    m.model = MagicMock(side_effect=lambda x, *a, **k:
                        torch.tensor([vals]).repeat(x.shape[0], 1))
    with pytest.raises(OrientationInferenceError, match="non-finite"):
        m.predict_batch(RGB, [[10, 10, 100, 100]])


def test_numpy_scalar_bboxes_are_accepted():
    """Detectors hand us np.float32/np.int64 — Real, but not int/float instances."""
    from app.models.wbia_orientation import resolve_bbox
    assert resolve_bbox([np.int64(10), np.int64(10), np.int64(50), np.int64(50)],
                        400, 300) == [10, 10, 50, 50]
    assert resolve_bbox([np.float32(10.9), np.float32(10.0),
                         np.float32(50.0), np.float32(50.0)], 400, 300) == [10, 10, 50, 50]


def test_bool_bbox_values_are_rejected():
    """bool is an int subclass; a coordinate of True is a bug, not a 1."""
    from app.models.wbia_orientation import resolve_bbox, OrientationInferenceError
    with pytest.raises(OrientationInferenceError, match="non-numeric"):
        resolve_bbox([True, 0, 10, 10], 400, 300)


@pytest.mark.parametrize("bbox", [
    [10.7, 10.2, 50.9, 50.9],
    [-0.9, -0.9, 500.5, 400.5],
    [379.6, 10.1, 50.8, 50.2],
])
def test_fractional_bboxes_agree_with_actual_numpy_slice(bbox):
    """Fractional w/h too — grounded against NumPy after the same int() rule."""
    from app.models.wbia_orientation import resolve_bbox
    img = np.zeros((300, 400, 3))
    x, y, w, h = (int(v) for v in bbox)
    actual = img[y:y + h, x:x + w]
    eff = resolve_bbox(bbox, 400, 300)
    assert (eff[2], eff[3]) == (actual.shape[1], actual.shape[0])


def test_predict_batch_runs_three_forwards_for_the_whole_image_not_per_bbox():
    """Batched: 3 TTA forwards TOTAL for N bboxes, not 3N."""
    m = _model(hflip=True, vflip=True)
    m.model.reset_mock()
    m.predict_batch(RGB, [[10, 10, 100, 100], [50, 50, 60, 60], [0, 0, 400, 300]])
    assert m.model.call_count == 3, "expected base+hflip+vflip on ONE stacked batch"


def test_batched_rows_stay_aligned_with_their_bboxes():
    """A misaligned row would attach one bbox's theta to another's crop."""
    from app.models.wbia_orientation import WbiaOrientationModel, compute_theta
    m = WbiaOrientationModel()
    m.model_id, m.device, m.imsize = "t", "cpu", (224, 224)
    m.hflip = m.vflip = False
    rows = torch.tensor([[0.5, 0.5, 1.0, 0.5, 0.1],      # -> 90 deg
                         [0.5, 0.5, 0.5, 0.0, 0.1],      # -> 0 deg
                         [0.5, 0.5, 0.0, 0.5, 0.1]])     # -> 270 deg
    logits = torch.log(rows.clamp(1e-9, 1 - 1e-9) / (1 - rows.clamp(1e-9, 1 - 1e-9)))
    m.model = MagicMock(side_effect=lambda x, *a, **k: logits)
    out = m.predict_batch(RGB, [[0, 0, 100, 100], [10, 10, 50, 50], [20, 20, 30, 30]])
    for r, expected in zip(out, rows.tolist()):
        assert r["theta"] == pytest.approx(compute_theta(expected), abs=1e-6)


def test_empty_bbox_list_returns_empty_without_inference():
    m = _model()
    m.model.reset_mock()
    assert m.predict_batch(RGB, []) == []
    assert m.model.call_count == 0


def test_preprocess_produces_the_reference_tensor_shape_and_dtype():
    """Pins the preprocessing contract the mocked backbone cannot: 224x224,
    float32 (the reference's .float() after skimage's float64), 3 channels."""
    m = _model()
    img = np.random.RandomState(2).rand(300, 400, 3)
    x = m._preprocess(img, [10, 10, 100, 100])
    assert x.shape == (1, 3, 224, 224)
    assert x.dtype == torch.float32


def test_preprocess_normalizes_with_imagenet_statistics():
    """A constant image maps to (value-mean)/std per channel — catches a dropped
    or altered Normalize."""
    m = _model()
    img = np.full((300, 400, 3), 0.485, dtype=np.float64)
    x = m._preprocess(img, [0, 0, 400, 300])
    assert x[0, 0].mean().item() == pytest.approx(0.0, abs=1e-4)


def test_preprocess_degenerate_crop_uses_the_full_image():
    """animal_wbia.py:25-28 — and the result must differ from a real crop."""
    m = _model()
    img = np.random.RandomState(3).rand(300, 400, 3)
    full = m._preprocess(img, [0, 0, 400, 300])
    degenerate = m._preprocess(img, [0, 0, 0, 0])       # empty -> full image
    assert torch.allclose(full, degenerate)


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


def test_sigmoid_of_inf_is_finite_so_only_nan_survives_the_head():
    """Documents why the NaN tests use NaN: an inf logit saturates to 1.0."""
    assert torch.sigmoid(torch.tensor([float("inf")])).item() == 1.0
    assert math.isnan(torch.sigmoid(torch.tensor([float("nan")])).item())


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
