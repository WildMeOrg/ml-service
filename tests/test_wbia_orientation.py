"""Tests for WbiaOrientationModel — the wbia-plugin-orientation theta port.

This port's failure mode is being SILENTLY wrong by an angle, which downstream is
indistinguishable from correct output — the same class of bug as issue #33. So
these tests pin the reference algorithm's arithmetic exactly, plus every failure
mode surfaced in design review:

  - fail-closed (a 0.0 fallback IS the bug being repaired; 0.0 is also valid)
  - boxes are clipped to the frame as WBIA's localizations step did; without it
    NumPy's non-clamping slice crops the FAR EDGE for a negative origin
  - integerization rule (int() truncation, then the clip)
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
#
# WBIA never let the orientation plugin see an overhanging box: Wildbook's
# lightnet detections went through the `localizations` table, which clips every
# box to the frame (wbia/core_images.py:1539-1557, boundary=True by default),
# and the plugin then read those stored, clipped boxes. The plugin's raw
# `image[y1:y1+h, x1:x1+w]` slice would resolve a negative origin from the FAR
# edge, but in WBIA that was unreachable. Reproducing the slice WITHOUT the clip
# turned a whale shark overhanging the top by 28 px into a crop of the bottom
# 28 rows (Sharkbook, 2026-11-25), which became a persisted sliver.


# The preflight gate's independent transcription of core_images.py:1539-1557,
# reused here rather than written a third time.
from scripts.preflight.run_gate import wbia_localization_boundary as _wbia_boundary


@pytest.mark.parametrize("bbox,expected", [
    ([10, 10, 50, 50], [10, 10, 50, 50]),        # ordinary
    ([-20, 10, 50, 50], [0, 10, 30, 50]),        # narrow negative -> clipped, not empty
    ([-20, 10, 500, 50], [0, 10, 400, 50]),      # WIDE negative -> clipped, NOT the far edge
    ([380, 10, 50, 50], [380, 10, 20, 50]),      # overrun -> truncated
    ([500, 400, 50, 50], [400, 300, 0, 0]),      # fully outside -> empty (full-frame fallback)
    ([-60, 10, 50, 50], [0, 10, 0, 50]),         # entirely left of frame -> empty
])
def test_resolve_bbox_clips_to_the_frame_like_wbia(bbox, expected):
    from app.models.wbia_orientation import resolve_bbox
    assert resolve_bbox(bbox, width=400, height=300) == expected


def test_sharkbook_guam2_overhang_is_the_whole_shark_not_the_bottom_strip():
    """whaleshark_v0 on a 2860x1464 image: [22, -28, 2839, 1492]. Unclipped, the
    -28 resolved from the bottom edge -> [22, 1436, 2838, 28]."""
    from app.models.wbia_orientation import resolve_bbox
    assert resolve_bbox([22, -28, 2839, 1492], 2860, 1464) == [22, 0, 2838, 1464]


@pytest.mark.parametrize("bbox", [
    [10, 10, 50, 50], [-20, 10, 500, 50], [380, 10, 50, 50], [500, 400, 50, 50],
    [-20, 10, 50, 50], [-5, -7, 420, 320], [100, -40, 30, 20],
])
def test_resolve_bbox_crops_what_wbia_cropped(bbox):
    """Grounded against WBIA's clip followed by the plugin's NumPy slice: same
    origin, same pixels."""
    from app.models.wbia_orientation import resolve_bbox
    img = np.arange(300 * 400).reshape(300, 400)
    x, y, w, h = _wbia_boundary(bbox, 400, 300)
    expected = img[y:y + h, x:x + w]
    ex, ey, ew, eh = resolve_bbox(bbox, 400, 300)
    got = img[ey:ey + eh, ex:ex + ew]
    assert got.shape == expected.shape and np.array_equal(got, expected), bbox


def test_fractional_overrun_keeps_its_last_pixel():
    """Integerize FIRST, then clip: x=399.6, w=2 is int [399, 2] -> 1 px wide.
    Clipping the floats first would give 0.4 -> int 0 -> a full-frame fallback."""
    from app.models.wbia_orientation import resolve_bbox
    assert resolve_bbox([399.6, 10, 2, 50], 400, 300) == [399, 10, 1, 50]


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


@pytest.mark.parametrize("bbox,expected", [
    ([-0.5, 10, 50, 50], [0, 10, 50, 50]),     # int(-0.5) == 0   (truncate toward zero)
    ([10.9, 10, 50, 50], [10, 10, 50, 50]),    # int(10.9) == 10
    ([-1.9, 10, 50, 50], [0, 10, 49, 50]),     # int(-1.9) == -1, then clipped to 0
])
def test_integerization_truncates_toward_zero_then_clips(bbox, expected):
    """int(), matching pipeline_router's detector-box rule, THEN the WBIA clip."""
    from app.models.wbia_orientation import resolve_bbox
    assert resolve_bbox(bbox, 400, 300) == expected


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
    assert set(r) == {"model_id", "theta", "theta_oriented", "oriented_bbox",
                      "coords_normalized", "effective_bbox", "orientation_status"}
    for forbidden in ("label", "probability", "class_id", "class", "predictions"):
        assert forbidden not in r
    assert math.isfinite(r["theta"])
    assert len(r["coords_normalized"]) == 5
    assert math.isfinite(r["theta_oriented"])
    assert len(r["oriented_bbox"]) == 4


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


def test_predict_batch_crops_an_overhanging_box_inside_the_frame():
    """The production regression at the model boundary: a detector box hanging
    28 px over the top must not crop the bottom rows. Checked on the PIXELS
    reaching the backbone, not just on the reported box."""
    m = _model()
    rows = m.predict_batch(RGB, [[22, -28, 350, 356],   # overhangs the top (RGB is 400x300)
                                 [22, 0, 350, 300],     # the clipped box itself
                                 [22, 272, 350, 28]])   # the old far-edge strip
    assert rows[0]["effective_bbox"] == [22, 0, 350, 300]
    batch = m.model.call_args_list[0].args[0]           # base (non-TTA) forward
    assert torch.equal(batch[0], batch[1])
    assert not torch.equal(batch[0], batch[2])


# ------------------------------------------------- EXIF orientation (frame)
#
# The detector (lightnet: cv2.imdecode) and MiewID (decode_image_rgb) work in
# the UPRIGHT frame, EXIF Orientation applied; so are the boxes the orienter is
# handed. imageio.imread without rotate=True returns the stored SENSOR frame,
# so for a portrait phone photo the orienter cropped the box out of a sideways
# image. WBIA rewrote EXIF-rotated images upright on import
# (wbia/algo/preproc/preproc_image.py:193-201), so the plugin it ran read
# upright pixels.

from tests.test_exif_orientation_decode import (  # noqa: E402
    ORACLE, _jpeg_with_orientation, _sensor_pixels)


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_orienter_crops_from_the_upright_frame(orientation):
    """The tensor reaching the backbone for a tagged JPEG equals the one for the
    same decoded pixels stored upright, untagged (lossless PNG)."""
    sensor = _sensor_pixels()                                    # 120 x 200
    tagged = _jpeg_with_orientation(sensor, orientation)
    decoded = np.array(Image.open(io.BytesIO(tagged)).convert("RGB"))
    upright = _png(np.ascontiguousarray(ORACLE[orientation](decoded)))
    h, w = ORACLE[orientation](decoded).shape[:2]
    bbox = [w // 4, h // 2, w // 2, h // 3]                      # valid in the upright frame

    m = _model()
    m.predict_batch(tagged, [bbox])
    m.predict_batch(upright, [bbox])
    a, b = (c.args[0] for c in m.model.call_args_list[:2])
    assert torch.equal(a, b)


def test_a_box_valid_only_upright_is_not_clipped_to_the_sensor_frame():
    """Portrait (EXIF 6): upright frame is 120 wide x 200 tall. A box at y=150
    exists only upright; in the sensor frame (200 x 120) it would be clipped
    away and fall back to the full frame."""
    tagged = _jpeg_with_orientation(_sensor_pixels(), 6)
    r = _model().predict_batch(tagged, [[10, 150, 80, 40]])[0]
    assert r["effective_bbox"] == [10, 150, 80, 40]


def test_sixteen_bit_png_is_not_flattened_to_eight_bits():
    """rotate=True must not change untagged decoding: an I;16 gradient keeps
    its tonal range rather than clipping at 255."""
    from app.models.wbia_orientation import _canonicalize_rgb
    import imageio.v2 as imageio
    g = (np.arange(60 * 80).reshape(60, 80) * 13).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(g).save(buf, format="PNG")
    m = _model()
    m.predict_batch(buf.getvalue(), [[0, 0, 80, 60]])
    seen = m.model.call_args_list[0].args[0]
    raw = _canonicalize_rgb(np.asarray(imageio.imread(io.BytesIO(buf.getvalue()))))
    assert raw.dtype == np.uint16 and raw.max() > 255
    assert torch.equal(seen, m._preprocess(raw, [0, 0, 80, 60]))
