"""Per-detection plausibility gate for wbia-orientation (issue #59).

When the regressor is unsure it can predict a near-zero half-width and/or push
the centre to the edge of the crop. oriented_box() faithfully turns that into a
long thin sliver (GrouperSpotter: 8 x 866 px, long/short ratio 108) that is
persisted, cropped and embedded by MiewID. Only an EXACTLY zero side was ever
caught. These tests pin the gate that flags such boxes, one detection at a
time, so the router can fall back for that detection alone.

Every sliver in the issue's sweep had long/short >= 13; the healthy population
is <= ~3. The siblings below are synthetic (their real coords were not kept),
built to the issue's description: ratio 1.1-2.5, on the fish.
"""
import math

import pytest
from shapely.geometry import Polygon

from app.models.wbia_orientation import (
    DEFAULT_MAX_ASPECT,
    DEFAULT_MIN_OVERLAP_FRAC,
    DEFAULT_MIN_SHORT_FRAC,
    oriented_box,
    plausibility_failures,
    validate_plausibility_limits,
)


def _gate(coords, eff, **limits):
    box, theta = oriented_box(coords, eff)
    assert box is not None
    return plausibility_failures(box, theta, eff, **limits)


def _coords(axis_deg, half_len, half_w, xc=0.5, yc=0.5):
    """Normalized [xc, yc, xt, yt, w] for a long axis at `axis_deg`."""
    r = math.radians(axis_deg)
    return [xc, yc, xc + half_len * math.cos(r), yc + half_len * math.sin(r), half_w]


# ------------------------------------------------------------- the defaults

def test_defaults_are_the_issue_values():
    assert DEFAULT_MAX_ASPECT == 6.0
    assert DEFAULT_MIN_SHORT_FRAC == 0.15
    assert DEFAULT_MIN_OVERLAP_FRAC == 0.3


# ----------------------------------------------- the polygon the gate uses

@pytest.mark.parametrize("axis_deg", [-70.0, -15.0, 0.0, 35.0, 80.0, 114.4, 179.0])
@pytest.mark.parametrize("eff", [[0, 0, 400, 400], [1477, 1076, 1206, 579]])
def test_overlap_uses_the_object_polygon_not_its_axis_aligned_bound(axis_deg, eff):
    """The gate rebuilds the object polygon from (oriented_bbox, theta). Check
    that against the corners oriented_box() itself constructs, independently:
    centre +/- half_l along the axis, +/- half_w across it, in image space."""
    from app.models.wbia_orientation import _object_polygon

    coords = _coords(axis_deg, 0.3, 0.08)
    box, theta = oriented_box(coords, eff)

    xc, yc, xt, yt, w = coords
    x1, y1, bw, bh = eff
    n = math.hypot(xt - xc, yt - yc)
    ux, uy = (xt - xc) / n, (yt - yc) / n
    cx, cy = xc * bw + x1, yc * bh + y1
    tx, ty = xt * bw + x1, yt * bh + y1
    wx, wy = (xt - w * uy) * bw + x1, (yt + w * ux) * bh + y1
    half_w = math.hypot(wx - tx, wy - ty)
    ax, ay = (tx - cx), (ty - cy)
    la = math.hypot(ax, ay)
    px, py = -ay / la * half_w, ax / la * half_w
    expected = Polygon([(cx + ax + px, cy + ay + py), (cx + ax - px, cy + ay - py),
                        (cx - ax - px, cy - ay - py), (cx - ax + px, cy - ay + py)])

    got = _object_polygon(box, theta)
    assert got.symmetric_difference(expected).area == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------- the issue's failure shapes

def test_worked_example_e6903062_is_rejected():
    """Goliath grouper, 1206x898 image: centre 4 px from the right edge,
    half_w ~4 px, half_l ~433 px, long axis at 204.4 - 90 = 114.4 deg."""
    eff = [0, 0, 1206, 898]
    cx, cy, half_l, axis = 1202.0, 457.0, 433.0, math.radians(114.4)
    tx, ty = cx + half_l * math.cos(axis), cy + half_l * math.sin(axis)
    coords = [cx / 1206, cy / 898, tx / 1206, ty / 898, 4.0 / 1000]
    box, _ = oriented_box(coords, eff)
    assert max(box[2], box[3]) / min(box[2], box[3]) > 50   # it really is a sliver

    reasons = _gate(coords, eff)
    assert "aspect" in reasons
    assert "short_side" in reasons


def test_band_along_the_bottom_edge_is_rejected():
    """Shape (a): cbb849d0, a band whose centre sits on the bottom edge."""
    eff = [0, 0, 2000, 1125]
    assert "aspect" in _gate(_coords(0.0, 0.48, 0.003, yc=0.995), eff)


def test_thin_line_beside_the_fish_is_rejected():
    """Shape (b): cfdf04b6, a short diagonal line in open water. Ratio 32."""
    eff = [750, 600, 1400, 900]
    assert "aspect" in _gate(_coords(27.0, 0.2, 0.004, xc=0.3, yc=0.4), eff)


def test_near_zero_width_is_rejected():
    assert {"aspect", "short_side"} <= set(_gate(_coords(30.0, 0.3, 0.003), [0, 0, 600, 400]))


def test_ordinary_proportions_mostly_off_the_crop_are_rejected():
    """The sigmoid keeps every coord in [0, 1], so the CENTRE is always inside
    the crop, but the box can still hang mostly outside it. Ratio ~3.3, short
    side well above the floor -- only the overlap check sees this."""
    reasons = _gate([0.99, 0.99, 0.5, 0.99, 0.15], [0, 0, 400, 400])
    assert reasons == ["low_overlap"]


# --------------------------------------------- healthy boxes pass unchanged

HEALTHY = [
    # (coords, eff): ratios 1.1-2.5, centred on the detection, various angles
    (_coords(0.0, 0.30, 0.27), [100, 80, 600, 400]),      # ~1.1
    (_coords(20.0, 0.30, 0.20), [0, 0, 800, 500]),        # ~1.5
    (_coords(-35.0, 0.32, 0.16), [40, 60, 700, 700]),     # ~2.0
    (_coords(45.0, 0.30, 0.12), [0, 0, 500, 500]),        # 2.5, diagonal: corners overhang
    (_coords(100.0, 0.30, 0.14), [300, 10, 400, 900]),    # anisotropic crop
    (_coords(170.0, 0.28, 0.15, xc=0.45, yc=0.55), [0, 0, 1206, 898]),
]


@pytest.mark.parametrize("coords,eff", HEALTHY)
def test_healthy_siblings_pass(coords, eff):
    assert _gate(coords, eff) == []


def test_small_healthy_box_passes():
    """There is deliberately no absolute pixel floor: a centred 12x24 box in a
    16x28 detection is a small animal, not a failure."""
    assert plausibility_failures([2.0, 2.0, 12.0, 24.0], 0.0, [0, 0, 16, 28]) == []


def test_healthy_box_overhanging_the_frame_passes():
    """Rotated boxes legitimately overhang the detection a little."""
    assert plausibility_failures([-10.0, 20.0, 300.0, 150.0], 0.6, [0, 0, 300, 250]) == []


# ------------------------------------------------- unconditional sub-pixel

def test_sub_pixel_is_rejected_even_with_every_limit_disabled():
    """A 0.4 px side rounds to 0, and the router 500s on that. It must never be
    reported as plausible, whatever the config says."""
    reasons = plausibility_failures([10.0, 10.0, 0.4, 0.4], 0.0, [0, 0, 100, 100],
                                    max_aspect=None, min_short_frac=0.0,
                                    min_overlap_frac=0.0)
    assert reasons == ["sub_pixel"]


# --------------------------------------------------- disabling each check

def test_each_check_can_be_disabled():
    coords, eff = _coords(30.0, 0.3, 0.003), [0, 0, 600, 400]
    assert "aspect" not in _gate(coords, eff, max_aspect=None)
    assert "short_side" not in _gate(coords, eff, min_short_frac=0.0)
    assert _gate([0.99, 0.99, 0.5, 0.99, 0.15], [0, 0, 400, 400],
                 min_overlap_frac=0.0) == []


def test_every_failing_check_is_reported_not_just_the_first():
    # Centre in the corner, long axis along x: half_l 196 px, half_w 32 px.
    # Aspect ~6.1, short side 64 < 0.2 * 400, ~27% of its area inside the crop.
    reasons = _gate([0.995, 0.995, 0.505, 0.995, 0.08], [0, 0, 400, 400],
                    min_short_frac=0.2)
    assert reasons == ["aspect", "short_side", "low_overlap"]


# ---------------------------------------------------------- config checks

def test_validate_limits_applies_defaults():
    assert validate_plausibility_limits("m") == {
        "max_aspect": 6.0, "min_short_frac": 0.15, "min_overlap_frac": 0.3}


def test_validate_limits_accepts_null_aspect_and_zero_fracs():
    assert validate_plausibility_limits(
        "m", max_aspect=None, min_short_frac=0, min_overlap_frac=0) == {
        "max_aspect": None, "min_short_frac": 0.0, "min_overlap_frac": 0.0}


@pytest.mark.parametrize("key,value", [
    ("max_aspect", 0.5), ("max_aspect", 0), ("max_aspect", float("inf")),
    ("max_aspect", "6"), ("max_aspect", True),
    ("min_short_frac", -0.1), ("min_short_frac", 1.5), ("min_short_frac", None),
    ("min_overlap_frac", float("nan")), ("min_overlap_frac", 1.01),
])
def test_validate_limits_rejects_bad_values(key, value):
    with pytest.raises(ValueError, match=key):
        validate_plausibility_limits("m", **{key: value})


# ------------------------------------------------- wired into the model

def test_healthy_row_is_marked_ok():
    from tests.test_wbia_orientation import RGB, _model
    r = _model(coords=(0.5, 0.5, 0.8, 0.5, 0.15)).predict_batch(RGB, [[10, 10, 300, 200]])[0]
    assert r["orientation_status"] == "ok"
    assert "orientation_reject_reasons" not in r


def test_implausible_row_is_flagged_not_raised():
    """One bad detection must not fail the request: the model returns the row,
    labelled, with the raw prediction kept for audit."""
    from tests.test_wbia_orientation import RGB, _model
    m = _model(coords=(0.5, 0.5, 0.9, 0.5, 0.003))
    rows = m.predict_batch(RGB, [[10, 10, 300, 200], [50, 40, 200, 150]])

    assert len(rows) == 2
    for r in rows:
        assert r["orientation_status"] == "rejected_implausible"
        assert "aspect" in r["orientation_reject_reasons"]
        assert len(r["oriented_bbox"]) == 4 and len(r["coords_normalized"]) == 5


def test_rejection_is_logged_with_what_is_needed_to_track_it(caplog):
    from tests.test_wbia_orientation import RGB, _model
    m = _model(coords=(0.5, 0.5, 0.9, 0.5, 0.003))
    with caplog.at_level("WARNING", logger="app.models.wbia_orientation"):
        m.predict_batch(RGB, [[10, 10, 300, 200]])
    msg = caplog.text
    assert "'t'" in msg and "aspect" in msg and "[10, 10, 300, 200]" in msg


def test_configured_limits_are_used():
    from tests.test_wbia_orientation import RGB, _model
    m = _model(coords=(0.5, 0.5, 0.9, 0.5, 0.003))
    m.limits = validate_plausibility_limits("t", max_aspect=None, min_short_frac=0)
    r = m.predict_batch(RGB, [[10, 10, 300, 200]])[0]
    assert r["orientation_status"] == "ok"


def test_load_rejects_a_bad_limit_before_touching_the_checkpoint():
    from app.models.wbia_orientation import WbiaOrientationModel
    with pytest.raises(ValueError, match="max_aspect"):
        WbiaOrientationModel().load(model_id="t", checkpoint_path="/nonexistent.pth",
                                    max_aspect=0.5)


def test_load_still_rejects_unknown_keys_and_names_the_new_ones():
    from app.models.wbia_orientation import WbiaOrientationModel
    with pytest.raises(ValueError, match="max_aspect.*min_short_frac.*min_overlap_frac"):
        WbiaOrientationModel().load(model_id="t", checkpoint_path="x", max_ratio=3)


def test_model_info_reports_the_limits():
    from app.models.wbia_orientation import WbiaOrientationModel
    m = WbiaOrientationModel()
    assert m.get_model_info()["plausibility"] == {
        "max_aspect": 6.0, "min_short_frac": 0.15, "min_overlap_frac": 0.3}
