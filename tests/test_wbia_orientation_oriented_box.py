"""`oriented_box` must reproduce WBIA's own persisted geometry, not invent one.

The authoritative reference is `wbia/core_annots.py:2467`, which builds the four
corners of the object-aligned box, de-rotates them by -theta about the centre, and
persists the axis-aligned bound:

    predicted_verts  = get_object_aligned_box(xc, yc, xt, yt, w)
    calculated_theta = arctan2(yt - yc, xt - xc) + deg2rad(90)
    predicted_rot    = rotation_around_mat3x3(calculated_theta * -1.0, xc, yc)
    aligned_verts    = transform_points_with_homography(predicted_rot, verts)
    predicted_bbox   = bboxes_from_vert_list([aligned_verts])[0]

These tests assert against an INDEPENDENT transcription of that algorithm rather
than hand-computed numbers, so they fail if the port drifts from the reference.

History: a previous revision emitted (w=long, h=short, theta=long-axis) -- it
dropped the reference's +90 and chose its own width/height assignment. That draws
the SAME rectangle, so every box-geometry check passed, but it rotates the MiewID
crop a quarter turn and embeddings stopped matching catalogs built through WBIA.
"""
import math

import numpy as np
import pytest

from app.models.wbia_orientation import compute_theta, oriented_box


# --------------------------------------------------------------------------
# Independent transcription of the reference (data_manipulation.py + core_annots)
# --------------------------------------------------------------------------
def _perp(p0, p1, dist):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    return p1 + dist * np.array([-u[1], u[0]]), p1 + dist * np.array([u[1], -u[0]])


def _along(p0, p1, dist):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    return p0 + dist * (p1 - p0) / np.linalg.norm(p1 - p0)


def reference_persisted(xc, yc, xt, yt, w):
    """wbia/core_annots.py:2467, transcribed. All inputs in IMAGE space."""
    c1, c2 = _perp([xc, yc], [xt, yt], w)
    dist = np.linalg.norm([xc - xt, yc - yt])
    t2 = _along([xt, yt], [xc, yc], 2 * dist)
    c3, c4 = _perp([xc, yc], list(t2), w)
    verts = np.array([c1, c2, c3, c4])
    theta = math.atan2(yt - yc, xt - xc) + math.radians(90)
    c, s = math.cos(-theta), math.sin(-theta)
    rot = (verts - [xc, yc]) @ np.array([[c, -s], [s, c]]).T + [xc, yc]
    x0, y0 = rot.min(axis=0)
    x1, y1 = rot.max(axis=0)
    return [x0, y0, x1 - x0, y1 - y0], theta


def to_image_space(coords, crop):
    """Mirror oriented_box's rescale so the reference receives the same inputs."""
    xc, yc, xt, yt, w = coords
    x1, y1, bw, bh = crop
    ux, uy = xt - xc, yt - yc
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    xw, yw = xt + w * -uy, yt + w * ux
    CX, CY = xc * bw + x1, yc * bh + y1
    TX, TY = xt * bw + x1, yt * bh + y1
    WX, WY = xw * bw + x1, yw * bh + y1
    return CX, CY, TX, TY, math.hypot(WX - TX, WY - TY)


SQUARE = [0, 0, 400, 400]
# A real, strongly anisotropic crop: the beluga that surfaced this bug.
ANISO = [1477, 1076, 1206, 579]


def _coords(angle_deg, half_len_frac=0.25, half_w_frac=0.10):
    r = math.radians(angle_deg)
    return [0.5, 0.5,
            0.5 + half_len_frac * math.cos(r),
            0.5 + half_len_frac * math.sin(r),
            half_w_frac]


@pytest.mark.parametrize("angle", [-70.0, -40.0, -15.0, 0.0, 13.79, 35.0, 60.0, 80.0, 179.0])
@pytest.mark.parametrize("crop", [SQUARE, ANISO], ids=["square", "anisotropic"])
def test_matches_wbia_reference(angle, crop):
    """Every output identical to the reference construction, to 1e-6."""
    coords = _coords(angle)
    got_box, got_theta = oriented_box(coords, crop)
    exp_box, exp_theta = reference_persisted(*to_image_space(coords, crop))

    assert got_theta == pytest.approx(exp_theta, abs=1e-12)
    for g, e in zip(got_box, exp_box):
        assert g == pytest.approx(e, abs=1e-6)


def test_theta_carries_the_reference_plus_90():
    """The +90 is the convention every WBIA-built catalog was stored in. Dropping
    it draws the same rectangle but rotates the MiewID crop 90 degrees."""
    coords = _coords(13.79)
    _, theta = oriented_box(coords, SQUARE)
    axis = math.atan2(coords[3] - coords[1], coords[2] - coords[0])

    assert math.degrees(theta - axis) == pytest.approx(90.0, abs=1e-9)


def test_plus_90_is_applied_in_image_space_not_normalized():
    """An angle is NOT preserved by anisotropic scaling, so the +90 must be added
    AFTER the rescale. On a 4:1 crop the two differ substantially."""
    coords = _coords(-45.0)
    crop = [0, 0, 400, 100]
    _, theta = oriented_box(coords, crop)

    normalized_then_90 = math.atan2(coords[3] - coords[1],
                                    coords[2] - coords[0]) + math.pi / 2
    assert abs(math.degrees(theta - normalized_then_90)) > 5.0
    _, exp_theta = reference_persisted(*to_image_space(coords, crop))
    assert theta == pytest.approx(exp_theta, abs=1e-12)


def test_width_is_not_forced_to_be_the_shorter_side():
    """The reference never guarantees width < height -- it is whatever the
    de-rotated bound gives. Sorting the dimensions would be a fresh bug."""
    seen_wider = False
    rs = np.random.RandomState(0)
    for _ in range(40):
        coords = [0.5, 0.5, rs.uniform(.15, .85), rs.uniform(.15, .85), rs.uniform(.04, .30)]
        crop = [0, 0, int(rs.uniform(200, 1500)), int(rs.uniform(200, 1500))]
        box, _ = oriented_box(coords, crop)
        exp, _ = reference_persisted(*to_image_space(coords, crop))
        for g, e in zip(box, exp):
            assert g == pytest.approx(e, abs=1e-6)
        if box[2] > box[3]:
            seen_wider = True
    assert seen_wider, "expected at least one w>h case; a sort would suppress them"


def test_compute_theta_is_left_faithful_to_upstream():
    """compute_theta stays byte-faithful so the host preflight keeps comparing
    like with like; oriented_box does its own image-space work."""
    assert math.degrees(compute_theta(_coords(13.79))) == pytest.approx(103.79, abs=0.01)


@pytest.mark.parametrize("coords,crop", [
    ([0.5, 0.5, 0.5, 0.5, 0.1], [0, 0, 400, 400]),   # centre == side point: no axis
    ([0.5, 0.5, 0.75, 0.5, 0.0], [0, 0, 400, 400]),  # zero width
    ([0.5, 0.5, 0.75, 0.5, 0.1], [0, 0, 0, 400]),    # degenerate crop
])
def test_degenerate_predictions_yield_no_box(coords, crop):
    """No axis means no angle. Returning None lets the caller fail closed rather
    than persist a fabricated orientation."""
    assert oriented_box(coords, crop) == (None, None)
