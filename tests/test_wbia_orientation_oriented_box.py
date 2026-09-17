"""The object-aligned box derived from the regressor's coordinates.

Regression cover for the quarter-turn bug: `compute_theta` is faithful to the
reference, but the reference's theta belongs to the OBJECT-ALIGNED box, not to
the detector's axis-aligned crop region. Emitting the reference theta beside the
axis-aligned bbox described two different rectangles -- Wildbook drew the
annotation 90deg off the animal, and `get_chip_from_img` windowed the animal
after rotating it out of that window.

Geometry is per the reference's own `get_object_aligned_box`
(utils/data_manipulation.py:47): "the center point (xc, yc), the side point
(xt, yt) and HALF width w".
"""
import math

import pytest

from app.models.wbia_orientation import compute_theta, oriented_box


def _coords_for(angle_deg, length, width, crop):
    """Normalized [xc, yc, xt, yt, w] for an animal of `length` x `width` lying
    at `angle_deg`, centred in a SQUARE crop (the model's own input space, where
    the normalized frame is isotropic and perpendicularity is preserved)."""
    x1, y1, bw, bh = crop
    r = math.radians(angle_deg)
    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
    tx, ty = cx + length / 2.0 * math.cos(r), cy + length / 2.0 * math.sin(r)
    px, py = tx + width / 2.0 * -math.sin(r), ty + width / 2.0 * math.cos(r)
    nx = lambda X: (X - x1) / bw      # noqa: E731
    ny = lambda Y: (Y - y1) / bh      # noqa: E731
    return [nx(cx), ny(cy), nx(tx), ny(ty),
            math.hypot(nx(px) - nx(tx), ny(py) - ny(ty))]


CROP = [100, 100, 800, 800]


@pytest.mark.parametrize("angle", [-70.0, -40.0, -15.0, 0.0, 13.79, 35.0, 60.0, 80.0])
def test_recovers_the_true_object_aligned_box(angle):
    """Long side, short side and angle all come back, at every orientation."""
    bbox, theta = oriented_box(_coords_for(angle, 560.0, 150.0, CROP), CROP)

    assert bbox is not None
    assert bbox[2] == pytest.approx(560.0, abs=1.0)    # long side
    assert bbox[3] == pytest.approx(150.0, abs=1.0)    # short side
    assert math.degrees(theta) == pytest.approx(angle, abs=0.1)


def test_theta_is_the_long_axis_not_the_upright_rotation():
    """The bug in one assertion: the reference theta is a quarter turn off the
    animal's own axis, which is what the annotation convention stores."""
    coords = _coords_for(13.79, 560.0, 150.0, CROP)
    _, theta = oriented_box(coords, CROP)

    assert math.degrees(theta) == pytest.approx(13.79, abs=0.1)
    assert math.degrees(compute_theta(coords)) == pytest.approx(103.79, abs=0.1)
    assert compute_theta(coords) - theta == pytest.approx(math.pi / 2, abs=1e-6)


def test_box_is_centred_on_the_animal():
    bbox, _ = oriented_box(_coords_for(13.79, 560.0, 150.0, CROP), CROP)

    assert bbox[0] + bbox[2] / 2 == pytest.approx(CROP[0] + CROP[2] / 2, abs=1.0)
    assert bbox[1] + bbox[3] / 2 == pytest.approx(CROP[1] + CROP[3] / 2, abs=1.0)


def test_long_side_is_twice_the_centre_to_side_point_distance():
    """Pins the reference's parametrisation: (xt, yt) is the SIDE point, so it
    is a half-length from the centre, and w is a HALF width."""
    bbox, _ = oriented_box([0.5, 0.5, 0.75, 0.5, 0.1], [0, 0, 400, 400])

    assert bbox[2] == pytest.approx(2 * 0.25 * 400)   # 200
    assert bbox[3] == pytest.approx(2 * 0.1 * 400)    # 80


@pytest.mark.parametrize("coords,crop", [
    ([0.5, 0.5, 0.5, 0.5, 0.1], [0, 0, 400, 400]),   # centre == side point: no axis
    ([0.5, 0.5, 0.75, 0.5, 0.0], [0, 0, 400, 400]),  # zero width
    ([0.5, 0.5, 0.75, 0.5, 0.1], [0, 0, 0, 400]),    # degenerate crop
])
def test_degenerate_predictions_yield_no_box(coords, crop):
    """No axis means no angle. Returning None lets the caller state 'no
    rotation' rather than persist a fabricated one."""
    assert oriented_box(coords, crop) == (None, None)
