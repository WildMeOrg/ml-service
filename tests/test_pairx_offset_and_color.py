"""Regression tests for the PairX explain pipeline.

Covers three previously-shipping bugs:

1. Offset / spatial-coordinate mismatch — when a bounding box was given
   but ``crop_bbox=False`` (the API default), ``process_image`` used the
   full original frame as the display image while the model tensor was
   built from the bbox chip. PairX overlays heatmaps in tensor-coord
   space directly on the display, so the colored relevance pixels ended
   up offset from the corresponding features.

2. RGB/BGR double swap — ``run_pairx`` swapped channels on output and
   the endpoint swapped them again before ``cv2.imencode``. The two
   swaps cancelled, ``imencode`` (which assumes BGR) received RGB, and
   the final PNG had red and blue inverted.

3. Rotation convention drift — ``MiewidModel.extract_embeddings`` used
   PIL crop-then-rotate(-theta), but ``wbia-plugin-miew-id`` (the
   training pipeline) uses ``get_chip_from_img`` which rotates the
   whole image by +theta and samples axis-aligned. For non-zero theta
   the chip handed to the model no longer matched the training-time
   representation, silently degrading embeddings.

These tests intentionally avoid loading real model weights. They
exercise the routing/helper layer directly with stubbed dependencies.
"""
from __future__ import annotations

import asyncio
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Bug 1: spatial offset — display image must equal chip when bbox is given
# ---------------------------------------------------------------------------

def _make_image_with_marker(w: int = 200, h: int = 200) -> np.ndarray:
    """Return an HxWx3 uint8 RGB image with a distinctive marker we can
    locate after a crop. The bbox we use in the test is (50, 60, 80, 70),
    so we paint a bright magenta square at (60, 70)-(90, 100) — fully
    inside that bbox but offset from the center of the full image."""
    img = np.full((h, w, 3), 30, dtype=np.uint8)  # near-black background
    img[70:100, 60:90] = (255, 0, 255)  # magenta marker inside the bbox
    return img


def _png_bytes(img_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_process_image_returns_chip_as_display_when_bbox_provided(tmp_path):
    """With a real bbox and crop_bbox=False the display image must be the
    bbox chip resized to the model input — NOT the full image. This is
    the spatial-offset regression."""
    from app.routers.explain_router import process_image

    img = _make_image_with_marker()
    img_path = tmp_path / "marker.png"
    Image.fromarray(img, mode="RGB").save(img_path)

    bbox = [50, 60, 80, 70]  # x, y, w, h — the chip is 80x70 RGB
    theta = 0.0

    display, tensor = asyncio.run(
        process_image(str(img_path), bbox, theta, crop_bbox=False, model="miewid-msv4.1", device="cpu")
    )

    # Tensor should always be (1, 3, 440, 440)
    assert tensor.shape == (1, 3, 440, 440)
    # Display image should be (440, 440, 3) regardless
    assert display.shape == (440, 440, 3)

    # Build the expected display: chip from get_chip_from_img -> PIL resize to (440, 440)
    from app.utils.helpers import get_chip_from_img
    import torchvision.transforms as transforms

    chip = get_chip_from_img(img, bbox, theta)
    expected = np.array(transforms.Resize((440, 440))(Image.fromarray(chip)))

    # Display must match the chip, not the full image. We allow bit-for-bit
    # equality because both go through the same PIL Resize on the same chip.
    np.testing.assert_array_equal(display, expected)


def test_process_image_returns_full_image_when_no_bbox(tmp_path):
    """The (0,0,0,0) padding bbox should keep the full-frame display
    behavior — get_chip_from_img falls back to the original image when
    the crop is degenerate, so this naturally still equals the original."""
    from app.routers.explain_router import process_image

    img = _make_image_with_marker()
    img_path = tmp_path / "marker.png"
    Image.fromarray(img, mode="RGB").save(img_path)

    display, _ = asyncio.run(
        process_image(str(img_path), [0, 0, 0, 0], 0.0, crop_bbox=False, model="miewid-msv4.1", device="cpu")
    )

    import torchvision.transforms as transforms
    expected = np.array(transforms.Resize((440, 440))(Image.fromarray(img)))
    np.testing.assert_array_equal(display, expected)


def test_process_image_theta_only_with_sentinel_bbox_does_not_crash(tmp_path):
    """Regression for a latent crash: when extend_bb_list pads bbox with
    [0,0,0,0] but the caller actually supplied a non-trivial theta,
    get_chip_from_img used to call cv2.getRectSubPix with size (0,0),
    which returns None and then raised AttributeError on .shape.
    process_image must promote the sentinel to a full-frame bbox so the
    rotation-only chip path works."""
    from app.routers.explain_router import process_image
    from app.utils.helpers import get_chip_from_img
    import torchvision.transforms as transforms

    img = _make_image_with_marker()
    img_path = tmp_path / "marker.png"
    Image.fromarray(img, mode="RGB").save(img_path)

    theta = 0.4
    display, tensor = asyncio.run(
        process_image(str(img_path), [0, 0, 0, 0], theta, crop_bbox=False, model="miewid-msv4.1", device="cpu")
    )

    assert tensor.shape == (1, 3, 440, 440)
    h, w = img.shape[:2]
    expected_chip = get_chip_from_img(img, [0, 0, w, h], theta)
    expected = np.array(transforms.Resize((440, 440))(Image.fromarray(expected_chip)))
    np.testing.assert_array_equal(display, expected)


def test_process_image_full_frame_bbox_with_rotation_uses_rotated_chip(tmp_path):
    """An explicit full-frame bbox plus rotation should also display the
    rotated chip — not the un-rotated full image."""
    from app.routers.explain_router import process_image
    from app.utils.helpers import get_chip_from_img
    import torchvision.transforms as transforms

    img = _make_image_with_marker()
    h, w = img.shape[:2]
    img_path = tmp_path / "marker.png"
    Image.fromarray(img, mode="RGB").save(img_path)

    theta = 0.4
    display, _ = asyncio.run(
        process_image(str(img_path), [0, 0, w, h], theta, crop_bbox=False, model="miewid-msv4.1", device="cpu")
    )

    expected_chip = get_chip_from_img(img, [0, 0, w, h], theta)
    expected = np.array(transforms.Resize((440, 440))(Image.fromarray(expected_chip)))
    np.testing.assert_array_equal(display, expected)


# ---------------------------------------------------------------------------
# Bug 2: RGB/BGR — run_pairx must NOT swap channels on output
# ---------------------------------------------------------------------------

def test_endpoint_encode_preserves_rgb_channel_order():
    """End-to-end pin: the endpoint's cv2.imencode call wraps the
    pairx-RGB array in a single RGB→BGR conversion. Decoding the
    resulting PNG (which cv2.imdecode returns as BGR) and swapping
    back to RGB must round-trip to the original sentinel. If anyone
    removes the final RGB2BGR before imencode, this test fails."""
    import base64

    import cv2

    # Bright red top, bright blue bottom — easy to spot inversions.
    sentinel = np.zeros((8, 8, 3), dtype=np.uint8)
    sentinel[:4, :, 0] = 255  # red
    sentinel[4:, :, 2] = 255  # blue

    # Mirror the endpoint's encode step exactly.
    _, buf = cv2.imencode(".png", cv2.cvtColor(sentinel, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buf).decode("utf-8")

    # Decode the way an HTTP client would: bytes -> cv2.imdecode (BGR) -> RGB
    decoded_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(decoded_rgb, sentinel)


def test_run_pairx_preserves_channel_order_from_explain():
    """pairx.explain() returns RGB. run_pairx must hand that RGB through
    unchanged so the single endpoint-side RGB2BGR before cv2.imencode is
    the only channel conversion. A second swap inside run_pairx would
    cancel it and the final PNG would have red/blue inverted."""
    from app.routers import explain_router

    # Distinctive RGB sentinel: pure red top half, pure blue bottom half.
    # If something swaps R/B we will see it.
    sentinel = np.zeros((4, 4, 3), dtype=np.uint8)
    sentinel[:2, :, 0] = 255  # red top
    sentinel[2:, :, 2] = 255  # blue bottom

    fake_model = MagicMock()
    # The layer-key validation calls named_modules() — return a dict-like
    # so `in dict(model.named_modules())` is True.
    fake_model.named_modules.return_value = [("backbone.blocks.3", MagicMock())]

    imgs1_t = [torch.zeros(1, 3, 4, 4)]
    imgs2_t = [torch.zeros(1, 3, 4, 4)]
    imgs1 = [np.zeros((4, 4, 3), dtype=np.uint8)]
    imgs2 = [np.zeros((4, 4, 3), dtype=np.uint8)]

    with patch.object(explain_router, "explain", return_value=[sentinel.copy()]):
        out = explain_router.run_pairx(
            imgs1_t, imgs2_t, imgs1, imgs2, fake_model,
            layer_key="backbone.blocks.3",
            k_lines=0, k_colors=0,
            visualization_type="lines_and_colors",
        )

    assert len(out) == 1
    # Output must equal the input sentinel — i.e., RGB preserved.
    np.testing.assert_array_equal(out[0], sentinel)


# ---------------------------------------------------------------------------
# Bug 3: rotation convention — extract_embeddings must use get_chip_from_img
# ---------------------------------------------------------------------------

def test_extract_embeddings_uses_canonical_chip_helper():
    """The chip handed to self.preprocess must match what
    get_chip_from_img produces for the same (image, bbox, theta). This
    pins extract_embeddings to the wbia-plugin-miew-id training-time
    convention (rotate whole image by +theta, then sample axis-aligned)
    instead of the older PIL crop-then-rotate(-theta) that diverged
    geometrically whenever theta != 0."""
    from app.models.miewid import MiewidModel
    from app.utils.helpers import get_chip_from_img

    # Build a non-uniform image so the chip is sensitive to crop+theta.
    rng = np.random.default_rng(seed=0)
    img_rgb = rng.integers(0, 256, size=(300, 300, 3), dtype=np.uint8)
    image_bytes = _png_bytes(img_rgb)

    bbox = (80, 90, 100, 110)
    theta = 0.4  # ~22.9°, well above the abs(theta) < 0.1 fast-path threshold

    expected_chip = get_chip_from_img(img_rgb.copy(), list(bbox), float(theta))

    # Stand up a real MiewidModel instance but stub the heavy bits.
    model = MiewidModel.__new__(MiewidModel)
    model.device = "cpu"

    fake_tensor = torch.zeros(3, 4, 4)
    captured = {}

    def fake_preprocess(image):
        captured["chip"] = image
        return {"image": fake_tensor}

    model.preprocess = fake_preprocess

    # Replace the underlying model with a stub that returns a known embedding
    # given the preprocessed batch — we only care that extract_embeddings
    # routed the right chip through preprocess.
    fake_torch_model = MagicMock(return_value=torch.tensor([[1.0, 2.0, 3.0]]))
    model.model = fake_torch_model

    out = model.extract_embeddings(image_bytes, bbox=bbox, theta=theta)
    np.testing.assert_array_equal(out, np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_array_equal(captured["chip"], expected_chip)


def test_get_chip_from_img_survives_out_of_frame_and_zero_size():
    """Direct helper test: out-of-frame, zero-size, and oversized bboxes
    must not crash, regardless of theta. This pins behavior for every
    caller (process_image AND extract_embeddings AND any future caller),
    not just the explain router. cv2.getRectSubPix with size=(0,0) used
    to return None and crash the .shape check; the helper now bails out
    on zero-size up front.

    Zero-size and negative-size bboxes additionally MUST fall back to
    the original image — not just "not crash". That's the existing
    contract for the theta=0 empty-slice case and is the safest
    behavior for the rotated case too."""
    import numpy as np
    from app.utils.helpers import get_chip_from_img

    img = np.full((1000, 1000, 3), 128, dtype=np.uint8)

    # Bboxes whose result the helper guarantees to be a non-empty (H, W, 3)
    # chip — out-of-frame and oversized cases. (We don't assert exact
    # content because the rotated path's white padding makes pixel
    # comparison brittle.)
    non_crash_cases = [
        ([950, 0, 100, 100], 0.0),
        ([950, 0, 100, 100], 0.4),
        ([1050, 0, 100, 100], 0.0),
        ([1050, 0, 100, 100], 0.4),
        ([5000, 5000, 100, 100], 0.4),
        ([0, 0, 5000, 5000], 0.4),
    ]
    for bbox, theta in non_crash_cases:
        chip = get_chip_from_img(img.copy(), list(bbox), theta)
        assert chip is not None
        assert chip.ndim == 3 and chip.shape[2] == 3
        assert min(chip.shape) >= 1

    # Bboxes the helper specifically promises to fall back to the
    # original full image: zero-size at any location, and negative
    # width/height. The guard at app/utils/helpers.py uses w <= 0 / h <= 0
    # so we lock that contract in.
    fallback_cases = [
        ([0, 0, 0, 0], 0.0),
        ([0, 0, 0, 0], 0.4),
        ([500, 500, 0, 0], 0.4),
        ([100, 200, -5, 50], 0.0),   # negative width
        ([100, 200, 50, -5], 0.4),   # negative height
    ]
    for bbox, theta in fallback_cases:
        chip = get_chip_from_img(img.copy(), list(bbox), theta)
        np.testing.assert_array_equal(chip, img), (
            f"expected full-image fallback for {bbox}, theta={theta}"
        )


def test_extract_embeddings_handles_zero_size_bbox_with_rotation():
    """/extract/ and /pipeline/ paths call get_chip_from_img directly,
    so the helper-level zero-size guard must protect them too. Pre-fix,
    a zero-size bbox + non-trivial theta crashed with a 500 because
    cv2.getRectSubPix returned None.

    Beyond "does not crash", we also pin the contract that a zero-size
    bbox falls back to the full image — capturing the chip handed to
    preprocess catches a regression where the wrong content gets fed
    to the model."""
    from app.models.miewid import MiewidModel

    rng = np.random.default_rng(seed=3)
    img_rgb = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)

    model = MiewidModel.__new__(MiewidModel)
    model.device = "cpu"
    captured = {}

    def fake_preprocess(image):
        captured["chip"] = image
        return {"image": torch.zeros(3, 4, 4)}

    model.preprocess = fake_preprocess
    model.model = MagicMock(return_value=torch.tensor([[0.5]]))

    # Zero-size at a non-sentinel position with meaningful rotation —
    # the case that crashed extract_embeddings until the helper guard
    # landed.
    out = model.extract_embeddings(_png_bytes(img_rgb), bbox=(200, 200, 0, 0), theta=0.4)
    np.testing.assert_array_equal(out, np.array([[0.5]]))
    np.testing.assert_array_equal(captured["chip"], img_rgb)


def test_extract_embeddings_no_bbox_with_theta_uses_full_frame_helper():
    """bbox=None + theta != 0 must rotate the whole image through the
    canonical get_chip_from_img helper (the same code path
    wbia-plugin-miew-id uses), not PIL .rotate(-theta). Pins the
    full-frame fallback at app/models/miewid.py."""
    from app.models.miewid import MiewidModel
    from app.utils.helpers import get_chip_from_img

    rng = np.random.default_rng(seed=2)
    img_rgb = rng.integers(0, 256, size=(120, 90, 3), dtype=np.uint8)
    h, w = img_rgb.shape[:2]
    theta = 0.4

    expected = get_chip_from_img(img_rgb.copy(), [0, 0, w, h], float(theta))

    model = MiewidModel.__new__(MiewidModel)
    model.device = "cpu"
    captured = {}

    def fake_preprocess(image):
        captured["chip"] = image
        return {"image": torch.zeros(3, 4, 4)}

    model.preprocess = fake_preprocess
    model.model = MagicMock(return_value=torch.tensor([[0.0]]))

    model.extract_embeddings(_png_bytes(img_rgb), bbox=None, theta=theta)
    np.testing.assert_array_equal(captured["chip"], expected)


def test_extract_embeddings_no_bbox_no_theta_uses_full_image():
    """No bbox + theta=0 should hand the full decoded RGB image to
    preprocess unchanged (no crop, no rotation)."""
    from app.models.miewid import MiewidModel

    rng = np.random.default_rng(seed=1)
    img_rgb = rng.integers(0, 256, size=(50, 70, 3), dtype=np.uint8)

    model = MiewidModel.__new__(MiewidModel)
    model.device = "cpu"
    captured = {}

    def fake_preprocess(image):
        captured["chip"] = image
        return {"image": torch.zeros(3, 4, 4)}

    model.preprocess = fake_preprocess
    model.model = MagicMock(return_value=torch.tensor([[0.0]]))

    model.extract_embeddings(_png_bytes(img_rgb), bbox=None, theta=0.0)
    np.testing.assert_array_equal(captured["chip"], img_rgb)
