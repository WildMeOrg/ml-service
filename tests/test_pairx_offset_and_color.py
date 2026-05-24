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


# ---------------------------------------------------------------------------
# Bug 2: RGB/BGR — run_pairx must NOT swap channels on output
# ---------------------------------------------------------------------------

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
