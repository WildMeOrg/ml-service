"""EXIF Orientation must be applied wherever an image is decoded with Pillow.

Whiskerbook elephants, Sep 2026: 219 portrait photos (EXIF Orientation 6/8)
carried bounding boxes in the upright frame — the frame cv2.imdecode produces
and the frame Wildbook displays (its derivatives are ImageMagick
``-auto-orient``ed) — but ``MiewidModel.extract_embeddings`` decoded the
original with ``PIL.Image.open`` (which never applies the tag) and cropped the
box out of the landscape sensor frame. Production therefore stored an
embedding of a sideways, mostly wrong region for 6.7% of that catalogue
(Rank-1 on those queries 22.5% vs 87.5% with the right crop).

The fix routes every Pillow decode through ``decode_image_rgb`` so the pixel
frame is the upright one. These tests pin:

  - the helper applies all eight EXIF orientations, checked against an
    independent numpy oracle (rot90 / flips), not against Pillow itself
  - the helper lands in the same frame as cv2's default decode, which every
    cv2-decoding model in this service already uses
  - the MiewID chip is cut from the upright frame, not the sensor frame
  - the ultralytics detector and MegaDetector are handed the upright image
    (ultralytics does not transpose PIL inputs itself)
"""
import io
import sys
import threading
import types
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from app.utils.helpers import decode_image_rgb, get_chip_from_img

ORIENTATION_TAG = 0x0112
ROTATE_90_CW = 6   # camera held for a portrait shot; pixels stored landscape
ROTATE_90_CCW = 8

# Independent oracle: what each EXIF Orientation value means for the stored
# (sensor) pixels, expressed with numpy only. Values per the EXIF spec.
ORACLE = {
    1: lambda a: a,
    2: lambda a: a[:, ::-1],                 # mirror horizontal
    3: lambda a: a[::-1, ::-1],              # rotate 180
    4: lambda a: a[::-1, :],                 # mirror vertical
    5: lambda a: np.transpose(a, (1, 0, 2)),                 # transpose
    6: lambda a: np.rot90(a, k=-1),          # rotate 90 clockwise
    7: lambda a: np.rot90(np.transpose(a, (1, 0, 2)), k=2),  # transverse
    8: lambda a: np.rot90(a, k=1),           # rotate 90 counter-clockwise
}


def _sensor_pixels(h=120, w=200):
    """Landscape sensor frame built from large flat colour blocks (quadrants
    plus a distinct top-left marker) so that a decoder-rounding-tolerant
    comparison is still sensitive to any flip or rotation."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2, : w // 2] = (200, 30, 30)
    img[: h // 2, w // 2:] = (30, 200, 30)
    img[h // 2:, : w // 2] = (30, 30, 200)
    img[h // 2:, w // 2:] = (220, 220, 40)
    img[5:35, 5:45] = 255          # top-left marker in the SENSOR frame
    return img


def _jpeg_with_orientation(rgb, orientation):
    exif = Image.Exif()
    exif[ORIENTATION_TAG] = orientation
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95, exif=exif.tobytes())
    return buf.getvalue()


def _block_means(a, n=6):
    """Mean colour of an n x n grid of blocks: robust to JPEG rounding, still
    sensitive to any flip/rotation of a block-coloured image."""
    h, w = a.shape[:2]
    return np.array([[a[i * h // n:(i + 1) * h // n, j * w // n:(j + 1) * w // n].reshape(-1, 3).mean(0)
                      for j in range(n)] for i in range(n)])


@pytest.mark.parametrize("orientation", sorted(ORACLE))
def test_decode_image_rgb_applies_every_exif_orientation(orientation):
    sensor = _sensor_pixels()
    data = _jpeg_with_orientation(sensor, orientation)

    got = decode_image_rgb(data)

    # Oracle on the SENSOR pixels as Pillow itself decodes them (so the only
    # difference between got and expected is the orientation handling).
    sensor_decoded = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    expected = ORACLE[orientation](sensor_decoded)
    assert got.dtype == np.uint8
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)
    # And the orientation really was applied (not just a shape no-op).
    if orientation != 1 and got.shape == sensor_decoded.shape:
        assert not np.array_equal(got, sensor_decoded)


@pytest.mark.parametrize("orientation", [ROTATE_90_CW, ROTATE_90_CCW])
def test_decode_image_rgb_matches_cv2_frame(orientation):
    """cv2.imdecode honours EXIF by default; the helper must land in the same
    frame. Compared on coarse block means so decoder rounding cannot matter."""
    data = _jpeg_with_orientation(_sensor_pixels(), orientation)
    got = decode_image_rgb(data)
    via_cv2 = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    assert via_cv2.shape == got.shape == (200, 120, 3)
    np.testing.assert_allclose(_block_means(got), _block_means(via_cv2), atol=6)


def test_decode_image_rgb_is_identity_without_orientation_tag():
    sensor = _sensor_pixels()
    buf = io.BytesIO()
    Image.fromarray(sensor).save(buf, format="PNG")
    got = decode_image_rgb(buf.getvalue())
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, sensor)


def test_decode_image_rgb_converts_to_rgb():
    buf = io.BytesIO()
    Image.fromarray(_sensor_pixels()).convert("L").save(buf, format="PNG")
    got = decode_image_rgb(buf.getvalue())
    assert got.shape == (120, 200, 3) and got.dtype == np.uint8


def _stubbed_miewid():
    from app.models.miewid import MiewidModel
    model = MiewidModel.__new__(MiewidModel)
    model.device = "cpu"
    model.inference_lock = threading.RLock()
    captured = {}

    def fake_preprocess(image):
        captured["chip"] = image
        return {"image": torch.zeros(3, 4, 4)}

    model.preprocess = fake_preprocess
    model.model = MagicMock(return_value=torch.tensor([[1.0, 2.0, 3.0]]))
    return model, captured


def test_extract_embeddings_crops_from_the_upright_frame():
    """The bbox is expressed in the upright frame (detector / Wildbook UI).
    The chip handed to preprocess must be that region of the upright image —
    and must NOT be the same coordinates cut from the un-rotated sensor frame,
    which is what production did before the fix."""
    data = _jpeg_with_orientation(_sensor_pixels(), ROTATE_90_CW)
    sensor_decoded = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    upright = ORACLE[ROTATE_90_CW](sensor_decoded)

    bbox = (10, 20, 60, 90)   # inside the 120-wide x 200-tall upright frame
    model, captured = _stubbed_miewid()
    out = model.extract_embeddings(data, bbox=bbox, theta=0.0)
    np.testing.assert_array_equal(out, np.array([[1.0, 2.0, 3.0]]))

    expected = get_chip_from_img(np.ascontiguousarray(upright), list(bbox), 0.0)
    np.testing.assert_array_equal(captured["chip"], expected)

    wrong_frame = get_chip_from_img(sensor_decoded.copy(), list(bbox), 0.0)
    assert not np.array_equal(captured["chip"], wrong_frame)


def test_extract_embeddings_full_frame_is_upright():
    data = _jpeg_with_orientation(_sensor_pixels(), ROTATE_90_CCW)
    model, captured = _stubbed_miewid()
    model.extract_embeddings(data)
    assert captured["chip"].shape == (200, 120, 3)


def test_yolo_ultralytics_receives_the_upright_image():
    """ultralytics.LoadPilAndNumpy does not exif-transpose a PIL image, so the
    detector must be handed one that is already upright; otherwise its boxes
    land in the sensor frame while Wildbook draws them on the auto-oriented
    derivative."""
    from app.models.yolo_ultralytics import YOLOUltralyticsModel

    m = YOLOUltralyticsModel.__new__(YOLOUltralyticsModel)
    m.model_info = {"imgsz": 640, "conf": 0.25, "device": "cpu", "dilation_factors": (1.0, 1.0)}
    m.model = MagicMock()
    m.model.predict.return_value = [MagicMock()]
    m._process_results = MagicMock(return_value={"predictions": []})

    data = _jpeg_with_orientation(_sensor_pixels(), ROTATE_90_CW)
    m.predict(data)

    handed = m.model.predict.call_args[0][0]
    assert isinstance(handed, Image.Image)
    assert handed.mode == "RGB"
    assert handed.size == (120, 200)   # (width, height) of the upright frame
    kw = m.model.predict.call_args[1]
    assert kw["imgsz"] == 640 and kw["conf"] == 0.25 and kw["device"] == "cpu"


@pytest.fixture
def megadetector_model(monkeypatch):
    """Import app.models.megadetector against a stubbed PytorchWildlife
    (same pattern as test_megadetector_checkpoint.py)."""
    detection = types.ModuleType("PytorchWildlife.models.detection")
    detection.MegaDetectorV6 = MagicMock()
    models = types.ModuleType("PytorchWildlife.models")
    models.detection = detection
    pw = types.ModuleType("PytorchWildlife")
    pw.models = models
    monkeypatch.setitem(sys.modules, "PytorchWildlife", pw)
    monkeypatch.setitem(sys.modules, "PytorchWildlife.models", models)
    monkeypatch.setitem(sys.modules, "PytorchWildlife.models.detection", detection)
    monkeypatch.delitem(sys.modules, "app.models.megadetector", raising=False)
    from app.models.megadetector import MegaDetectorModel
    yield MegaDetectorModel
    monkeypatch.delitem(sys.modules, "app.models.megadetector", raising=False)


def test_megadetector_bytes_to_numpy_is_upright_bgr_uint8(megadetector_model):
    m = megadetector_model.__new__(megadetector_model)
    data = _jpeg_with_orientation(_sensor_pixels(), ROTATE_90_CW)
    sensor_decoded = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    expected_bgr = ORACLE[ROTATE_90_CW](sensor_decoded)[..., ::-1]

    got = m._bytes_to_numpy(data)

    assert got.dtype == np.uint8
    assert got.shape == (200, 120, 3)
    np.testing.assert_array_equal(got, expected_bgr)
