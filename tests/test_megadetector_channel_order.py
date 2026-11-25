"""Red-image regression test for MegaDetector channel order.

MegaDetector (v5 and v6) is trained on RGB. Feeding it BGR doesn't crash
and the detections still look plausible, but recall drops a lot (one
published comparison found ~30-50% more errors). It's an easy mistake to
make here because the libraries disagree:

* ``MegaDetectorModel._bytes_to_numpy`` decodes RGB and then deliberately
  converts it to BGR.
* PytorchWildlife's ``single_image_detection`` hands that array straight to
  Ultralytics' ``DetectionPredictor.stream_inference``.
* Ultralytics' ``preprocess`` assumes every numpy array is BGR (OpenCV
  convention) and flips it to RGB (``im[..., ::-1]``).

PytorchWildlife's own docstrings say numpy input should be RGB. If someone
"fixes" our RGB2BGR to match them, the network silently starts seeing BGR.
This test sends a pure-red image through our decode and the real Ultralytics
preprocess and checks that red arrives in tensor channel 0.

PytorchWildlife can't be imported in the test environment, so it is stubbed
the same way test_megadetector_checkpoint.py does it. Ultralytics is real.
"""
import io
import sys
import types

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
detect = pytest.importorskip("ultralytics.models.yolo.detect")

SIZE = 64  # multiple of the stride, so letterbox adds no grey padding


@pytest.fixture
def megadetector_cls(monkeypatch):
    """Import app.models.megadetector against a stubbed PytorchWildlife."""
    detection = types.ModuleType("PytorchWildlife.models.detection")
    detection.MegaDetectorV6 = object
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


def _red_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (SIZE, SIZE), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _ultralytics_preprocess(arr: np.ndarray) -> torch.Tensor:
    """Run Ultralytics' real DetectionPredictor.preprocess on one image,
    which is the step PytorchWildlife's stream_inference reaches. Only the
    attributes preprocess/pre_transform read are filled in, so no weights
    are needed."""
    predictor = detect.DetectionPredictor()
    predictor.imgsz = (SIZE, SIZE)
    predictor.device = torch.device("cpu")
    predictor.model = types.SimpleNamespace(fp16=False, stride=32, format="pt")
    return predictor.preprocess([arr])


def test_red_image_reaches_network_as_rgb(megadetector_cls):
    """Pure red in -> tensor channel 0 (R) is 1.0 and channels 1-2 are 0."""
    captured = {}

    class FakeDetector:
        def single_image_detection(self, img, *args, **kwargs):
            captured["img"] = img
            return {"detections": None, "labels": []}

    model = megadetector_cls()
    model.model = FakeDetector()
    model.model_info = {"confidence_threshold": 0.5}

    model.predict(_red_png())

    tensor = _ultralytics_preprocess(captured["img"])
    assert tensor.shape == (1, 3, SIZE, SIZE)
    means = tensor[0].mean(dim=(1, 2)).tolist()
    assert means == pytest.approx([1.0, 0.0, 0.0], abs=1e-6), (
        f"network sees per-channel means {means}; red should be in channel 0. "
        "If red ended up in channel 2, MegaDetector is being fed BGR. Check "
        "the RGB->BGR conversion in MegaDetectorModel._bytes_to_numpy."
    )


def test_ultralytics_still_treats_numpy_as_bgr():
    """Pins the library assumption the test above depends on. If an
    Ultralytics upgrade stops flipping numpy input, this fails and
    _bytes_to_numpy has to change."""
    bgr_red = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    bgr_red[..., 2] = 255  # red in OpenCV's BGR layout
    tensor = _ultralytics_preprocess(bgr_red)
    assert tensor[0, 0].mean().item() == pytest.approx(1.0)
    assert tensor[0, 2].mean().item() == pytest.approx(0.0)
