"""Tests for the RF-DETR detector handler (app/models/rfdetr_model.py).

The ``rfdetr`` package needs torch, transformers and a checkpoint, none of which
the test environment has, so a stub module captures the constructor and predict
calls and hands back canned detections in the shape ``rfdetr`` returns (a
supervision ``Detections``: ``xyxy``, ``confidence``, ``class_id`` arrays).
"""
import importlib
import io
import sys
import types

import numpy as np
import pytest
from PIL import Image

ORIENTATION_TAG = 274
VARIANT_CLASSES = ("RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRBase", "RFDETRLarge")


class FakeDetections:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.confidence = None if confidence is None else np.asarray(confidence, dtype=np.float32)
        self.class_id = None if class_id is None else np.asarray(class_id, dtype=np.int64)

    def __len__(self):
        return len(self.xyxy)


@pytest.fixture
def rfdetr_model(monkeypatch):
    """Import app.models.rfdetr_model against a stubbed rfdetr package."""
    captured = {"detections": FakeDetections([], [], [])}

    class FakeRFDETR:
        def __init__(self, **kwargs):
            captured["ctor"] = dict(kwargs)
            captured["ctor_class"] = type(self).__name__

        def predict(self, image, threshold=0.5, **kwargs):
            captured["predict"] = {"image_size": image.size, "threshold": threshold}
            return captured["detections"]

    mod = types.ModuleType("rfdetr")
    for name in VARIANT_CLASSES:
        setattr(mod, name, type(name, (FakeRFDETR,), {}))
    monkeypatch.setitem(sys.modules, "rfdetr", mod)
    monkeypatch.delitem(sys.modules, "app.models.rfdetr_model", raising=False)

    from app.models.rfdetr_model import RFDETRModel
    yield RFDETRModel, captured
    monkeypatch.delitem(sys.modules, "app.models.rfdetr_model", raising=False)


def _jpeg(width=200, height=120, orientation=None):
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, : width // 2] = (255, 0, 0)
    buf = io.BytesIO()
    if orientation is None:
        Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    else:
        exif = Image.Exif()
        exif[ORIENTATION_TAG] = orientation
        Image.fromarray(rgb).save(buf, format="JPEG", quality=95, exif=exif.tobytes())
    return buf.getvalue()


def _loaded(RFDETRModel, tmp_path, **overrides):
    weights = tmp_path / "cfd-rf-detr-medium-1024.pth"
    weights.write_bytes(b"w")
    cfg = dict(model_path=str(weights), device="cuda", model_id="cfd", variant="medium",
               resolution=1024, conf=0.3, class_names=["fish"])
    cfg.update(overrides)
    model = RFDETRModel()
    model.load(**cfg)
    return model, weights


# -- load ---------------------------------------------------------------------

def test_load_builds_the_requested_variant_with_weights_device_and_resolution(rfdetr_model, tmp_path):
    RFDETRModel, captured = rfdetr_model
    model, weights = _loaded(RFDETRModel, tmp_path)

    assert captured["ctor_class"] == "RFDETRMedium"
    assert captured["ctor"] == {"pretrain_weights": str(weights), "device": "cuda", "resolution": 1024}
    info = model.get_model_info()
    assert info["model_type"] == "rfdetr"
    assert info["variant"] == "medium" and info["resolution"] == 1024 and info["conf"] == 0.3
    assert info["class_names"] == {"0": "fish"}


def test_load_without_resolution_leaves_it_to_the_package_default(rfdetr_model, tmp_path):
    RFDETRModel, captured = rfdetr_model
    _loaded(RFDETRModel, tmp_path, resolution=None, variant="small")
    assert captured["ctor_class"] == "RFDETRSmall"
    assert "resolution" not in captured["ctor"]


def test_checkpoint_path_is_an_alias_for_model_path(rfdetr_model, tmp_path):
    RFDETRModel, captured = rfdetr_model
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"w")
    model = RFDETRModel()
    model.load(model_path="", device="cpu", model_id="cfd", class_names=["fish"],
               checkpoint_path=str(weights))
    assert captured["ctor"]["pretrain_weights"] == str(weights)


def test_class_names_dict_with_string_keys_is_accepted(rfdetr_model, tmp_path):
    RFDETRModel, _ = rfdetr_model
    model, _ = _loaded(RFDETRModel, tmp_path, class_names={"0": "fish", "3": "turtle"})
    assert model.class_names == {0: "fish", 3: "turtle"}


@pytest.mark.parametrize("bad", [
    dict(variant="huge"),
    dict(class_names=None),
    dict(class_names=[]),
    dict(class_names={"zero": "fish"}),
    dict(class_names={"0": "fish", "00": "turtle"}),   # two keys, one id: would silently overwrite
    dict(class_names={1.5: "fish"}),                    # int() would truncate to 1
    dict(class_names={"-1": "fish"}),
    dict(class_names={True: "fish"}),
    dict(class_names=[""]),
    dict(class_names=["   "]),
    dict(class_names=[None]),
    dict(class_names="fish"),
    dict(model_path=""),
    dict(imgsz=640),           # a yolo key, not an rfdetr one: typo must fail at startup
])
def test_bad_config_fails_at_load(rfdetr_model, tmp_path, bad):
    RFDETRModel, _ = rfdetr_model
    with pytest.raises(ValueError):
        _loaded(RFDETRModel, tmp_path, **bad)


def test_class_names_are_stripped_and_ints_or_digit_strings_both_work(rfdetr_model, tmp_path):
    RFDETRModel, _ = rfdetr_model
    model, _ = _loaded(RFDETRModel, tmp_path, class_names={0: " fish ", "7": "turtle"})
    assert model.class_names == {0: "fish", 7: "turtle"}


def test_constructor_failure_propagates_through_the_model_handler(rfdetr_model, tmp_path):
    """rfdetr's strict state-dict load raises on a variant/checkpoint mismatch; that must
    surface from ModelHandler.load_model (which is what aborts startup), not be swallowed."""
    RFDETRModel, captured = rfdetr_model
    import rfdetr as stub

    def boom(self, **kwargs):
        raise RuntimeError("size mismatch for backbone")
    stub.RFDETRMedium.__init__ = boom

    from app.models.model_handler import ModelHandler
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"w")
    with pytest.raises(Exception, match="size mismatch"):
        ModelHandler().load_model("cfd", "rfdetr", model_path=str(weights), device="cpu",
                                  variant="medium", class_names=["fish"])


# -- predict ------------------------------------------------------------------

def test_predict_converts_corner_boxes_and_maps_class_names(rfdetr_model, tmp_path):
    RFDETRModel, captured = rfdetr_model
    model, _ = _loaded(RFDETRModel, tmp_path)
    captured["detections"] = FakeDetections([[10, 20, 110, 220], [0, 0, 5, 5]], [0.9, 0.4], [0, 0])

    out = model.predict(_jpeg())

    assert out["bboxes"] == [[10.0, 20.0, 100.0, 200.0], [0.0, 0.0, 5.0, 5.0]]
    assert out["thetas"] == [0.0, 0.0]
    assert out["scores"] == pytest.approx([0.9, 0.4])
    assert out["class_ids"] == [0, 0]
    assert out["class_names"] == ["fish", "fish"]
    assert all(type(v) is float for b in out["bboxes"] for v in b)
    assert all(type(s) is float for s in out["scores"])       # native, not numpy.float32
    assert all(type(i) is int for i in out["class_ids"])      # native, not numpy.int64
    assert captured["predict"]["threshold"] == 0.3   # the configured conf


def test_predict_maps_noncontiguous_ids_to_their_own_labels(rfdetr_model, tmp_path):
    RFDETRModel, captured = rfdetr_model
    model, _ = _loaded(RFDETRModel, tmp_path, class_names={"0": "fish", "3": "turtle"})
    captured["detections"] = FakeDetections([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]], [0.5, 0.6, 0.7], [3, 0, 3])
    out = model.predict(_jpeg())
    assert out["class_ids"] == [3, 0, 3]
    assert out["class_names"] == ["turtle", "fish", "turtle"]


@pytest.mark.parametrize("confidence, class_id", [(None, [0]), ([0.9], None)])
def test_predict_refuses_detections_missing_confidence_or_class_id(rfdetr_model, tmp_path, confidence, class_id):
    RFDETRModel, captured = rfdetr_model
    model, _ = _loaded(RFDETRModel, tmp_path)
    captured["detections"] = FakeDetections([[1, 1, 2, 2]], confidence, class_id)
    with pytest.raises(ValueError, match="without a confidence or class id"):
        model.predict(_jpeg())


def test_predict_with_empty_detections_and_none_fields_returns_empty_lists(rfdetr_model, tmp_path):
    RFDETRModel, captured = rfdetr_model
    model, _ = _loaded(RFDETRModel, tmp_path)
    captured["detections"] = FakeDetections([], None, None)
    out = model.predict(_jpeg())
    assert out == {"bboxes": [], "thetas": [], "scores": [], "class_ids": [], "class_names": []}


def test_predict_conf_override_reaches_the_threshold_and_config_keys_are_tolerated(rfdetr_model, tmp_path):
    """The routers copy the whole model config into predict kwargs, plus request
    overrides; only conf is meaningful per request."""
    RFDETRModel, captured = rfdetr_model
    model, _ = _loaded(RFDETRModel, tmp_path)

    model.predict(_jpeg(), conf=0.05, variant="medium", resolution=1024,
                  class_names=["fish"], model_id="cfd")

    assert captured["predict"]["threshold"] == 0.05


def test_predict_refuses_an_unmapped_class_id(rfdetr_model, tmp_path):
    RFDETRModel, captured = rfdetr_model
    model, _ = _loaded(RFDETRModel, tmp_path)
    captured["detections"] = FakeDetections([[1, 1, 2, 2]], [0.9], [7])
    with pytest.raises(ValueError, match="class id 7"):
        model.predict(_jpeg())


def test_predict_with_no_detections_returns_empty_lists(rfdetr_model, tmp_path):
    RFDETRModel, captured = rfdetr_model
    model, _ = _loaded(RFDETRModel, tmp_path)
    captured["detections"] = FakeDetections([], [], [])
    out = model.predict(_jpeg())
    assert out == {"bboxes": [], "thetas": [], "scores": [], "class_ids": [], "class_names": []}


def test_predict_hands_the_model_the_exif_upright_frame(rfdetr_model, tmp_path):
    """A 200x120 sensor frame tagged Orientation 6 displays as 120x200; the
    detector must see the displayed frame so its boxes match Wildbook's."""
    RFDETRModel, captured = rfdetr_model
    model, _ = _loaded(RFDETRModel, tmp_path)

    model.predict(_jpeg(width=200, height=120, orientation=6))
    assert captured["predict"]["image_size"] == (120, 200)

    model.predict(_jpeg(width=200, height=120))
    assert captured["predict"]["image_size"] == (200, 120)


def test_predict_before_load_fails(rfdetr_model):
    RFDETRModel, _ = rfdetr_model
    with pytest.raises(ValueError, match="not loaded"):
        RFDETRModel().predict(_jpeg())


# -- registry -----------------------------------------------------------------

def test_registry_entry_resolves_to_the_handler(rfdetr_model):
    from app.models.model_handler import MODEL_REGISTRY
    entry = MODEL_REGISTRY["rfdetr"]
    module = importlib.import_module(entry["module"])
    assert getattr(module, entry["class"]).__name__ == "RFDETRModel"
