"""Tests for the YOLO detector resolving weights through get_checkpoint_path.

Every other model type resolved its weights through the shared resolver; the
Ultralytics loader handed its configured path straight to YOLO(). That made the
detector the one weight that could not live in an object store, so a
MODEL_BASE pointing at https:// worked for every model except the detector.

ultralytics isn't importable in the test environment, so a stub captures the
constructor call.
"""
import sys
import types

import pytest


@pytest.fixture
def yolo_module(monkeypatch):
    captured = {}

    class FakeYOLO:
        def __init__(self, path):
            captured["path"] = path

        def to(self, device):
            captured["device"] = device
            return self

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.delitem(sys.modules, "app.models.yolo_ultralytics", raising=False)

    import app.models.yolo_ultralytics as module
    return module, captured


def test_url_weight_is_downloaded_before_load(yolo_module, monkeypatch):
    """A URL must be fetched and the cached local path handed to YOLO()."""
    module, captured = yolo_module
    monkeypatch.setattr(module, "get_checkpoint_path",
                        lambda p: "/tmp/checkpoints/detect.pt")

    model = module.YOLOUltralyticsModel()
    model.load("https://storage.googleapis.com/bucket/models/detect.pt", "cpu")

    assert captured["path"] == "/tmp/checkpoints/detect.pt", \
        "YOLO() must receive the resolved local path, not the URL"


def test_local_path_is_passed_through(yolo_module, tmp_path):
    """The default /datasets bind-mount case must keep working unchanged."""
    module, captured = yolo_module
    weight = tmp_path / "detect.pt"
    weight.write_bytes(b"stub")

    model = module.YOLOUltralyticsModel()
    model.load(str(weight), "cpu")

    assert captured["path"] == str(weight)
    assert captured["device"] == "cpu"


def test_missing_local_weight_raises_filenotfound(yolo_module):
    """A typo'd path should fail as FileNotFoundError, not an opaque framework error."""
    module, _ = yolo_module

    model = module.YOLOUltralyticsModel()
    with pytest.raises(FileNotFoundError):
        model.load("/datasets/does-not-exist.pt", "cpu")
