"""Tests for the YOLO detector resolving URL weights through the shared resolver.

Every other model type resolved its weights through checkpoint_utils; the
Ultralytics loader handed its configured path straight to YOLO(). That made the
detector the one weight that could not live in an object store, so a
MODEL_BASE pointing at https:// worked for every model except the detector.

Non-URL specs must still reach Ultralytics untouched: it accepts bare hub names
like "yolov8n.pt" and downloads them itself, so resolving those would convert a
working deployment into a FileNotFoundError.

ultralytics isn't importable in the test environment, so a stub captures the
constructor call.
"""
import sys
import types

import pytest


MODULE = "app.models.yolo_ultralytics"


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

    # monkeypatch.delitem records nothing to restore when the key was absent,
    # so a fake-backed module would stay cached in sys.modules (and bound on the
    # parent package) for every later test. Restore both explicitly.
    import app.models as package
    sentinel = object()
    previous_module = sys.modules.get(MODULE, sentinel)
    previous_attr = getattr(package, "yolo_ultralytics", sentinel)
    sys.modules.pop(MODULE, None)

    try:
        import app.models.yolo_ultralytics as module
        yield module, captured
    finally:
        if previous_module is sentinel:
            sys.modules.pop(MODULE, None)
        else:
            sys.modules[MODULE] = previous_module
        if previous_attr is sentinel:
            if hasattr(package, "yolo_ultralytics"):
                delattr(package, "yolo_ultralytics")
        else:
            setattr(package, "yolo_ultralytics", previous_attr)


def test_url_weight_is_resolved_before_load(yolo_module, monkeypatch):
    """A URL must be handed to the resolver, and its cached path given to YOLO()."""
    module, captured = yolo_module
    seen = {}

    def fake_resolver(path):
        seen["arg"] = path
        return "/tmp/checkpoints/detect.pt"

    monkeypatch.setattr(module, "get_checkpoint_path", fake_resolver)

    model = module.YOLOUltralyticsModel()
    model.load("https://storage.googleapis.com/bucket/models/detect.pt", "cpu")

    assert seen["arg"] == "https://storage.googleapis.com/bucket/models/detect.pt", \
        "the resolver must receive the configured URL unchanged"
    assert captured["path"] == "/tmp/checkpoints/detect.pt", \
        "YOLO() must receive the resolved local path, not the URL"


def test_http_url_also_resolved(yolo_module, monkeypatch):
    module, captured = yolo_module
    monkeypatch.setattr(module, "get_checkpoint_path", lambda p: "/tmp/c/d.pt")

    module.YOLOUltralyticsModel().load("http://example.invalid/d.pt", "cpu")

    assert captured["path"] == "/tmp/c/d.pt"


def test_local_path_bypasses_the_resolver(yolo_module, monkeypatch, tmp_path):
    """The /datasets bind-mount case must keep working, resolver untouched."""
    module, captured = yolo_module
    monkeypatch.setattr(module, "get_checkpoint_path",
                        lambda p: pytest.fail("resolver must not run for a local path"))
    weight = tmp_path / "detect.pt"
    weight.write_bytes(b"stub")

    module.YOLOUltralyticsModel().load(str(weight), "cpu")

    assert captured["path"] == str(weight)
    assert captured["device"] == "cpu"


def test_bare_hub_name_reaches_ultralytics_untouched(yolo_module, monkeypatch):
    """Regression guard: Ultralytics auto-downloads hub names; do not intercept."""
    module, captured = yolo_module
    monkeypatch.setattr(module, "get_checkpoint_path",
                        lambda p: pytest.fail("resolver must not run for a hub name"))

    module.YOLOUltralyticsModel().load("yolov8n.pt", "cpu")

    assert captured["path"] == "yolov8n.pt", \
        "resolving hub names would break configs that rely on Ultralytics' download"


def test_missing_local_path_is_left_to_ultralytics(yolo_module, monkeypatch):
    """Not our error to raise: the framework already reports its own."""
    module, captured = yolo_module
    monkeypatch.setattr(module, "get_checkpoint_path",
                        lambda p: pytest.fail("resolver must not run for a local path"))

    module.YOLOUltralyticsModel().load("/datasets/does-not-exist.pt", "cpu")

    assert captured["path"] == "/datasets/does-not-exist.pt"


def test_model_info_records_the_configured_path(yolo_module, monkeypatch):
    """Reported provenance should be what the operator configured, not the cache."""
    module, _ = yolo_module
    monkeypatch.setattr(module, "get_checkpoint_path", lambda p: "/tmp/checkpoints/d.pt")

    model = module.YOLOUltralyticsModel()
    model.load("https://example.invalid/models/d.pt", "cpu")

    assert model.model_info["model_path"] == "https://example.invalid/models/d.pt"
