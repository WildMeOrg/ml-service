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
PACKAGE = "app.models"
_SENTINEL = object()


@pytest.fixture
def yolo_module(monkeypatch):
    captured = {}

    class FakeYOLO:
        def __init__(self, path):
            captured["path"] = path

        def to(self, device):
            captured["device"] = device
            return self

    # Snapshot BEFORE stubbing. app/models/__init__.py does
    # `from .yolo_ultralytics import YOLOUltralyticsModel`, so if the package
    # itself is cold, importing it below would build it against the stub and
    # leave a fake-backed class bound as app.models.YOLOUltralyticsModel for
    # every later test. monkeypatch.delitem also records nothing to restore
    # when a key was absent, so both entries are handled by hand.
    saved = {name: sys.modules.get(name, _SENTINEL) for name in (PACKAGE, MODULE)}
    # `import app.models` also binds a `models` attribute on the `app` package.
    # Dropping the sys.modules entry alone leaves `from app import models`
    # returning the discarded, stub-backed package object.
    import app as app_pkg
    saved_app_attr = getattr(app_pkg, "models", _SENTINEL)

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    sys.modules.pop(MODULE, None)

    try:
        import app.models.yolo_ultralytics as module
        yield module, captured
    finally:
        for name, previous in saved.items():
            if previous is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        if saved_app_attr is _SENTINEL:
            if hasattr(app_pkg, "models"):
                delattr(app_pkg, "models")
        else:
            setattr(app_pkg, "models", saved_app_attr)
        # Rebind the parent's attribute to whatever module is authoritative now,
        # so app.models.yolo_ultralytics and app.models.YOLOUltralyticsModel do
        # not keep pointing at this test's stub-backed import.
        package = sys.modules.get(PACKAGE)
        restored = sys.modules.get(MODULE)
        if package is not None:
            if restored is None:
                if hasattr(package, "yolo_ultralytics"):
                    delattr(package, "yolo_ultralytics")
            else:
                setattr(package, "yolo_ultralytics", restored)
                if hasattr(restored, "YOLOUltralyticsModel"):
                    setattr(package, "YOLOUltralyticsModel",
                            restored.YOLOUltralyticsModel)


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


def test_fixture_leaves_the_package_unstubbed(yolo_module):
    """Guard the fixture itself: a later test must not inherit FakeYOLO.

    app/models/__init__.py re-exports YOLOUltralyticsModel, so a fixture that
    snapshotted after stubbing would leave that export bound to a fake-backed
    module for the rest of the session.
    """
    module, _ = yolo_module
    assert module.YOLO.__name__ == "FakeYOLO", "the stub is active inside the test"


def test_no_fake_backed_package_survives_teardown():
    """Runs after the fixture torn down above; ordering is intentional.

    Checks the parent-attribute path as well as sys.modules: a cold import
    inside the fixture binds `app.models`, and restoring only sys.modules would
    leave `from app import models` handing back the stub-backed package.
    """
    import app as app_pkg

    candidates = [sys.modules.get(MODULE), getattr(app_pkg, "models", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        yolo = getattr(candidate, "YOLO", None)
        assert yolo is None or yolo.__name__ != "FakeYOLO", \
            f"the ultralytics stub leaked out of the fixture via {candidate!r}"
        model_cls = getattr(candidate, "YOLOUltralyticsModel", None)
        if model_cls is not None:
            leaked = model_cls.load.__globals__.get("YOLO")
            assert leaked is None or leaked.__name__ != "FakeYOLO", \
                "a fake-backed YOLOUltralyticsModel survived teardown"
