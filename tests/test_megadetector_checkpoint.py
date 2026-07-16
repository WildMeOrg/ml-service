"""Tests for explicit checkpoint_path support in the MegaDetector loader.

Every other model type resolves its weights through
checkpoint_utils.get_checkpoint_path (local path or URL). MegaDetector
was the exception: it always let PytorchWildlife fetch weights from its
hardcoded zenodo URL at load time — a hidden internet download on every
cold start that MODEL_BASE cannot control or pin.

PytorchWildlife needs GPU/network extras that aren't importable in the
test environment, so a stub module captures the constructor call.
"""
import sys
import types

import pytest


@pytest.fixture
def megadetector_model(monkeypatch):
    """Import app.models.megadetector against a stubbed PytorchWildlife."""
    captured = {}

    class FakeMegaDetectorV6:
        def __init__(self, weights=None, device="cpu", pretrained=True,
                     version="yolov9c"):
            captured.update(weights=weights, device=device,
                            pretrained=pretrained, version=version)

    detection = types.ModuleType("PytorchWildlife.models.detection")
    detection.MegaDetectorV6 = FakeMegaDetectorV6
    models = types.ModuleType("PytorchWildlife.models")
    models.detection = detection
    pw = types.ModuleType("PytorchWildlife")
    pw.models = models

    monkeypatch.setitem(sys.modules, "PytorchWildlife", pw)
    monkeypatch.setitem(sys.modules, "PytorchWildlife.models", models)
    monkeypatch.setitem(sys.modules, "PytorchWildlife.models.detection", detection)
    monkeypatch.delitem(sys.modules, "app.models.megadetector", raising=False)

    from app.models.megadetector import MegaDetectorModel
    yield MegaDetectorModel, captured
    monkeypatch.delitem(sys.modules, "app.models.megadetector", raising=False)


def test_checkpoint_path_reaches_pytorchwildlife(megadetector_model, tmp_path):
    MegaDetectorModel, captured = megadetector_model
    weights = tmp_path / "MDV6-yolov10-e-1280.pt"
    weights.write_bytes(b"w")

    model = MegaDetectorModel()
    model.load(model_path="", device="cpu", model_id="MDV6-yolov10-e",
               checkpoint_path=str(weights))

    assert captured["weights"] == str(weights)
    assert captured["version"] == "MDV6-yolov10-e"


def test_absent_checkpoint_path_keeps_auto_download(megadetector_model):
    """Without checkpoint_path the current behavior is preserved:
    PytorchWildlife receives weights=None and fetches its own."""
    MegaDetectorModel, captured = megadetector_model

    model = MegaDetectorModel()
    model.load(model_path="", device="cpu", model_id="MDV6-yolov10-e", conf=0.1)

    assert captured["weights"] is None
    assert captured["pretrained"] is True


def test_missing_local_checkpoint_fails_fast(megadetector_model):
    """A configured-but-absent local weight is a config error, not a cue
    to silently download something else."""
    MegaDetectorModel, captured = megadetector_model

    model = MegaDetectorModel()
    with pytest.raises(Exception) as excinfo:
        model.load(model_path="", device="cpu", model_id="MDV6-yolov10-e",
                   checkpoint_path="/nonexistent/weights.pt")
    assert "not found" in str(excinfo.value).lower()
    assert captured == {}, "PytorchWildlife must not be constructed on bad config"
