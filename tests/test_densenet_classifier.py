"""Loader + predict tests for DenseNetClassifierModel.

We don't need real DenseNet weights — we mock torch.load to return
synthetic state dicts whose classifier-head shape implies num_classes."""
from collections import OrderedDict
from unittest.mock import patch

import pytest
import torch


def _fake_densenet_state(num_classes: int, classes=None, hrnet=False) -> dict:
    """Build a synthetic state-dict that DenseNetClassifierModel.load can
    detect as either DenseNet201 (1920-feature head) or HRNet-W32 (2048).
    The classifier weight shape is (num_classes, feat_dim).

    We rely on the classifier.weight feature-dimension fallback in
    _detect_arch_and_num_classes (1920 -> densenet201, 2048 -> hrnet_w32)
    rather than adding architecture-specific keys that would cause
    shape-mismatch errors in load_state_dict (strict=False only skips
    missing/extra keys, not shape mismatches).
    """
    feat_dim = 2048 if hrnet else 1920
    state = OrderedDict()
    state["classifier.weight"] = torch.zeros(num_classes, feat_dim)
    state["classifier.bias"] = torch.zeros(num_classes)
    return {"state": state, "classes": classes} if classes else {"state": state}


# Patch both torch.load and get_checkpoint_path (which validates file existence)
_PATCH_CKPT = "app.models.densenet_classifier.get_checkpoint_path"


def test_load_single_checkpoint_normalizes_to_list():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b", "c"])
    with patch("torch.load", return_value=fake), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetClassifierModel()
        m.load(model_id="t", checkpoint_path="/fake/path.weights", device="cpu")
        assert len(m.models) == 1
        assert m.label_map == {0: "a", 1: "b", 2: "c"}
        assert m.compound_labels is False


def test_load_ensemble_three_matching_checkpoints():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b", "c"])
    with patch("torch.load", return_value=fake), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetClassifierModel()
        m.load(model_id="t", checkpoint_paths=[
            "/fake/0.weights", "/fake/1.weights", "/fake/2.weights"
        ], device="cpu")
        assert len(m.models) == 3


def test_load_ensemble_mismatched_num_classes_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake_a = _fake_densenet_state(3, classes=["a", "b", "c"])
    fake_b = _fake_densenet_state(4, classes=["a", "b", "c", "d"])
    with patch("torch.load", side_effect=[fake_a, fake_b]), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="num_classes"):
            m.load(model_id="t", checkpoint_paths=[
                "/fake/0.weights", "/fake/1.weights"], device="cpu")


def test_load_ensemble_mismatched_class_order_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake_a = _fake_densenet_state(3, classes=["a", "b", "c"])
    fake_b = _fake_densenet_state(3, classes=["a", "c", "b"])
    with patch("torch.load", side_effect=[fake_a, fake_b]), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="classes"):
            m.load(model_id="t", checkpoint_paths=[
                "/fake/0.weights", "/fake/1.weights"], device="cpu")


def test_load_stale_classes_list_shorter_than_num_classes_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b"])
    with patch("torch.load", return_value=fake), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="classes"):
            m.load(model_id="t", checkpoint_path="/fake/0.weights", device="cpu")


def test_load_stale_classes_list_longer_than_num_classes_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b", "c", "d"])
    with patch("torch.load", return_value=fake), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="classes"):
            m.load(model_id="t", checkpoint_path="/fake/0.weights", device="cpu")


def test_load_missing_classes_without_label_map_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3)
    with patch("torch.load", return_value=fake), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="classes"):
            m.load(model_id="t", checkpoint_path="/fake/0.weights", device="cpu")


def test_load_mixed_ensemble_some_have_classes_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake_with = _fake_densenet_state(3, classes=["a", "b", "c"])
    fake_without = _fake_densenet_state(3)
    with patch("torch.load", side_effect=[fake_with, fake_without]), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="classes"):
            m.load(model_id="t", checkpoint_paths=[
                "/fake/0.weights", "/fake/1.weights"], device="cpu")


def test_load_explicit_label_map_overrides_classes():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake_a = _fake_densenet_state(3, classes=["x", "y", "z"])
    fake_b = _fake_densenet_state(3, classes=["x", "y", "z"])
    with patch("torch.load", side_effect=[fake_a, fake_b]), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetClassifierModel()
        m.load(model_id="t",
               checkpoint_paths=["/fake/0.weights", "/fake/1.weights"],
               label_map={"0": "a", "1": "b", "2": "c"},
               device="cpu")
        assert m.label_map == {0: "a", 1: "b", 2: "c"}


def test_load_explicit_label_map_missing_index_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b", "c"])
    with patch("torch.load", return_value=fake), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="label_map"):
            m.load(model_id="t", checkpoint_path="/fake/0.weights",
                   label_map={"0": "x", "1": "y"}, device="cpu")


def test_load_ensemble_indices_subset():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b", "c"])
    with patch("torch.load", return_value=fake) as mock_load, \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetClassifierModel()
        m.load(model_id="t", checkpoint_paths=[
            "/fake/0.weights", "/fake/1.weights", "/fake/2.weights"
        ], ensemble_indices=[0], device="cpu")
        assert len(m.models) == 1
        assert mock_load.call_count == 1
        loaded_path = mock_load.call_args_list[0][0][0]
        assert "0.weights" in str(loaded_path)


def test_load_ensemble_indices_out_of_range_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b", "c"])
    with patch("torch.load", return_value=fake), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="ensemble_indices"):
            m.load(model_id="t", checkpoint_paths=["/fake/0.weights"],
                   ensemble_indices=[0, 5], device="cpu")


def test_load_compound_false_with_colon_label_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(2, classes=["a:b", "c:d"])
    with patch("torch.load", return_value=fake), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="compound_labels"):
            m.load(model_id="t", checkpoint_path="/fake/0.weights",
                   compound_labels=False, device="cpu")


# ---------------------------------------------------------------------------
# Predict / ensemble averaging tests
# ---------------------------------------------------------------------------

def test_predict_ensemble_averaging_matches_expected_softmax():
    """The crux of the design: averaging post-softmax across N members
    must produce mathematically-correct probabilities, NOT double-softmax."""
    from app.models.densenet_classifier import DenseNetClassifierModel
    import torch.nn as nn

    class FakeModel(nn.Module):
        def __init__(self, logits):
            super().__init__()
            self._logits = torch.tensor(logits, dtype=torch.float32)
        def forward(self, x):
            return self._logits.unsqueeze(0)
        def eval(self): return self
        def to(self, *a, **kw): return self

    m = DenseNetClassifierModel()
    m.models = [FakeModel([5.0, 0.0, 0.0]), FakeModel([0.0, 0.0, 5.0])]
    m.label_map = {0: "a", 1: "b", 2: "c"}
    m.compound_labels = False
    m.sentinel_prefixes = ["species"]
    m.device = "cpu"
    m._preprocess = lambda *a, **kw: torch.zeros(1, 3, 224, 224)

    result = m.predict(b"fakebytes")
    preds = result["predictions"]
    by_idx = {p["index"]: p for p in preds}
    assert abs(by_idx[0]["probability"] - 0.4967) < 1e-3, by_idx
    assert abs(by_idx[2]["probability"] - 0.4967) < 1e-3, by_idx
    assert abs(by_idx[1]["probability"] - 0.0067) < 1e-3, by_idx
    assert by_idx[0]["probability"] > 0.4, "looks like double-softmax"


def test_predict_compound_labels_emits_per_prediction_parsing():
    from app.models.densenet_classifier import DenseNetClassifierModel
    import torch.nn as nn

    class FakeModel(nn.Module):
        def __init__(self, logits):
            super().__init__()
            self._logits = torch.tensor(logits, dtype=torch.float32)
        def forward(self, x):
            return self._logits.unsqueeze(0)
        def eval(self): return self

    m = DenseNetClassifierModel()
    m.models = [FakeModel([5.0, 0.0, 0.0])]
    m.label_map = {
        0: "salamander_fire_adult:up",
        1: "salamander_fire_juvenile:left",
        2: "salamander_fire_juvenile:right",
    }
    m.compound_labels = True
    m.sentinel_prefixes = ["species"]
    m.device = "cpu"
    m._preprocess = lambda *a, **kw: torch.zeros(1, 3, 224, 224)

    result = m.predict(b"fakebytes")
    for p in result["predictions"]:
        assert p["species"] is not None
        assert p["viewpoint"] in ("up", "left", "right")
    top = result["predictions"][0]
    assert top["species"] == "salamander_fire_adult"
    assert top["viewpoint"] == "up"


def test_predict_single_member_no_implicit_softmax_wrap():
    from app.models.densenet_classifier import DenseNetClassifierModel
    import torch.nn as nn

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return torch.tensor([[5.0, 0.0, 0.0]])
        def eval(self): return self

    m = DenseNetClassifierModel()
    m.models = [FakeModel()]
    m.label_map = {0: "a", 1: "b", 2: "c"}
    m.compound_labels = False
    m.sentinel_prefixes = ["species"]
    m.device = "cpu"
    m._preprocess = lambda *a, **kw: torch.zeros(1, 3, 224, 224)

    result = m.predict(b"fakebytes")
    assert abs(result["probability"] - 0.9866) < 1e-3, result
