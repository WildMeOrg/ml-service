"""Label-resolution tests for DenseNetOrientationModel.

Regression cover for issue #33: the loader used to invent a class list
(['down','front','left','right','up']) whenever a checkpoint declared no classes
and happened to have 5 outputs. Every checkpoint deployed under this type was in
fact a wbia-plugin-orientation oriented-bbox REGRESSOR whose 5 outputs are
[xc, yc, xt, yt, w] -- coordinates for deriving theta, not class logits. The
fabricated labels were indistinguishable from real predictions downstream and
Wildbook stored them as viewpoints.

We mock torch.load rather than ship real HRNet/DenseNet weights.
"""
from collections import OrderedDict
from unittest.mock import patch

import pytest
import torch

_PATCH_CKPT = "app.models.densenet_orientation.get_checkpoint_path"


def _state(num_classes: int, hrnet: bool = True) -> OrderedDict:
    """Synthetic state dict; classifier feature dim implies the architecture
    (2048 -> hrnet_w32, 1920 -> densenet201) via _detect_architecture."""
    feat = 2048 if hrnet else 1920
    st = OrderedDict()
    st["classifier.weight"] = torch.zeros(num_classes, feat)
    st["classifier.bias"] = torch.zeros(num_classes)
    return st


def _wrapped(num_classes, classes, hrnet=True):
    return {"state": _state(num_classes, hrnet), "classes": classes}


def _raw(num_classes, hrnet=True):
    """A bare state_dict — exactly the shape of orientation.whaleshark.v3.pth."""
    return _state(num_classes, hrnet)


class _StubBackbone(torch.nn.Module):
    """Minimal real Module whose state_dict keys are exactly
    classifier.{weight,bias} — matching the synthetic checkpoints above, so
    load_state_dict() succeeds at its default strict=True. It must be a real
    nn.Module (not a MagicMock) because load() wraps .classifier in an
    nn.Sequential with a Softmax."""

    def __init__(self, feat: int, num_classes: int):
        super().__init__()
        self.classifier = torch.nn.Linear(feat, num_classes)


def _load(checkpoint, **kwargs):
    """Load with the backbone stubbed; these tests pin label resolution only."""
    from app.models.densenet_orientation import DenseNetOrientationModel

    def _timm(*_a, **kw):
        return _StubBackbone(2048, kw.get("num_classes", 5))

    with patch("torch.load", return_value=checkpoint), \
         patch(_PATCH_CKPT, side_effect=lambda p: p), \
         patch("timm.create_model", side_effect=_timm):
        m = DenseNetOrientationModel()
        m.load(model_id=kwargs.pop("model_id", "t"),
               checkpoint_path="/fake/ckpt.pth", device="cpu", **kwargs)
    return m


# --------------------------------------------------------------------------
# The regression: never fabricate labels
# --------------------------------------------------------------------------

def test_raw_checkpoint_with_5_outputs_is_rejected_not_guessed():
    """Issue #33: this is the whaleshark_v3 / leopard_shark_v0-orient shape.

    5 outputs + no classes used to silently become ['down','front','left',
    'right','up']. Those are all VALID Wildbook viewpoints, so nothing
    downstream could reject them.
    """
    with pytest.raises(ValueError, match="declares no 'classes'"):
        _load(_raw(5), model_id="whaleshark_v3")


def test_rejection_message_is_actionable():
    with pytest.raises(ValueError) as e:
        _load(_raw(5), model_id="whaleshark_v3")
    msg = str(e.value)
    assert "whaleshark_v3" in msg, "must name the offending model"
    assert "label_map" in msg, "must say how to fix it"
    assert "regressor" in msg, "must explain why a label_map may not be the answer"


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 8])
def test_no_output_count_is_ever_guessed(n):
    """The old code guessed at n==5 and emitted 'class_i' otherwise. Neither
    is acceptable: both fabricate meaning the checkpoint never carried."""
    with pytest.raises(ValueError, match="declares no 'classes'"):
        _load(_raw(n))


def test_generic_class_i_labels_are_not_emitted_either():
    with pytest.raises(ValueError):
        _load(_raw(7))


def test_default_orientation_classes_constant_is_gone():
    """Guard against the guess being reintroduced."""
    import app.models.densenet_orientation as mod
    assert not hasattr(mod, "DEFAULT_ORIENTATION_CLASSES"), \
        "the fabricated default class list must not come back (issue #33)"


# --------------------------------------------------------------------------
# Legitimate paths still work
# --------------------------------------------------------------------------

def test_checkpoint_classes_are_used():
    m = _load(_wrapped(3, ["left", "right", "back"]))
    assert m.label_map == {0: "left", 1: "right", 2: "back"}


def test_config_label_map_is_used():
    m = _load(_raw(3), label_map={0: "left", 1: "right", 2: "back"})
    assert m.label_map == {0: "left", 1: "right", 2: "back"}


def test_config_label_map_wins_over_checkpoint_classes():
    m = _load(_wrapped(2, ["a", "b"]), label_map={0: "left", 1: "right"})
    assert m.label_map == {0: "left", 1: "right"}


def test_config_label_map_accepts_string_keys():
    """model_config.json is JSON, so keys arrive as strings."""
    m = _load(_raw(2), label_map={"0": "left", "1": "right"})
    assert m.label_map == {0: "left", 1: "right"}


# --------------------------------------------------------------------------
# Stale / malformed metadata
# --------------------------------------------------------------------------

def test_checkpoint_classes_length_mismatch_raises():
    with pytest.raises(ValueError, match="stale metadata"):
        _load(_wrapped(5, ["left", "right"]))


def test_config_label_map_wrong_keys_raises():
    with pytest.raises(ValueError, match="label_map keys must be exactly"):
        _load(_raw(3), label_map={0: "left", 1: "right", 5: "back"})


def test_config_label_map_too_short_raises():
    with pytest.raises(ValueError, match="label_map keys must be exactly"):
        _load(_raw(3), label_map={0: "left", 1: "right"})


# --------------------------------------------------------------------------
# Metadata validation hardening (Codex review, round 1)
# --------------------------------------------------------------------------

def test_checkpoint_classes_as_bare_string_is_rejected():
    """enumerate("abc") would silently yield 'a','b','c' for a 3-output head."""
    with pytest.raises(ValueError, match="must be a list or tuple"):
        _load({"state": _state(3), "classes": "abc"})


def test_checkpoint_classes_wrong_type_is_rejected():
    with pytest.raises(ValueError, match="must be a list or tuple"):
        _load({"state": _state(2), "classes": {"0": "left", "1": "right"}})


def test_label_map_not_a_mapping_is_rejected():
    with pytest.raises(ValueError, match="must be a mapping"):
        _load(_raw(2), label_map=["left", "right"])


def test_label_map_duplicate_keys_after_coercion_are_rejected():
    """JSON permits "0" and "00" as distinct keys; both coerce to 0."""
    with pytest.raises(ValueError, match="duplicate index"):
        _load(_raw(2), label_map={"0": "left", "00": "other", "1": "right"})


def test_label_map_non_integer_key_is_rejected():
    with pytest.raises(ValueError, match="not an integer index"):
        _load(_raw(2), label_map={"left": "left", "1": "right"})


@pytest.mark.parametrize("bad", [None, 5, "", "   ", ["left"]])
def test_non_string_or_empty_labels_are_rejected(bad):
    with pytest.raises(ValueError, match="non-empty string"):
        _load(_raw(2), label_map={0: bad, 1: "right"})


def test_checkpoint_classes_with_non_string_entry_is_rejected():
    with pytest.raises(ValueError, match="non-empty string"):
        _load({"state": _state(2), "classes": ["left", None]})
