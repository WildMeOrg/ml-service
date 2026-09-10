"""Loader + decision-table tests for DenseNetWildDogCascadeModel.

The decision table is a direct port of WBIA's wild dog labeler
(wbia/core_annots.py:2200-2217); these tests pin every branch of it so a
future refactor can't silently change ACW's labelling behaviour.

Like test_densenet_classifier.py, we mock torch.load rather than ship real
DenseNet weights.
"""
from collections import OrderedDict
from unittest.mock import patch

import pytest
import torch

# Real label spaces, read off the actual ACW checkpoints.
V1_COAT_CLASSES = [
    "wild_dog+tail_general:ignore",
    "wild_dog+tail_long_black:ignore",
    "wild_dog+tail_long_white:ignore",
    "wild_dog+tail_multi_black:ignore",
    "wild_dog+tail_short_black:ignore",
    "wild_dog+tail_standard:ignore",
    "wild_dog_dark:left", "wild_dog_dark:other", "wild_dog_dark:right",
    "wild_dog_general:left", "wild_dog_general:other", "wild_dog_general:right",
    "wild_dog_puppy:ignore",
    "wild_dog_standard:left", "wild_dog_standard:other", "wild_dog_standard:right",
    "wild_dog_tan:left", "wild_dog_tan:other", "wild_dog_tan:right",
]

V2_VIEWPOINT_CLASSES = [
    "wild_dog+tail_general:ignore",
    "wild_dog+tail_long:ignore",
    "wild_dog+tail_multi_black:ignore",
    "wild_dog+tail_short_black:ignore",
    "wild_dog+tail_standard:ignore",
    "wild_dog:back", "wild_dog:backleft", "wild_dog:backright",
    "wild_dog:down", "wild_dog:front", "wild_dog:frontleft",
    "wild_dog:frontright", "wild_dog:left", "wild_dog:right",
    "wild_dog:up", "wild_dog:upback", "wild_dog:upfront",
    "wild_dog:upleft", "wild_dog:upright",
    "wild_dog_puppy:ignore",
]

V3_ROUTER_CLASSES = [
    "wild_dog+tail_double_black_brown:ignore",
    "wild_dog+tail_double_black_white:ignore",
    "wild_dog+tail_general:ignore",
    "wild_dog+tail_long_black:ignore",
    "wild_dog+tail_long_white:ignore",
    "wild_dog+tail_short_black:ignore",
    "wild_dog+tail_standard:ignore",
    "wild_dog+tail_triple_black:ignore",
    "wild_dog:ignore",
]

_PATCH_CKPT = "app.models.densenet_classifier.get_checkpoint_path"


def _fake_densenet_state(classes) -> dict:
    state = OrderedDict()
    state["classifier.weight"] = torch.zeros(len(classes), 1920)
    state["classifier.bias"] = torch.zeros(len(classes))
    return {"state": state, "classes": list(classes)}


def _role(path_marker, classes, heads=3):
    return {"checkpoint_paths": [f"/fake/{path_marker}/labeler.{i}.weights"
                                for i in range(heads)]}


def _torch_load_by_path(mapping):
    """Return a torch.load side_effect that picks a fake checkpoint by path.

    Markers are matched longest-first: short markers are substrings of longer
    ones ('v1' is inside 'v1trunc'), and matching in dict order would silently
    hand back the wrong fixture.
    """
    ordered = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    def _load(path, *args, **kwargs):
        for marker, classes in ordered:
            if marker in str(path):
                return _fake_densenet_state(classes)
        raise AssertionError(f"unexpected checkpoint path: {path}")
    return _load


DEFAULT_MAPPING = {
    "v1": V1_COAT_CLASSES,
    "v2": V2_VIEWPOINT_CLASSES,
    "v3": V3_ROUTER_CLASSES,
}


def _load_cascade(mapping=None, **overrides):
    from app.models.densenet_wilddog_cascade import DenseNetWildDogCascadeModel
    mapping = mapping or DEFAULT_MAPPING
    spec = {
        "router": _role("v3", V3_ROUTER_CLASSES),
        "coat": _role("v1", V1_COAT_CLASSES),
        "viewpoint": _role("v2", V2_VIEWPOINT_CLASSES),
    }
    spec.update(overrides)
    with patch("torch.load", side_effect=_torch_load_by_path(mapping)), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        m = DenseNetWildDogCascadeModel()
        m.load(model_id="wd-cascade", device="cpu", **spec)
    return m


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------

def test_load_wires_three_roles_with_correct_label_spaces():
    m = _load_cascade()
    assert set(m.members) == {"router", "coat", "viewpoint"}
    assert len(m.members["router"].label_map) == 9
    assert len(m.members["coat"].label_map) == 19
    assert len(m.members["viewpoint"].label_map) == 20
    # Each role is itself a 3-head ensemble.
    for role in ("router", "coat", "viewpoint"):
        assert len(m.members[role].models) == 3


def test_load_enables_compound_labels_by_default():
    m = _load_cascade()
    for role in ("router", "coat", "viewpoint"):
        assert m.members[role].compound_labels is True


@pytest.mark.parametrize("missing", ["router", "coat", "viewpoint"])
def test_load_missing_role_raises(missing):
    with pytest.raises(ValueError, match="missing required role"):
        _load_cascade(**{missing: None})


def test_load_rejects_swapped_router_and_coat():
    """v3 in the coat slot must fail loudly, not mislabel silently."""
    with pytest.raises(ValueError, match="missing .* label"):
        _load_cascade(
            router=_role("v1", V1_COAT_CLASSES),
            coat=_role("v3", V3_ROUTER_CLASSES),
        )


def test_load_rejects_swapped_router_and_viewpoint():
    """The subtle one: v2 and v3 BOTH emit raw species 'wild_dog', so a
    species-level check cannot catch this swap. Regression test for a real
    defect — only v2 has wild_dog:front, only v3 has the triple_black tail."""
    with pytest.raises(ValueError, match="missing .* label"):
        _load_cascade(
            router=_role("v2", V2_VIEWPOINT_CLASSES),
            viewpoint=_role("v3", V3_ROUTER_CLASSES),
        )


def test_load_rejects_swapped_coat_and_viewpoint():
    with pytest.raises(ValueError, match="missing .* label"):
        _load_cascade(
            coat=_role("v2", V2_VIEWPOINT_CLASSES),
            viewpoint=_role("v1", V1_COAT_CLASSES),
        )


def test_load_rejects_unrelated_model_in_coat_slot():
    cheetah = ["cheetah:left", "cheetah:right"]
    mapping = dict(DEFAULT_MAPPING, cheetah=cheetah)
    with pytest.raises(ValueError, match="missing .* label"):
        _load_cascade(mapping=mapping, coat=_role("cheetah", cheetah))


def _required_cases():
    """Every individual required label, per role — dropping ANY must fail."""
    from app.models.densenet_wilddog_cascade import REQUIRED_LABELS
    real = {"router": V3_ROUTER_CLASSES, "coat": V1_COAT_CLASSES,
            "viewpoint": V2_VIEWPOINT_CLASSES}
    return [(role, real[role], label)
            for role in ("router", "coat", "viewpoint")
            for label in sorted(REQUIRED_LABELS[role][0])]


@pytest.mark.parametrize("role,full,dropped", _required_cases())
def test_load_rejects_role_missing_any_single_required_label(role, full, dropped):
    """Exhaustive: a checkpoint missing even one required label is rejected.

    A missing label wouldn't crash — the corresponding branch would just never
    fire — so this is exactly the silent-degradation case load-time validation
    exists to catch.
    """
    truncated = [c for c in full if c != dropped]
    marker = f"trunc-{role}"
    mapping = dict(DEFAULT_MAPPING, **{marker: truncated})
    with pytest.raises(ValueError, match="missing .* label"):
        _load_cascade(mapping=mapping, **{role: _role(marker, truncated)})


@pytest.mark.parametrize("bad_label", [
    "wild_dog_nocolon",   # no colon      -> species=None at inference
    ":ignore",            # empty species
    "wild_dog:",          # empty viewpoint
    "wild_dog:a:b",       # two colons    -> viewpoint would become 'a:b'
])
def test_load_rejects_malformed_labels(bad_label):
    bad = list(V3_ROUTER_CLASSES) + [bad_label]
    mapping = dict(DEFAULT_MAPPING, v3bad=bad)
    with pytest.raises(ValueError, match="non-compound label"):
        _load_cascade(mapping=mapping, router=_role("v3bad", bad))


def test_load_rejects_unknown_role_config_key():
    spec = _role("v3", V3_ROUTER_CLASSES)
    spec["chekpoint_paths"] = ["/typo"]
    with pytest.raises(ValueError, match="unknown config key"):
        _load_cascade(router=spec)


def test_load_rejects_unknown_top_level_config_key():
    with pytest.raises(ValueError, match="unknown config key"):
        _load_cascade(imgsz=640)


def test_failed_load_leaves_no_partial_members():
    from app.models.densenet_wilddog_cascade import DenseNetWildDogCascadeModel
    m = DenseNetWildDogCascadeModel()
    with patch("torch.load", side_effect=_torch_load_by_path(DEFAULT_MAPPING)), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        with pytest.raises(ValueError):
            m.load(model_id="wd", device="cpu",
                   router=_role("v1", V1_COAT_CLASSES),   # wrong -> rejected
                   coat=_role("v3", V3_ROUTER_CLASSES),
                   viewpoint=_role("v2", V2_VIEWPOINT_CLASSES))
    assert m.members == {}, "a rejected load must not leave members wired up"


def test_failed_reload_does_not_corrupt_an_already_loaded_cascade():
    """A failed reload must not pair the old members with the new metadata."""
    m = _load_cascade()
    good_members = m.members
    with patch("torch.load", side_effect=_torch_load_by_path(DEFAULT_MAPPING)), \
         patch(_PATCH_CKPT, side_effect=lambda p: p):
        with pytest.raises(ValueError):
            m.load(model_id="new-id", device="cpu", img_size=999,
                   router=_role("v1", V1_COAT_CLASSES),   # wrong -> rejected
                   coat=_role("v3", V3_ROUTER_CLASSES),
                   viewpoint=_role("v2", V2_VIEWPOINT_CLASSES))
    assert m.members is good_members, "members must be untouched"
    assert m.model_id == "wd-cascade", "model_id must not adopt the failed load"
    assert m.img_size == 224, "img_size must not adopt the failed load"


# --------------------------------------------------------------------------
# Decision table — core_annots.py:2200-2217
# --------------------------------------------------------------------------

class _StubMember:
    """Stands in for a loaded DenseNetClassifierModel role."""

    def __init__(self, species, viewpoint, probability):
        self.label_map = {0: f"{species}:{viewpoint}"}
        self.models = [None]
        self._top = {
            "label": f"{species}:{viewpoint}",
            "probability": probability,
            "index": 0,
            "species": species,
            "viewpoint": viewpoint,
        }

    def predict(self, image_bytes, bbox=None, theta=0.0, **kwargs):
        return {"predictions": [self._top]}


def _cascade_with(router, coat, viewpoint):
    from app.models.densenet_wilddog_cascade import DenseNetWildDogCascadeModel
    m = DenseNetWildDogCascadeModel()
    m.model_id = "wd-cascade"
    m.members = {"router": _StubMember(*router),
                 "coat": _StubMember(*coat),
                 "viewpoint": _StubMember(*viewpoint)}
    return m


def test_body_all_three_agree_uses_coat_species_and_v2_viewpoint():
    """The whole point of the cascade: coat from v1, viewpoint from v2."""
    m = _cascade_with(
        router=("wild_dog", "ignore", 0.9),
        coat=("wild_dog_standard", "left", 0.8),
        viewpoint=("wild_dog", "frontleft", 0.7),
    )
    out = m.predict(b"img")
    assert out["predictions"][0]["species"] == "wild_dog_standard"
    # 'frontleft' exists only in v2's label space, never in v1's.
    assert out["predictions"][0]["viewpoint"] == "frontleft"
    assert out["class"] == "wild_dog_standard:frontleft"


def test_body_with_puppy_viewpoint_member_still_counts_as_body():
    """v2's wild_dog_puppy is a body class too (flag2)."""
    m = _cascade_with(
        router=("wild_dog", "ignore", 0.9),
        coat=("wild_dog_tan", "right", 0.8),
        viewpoint=("wild_dog_puppy", "ignore", 0.7),
    )
    out = m.predict(b"img")
    assert out["predictions"][0]["species"] == "wild_dog_tan"
    assert out["predictions"][0]["viewpoint"] == "ignore"


def test_body_with_puppy_coat_uses_puppy_species():
    """v1's wild_dog_puppy is in the coat body set (flag1)."""
    m = _cascade_with(
        router=("wild_dog", "ignore", 0.9),
        coat=("wild_dog_puppy", "ignore", 0.8),
        viewpoint=("wild_dog", "backright", 0.7),
    )
    out = m.predict(b"img")
    assert out["predictions"][0]["species"] == "wild_dog_puppy"
    assert out["predictions"][0]["viewpoint"] == "backright"


def test_body_but_coat_says_tail_is_ambiguous():
    m = _cascade_with(
        router=("wild_dog", "ignore", 0.9),
        coat=("wild_dog+tail_general", "ignore", 0.8),
        viewpoint=("wild_dog", "left", 0.7),
    )
    out = m.predict(b"img")
    assert out["predictions"][0]["species"] == "wild_dog_ambiguous"
    assert out["predictions"][0]["viewpoint"] == "ambiguous"


def test_body_but_viewpoint_says_tail_is_ambiguous():
    m = _cascade_with(
        router=("wild_dog", "ignore", 0.9),
        coat=("wild_dog_dark", "left", 0.8),
        viewpoint=("wild_dog+tail_standard", "ignore", 0.7),
    )
    out = m.predict(b"img")
    assert out["predictions"][0]["species"] == "wild_dog_ambiguous"


def test_tail_all_agree_uses_router_tail_class_and_ignore_viewpoint():
    m = _cascade_with(
        router=("wild_dog+tail_triple_black", "ignore", 0.9),
        coat=("wild_dog+tail_general", "ignore", 0.8),
        viewpoint=("wild_dog+tail_standard", "ignore", 0.7),
    )
    out = m.predict(b"img")
    # v3 is the only model that knows tail_triple_black.
    assert out["predictions"][0]["species"] == "wild_dog+tail_triple_black"
    assert out["predictions"][0]["viewpoint"] == "ignore"


def test_tail_but_coat_says_body_is_tail_ambiguous():
    m = _cascade_with(
        router=("wild_dog+tail_long_white", "ignore", 0.9),
        coat=("wild_dog_standard", "left", 0.8),
        viewpoint=("wild_dog+tail_general", "ignore", 0.7),
    )
    out = m.predict(b"img")
    assert out["predictions"][0]["species"] == "wild_dog+tail_ambiguous"
    assert out["predictions"][0]["viewpoint"] == "ambiguous"


def test_tail_but_viewpoint_says_body_is_tail_ambiguous():
    m = _cascade_with(
        router=("wild_dog+tail_long_white", "ignore", 0.9),
        coat=("wild_dog+tail_general", "ignore", 0.8),
        viewpoint=("wild_dog", "front", 0.7),
    )
    out = m.predict(b"img")
    assert out["predictions"][0]["species"] == "wild_dog+tail_ambiguous"


# --------------------------------------------------------------------------
# Score + output contract
# --------------------------------------------------------------------------

def test_score_is_mean_of_three_member_probabilities():
    m = _cascade_with(
        router=("wild_dog", "ignore", 0.9),
        coat=("wild_dog_standard", "left", 0.6),
        viewpoint=("wild_dog", "left", 0.3),
    )
    out = m.predict(b"img")
    assert out["predictions"][0]["probability"] == pytest.approx(0.6)
    assert out["probability"] == pytest.approx(0.6)


def test_output_matches_classify_slot_contract():
    """pipeline_router reads predictions[0].{label,probability,index,species,viewpoint}."""
    m = _cascade_with(
        router=("wild_dog", "ignore", 0.9),
        coat=("wild_dog_general", "other", 0.8),
        viewpoint=("wild_dog", "upright", 0.7),
    )
    out = m.predict(b"img")
    assert out["predictions"], "router requires a non-empty predictions list"
    top = out["predictions"][0]
    for key in ("label", "probability", "index", "species", "viewpoint"):
        assert key in top, f"classify contract requires predictions[0].{key}"
    assert top["label"] == "wild_dog_general:upright"


def test_cascade_diagnostics_report_each_member():
    m = _cascade_with(
        router=("wild_dog", "ignore", 0.9),
        coat=("wild_dog_general", "other", 0.8),
        viewpoint=("wild_dog", "upright", 0.7),
    )
    out = m.predict(b"img")
    assert out["cascade"]["router"]["species"] == "wild_dog"
    assert out["cascade"]["coat"]["species"] == "wild_dog_general"
    assert out["cascade"]["viewpoint"]["viewpoint"] == "upright"


def test_ambiguous_classes_are_reachable_only_via_disagreement():
    """No member has an 'ambiguous' class; it must come from the cascade."""
    from app.models.densenet_wilddog_cascade import (
        AMBIGUOUS_BODY, AMBIGUOUS_TAIL,
    )
    for classes in (V1_COAT_CLASSES, V2_VIEWPOINT_CLASSES, V3_ROUTER_CLASSES):
        species = {c.split(":", 1)[0] for c in classes}
        assert AMBIGUOUS_BODY not in species
        assert AMBIGUOUS_TAIL not in species


def test_registered_in_model_registry():
    from app.models.model_handler import MODEL_REGISTRY
    entry = MODEL_REGISTRY["densenet-wilddog-cascade"]
    assert entry["class"] == "DenseNetWildDogCascadeModel"


def test_not_confusable_with_plain_densenet_classifier():
    """The cascade must be its own type: DenseNetClassifierModel's load()
    contract (checkpoint_path/checkpoint_paths) is not the cascade's, so it
    must not be substitutable for one."""
    from app.models.densenet_wilddog_cascade import DenseNetWildDogCascadeModel
    from app.models.densenet_classifier import DenseNetClassifierModel
    assert not isinstance(DenseNetWildDogCascadeModel(), DenseNetClassifierModel)
