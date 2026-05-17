# DenseNet Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `densenet-classifier` model_type to ml-service that loads N ensemble DenseNet checkpoints, supports compound `species:viewpoint` label parsing, and wires into `/pipeline/` as a valid classify-slot model alongside the existing EfficientNet path.

**Architecture:** Codex-reviewed design at `docs/plans/2026-05-16-densenet-classifier-design.md` (commit `f53f8b6`). New sibling class to `DenseNetOrientationModel`, ensemble-averaged softmax, shared parser helper used by both DenseNet-classifier and EfficientNet, router accepts both EfficientNet and DenseNet-classifier in the classify slot, parsed `iaClass`/`viewpoint` promoted from `predictions[0]` to top-level of each result.

**Tech Stack:** Python 3.11, PyTorch, FastAPI, pytest, timm (for HRNet), torchvision (for DenseNet201). Existing ml-service infrastructure (BaseModel ABC, ModelHandler registry).

---

## File Structure

```
app/utils/label_parsing.py                   # NEW: shared parse_class_label helper
app/models/densenet_classifier.py            # NEW: DenseNetClassifierModel
app/models/model_handler.py                  # MODIFY: register new type
app/models/efficientnet.py                   # MODIFY: delegate to shared helper
app/routers/pipeline_router.py               # MODIFY: accept new type + read predictions[0]
tests/test_label_parsing.py                  # NEW: layer-1 unit tests
tests/test_densenet_classifier.py            # NEW: layer-2 loader + predict tests
tests/test_pipeline_router_classifier.py     # NEW: layer-3 FastAPI integration tests
tests/test_efficientnet_sentinel.py          # NEW: EfficientNet regression test for sentinel suppression
```

---

### Task 1: Shared label-parsing helper (TDD)

**Files:**
- Create: `app/utils/label_parsing.py`
- Test: `tests/test_label_parsing.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_label_parsing.py
import pytest
from app.utils.label_parsing import parse_class_label


def test_compound_label_parses_species_and_viewpoint():
    assert parse_class_label("salamander_fire_adult:up", compound_labels=True) == \
        ("salamander_fire_adult", "up")


def test_compound_label_default_sentinel_suppresses_species():
    assert parse_class_label("species:left", compound_labels=True) == (None, "left")


def test_compound_label_explicit_empty_sentinel_keeps_species():
    assert parse_class_label("species:left", compound_labels=True,
                             sentinel_prefixes=[]) == ("species", "left")


def test_compound_label_custom_sentinel_list():
    assert parse_class_label("species:left", compound_labels=True,
                             sentinel_prefixes=["species", "viewpoint"]) == (None, "left")


def test_compound_true_no_colon_returns_viewpoint_only():
    assert parse_class_label("up", compound_labels=True) == (None, "up")


def test_compound_false_no_colon_returns_viewpoint_only():
    assert parse_class_label("up", compound_labels=False) == (None, "up")


def test_compound_false_with_colon_raises_value_error():
    with pytest.raises(ValueError) as exc_info:
        parse_class_label("salamander_fire_adult:up", compound_labels=False)
    assert "compound_labels" in str(exc_info.value).lower()


def test_compound_label_only_one_colon_splits_correctly():
    # Multi-colon labels split on first colon only.
    assert parse_class_label("a:b:c", compound_labels=True) == ("a", "b:c")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/c/ml-service
python3 -m pytest tests/test_label_parsing.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.utils.label_parsing'`

- [ ] **Step 3: Write the minimal implementation**

```python
# app/utils/label_parsing.py
"""Shared parser for class labels of the form `<species>:<viewpoint>`.

Used by both DenseNetClassifierModel and EfficientNetModel so compound-
label parsing behavior is identical across model types.
"""
from typing import List, Optional, Tuple

DEFAULT_SENTINEL_PREFIXES = ["species"]


def parse_class_label(
    label: str,
    compound_labels: bool,
    sentinel_prefixes: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Parse a class label into (species, viewpoint).

    Args:
        label: the raw class label string.
        compound_labels: whether labels are expected to be `<species>:<viewpoint>`.
        sentinel_prefixes: list of placeholder prefixes whose appearance suppresses
            species emission. Defaults to ['species'] (matches the deployed
            efficientnet-classifier convention where `species:` is a literal
            namespace, not a real species name). Pass [] to disable.

    Returns:
        (species, viewpoint). species is None when:
          - compound_labels is False and the label has no colon, OR
          - compound_labels is True but the label has no colon, OR
          - the prefix matches a sentinel.
        viewpoint is always populated (the whole label when no colon, or the
        suffix after the first colon when colon-bearing).

    Raises:
        ValueError: if compound_labels is False but the label contains ':'.
            This catches a config mistake (operator forgot to set
            compound_labels: true) at the earliest opportunity.
    """
    if sentinel_prefixes is None:
        sentinel_prefixes = DEFAULT_SENTINEL_PREFIXES

    if ":" not in label:
        return (None, label)

    if not compound_labels:
        raise ValueError(
            f"Label {label!r} contains ':' but compound_labels=False. "
            f"Either set compound_labels: true in the model config, "
            f"or fix the checkpoint to use non-compound labels."
        )

    prefix, suffix = label.split(":", 1)
    if prefix in sentinel_prefixes:
        return (None, suffix)
    return (prefix, suffix)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/c/ml-service
python3 -m pytest tests/test_label_parsing.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/ml-service
perl -i -pe 's/\r\n/\n/g' app/utils/label_parsing.py tests/test_label_parsing.py
git add app/utils/label_parsing.py tests/test_label_parsing.py
git commit -m "feat: shared parse_class_label helper for compound species:viewpoint labels"
```

---

### Task 2: EfficientNet delegation to shared helper

**Files:**
- Modify: `app/models/efficientnet.py` (the `parse_compound_labels` path, around line 246)
- Test: `tests/test_efficientnet_sentinel.py`

- [ ] **Step 1: Locate the current compound-parsing logic**

Run: `grep -n "parse_compound_labels\|':' in\|split(':'" /mnt/c/ml-service/app/models/efficientnet.py`

Expected output includes a line around 246 where EfficientNet manually splits on `:` and sets `entry['species'] = parts[0]`. Read the surrounding ~10 lines to understand context.

- [ ] **Step 2: Write the failing regression test**

```python
# tests/test_efficientnet_sentinel.py
"""Regression: EfficientNet's parse_compound_labels path must use the shared
parser, which suppresses sentinel prefixes like 'species'. Before this fix,
EfficientNet would set entry['species'] = 'species' (literal namespace) for
the deployed efficientnet-classifier whose labels are 'species:up' etc."""
import pytest


def test_efficientnet_sentinel_suppression(monkeypatch):
    """When a prediction's label is 'species:up' and the model has
    parse_compound_labels=True, the per-prediction `species` field must be
    None (sentinel suppression), not the literal string 'species'."""
    from app.models.efficientnet import EfficientNetModel

    # Build a minimal EfficientNetModel state. Don't actually load weights;
    # exercise just the post-inference parsing path.
    model = EfficientNetModel()
    model.label_map = {0: "species:up", 1: "species:down"}
    model.multi_label = False
    model.parse_compound_labels = True

    # Synthetic logits → predictions[0] should be 'species:up'.
    # Whatever helper EfficientNet uses for label-formatting, after this
    # task it should delegate to parse_class_label with the default
    # sentinel list ['species'].
    import torch
    fake_logits = torch.tensor([[5.0, 0.0]])
    result = model._format_predictions(fake_logits)  # method name TBD by Task 2 Step 3
    top = result["predictions"][0]
    assert top["label"] == "species:up"
    assert top.get("species") is None, \
        f"expected sentinel 'species' suppressed, got {top.get('species')!r}"
    assert top.get("viewpoint") == "up"
```

Note: the exact method name `_format_predictions` may differ in the actual code; in Step 3 you align the test to the real method by reading EfficientNet's existing post-inference code path.

- [ ] **Step 3: Refactor EfficientNet to use the shared helper**

Read `app/models/efficientnet.py` to find where compound parsing happens (around line 240-250 currently). Replace the manual `parts = label.split(':')` block with a call to `parse_class_label`:

```python
# Before (in efficientnet.py predict path, approx line 240-250):
if self.parse_compound_labels and ':' in entry['label']:
    parts = entry['label'].split(':')
    entry['species'] = parts[0]
    entry['viewpoint'] = parts[1] if len(parts) > 1 else None

# After:
from app.utils.label_parsing import parse_class_label
...
if self.parse_compound_labels:
    species, viewpoint = parse_class_label(
        entry['label'],
        compound_labels=True,
        sentinel_prefixes=getattr(self, 'sentinel_prefixes', None),
    )
    entry['species'] = species
    entry['viewpoint'] = viewpoint
```

The `getattr` with `None` default makes `sentinel_prefixes` optional: existing EfficientNet configs that don't specify it get the default `["species"]` from `parse_class_label`. Adding `sentinel_prefixes` to EfficientNet's `__init__`/`load` signature is a follow-up — out of scope for this commit beyond the `getattr` fallback.

- [ ] **Step 4: Run the regression test**

```bash
cd /mnt/c/ml-service
python3 -m pytest tests/test_efficientnet_sentinel.py -v
```

Expected: PASS. If the test references a method name that doesn't exist in `EfficientNetModel`, replace it with the actual method name discovered in Step 1.

- [ ] **Step 5: Run any existing EfficientNet tests to confirm no regression**

```bash
cd /mnt/c/ml-service
python3 -m pytest tests/test_efficientnet_config.py tests/test_classify_endpoint.py -v
```

Expected: existing tests pass (the shared parser preserves backward behavior for non-sentinel labels).

- [ ] **Step 6: Commit**

```bash
cd /mnt/c/ml-service
perl -i -pe 's/\r\n/\n/g' app/models/efficientnet.py tests/test_efficientnet_sentinel.py
git add app/models/efficientnet.py tests/test_efficientnet_sentinel.py
git commit -m "fix: EfficientNet compound-label parsing delegates to shared helper

Stops emitting literal 'species' as the species field for labels of the
form 'species:<viewpoint>' (sentinel suppression). The deployed
efficientnet-classifier checkpoint uses this exact label scheme, so this
fixes a long-standing data-quality leak in addition to centralizing the
parser for the upcoming DenseNetClassifierModel."
```

---

### Task 3: DenseNetClassifierModel module (TDD)

**Files:**
- Create: `app/models/densenet_classifier.py`
- Test: `tests/test_densenet_classifier.py`

This task implements the loader + predict separately to keep TDD steps small. Three sub-tasks: (a) loader with single-checkpoint path, (b) ensemble loading + validation, (c) predict with ensemble averaging.

- [ ] **Step 1: Write the failing loader tests (single + ensemble)**

```python
# tests/test_densenet_classifier.py
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
    The classifier weight shape is (num_classes, feat_dim)."""
    feat_dim = 2048 if hrnet else 1920
    # Architecture-detector keys (see densenet_orientation.py:120-130).
    arch_key = ("classifier.weight" if not hrnet
                else "classifier.weight")  # both use same final layer name
    state = OrderedDict()
    state[arch_key] = torch.zeros(num_classes, feat_dim)
    state["classifier.bias"] = torch.zeros(num_classes)
    # Sprinkle in enough non-trivial keys for arch detection to pick the
    # right backbone. (Adjust based on what densenet_orientation.py:120-130
    # actually checks — placeholder for now; the real test will use the
    # exact key names that detector inspects.)
    return {"state": state, "classes": classes} if classes else {"state": state}


def test_load_single_checkpoint_normalizes_to_list():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b", "c"])
    with patch("torch.load", return_value=fake):
        m = DenseNetClassifierModel()
        m.load(model_id="t", checkpoint_path="/fake/path.weights",
               device="cpu")
        assert len(m.models) == 1
        assert m.label_map == {0: "a", 1: "b", 2: "c"}
        assert m.compound_labels is False


def test_load_ensemble_three_matching_checkpoints():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b", "c"])
    with patch("torch.load", return_value=fake):
        m = DenseNetClassifierModel()
        m.load(model_id="t", checkpoint_paths=[
            "/fake/0.weights", "/fake/1.weights", "/fake/2.weights"
        ], device="cpu")
        assert len(m.models) == 3


def test_load_ensemble_mismatched_num_classes_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake_a = _fake_densenet_state(3, classes=["a", "b", "c"])
    fake_b = _fake_densenet_state(4, classes=["a", "b", "c", "d"])
    with patch("torch.load", side_effect=[fake_a, fake_b]):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="num_classes"):
            m.load(model_id="t", checkpoint_paths=[
                "/fake/0.weights", "/fake/1.weights"], device="cpu")


def test_load_ensemble_mismatched_class_order_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake_a = _fake_densenet_state(3, classes=["a", "b", "c"])
    fake_b = _fake_densenet_state(3, classes=["a", "c", "b"])
    with patch("torch.load", side_effect=[fake_a, fake_b]):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="classes"):
            m.load(model_id="t", checkpoint_paths=[
                "/fake/0.weights", "/fake/1.weights"], device="cpu")


def test_load_stale_classes_list_shorter_than_num_classes_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b"])   # only 2 names
    with patch("torch.load", return_value=fake):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="classes"):
            m.load(model_id="t", checkpoint_path="/fake/0.weights",
                   device="cpu")


def test_load_stale_classes_list_longer_than_num_classes_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b", "c", "d"])
    with patch("torch.load", return_value=fake):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="classes"):
            m.load(model_id="t", checkpoint_path="/fake/0.weights",
                   device="cpu")


def test_load_missing_classes_without_label_map_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3)  # no 'classes' key
    with patch("torch.load", return_value=fake):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="classes"):
            m.load(model_id="t", checkpoint_path="/fake/0.weights",
                   device="cpu")


def test_load_mixed_ensemble_some_have_classes_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake_with = _fake_densenet_state(3, classes=["a", "b", "c"])
    fake_without = _fake_densenet_state(3)
    with patch("torch.load", side_effect=[fake_with, fake_without]):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="classes"):
            m.load(model_id="t", checkpoint_paths=[
                "/fake/0.weights", "/fake/1.weights"], device="cpu")


def test_load_explicit_label_map_overrides_classes():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake_a = _fake_densenet_state(3, classes=["x", "y", "z"])  # wrong names
    fake_b = _fake_densenet_state(3, classes=["x", "y", "z"])
    with patch("torch.load", side_effect=[fake_a, fake_b]):
        m = DenseNetClassifierModel()
        m.load(model_id="t",
               checkpoint_paths=["/fake/0.weights", "/fake/1.weights"],
               label_map={"0": "a", "1": "b", "2": "c"},
               device="cpu")
        # Int coercion of string keys.
        assert m.label_map == {0: "a", 1: "b", 2: "c"}


def test_load_explicit_label_map_missing_index_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b", "c"])
    with patch("torch.load", return_value=fake):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="label_map"):
            m.load(model_id="t", checkpoint_path="/fake/0.weights",
                   label_map={"0": "x", "1": "y"},  # missing index 2
                   device="cpu")


def test_load_ensemble_indices_subset():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b", "c"])
    with patch("torch.load", return_value=fake) as mock_load:
        m = DenseNetClassifierModel()
        m.load(model_id="t", checkpoint_paths=[
            "/fake/0.weights", "/fake/1.weights", "/fake/2.weights"
        ], ensemble_indices=[0], device="cpu")
        assert len(m.models) == 1
        assert mock_load.call_count == 1
        # Confirm the right checkpoint was loaded.
        loaded_path = mock_load.call_args_list[0][0][0]
        assert "0.weights" in str(loaded_path)


def test_load_ensemble_indices_out_of_range_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(3, classes=["a", "b", "c"])
    with patch("torch.load", return_value=fake):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="ensemble_indices"):
            m.load(model_id="t", checkpoint_paths=[
                "/fake/0.weights"
            ], ensemble_indices=[0, 5], device="cpu")


def test_load_compound_false_with_colon_label_raises():
    from app.models.densenet_classifier import DenseNetClassifierModel
    fake = _fake_densenet_state(2, classes=["a:b", "c:d"])
    with patch("torch.load", return_value=fake):
        m = DenseNetClassifierModel()
        with pytest.raises(ValueError, match="compound_labels"):
            m.load(model_id="t", checkpoint_path="/fake/0.weights",
                   compound_labels=False, device="cpu")
```

- [ ] **Step 2: Run loader tests to verify they fail**

```bash
cd /mnt/c/ml-service
python3 -m pytest tests/test_densenet_classifier.py -v
```

Expected: `ModuleNotFoundError` for all tests.

- [ ] **Step 3: Implement the DenseNetClassifierModel loader**

Read `app/models/densenet_orientation.py` end-to-end first to lift architecture-detection logic (HRNet-W32 vs DenseNet201) and preprocessing (`_preprocess`). Then create:

```python
# app/models/densenet_classifier.py
"""DenseNet classifier with ensemble support and compound-label parsing.

Sibling to DenseNetOrientationModel — same checkpoint format and arch
detection, but oriented at the classify slot (top-level iaClass +
viewpoint promotion via shared parser) rather than the orientation slot.

Design: docs/plans/2026-05-16-densenet-classifier-design.md
"""
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from app.models.base_model import BaseModel
from app.utils.checkpoint_utils import get_checkpoint_path
from app.utils.label_parsing import parse_class_label

logger = logging.getLogger(__name__)


class DenseNetClassifierModel(BaseModel):
    """Ensemble-averaging DenseNet classifier for compound or
    pure-viewpoint label sets."""

    def __init__(self):
        super().__init__()
        self.models: List[nn.Module] = []
        self.label_map: Dict[int, str] = {}
        self.compound_labels: bool = False
        self.sentinel_prefixes: List[str] = ["species"]
        self.architecture: str = "densenet201"
        self.img_size: int = 224
        self.device: str = "cpu"

    def load(self, model_path: str = "", device: str = "cpu",
             model_id: str = "",
             checkpoint_path: Optional[str] = None,
             checkpoint_paths: Optional[List[str]] = None,
             img_size: int = 224,
             label_map: Optional[Dict] = None,
             compound_labels: bool = False,
             sentinel_prefixes: Optional[List[str]] = None,
             ensemble_indices: Optional[List[int]] = None,
             **kwargs) -> None:
        self.device = device
        self.img_size = img_size
        self.compound_labels = bool(compound_labels)
        if sentinel_prefixes is not None:
            self.sentinel_prefixes = list(sentinel_prefixes)

        # --- Resolve checkpoint paths ---
        if checkpoint_path and checkpoint_paths:
            raise ValueError(
                "Provide exactly one of checkpoint_path / checkpoint_paths"
            )
        if not checkpoint_path and not checkpoint_paths:
            raise ValueError(
                "Either checkpoint_path or checkpoint_paths is required"
            )
        paths = checkpoint_paths or [checkpoint_path]

        # --- Apply ensemble_indices subset selection ---
        if ensemble_indices is not None:
            for i in ensemble_indices:
                if not (0 <= i < len(paths)):
                    raise ValueError(
                        f"ensemble_indices contains out-of-range index "
                        f"{i} for checkpoint_paths of length {len(paths)}"
                    )
            paths = [paths[i] for i in ensemble_indices]

        # --- Load all checkpoints ---
        checkpoints = []
        for p in paths:
            actual = get_checkpoint_path(p)
            ck = torch.load(actual, map_location=device, weights_only=False)
            checkpoints.append(ck)

        # --- Determine num_classes and arch from the first checkpoint ---
        first_state = _state_dict_of(checkpoints[0])
        num_classes, self.architecture = _detect_arch_and_num_classes(first_state)

        # --- Validate every member matches num_classes ---
        for i, ck in enumerate(checkpoints[1:], start=1):
            n_i, _ = _detect_arch_and_num_classes(_state_dict_of(ck))
            if n_i != num_classes:
                raise ValueError(
                    f"Ensemble checkpoint {i} has num_classes={n_i}, "
                    f"first checkpoint has num_classes={num_classes}. "
                    f"All members must share num_classes."
                )

        # --- Resolve label map ---
        if label_map is not None:
            # Coerce string JSON keys to int.
            coerced = {int(k): v for k, v in label_map.items()}
            expected = set(range(num_classes))
            actual = set(coerced.keys())
            if actual != expected:
                raise ValueError(
                    f"label_map keys must be exactly {sorted(expected)}; "
                    f"got {sorted(actual)}"
                )
            self.label_map = coerced
        else:
            classes_lists = [ck.get("classes") for ck in checkpoints]
            if any(c is None for c in classes_lists):
                raise ValueError(
                    "Every checkpoint must carry a 'classes' list when no "
                    "explicit label_map is provided"
                )
            first_classes = classes_lists[0]
            if len(first_classes) != num_classes:
                raise ValueError(
                    f"Checkpoint 'classes' list has length "
                    f"{len(first_classes)} but classifier head has "
                    f"num_classes={num_classes} — stale metadata"
                )
            for i, c in enumerate(classes_lists[1:], start=1):
                if list(c) != list(first_classes):
                    raise ValueError(
                        f"Ensemble checkpoint {i} 'classes' differs from "
                        f"first checkpoint. Averaging by index requires "
                        f"identical class order across all members."
                    )
            self.label_map = {i: c for i, c in enumerate(first_classes)}

        # --- Validate compound_labels vs label content ---
        any_colon = any(":" in lbl for lbl in self.label_map.values())
        if any_colon and not self.compound_labels:
            raise ValueError(
                "label_map contains ':' but compound_labels=False. "
                "Set compound_labels: true to enable species:viewpoint "
                "parsing, or fix the labels."
            )
        if self.compound_labels and not any_colon:
            logger.warning(
                f"Model '{model_id}': compound_labels=true but no label "
                f"contains ':' — every emitted prediction will have "
                f"species=None, viewpoint=label."
            )

        # --- Build model instances and load weights ---
        # NOTE: NO nn.Softmax wrap during load. Softmax is applied once
        # per ensemble member inside predict(). Wrapping here would cause
        # double-softmax once predict() also applies softmax.
        self.models = []
        for ck in checkpoints:
            backbone = _build_backbone(self.architecture, num_classes)
            state = _state_dict_of(ck)
            cleaned = _strip_module_prefix(state)
            backbone.load_state_dict(cleaned, strict=False)
            backbone.to(device).eval()
            self.models.append(backbone)

        logger.info(
            f"Loaded DenseNetClassifierModel '{model_id}': "
            f"{len(self.models)}-member ensemble, num_classes={num_classes}, "
            f"compound_labels={self.compound_labels}"
        )

    def predict(self, image_bytes: bytes,
                bbox: Optional[List[int]] = None,
                theta: float = 0.0) -> Dict[str, Any]:
        # Preprocess once; same input for every ensemble member.
        inputs = self._preprocess(image_bytes, bbox, theta)
        summed = None
        with torch.no_grad():
            for m in self.models:
                probs = torch.softmax(m(inputs), dim=-1)
                summed = probs if summed is None else summed + probs
        avg = summed / len(self.models)
        return self._format_output(avg)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "densenet-classifier",
            "device": self.device,
            "img_size": self.img_size,
            "num_classes": len(self.label_map),
            "label_map": self.label_map,
            "compound_labels": self.compound_labels,
            "ensemble_size": len(self.models),
            "architecture": self.architecture,
        }

    # ---- helpers ----

    def _preprocess(self, image_bytes, bbox, theta):
        # Reuse DenseNetOrientationModel's preprocessing as the single
        # source of truth — same crop/resize/normalize pipeline. Import
        # locally to avoid circular dependency at module load.
        from app.models.densenet_orientation import DenseNetOrientationModel
        return DenseNetOrientationModel._preprocess_tensor(
            image_bytes, bbox, theta, self.img_size, self.device
        )

    def _format_output(self, avg: torch.Tensor) -> Dict[str, Any]:
        # Top-K=3 (or num_classes if smaller). Each entry parsed.
        k = min(3, avg.shape[-1])
        top_probs, top_idxs = torch.topk(avg, k, dim=-1)
        top_probs = top_probs[0].tolist()
        top_idxs = top_idxs[0].tolist()

        predictions = []
        for prob, idx in zip(top_probs, top_idxs):
            label = self.label_map[int(idx)]
            species, viewpoint = parse_class_label(
                label, self.compound_labels, self.sentinel_prefixes
            )
            predictions.append({
                "label": label,
                "probability": float(prob),
                "index": int(idx),
                "species": species,
                "viewpoint": viewpoint,
            })

        top = predictions[0]
        return {
            "class": top["label"],
            "probability": top["probability"],
            "class_id": top["index"],
            "predictions": predictions,
        }


# ---- module-private helpers ----

def _state_dict_of(checkpoint):
    """WBIA checkpoints are dicts with a 'state' key; raw state-dicts
    are also accepted."""
    if isinstance(checkpoint, dict) and "state" in checkpoint:
        return checkpoint["state"]
    return checkpoint


def _strip_module_prefix(state_dict):
    """Remove 'module.' prefix from DataParallel-wrapped state-dicts."""
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v
    return cleaned


def _detect_arch_and_num_classes(state):
    """Lift the architecture-detection convention from
    DenseNetOrientationModel.load(). Returns (num_classes, arch_str)."""
    cleaned = _strip_module_prefix(state)
    # HRNet checkpoints often have keys with 'transition' or 'stage' prefixes;
    # DenseNet201 checkpoints have 'features.norm5.weight' or similar.
    is_hrnet = any("transition" in k or "stage" in k for k in cleaned)
    arch = "hrnet_w32" if is_hrnet else "densenet201"

    classifier_keys = [k for k in cleaned if "classifier.weight" in k]
    if not classifier_keys:
        raise ValueError("No classifier.weight key in checkpoint state-dict")
    n = cleaned[classifier_keys[-1]].shape[0]
    return n, arch


def _build_backbone(architecture, num_classes):
    """Build a fresh model with the given architecture and num_classes
    output. No Softmax tail."""
    if architecture == "hrnet_w32":
        import timm
        model = timm.create_model("hrnet_w32", pretrained=False,
                                  num_classes=num_classes)
    else:
        from torchvision import models
        model = models.densenet201(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    return model
```

- [ ] **Step 4: Run loader tests**

```bash
cd /mnt/c/ml-service
python3 -m pytest tests/test_densenet_classifier.py -v
```

Expected: most pass. Some may fail because the synthetic state-dict in `_fake_densenet_state` doesn't have enough keys to satisfy `_detect_arch_and_num_classes`. If failing, adjust `_fake_densenet_state` to include keys that match the detector (`features.norm5.weight` for densenet path).

- [ ] **Step 5: Write the predict / ensemble averaging test**

Append to `tests/test_densenet_classifier.py`:

```python
def test_predict_ensemble_averaging_matches_expected_softmax():
    """The crux of the design: averaging post-softmax across N members
    must produce mathematically-correct probabilities, NOT double-softmax."""
    from app.models.densenet_classifier import DenseNetClassifierModel
    import torch.nn as nn

    # Build two synthetic ensemble members that always return canned logits.
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

    # Stub _preprocess to bypass image-bytes decoding.
    m._preprocess = lambda *a, **kw: torch.zeros(1, 3, 224, 224)

    result = m.predict(b"fakebytes")

    # Expected probabilities:
    #   softmax([5,0,0]) ≈ [0.9866, 0.0067, 0.0067]
    #   softmax([0,0,5]) ≈ [0.0067, 0.0067, 0.9866]
    #   averaged       ≈ [0.4967, 0.0067, 0.4967]
    # Top-3 should have classes 0 and 2 with prob ~0.4967, class 1 ~0.0067.
    preds = result["predictions"]
    by_idx = {p["index"]: p for p in preds}
    assert abs(by_idx[0]["probability"] - 0.4967) < 1e-3, by_idx
    assert abs(by_idx[2]["probability"] - 0.4967) < 1e-3, by_idx
    assert abs(by_idx[1]["probability"] - 0.0067) < 1e-3, by_idx
    # Sanity: not double-softmaxed (would give ~0.39 / 0.30 / 0.30).
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
    # Every entry parsed, not just top-1.
    for p in result["predictions"]:
        assert p["species"] is not None
        assert p["viewpoint"] in ("up", "left", "right")
    # Top-1 is the adult:up class.
    top = result["predictions"][0]
    assert top["species"] == "salamander_fire_adult"
    assert top["viewpoint"] == "up"


def test_predict_single_member_no_implicit_softmax_wrap():
    """Single-member ensemble must not double-softmax. softmax([5,0,0])
    top probability is ~0.9866, not softmax-of-softmax (~0.4 something)."""
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
```

- [ ] **Step 6: Run predict tests**

```bash
cd /mnt/c/ml-service
python3 -m pytest tests/test_densenet_classifier.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
cd /mnt/c/ml-service
perl -i -pe 's/\r\n/\n/g' app/models/densenet_classifier.py tests/test_densenet_classifier.py
git add app/models/densenet_classifier.py tests/test_densenet_classifier.py
git commit -m "feat: DenseNetClassifierModel with ensemble + compound-label support"
```

---

### Task 4: Register DenseNetClassifierModel in ModelHandler

**Files:**
- Modify: `app/models/model_handler.py` (MODEL_REGISTRY dict, around line 5-40)

- [ ] **Step 1: Add the registry entry**

Read the current registry to find the right insertion point (just after the existing `densenet-orientation` entry).

```python
# app/models/model_handler.py — add to MODEL_REGISTRY:
'densenet-classifier': {
    'module': 'app.models.densenet_classifier',
    'class': 'DenseNetClassifierModel'
},
```

Place it just after the existing `'densenet-orientation'` entry to keep the related DenseNet entries co-located.

- [ ] **Step 2: Verify the registry loads cleanly**

```bash
cd /mnt/c/ml-service
python3 -c "from app.models.model_handler import MODEL_REGISTRY; print('densenet-classifier:', MODEL_REGISTRY.get('densenet-classifier'))"
```

Expected: `densenet-classifier: {'module': 'app.models.densenet_classifier', 'class': 'DenseNetClassifierModel'}`

- [ ] **Step 3: Commit**

```bash
cd /mnt/c/ml-service
perl -i -pe 's/\r\n/\n/g' app/models/model_handler.py
git add app/models/model_handler.py
git commit -m "feat: register densenet-classifier in MODEL_REGISTRY"
```

---

### Task 5: pipeline_router edits (accept new model type + emit parsed fields)

**Files:**
- Modify: `app/routers/pipeline_router.py` (around lines 8-10, 93, 264-298)
- Test: `tests/test_pipeline_router_classifier.py`

- [ ] **Step 1: Write the failing integration tests**

```python
# tests/test_pipeline_router_classifier.py
"""Layer-3 integration tests for the new classify-slot model type."""
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


def _make_app_with_models(predict_model, classify_model, extract_model):
    """Build an app with monkey-patched ModelHandler."""
    from app.main import app  # adjust if main module is different
    handler = MagicMock()
    handler.get_model.side_effect = lambda mid: {
        "p": predict_model, "c": classify_model, "e": extract_model
    }.get(mid)
    handler.get_model_info.side_effect = lambda mid: {
        "p": {"config": {}}, "c": {"config": {}},
        "e": {"config": {"version": 4.1}},
    }.get(mid)
    app.state.model_handler = handler
    return TestClient(app)


def test_pipeline_classify_densenet_classifier_emits_top_level_iaclass_and_viewpoint():
    from app.models.densenet_classifier import DenseNetClassifierModel
    from app.models.efficientnet import EfficientNetModel
    from app.models.miewid import MiewidModel
    from app.models.yolo_ultralytics import YOLOUltralyticsModel

    pm = MagicMock(spec=YOLOUltralyticsModel)
    pm.predict.return_value = {
        "predictions": [{
            "bbox": [0, 0, 10, 10],
            "theta": 0.0,
            "score": 0.9,
            "class": "detection",
            "class_id": 0
        }]
    }
    cm = MagicMock(spec=DenseNetClassifierModel)
    cm.predict.return_value = {
        "class": "salamander_fire_adult:up",
        "probability": 0.95,
        "class_id": 0,
        "predictions": [{
            "label": "salamander_fire_adult:up", "probability": 0.95,
            "index": 0,
            "species": "salamander_fire_adult", "viewpoint": "up",
        }, {
            "label": "salamander_fire_juvenile:left", "probability": 0.03,
            "index": 1,
            "species": "salamander_fire_juvenile", "viewpoint": "left",
        }],
    }
    em = MagicMock(spec=MiewidModel)
    em.extract_embeddings.return_value = __import__("numpy").array([[0.1] * 2152])

    client = _make_app_with_models(pm, cm, em)
    resp = client.post("/pipeline/", json={
        "predict_model_id": "p",
        "classify_model_id": "c",
        "extract_model_id": "e",
        "image_uri": "data:image/png;base64,iVBORw0KGgo=",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert len(body["results"]) == 1
    r = body["results"][0]
    assert r["iaClass"] == "salamander_fire_adult"
    assert r["viewpoint"] == "up"
    # raw classification kept too
    assert r["classification"]["class"] == "salamander_fire_adult:up"


def test_pipeline_classify_densenet_classifier_pure_viewpoint_omits_iaclass():
    from app.models.densenet_classifier import DenseNetClassifierModel
    from app.models.miewid import MiewidModel
    from app.models.yolo_ultralytics import YOLOUltralyticsModel

    pm = MagicMock(spec=YOLOUltralyticsModel)
    pm.predict.return_value = {
        "predictions": [{"bbox": [0, 0, 10, 10], "theta": 0.0,
                         "score": 0.9, "class": "detection", "class_id": 0}]
    }
    cm = MagicMock(spec=DenseNetClassifierModel)
    cm.predict.return_value = {
        "class": "up", "probability": 0.8, "class_id": 0,
        "predictions": [
            {"label": "up", "probability": 0.8, "index": 0,
             "species": None, "viewpoint": "up"},
        ],
    }
    em = MagicMock(spec=MiewidModel)
    em.extract_embeddings.return_value = __import__("numpy").array([[0.1] * 2152])

    client = _make_app_with_models(pm, cm, em)
    resp = client.post("/pipeline/", json={
        "predict_model_id": "p", "classify_model_id": "c",
        "extract_model_id": "e",
        "image_uri": "data:image/png;base64,iVBORw0KGgo=",
    })
    body = resp.json()
    r = body["results"][0]
    assert r["viewpoint"] == "up"
    assert "iaClass" not in r


def test_pipeline_classify_densenet_orientation_rejected_with_400():
    from app.models.densenet_orientation import DenseNetOrientationModel
    from app.models.miewid import MiewidModel
    from app.models.yolo_ultralytics import YOLOUltralyticsModel

    pm = MagicMock(spec=YOLOUltralyticsModel)
    cm = MagicMock(spec=DenseNetOrientationModel)
    em = MagicMock(spec=MiewidModel)

    client = _make_app_with_models(pm, cm, em)
    resp = client.post("/pipeline/", json={
        "predict_model_id": "p", "classify_model_id": "c",
        "extract_model_id": "e", "image_uri": "data:image/png;base64,iVBORw0KGgo=",
    })
    assert resp.status_code == 400
```

- [ ] **Step 2: Run integration tests to verify they fail**

```bash
cd /mnt/c/ml-service
python3 -m pytest tests/test_pipeline_router_classifier.py -v
```

Expected: failures — router doesn't yet accept DenseNetClassifierModel in classify slot.

- [ ] **Step 3: Edit pipeline_router.py imports and type check**

Find the existing imports near the top of `app/routers/pipeline_router.py`:

```python
# Add import alongside the existing model imports:
from app.models.densenet_classifier import DenseNetClassifierModel
```

Find the classify-model type check (around line 93). Replace:

```python
if not isinstance(classify_model, EfficientNetModel):
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Model '{pipeline_request.classify_model_id}' is not an EfficientNet model. Only EfficientNet models support classification."
    )
```

with:

```python
if not isinstance(classify_model, (EfficientNetModel, DenseNetClassifierModel)):
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Model '{pipeline_request.classify_model_id}' must be "
               f"EfficientNet or DenseNet-classifier for the classify slot."
    )
```

- [ ] **Step 4: Edit per-bbox result construction**

Find the existing classify-result extraction (around line 264-280). Add top-level parsed-field extraction:

```python
# Find the existing block that reads classify_result['predictions'][0] and
# builds top_classification. After that block, add:

top_species = None
top_viewpoint = None
if isinstance(classify_result, dict) and 'predictions' in classify_result \
        and classify_result['predictions']:
    top_class = classify_result['predictions'][0]
    # ... existing top_classification assignment unchanged ...
    top_species = top_class.get('species')
    top_viewpoint = top_class.get('viewpoint')
```

Find the existing `bbox_result = { ... }` dict construction (around line 286-298). After it, add:

```python
# After bbox_result is constructed but before pipeline_results.append:
if top_species is not None:
    bbox_result['iaClass'] = top_species
if top_viewpoint is not None:
    bbox_result['viewpoint'] = top_viewpoint
```

- [ ] **Step 5: Run integration tests to verify they pass**

```bash
cd /mnt/c/ml-service
python3 -m pytest tests/test_pipeline_router_classifier.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Run the full test suite to check for regressions**

```bash
cd /mnt/c/ml-service
python3 -m pytest tests/ -v --ignore=tests/test_wbia_compat.py --ignore=tests/test_wbia_stress.py --ignore=tests/test_stress.py
```

Expected: no failures (the test scripts that need a live ml-service stay ignored).

- [ ] **Step 7: Commit**

```bash
cd /mnt/c/ml-service
perl -i -pe 's/\r\n/\n/g' app/routers/pipeline_router.py tests/test_pipeline_router_classifier.py
git add app/routers/pipeline_router.py tests/test_pipeline_router_classifier.py
git commit -m "feat: /pipeline/ accepts densenet-classifier; emits parsed iaClass+viewpoint"
```

---

### Task 6: Final verification + Codex code review

**Files:** (none new)

- [ ] **Step 1: Run the full test suite one more time**

```bash
cd /mnt/c/ml-service
python3 -m pytest tests/ -v --ignore=tests/test_wbia_compat.py --ignore=tests/test_wbia_stress.py --ignore=tests/test_stress.py
```

Expected: all pass.

- [ ] **Step 2: Inspect git log of new commits**

```bash
cd /mnt/c/ml-service
git log --oneline main ^origin/main
```

Expected: 5 commits — task 1 (shared parser), task 2 (efficientnet delegation), task 3 (densenet-classifier model), task 4 (registry), task 5 (router edits).

- [ ] **Step 3: Send the consolidated diff to Codex for code review**

Compose a review prompt at `/tmp/codex-densenet-classifier-code-review.md` covering:
- Pointer to design doc commit (`f53f8b6`).
- Diff range: `git diff origin/main..HEAD -- app/ tests/`.
- Explicit ask: review each commit against the design, flag deviations,
  ensure test coverage matches the design's layer-1/2/3 plan.

Run: `cat /tmp/codex-densenet-classifier-code-review.md | codex exec --skip-git-repo-check -`

- [ ] **Step 4: Address Codex code-review findings**

If Codex finds issues, create a single follow-up commit (or amend the
relevant task's commit) that resolves them. Re-run pytest. If structural
changes are needed, re-send to Codex for verify.

- [ ] **Step 5: Re-verify against the live ml-service deployment (manual)**

Once code review is clean, deploy the new model_type to a dev ml-service
and run the existing `MlServiceLiveIntegrationTest` against it with a
salamander_fire_v2 IA.json `_mlservice_conf` entry that points
`classify_model_id` at the new densenet-classifier entry. Out of scope
for the unit-test pass; tracked as the next operational task.

---

## Self-review checklist

- [x] **Spec coverage:** every section of the design doc maps to a task — shared helper (Task 1), EfficientNet delegation (Task 2), DenseNetClassifierModel (Task 3), registry (Task 4), router (Task 5), Codex review (Task 6).
- [x] **Placeholder scan:** no "TBD" / "implement later" / abstract "handle edge cases" placeholders. All code blocks contain real code.
- [x] **Type consistency:** field names across tests and implementation match — `compound_labels`, `sentinel_prefixes`, `ensemble_indices`, `predictions[*].species`, `predictions[*].viewpoint`. Top-level result fields are `iaClass`, `viewpoint`. Helper is `parse_class_label` everywhere.

## Execution Handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch with checkpoints.
