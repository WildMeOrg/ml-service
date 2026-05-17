# Design: `densenet-classifier` model type for compound species:viewpoint labels

## Context

ml-service's `/pipeline/` endpoint currently restricts `classify_model_id` to
EfficientNet (`pipeline_router.py:93` does `isinstance(classify_model,
EfficientNetModel)`). The Wildbook v2 migration (`migrate-ml-service-v2`
branch) needs to port legacy WBIA labelers that use **DenseNet** with
compound `species:viewpoint` class labels — specifically `salamander_fire_v2`,
which has three classes:

```
['salamander_fire_adult:up', 'salamander_fire_juvenile:left', 'salamander_fire_juvenile:right']
```

WBIA queries this single model twice (`wbia/web/apis_detect.py:925-934`):
once with `get_property('labeler', ..., 'viewpoint')` to set the
annotation's viewpoint, and again — when `use_labeler_species: true` — with
`get_property('labeler', ..., 'species')` to set the annotation's species.
ml-service has no equivalent today: there is no DenseNet classifier model
type, and the router rejects DenseNet from the classify slot.

The corresponding Wildbook v2 consumer side (`MlServiceProcessor.persist
Detections` at `src/main/java/org/ecocean/ia/MlServiceProcessor.java:320-341`)
already reads top-level `viewpoint` and `iaClass` from each `/pipeline/`
result, but neither field exists in the current ml-service response shape.
The v2 client therefore creates annotations with null viewpoint and null
iaClass today — a pre-existing gap surfaced by this work.

## Goal

Enable ml-service to serve DenseNet labelers with compound or pure-viewpoint
class labels in the classify slot. Emit parsed species and viewpoint at the
top level of each `/pipeline/` result so Wildbook v2 can populate annotation
fields without code changes.

## Non-goals

- Migrating existing `densenet-orientation` users. They stay on that model
  type, loaded via `orientation_model_id`, emitting into `orientation.label`.
- **EfficientNet without `parse_compound_labels` is unchanged.** Its
  existing predict path keeps producing the same per-prediction dicts.
- A pure-continuous-theta orientation predictor. ml-service does not have
  one today; out of scope here.
- Wildbook-side changes. The v2 contract already reads the right top-level
  fields with `optString(..., null)` fallbacks.

## Incidental changes to EfficientNet

EfficientNet's existing `parse_compound_labels=True` path (`efficientnet.py:246`)
is delegated to the new shared `parse_class_label` helper. Effects:

- Per-prediction `species` and `viewpoint` fields continue to appear on
  each entry of `predictions[*]`, same shape as before.
- Sentinel suppression now applies (the deployed `"species:up"`-style
  classifier no longer leaks a literal species `"species"` — `species`
  becomes `None`, `viewpoint` stays correct).
- Router promotion of top-1 `species`/`viewpoint` to result-level
  `iaClass`/`viewpoint` now works for EfficientNet too (because the
  router reads from `predictions[0]`, single source of truth across both
  model types). No model-type branch in the router.

These are deliberate, additive, and covered by the Layer 3 EfficientNet
regression test.

## Design

### New module: `app/models/densenet_classifier.py`

`DenseNetClassifierModel(BaseModel)` — mirrors `DenseNetOrientationModel`'s
structure with three additions: ensemble loading, compound-label parsing,
and a different output shape suited for the classify slot.

**Load signature** (must accept every key the model_config.json example
uses, because `ModelHandler.load_model` passes config kwargs directly into
`load()` at `model_handler.py:74`):

```python
def load(self, model_path: str = "", device: str = 'cpu', model_id: str = "",
         checkpoint_path: str = None,
         checkpoint_paths: list = None,
         img_size: int = 224,
         label_map: dict = None,
         compound_labels: bool = False,
         sentinel_prefixes: list = None,    # default ["species"] in body
         ensemble_indices: list = None,     # default: load all
         **kwargs):                          # tolerate forward-compatible config keys
```

Loader behavior:

- Exactly one of `checkpoint_path` / `checkpoint_paths` must be provided.
  `checkpoint_path` is normalized internally into a single-element
  `checkpoint_paths` list.
- Optional `ensemble_indices: list[int]` config selects a subset of
  `checkpoint_paths` to actually load (for real-time fallback parity with
  WBIA's single-member usage). Default: load all configured paths.
- Each selected checkpoint is downloaded (if URL) and loaded via
  `torch.load`.
- Architecture is detected from the first checkpoint's state-dict keys
  (HRNet-W32 vs DenseNet201), reusing the logic from
  `densenet_orientation.py:120-130`.
- `num_classes` is derived from the first checkpoint's classifier weight
  shape. **All subsequent checkpoints must match `num_classes` exactly.**
- **When no explicit `label_map` is provided**, EVERY checkpoint must
  carry a `classes` list (WBIA format), AND `len(classes) == num_classes`
  (derived from the classifier weight shape), AND all checkpoints must
  match the first checkpoint's `classes` exactly (same length, same
  names, same order). Mixed "some have classes, some don't" is a
  ValueError at load time. A stale `classes` list shorter or longer than
  `num_classes` is also a ValueError — averaging by index demands every
  emitted label-index lookup succeed; a stale `classes` list of length
  `n != num_classes` would KeyError silently at predict time or worse,
  silently mislabel valid predictions. (EfficientNet documents the same
  risk at `efficientnet.py:104` but only truncates the label_map — for
  the ensemble loader we prefer fail-fast.)
- **When an explicit config `label_map` is provided**, it overrides
  whatever class names the checkpoints carry. The loader validates:
  - `num_classes` matches across every member,
  - `label_map` keys cover exactly `{0, 1, ..., num_classes - 1}` —
    no missing indices, no extras. Stringly-keyed JSON config values
    are coerced to int before this check.
  - The checkpoints' own `classes` lists, if any, are NOT cross-checked
    when `label_map` is explicit. Operator's choice has been to
    intentionally rebrand.
- **The classifier head is NOT wrapped in `nn.Softmax` at load time.**
  Models output raw logits. The ensemble-averaging step in `predict()` is
  the single softmax site, so we never double-softmax. (This differs
  intentionally from `DenseNetOrientationModel.load()` line 131, which
  appends Softmax during load. The orientation model is single-checkpoint
  and applies softmax inside the model graph; for an ensemble we must
  softmax outside so the average is mathematically meaningful.)
- All loaded `nn.Module` instances are stored as `self.models` (list);
  `self.compound_labels` carries the flag; `self.sentinel_prefixes`
  carries the configurable suppression list (default `["species"]`).

**Inference:**

```python
def predict(self, image_bytes, bbox=None, theta=0.0):
    # Same preprocessing as DenseNetOrientationModel.predict.
    inputs = self._preprocess(image_bytes, bbox, theta)
    summed = None
    with torch.no_grad():
        for m in self.models:
            # Models are loaded WITHOUT a Softmax tail (see loader note).
            # We softmax each member's raw logits here, then sum.
            probs = torch.softmax(m(inputs), dim=-1)
            summed = probs if summed is None else summed + probs
    avg = summed / len(self.models)
    return self._format_output(avg)
```

`_format_output(avg)` builds the result dict. Top-K (K=3 by default to
mirror EfficientNet) `predictions` list is always included. **Every entry
in `predictions` carries its own parsed `species` and `viewpoint`**, not
just the top-1 — `wbia_compat_router.py:308` and other downstream
consumers may inspect runner-ups.

Parsing rules (single source of truth, factored into a shared helper
`parse_class_label(label, compound_labels, sentinel_prefixes)`):

| Input | `compound_labels` | Output `(species, viewpoint)` |
|---|---|---|
| `"salamander_fire_adult:up"` | `True`  | `("salamander_fire_adult", "up")` |
| `"species:left"` (sentinel)   | `True`  | `(None, "left")` |
| `"up"` (no colon)             | `True`  | `(None, "up")` |
| `"up"`                        | `False` | `(None, "up")` |
| `"salamander_fire_adult:up"`  | `False` | **load-time error**: colon-bearing label encountered with `compound_labels: False` is a config mistake — fail-fast at load with a clear message, do not silently emit `(None, None)`. |

The helper lives in a small new module (e.g. `app/utils/label_parsing.py`)
and is also imported by `EfficientNetModel`'s compound-parsing path, so
the sentinel suppression applies uniformly — addressing the gap where
EfficientNet today produces literal species `"species"` from `"species:up"`-
style labels.

Top-level `class` / `probability` / `class_id` are also surfaced for
direct-call ergonomics (e.g. `/classify/` or `wbia_compat_router`), but
the router promotes `iaClass`/`viewpoint` from `predictions[0]`, not from
the top level of this dict. Single source of truth = each `predictions[*]`
entry.

```python
return {
  'class': top_label,
  'probability': top_prob,
  'class_id': top_idx,
  'predictions': [
    {'label': ..., 'probability': ..., 'index': ..., 'species': ..., 'viewpoint': ...},
    ...  # every entry parsed, not just top-1
  ],
}
```

EfficientNet's existing `predict()` already puts parsed `species`/`viewpoint`
on each `predictions[*]` entry when `parse_compound_labels=True`
(`efficientnet.py:246`), so once the shared `parse_class_label` helper is
in use, the router gets uniform behavior across both model types without
the router needing model-type branches.

### Registry: `app/models/model_handler.py`

Add one entry to `MODEL_REGISTRY`:

```python
'densenet-classifier': {
    'module': 'app.models.densenet_classifier',
    'class': 'DenseNetClassifierModel'
}
```

### Router: `app/routers/pipeline_router.py`

Three edits:

**(1) Import + relax classify-slot type check (~line 8-10, 93):**

```python
from app.models.densenet_classifier import DenseNetClassifierModel  # NEW

if not isinstance(classify_model, (EfficientNetModel, DenseNetClassifierModel)):
    raise HTTPException(status_code=400, detail=
        f"Model '{pipeline_request.classify_model_id}' must be EfficientNet "
        f"or DenseNet-classifier for the classify slot.")
```

`orientation_model_id`'s `DenseNetOrientationModel` check stays as-is
(the orient slot is unchanged).

**(2) Per-bbox result construction (~line 264-298):**

```python
top_species  = None
top_viewpoint = None
if isinstance(classify_result, dict) and 'predictions' in classify_result \
        and classify_result['predictions']:
    top_class = classify_result['predictions'][0]
    top_classification = {
        'class':       top_class.get('label'),
        'probability': top_class.get('probability'),
        'class_id':    top_class.get('index'),
    }
    # Read parsed fields from the top-1 prediction. Single source of
    # truth: both DenseNetClassifier and EfficientNet (existing
    # parse_compound_labels path) put parsed `species`/`viewpoint` on
    # each entry of `predictions[*]`. The router does not need to know
    # which model type produced the result.
    top_species   = top_class.get('species')
    top_viewpoint = top_class.get('viewpoint')
```

**(3) Add to `bbox_result` (~line 285-298):**

```python
if top_species is not None:
    bbox_result['iaClass'] = top_species
if top_viewpoint is not None:
    bbox_result['viewpoint'] = top_viewpoint
```

Fields omitted entirely when null — keeps the JSON shape clean and aligns
with `optString(..., null)` semantics on the Wildbook side.

### Response shape (additive)

Each `/pipeline/` result MAY now carry, in addition to the existing fields:

- `iaClass: str` — parsed species portion of a compound label, when
  applicable. Equals the value Wildbook v2 reads to populate annotation
  iaClass.
- `viewpoint: str` — top-1 viewpoint label. For pure-viewpoint models this
  is just the top class.

These are top-level on each result, NOT nested under `classification`.
`classification: {class, probability, class_id}` continues to carry the
raw top-1 label unchanged. `pipeline_results` (legacy alias) and other
v2-contract fields (`success`, `results`, `embedding_model_id`,
`embedding_model_version`) are unaffected.

### `model_config.json` example

```json
{
    "model_id": "salamander_fire_v2_label",
    "model_type": "densenet-classifier",
    "checkpoint_paths": [
        "/datasets/labeler.salamander_fire.v2/labeler.0.weights",
        "/datasets/labeler.salamander_fire.v2/labeler.1.weights",
        "/datasets/labeler.salamander_fire.v2/labeler.2.weights"
    ],
    "img_size": 224,
    "compound_labels": true,
    "sentinel_prefixes": ["species"]
}
```

`sentinel_prefixes` is optional; defaults to `["species"]`. Set it to a
broader list if your deployment has other placeholder namespaces; set it
to `[]` to disable sentinel suppression entirely.

`ensemble_indices: [0]` (optional) would load only `labeler.0.weights` for
a single-member real-time fallback — matches WBIA's behavior when
configured for a single ensemble member.

A viewpoint-only DenseNet (most species) would simply omit `compound_labels`
and use either a single `checkpoint_path` or an ensemble list.

## Backwards compatibility

| Path | Effect |
|---|---|
| Existing EfficientNet `classify_model_id` calls | Unchanged (isinstance check now accepts a tuple). |
| Existing `orientation_model_id` calls with `densenet-orientation` | Untouched. |
| Wildbook v2 `MlServiceProcessor.persistDetections` | Already reads `result.optString("iaClass", ...)` / `viewpoint`. With this change these fields appear on responses that use a compound DenseNet classifier; otherwise still absent (and `optString` defaults to null). |
| Legacy Wildbook (pre-v2) consumers | Unaffected; they don't look at these fields. |

## Failure modes

- `checkpoint_path` AND `checkpoint_paths` both set, OR neither set →
  ValueError at load time.
- Ensemble members with mismatched `num_classes` → ValueError at load time.
- Ensemble members with matching `num_classes` but mismatched `classes`
  ordering → ValueError at load time (averaging would corrupt
  probabilities silently if allowed).
- `compound_labels: true` but no label contains `":"` → load-time WARN log;
  predict still works (every label becomes viewpoint-only with
  `species = None`).
- `compound_labels: false` but at least one label contains `":"` → load-
  time ValueError. This is a config mistake — either the operator forgot
  the flag, or the checkpoint was misconfigured. Fail fast with a clear
  message rather than silently emit `(None, None)`.
- `ensemble_indices` references an index out of range → ValueError at
  load time.
- Inference failure on any single ensemble member → entire predict raises
  (consistent with how the existing extract path treats embedding failures
  after the v2 contract fix in commit `3fbed82`).
- Classify model is neither EfficientNet nor DenseNetClassifier → HTTP 400
  (existing pattern).

## Tests

Three layers; first two are fast and require no real checkpoint files.

**Layer 1 — pure-function helpers (`test_label_parsing.py`):**

- `parse_class_label("salamander_fire_adult:up", compound_labels=True)`
  → `("salamander_fire_adult", "up")`
- `parse_class_label("species:left", compound_labels=True)` →
  `(None, "left")` (default sentinel suppression)
- `parse_class_label("species:left", compound_labels=True,
  sentinel_prefixes=[])` → `("species", "left")` (suppression disabled)
- `parse_class_label("species:left", compound_labels=True,
  sentinel_prefixes=["species", "viewpoint"])` → `(None, "left")` (custom
  sentinel list)
- `parse_class_label("up", compound_labels=True)` → `(None, "up")`
- `parse_class_label("up", compound_labels=False)` → `(None, "up")`
- `parse_class_label("salamander_fire_adult:up", compound_labels=False)`
  → **raises `ValueError`** (config mistake; fail fast)

**Layer 2 — loader / ensemble (mocked `torch.load`):**

- Loading 3 ensemble checkpoints with matching `num_classes` AND matching
  `classes` order produces `len(self.models) == 3`.
- Loading 2 checkpoints with mismatched classifier weight shapes raises
  ValueError.
- Loading 2 checkpoints with matching `num_classes` but different
  `classes` order (e.g., `[A, B, C]` vs `[A, C, B]`) raises ValueError.
- Explicit `label_map` overrides checkpoint `classes` and skips the
  cross-checkpoint class-order check (still requires matching
  `num_classes`).
- Single `checkpoint_path` normalizes to a length-1 list internally.
- `ensemble_indices: [0]` loads only the first checkpoint.
- `ensemble_indices: [5]` against a 3-element `checkpoint_paths` raises
  ValueError.
- Predict averages softmax across N members with a deterministic mock
  state-dict pair: model A always predicts class 0 with logits
  `[5, 0, 0]`, model B always predicts class 2 with logits `[0, 0, 5]`.
  **Assert numeric probabilities, not just ranking** (ranking alone would
  not catch an accidental double-softmax):
  - softmax of `[5, 0, 0]` ≈ `[0.9866, 0.0067, 0.0067]`
  - softmax of `[0, 0, 5]` ≈ `[0.0067, 0.0067, 0.9866]`
  - averaged ≈ `[0.4967, 0.0067, 0.4967]`
  - assert each component within ±1e-3 of the expected value.
- Predict on a single-member ensemble produces identical results to the
  same model used standalone (no implicit Softmax wrap mid-load). Assert
  that for raw logits `[5, 0, 0]` the returned top probability is
  `0.9866 ± 1e-3`, not `softmax(softmax([5,0,0]))[0]`.
- Predict with `compound_labels=True` against a `salamander_fire_v2`-shaped
  mock label list: every entry in `predictions[*]` has both `species` and
  `viewpoint` parsed.
- **Load with `compound_labels=False` and a colon-bearing label list (e.g.
  `['salamander_fire_adult:up']`) raises ValueError at load time** (not
  at first predict). Layer-1 covers the parser-function case; this layer
  covers the loader-validation path.
- **`label_map` validation**: explicit `label_map={"0": "x", "1": "y"}`
  on a 3-class checkpoint raises ValueError (missing index 2). Stringly-
  keyed JSON values are coerced to int before validation.
- **Stale `classes` list (no explicit `label_map`)**: checkpoint with
  classifier weight shape `(3, 1920)` AND `classes = ['a', 'b']` (length
  2) raises ValueError. Same for length 4. Symmetric coverage of
  shorter-than and longer-than `num_classes`.
- **Checkpoint with no `classes` field AND no explicit `label_map`**:
  raises ValueError (`classes` is required for index→label resolution).
- **Mixed ensemble**: first checkpoint has `classes`, second has none →
  ValueError.

**Layer 3 — FastAPI router integration (`TestClient` + monkeypatched
ModelHandler):**

- `classify_model_id` → densenet-classifier (compound) → response per-result
  has top-level `iaClass` and `viewpoint`. Per-prediction parsing is
  asserted via `original_classify[i]['predictions'][*]['species'/'viewpoint']`
  (the per-bbox classify result block emitted in `/pipeline/` response),
  not via `result.classification.predictions` (`result.classification` is
  top-1 only).
- `classify_model_id` → densenet-classifier (non-compound, all colon-free
  labels) → response has top-level `viewpoint` only, no `iaClass`.
- `classify_model_id` → existing efficientnet-classifier (no
  `parse_compound_labels`) → existing behavior preserved (no
  `iaClass`/`viewpoint` on result).
- `classify_model_id` → existing efficientnet-classifier WITH
  `parse_compound_labels=True` and `species:up`-style labels → result
  shows top-level `viewpoint` only (sentinel suppression now applies
  uniformly via the shared helper). Regression coverage for finding #5.
  Assertion uses `result['viewpoint']` directly; `iaClass` MUST be absent.
- `classify_model_id` → densenet-orientation → 400 (rejected; orient lives
  in a different slot).

Live integration (running ml-service against real `salamander_fire_v2`
weights) is out of scope for unit tests; covered by hand-test after deploy.

## Codex review gates

Following the workflow already established on the Wildbook v2 branch:

1. **Design review** — Send this document to Codex for review BEFORE any
   code is written. Revise based on Codex feedback. No code lands until
   Codex green-lights the design.
2. **Code review** — Single implementation commit (new module + registry
   entry + router edits + tests). Send the diff to Codex for review BEFORE
   pushing. Iterate on findings until clean.
3. **Post-commit verify** — If code review surfaces a follow-up fix,
   re-review after the fix.

## Resolved questions (Codex design review, 2026-05-16)

Original design v1 was reviewed and the following decisions were locked in
based on Codex feedback. Each is now reflected in the sections above.

1. **Module structure**: new sibling `DenseNetClassifierModel`, not
   inheritance from `DenseNetOrientationModel`. The orientation class
   bakes in orientation response shape and softmax-on-load behavior;
   subclassing would couple two different contracts. DRY via small shared
   helpers later if needed.
2. **Sentinel handling**: configurable per-model via `sentinel_prefixes`
   (default `["species"]`). Sentinel suppression is implemented in a
   shared `parse_class_label` helper used by BOTH DenseNet-classifier AND
   EfficientNet, so the existing EfficientNet `"species:up"` literal-
   species bug is fixed in the same change.
3. **Per-prediction parsing**: every entry in `predictions[*]` carries its
   own `species` and `viewpoint`, not just top-1. `/pipeline/` promotes
   top-1 to top-level `iaClass`/`viewpoint`. `wbia_compat_router` and
   future consumers that inspect runner-ups get parsed fields too.
4. **Ensemble shape**: load all configured checkpoints by default. Optional
   `ensemble_indices: list[int]` selects a subset (parity with WBIA's
   single-member usage as a real-time fallback).
5. **Softmax ownership**: models load WITHOUT a Softmax tail. Softmax is
   applied exactly once during ensemble averaging in `predict()`. This
   intentionally differs from `DenseNetOrientationModel.load()`'s in-graph
   Softmax wrap — averaging post-softmax is mathematically correct only
   when each member is softmaxed independently, which is what we do.
6. **Label-order validation**: load-time check that every ensemble member
   has the same `classes` list in the same order (when `classes` is
   present). Averaging by index demands index→class agreement; mismatched
   order would corrupt averaged probabilities silently. Explicit
   `label_map` config override bypasses the cross-checkpoint class check
   (still requires matching `num_classes`).
7. **`compound_labels: false` + colon-bearing label**: load-time
   `ValueError` rather than silent `(None, None)`. Fail fast on config
   mistakes.
8. **Response-shape change**: keep `classification.class` as the raw
   label; add parsed `iaClass`/`viewpoint` at top level. No mutation of
   the existing fields.
