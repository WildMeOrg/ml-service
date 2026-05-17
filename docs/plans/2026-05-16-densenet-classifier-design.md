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
- A symmetric tweak to EfficientNet so it can also surface
  `species`/`viewpoint`. The router accepts whatever shape the classify
  model returns; EfficientNet enhancement is a follow-up if desired.
- A pure-continuous-theta orientation predictor. ml-service does not have
  one today; out of scope here.
- Wildbook-side changes. The v2 contract already reads the right top-level
  fields with `optString(..., null)` fallbacks.

## Design

### New module: `app/models/densenet_classifier.py`

`DenseNetClassifierModel(BaseModel)` — mirrors `DenseNetOrientationModel`'s
structure with three additions: ensemble loading, compound-label parsing,
and a different output shape suited for the classify slot.

**Load signature:**

```python
def load(self, model_path: str = "", device: str = 'cpu', model_id: str = "",
         checkpoint_path: str = None,
         checkpoint_paths: list = None,
         img_size: int = 224,
         label_map: dict = None,
         compound_labels: bool = False):
```

Loader behavior:

- Exactly one of `checkpoint_path` / `checkpoint_paths` must be provided.
  `checkpoint_path` is normalized internally into a single-element
  `checkpoint_paths` list.
- Each checkpoint is downloaded (if URL) and loaded via `torch.load`.
- Architecture is detected from the first checkpoint's state-dict keys
  (HRNet-W32 vs DenseNet201), reusing the logic from
  `densenet_orientation.py:120-130`.
- `num_classes` is derived from the first checkpoint's classifier weight
  shape. **All subsequent checkpoints must match `num_classes`** — error
  at load if mismatched.
- `label_map` resolves in order: explicit config arg, checkpoint `classes`
  field (WBIA format), generic `class_{i}` fallback. (Same precedence as
  the orientation model.)
- All loaded `nn.Module` instances are stored as `self.models` (list);
  `self.compound_labels` carries the flag.

**Inference:**

```python
def predict(self, image_bytes, bbox=None, theta=0.0):
    # Same preprocessing as DenseNetOrientationModel.predict.
    inputs = self._preprocess(image_bytes, bbox, theta)
    summed = None
    with torch.no_grad():
        for m in self.models:
            probs = torch.softmax(m(inputs), dim=-1)
            summed = probs if summed is None else summed + probs
    avg = summed / len(self.models)
    return self._format_output(avg)
```

`_format_output(avg)` builds the result dict. Top-K (K=3 by default to
mirror EfficientNet) `predictions` list is always included. For each
prediction:

- `label`, `probability`, `index` populated as usual.
- If `self.compound_labels` AND `":"` in label:
  - Split once: `prefix, suffix = label.split(":", 1)`.
  - If `prefix == "species"` (the literal sentinel namespace used by some
    deployed efficientnet checkpoints): set `species = None`, `viewpoint =
    suffix`.
  - Else: `species = prefix`, `viewpoint = suffix`.
- Else (no compound): `species = None`, `viewpoint = label`.

Top-1's `species` and `viewpoint` are also surfaced at the top level of
the returned dict for the router's convenience:

```python
return {
  'class': top_label,
  'probability': top_prob,
  'class_id': top_idx,
  'species': top_species,        # may be None
  'viewpoint': top_viewpoint,    # may be None
  'predictions': [{label, probability, index, species, viewpoint}, ...],
}
```

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
    top_species   = classify_result.get('species')
    top_viewpoint = classify_result.get('viewpoint')
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
    "compound_labels": true
}
```

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
- `compound_labels: true` but no label contains `":"` → load-time WARN log;
  predict still works (returns viewpoint = full label, species = None).
- Inference failure on any single ensemble member → entire predict raises
  (consistent with how the existing extract path treats embedding failures
  after the v2 contract fix in commit `3fbed82`).
- Classify model is neither EfficientNet nor DenseNetClassifier → HTTP 400
  (existing pattern).

## Tests

Three layers; first two are fast and require no real checkpoint files.

**Layer 1 — pure-function helpers (`test_densenet_classifier.py`):**

- `_parse_compound_label("salamander_fire_adult:up", compound=True)`
  → `("salamander_fire_adult", "up")`
- `_parse_compound_label("species:left", compound=True)` → `(None, "left")`
- `_parse_compound_label("up", compound=True)` → `(None, "up")`
- `_parse_compound_label("salamander_fire_adult:up", compound=False)`
  → `(None, None)` (raw label preserved separately)
- Sentinel namespace list (currently just `"species"`) is centralized in
  one constant so it can grow without touching the router.

**Layer 2 — loader / ensemble (mocked torch.load):**

- Loading 3 ensemble checkpoints with matching `num_classes` produces
  `len(self.models) == 3`.
- Loading 2 checkpoints with mismatched classifier weight shapes raises
  ValueError.
- Single `checkpoint_path` path normalizes to length-1 list.
- Predict averages softmax across N members (test with deterministic
  mock state-dicts that produce predictable logits).

**Layer 3 — FastAPI router integration (`TestClient` + monkeypatched
ModelHandler):**

- `classify_model_id` → densenet-classifier (compound) → response per-result
  has top-level `iaClass` and `viewpoint`.
- `classify_model_id` → densenet-classifier (non-compound) → response has
  top-level `viewpoint` only, no `iaClass`.
- `classify_model_id` → existing efficientnet-classifier → existing
  behavior preserved (no `iaClass`/`viewpoint` on result).
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

## Open questions for Codex

1. Is `DenseNetClassifierModel` truly enough of a separate concept to justify
   a new module, or should it derive from `DenseNetOrientationModel` (share
   90% of the load/preprocess code)?
2. The sentinel namespace `"species"` — is one literal sufficient, or
   should the suppression list be configurable per-model?
3. Ensemble support: should `_format_output` expose per-member predictions
   (top-K from each model) alongside the averaged top-K, or is the averaged
   view sufficient for all consumers?
4. Response-shape change: is adding `iaClass` and `viewpoint` at the top
   level (rather than nested under `classification`) acceptable, given the
   existing /pipeline/ response already has flat fields like `bbox`,
   `theta`, `embedding`, etc.?
