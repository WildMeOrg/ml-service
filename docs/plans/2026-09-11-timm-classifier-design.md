# Design: `timm-classifier` model type (DeepFaune v1.5 / DINOv3 ViT-L)

## Context

[EUR-DF-v1-5](https://huggingface.co/Addax-Data-Science/EUR-DF-v1-5) is
**DeepFaune v1.5** (CNRS), repackaged by Addax Data Science for AddaxAI. The
weights are upstream's, unmodified: same filename, same 1,212,611,300 bytes as
`https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.5/`, and SHA-256
matches on three sampled 1 MiB ranges (head/middle/tail).

Verified properties of the model:

| Property | Value |
|---|---|
| backbone | `vit_large_patch16_dinov3.lvd1689m` (timm), 303.1M params |
| pooling | `global_pool="token"` — **not** timm's DINOv3 default of `"avg"` |
| input | 224x224 (timm default cfg for this arch is 256) |
| preprocessing | BICUBIC resize, ImageNet mean/std, square-expanded crop |
| head | softmax over 40 European species/groups |
| checkpoint | `{'args': {...}, 'state_dict': ..., 'transform': Compose}`, keys prefixed `base_model.` |
| licence | CC-BY-SA-4.0 (upstream `classifTools.py` is CeCILL) |

None of the nine types in `MODEL_REGISTRY` can serve it:

1. `efficientnetv2` is the only generic timm classifier, and its `ImgClassifier`
   hardcodes `self.model.classifier` (`app/models/efficientnet.py:26-27`). timm
   ViTs expose `.head` — `AttributeError` at load.
2. It calls `timm.create_model` without `global_pool`. **This is the dangerous
   part:** `"token"` and `"avg"` produce state dicts with identical keys, so a
   strict load of DeepFaune weights under the wrong pooling reports 0 missing /
   0 unexpected keys and then classifies off the wrong feature. Upstream
   flags this in `classifTools.py:179`; Addax repeats it in their model card.
   Nothing raises. Silent wrong species.
3. Checkpoint parsing expects `{'state', 'classes'}` or a bare state dict.
4. Preprocessing is albumentations bilinear; the reference is torchvision
   BICUBIC, and the crop is square-expanded rather than the raw bbox rect.

## Goal

A config-driven `timm-classifier` model type that can serve any single-head timm
classifier, with DeepFaune v1.5 as its first consumer — one entry in
`model_config.json`, no new Python per model. The IoT sea-turtle and
Grouperspotter ports queue up behind the same type.

## Non-goals

- **The bird head.** Upstream v1.5 also ships
  `...-bird_head.pt` (8.1 MB), an MLP over the class-token embedding splitting
  `bird` into 8 groups. Addax deliberately omit it (a second head means a new
  model id, and DeepFaune has more heads coming: small mustelids, sex, age).
  We match the packaged model: 40 classes, `bird` terminal.
- The DeepFaune YOLOv8s detector. We run MegaDetector.
- Sequence-level logit averaging (upstream `predictTools.py`). Wildbook
  classifies per annotation; there is no burst context.
- fp16. `.half()` would halve resident VRAM to 579 MiB, but DeepFaune publish no
  fp16 accuracy numbers and no existing wrapper uses it. fp32, like every other
  model in the service.
- Any change to `efficientnetv2` behaviour. It keeps its albumentations path.

## Design

### New module: `app/models/timm_classifier.py`

`TimmClassifierModel(BaseModel)`. Config keys (all via `model_config.json`):

| key | default | notes |
|---|---|---|
| `model_arch` | required | timm architecture name |
| `checkpoint_path` | required | local path or URL, via `get_checkpoint_path` |
| `img_size` | `224` | square input |
| `global_pool` | `None` | `None` = timm's default for the arch; always logged and surfaced in `get_model_info` |
| `labels` / `label_map` | one required | ordered list, or `{index: label}` |
| `state_dict_key` | auto | tries `state_dict`, `state`, `model`, then the raw dict |
| `strip_prefix` | `None` | e.g. `base_model.` |
| `interpolation` | `bicubic` | `bicubic` or `bilinear` |
| `checkpoint_sha256` | `None` | optional digest verified against the resolved file |
| `verify_checkpoint_transform` | `True` | cross-check `img_size`/`mean`/`std` against preprocessing the checkpoint carries |
| `mean` / `std` | ImageNet | per-channel |
| `square_crop` | `False` | expand the short side of the bbox, then clip — DeepFaune's `cropSquareCVtoPIL` |
| `multi_label` | `False` | `False` = softmax + argmax; `True` = sigmoid + threshold |
| `threshold` | `0.5` | multi-label only |
| `label_mode` | `species` | `species` -> emit `species=label`, `viewpoint=None`; `viewpoint` -> the inverse; `compound` -> delegate to `parse_class_label` |
| `sentinel_prefixes` | `None` | `compound` mode only; passed through to `parse_class_label` |

Unknown keys are a **load-time error**. `app/main.py:87` forwards every
config key except `model_id`/`model_type` into `load(**kwargs)`, so a
misspelled `global_pool` would be swallowed, timm would silently fall back to
`"avg"`, and the strict state-dict load would still succeed — the headline
failure mode reached by typo. Any leftover `kwargs` raises, naming the unknown
keys. Keys beginning with `_` are exempt, matching the `_note` convention the
tracked `model_config.json` already uses for comments.

`label_mode` exists because `/pipeline/` promotes a classification to an
annotation's `iaClass` **only** from `predictions[0]['species']`
(`app/routers/pipeline_router.py:456-457`), and the existing wrappers populate
`species` only under `parse_compound_labels=True` — a path that assumes
`species:viewpoint` labels and treats a colon-free label as a *viewpoint*.
DeepFaune's labels are bare species names, so reusing that flag would emit
`species=None, viewpoint="red deer"`: backwards, and `iaClass` would never be
set. `label_mode: species` is the fix.

Label validation, before the model is built: exactly one of `labels` /
`label_map` (both, or neither, is an error); `labels` must be a real
list/tuple of non-empty strings, never a bare string; `label_map` keys must
coerce to exactly `0..N-1` with no duplicates after coercion (`"1"` and `"01"`
collide and are rejected). Loading must not be able to produce a label map
that silently mis-associates an output index.

Load sequence:

1. `get_checkpoint_path(checkpoint_path)` (downloads + caches URLs).
2. `torch.load(path, map_location='cpu', weights_only=False)`.
   **`map_location='cpu'` is load-bearing.** Measured on an RTX 5080: loading a
   1.2 GB checkpoint with `map_location=self.device` — what
   `efficientnet.py:104` and `densenet_classifier.py:80` both do — peaks at
   **2315 MiB** of VRAM because `torch.load` materialises a second full copy on
   the GPU before `load_state_dict` copies into the already-allocated params.
   Via CPU the peak is 1157 MiB, equal to steady state.
3. If the checkpoint carries `args['num_classes']`, cross-check it against the
   configured label count and raise on mismatch (upstream does the same).
4. Strip `strip_prefix` from keys if configured.
5. `timm.create_model(model_arch, pretrained=False, num_classes=N, **pool_kwargs)`,
   then `load_state_dict(..., strict=True)`. Strict, always: given failure mode
   (2) above, a lenient load is how you ship a silently wrong classifier.
6. `.to(device).eval()`.

`predict()` returns the **same dict shape as `EfficientNetModel`** —
`model_id`, `predictions[]` (`label`, `index`, `probability`, plus
`species`/`viewpoint` when compound), `all_probabilities`, `threshold`, `bbox`,
`theta` — so `/classify/` and `/pipeline/` need no response-shape changes and
the router's existing top-1 promotion to `iaClass`/`viewpoint` works unchanged.

Preprocessing mirrors `EfficientNetModel._preprocess_image`'s ordering (decode →
bbox crop → rotate by theta → resize → normalise) with two configurable
differences: optional square-crop expansion before the rotate, and torchvision
BICUBIC instead of albumentations bilinear.

### Registry and router

- `MODEL_REGISTRY['timm-classifier']` in `app/models/model_handler.py`.
- `TimmClassifierModel` added to the classify-slot allowlist at
  `app/routers/pipeline_router.py:107-109`, and to the `classify_model_id`
  field description.

### Dependency bump

`requirements.txt`: `timm==1.0.19` -> `timm==1.0.25`. Required, not optional —
**1.0.19 contains no DINOv3 at all** (no `.py` in the wheel mentions it). DINOv3
first appears in timm 1.0.20, in `timm/models/eva.py` and
`timm/layers/pos_embed_sincos.py` (it is a RoPE/EVA-family ViT, not
`vision_transformer.py`).

Regression evidence for the bump: `efficientnetv2_rw_m` (miewid), `hrnet_w32`
(both orientation paths and the densenet classifier fallback),
`tf_efficientnet_b4_ns` (efficientnetv2) and `densenet201` all produce
**identical state-dict key/shape signatures** under 1.0.19 and 1.0.25 —
so existing checkpoints keep loading strictly.

Torch is untouched: `docker/dockerfile:27` pins torch 2.1.2 / torchvision
0.16.2 and timm 1.0.20+ declares no torch floor. Checked at module level:
`vit_large_patch16_dinov3` instantiates only `Conv2d`, `Linear`, `LayerNorm`,
`GELU`, `Mlp`, `PatchEmbed` and `RotaryEmbeddingDinoV3` — no RMSNorm, so
timm's `F.rms_norm` path (torch >= 2.4, and runtime-guarded by
`has_torch_rms_norm` in any case) is never reached; attention is
`scaled_dot_product_attention` (torch 2.0+). Final confirmation is the docker
image build in CI.

## Resource cost

Measured (RTX 5080, torch 2.10+cu128, fp32, 224x224, `no_grad`):

| | GPU |
|---|---|
| weights resident | 1157 MiB (303.1M x 4 B) |
| + 1 concurrent classify | 1176 MiB peak (18 MiB activations) |
| + 2 concurrent (`MAX_CONCURRENT_CLASSIFICATIONS`) | ~1185 MiB peak |

Models load eagerly at startup (`app/main.py`), so **~1.2 GiB is resident from
boot** on top of whatever else that host's registry pins. Activations are noise.
Host RAM sees a ~1.2 GiB transient during `torch.load`.

## Config entry

```json
{
  "model_id": "deepfaune-v1.5",
  "model_type": "timm-classifier",
  "model_arch": "vit_large_patch16_dinov3.lvd1689m",
  "checkpoint_path": "https://huggingface.co/Addax-Data-Science/EUR-DF-v1-5/resolve/e5c04750c44cb2ecbfa4f86624c37ae438f0b3dd/deepfaune-vit_large_patch16_dinov3.lvd1689m.pt",
  "img_size": 224,
  "global_pool": "token",
  "strip_prefix": "base_model.",
  "square_crop": true,
  "multi_label": false,
  "labels": ["bison", "badger", "..."]
}
```

The checkpoint URL is **pinned to a commit SHA**, not `main`.
`download_checkpoint` caches by URL hash, so a mutable URL whose content changes
would be served from stale cache forever. HF is preferred over the canonical
pbil URL for exactly this reason: it publishes an LFS SHA-256 and immutable
revisions, where pbil is an Apache index whose files can be replaced in place.

## Test plan

1. `global_pool` reaches `timm.create_model` verbatim (mocked) — the headline
   failure mode.
2. Unknown/misspelled config keys raise at load, naming the offenders;
   `_`-prefixed keys are accepted. Covers misspelled `global_pool`,
   preprocessing keys and label keys.
3. `torch.load` is called with `map_location='cpu'` (mocked) — the 2.3 GiB trap.
4. `state_dict` extraction across the three layouts + `strip_prefix`; the raw
   fallback is accepted only when the object really is a tensor state dict.
5. `args['num_classes']` disagreeing with the label count raises; head-width
   mismatch raises under strict loading.
6. Label validation: neither/both label forms, sparse or out-of-range
   `label_map` keys, `"1"`/`"01"` coercion collision, a string passed as
   `labels`, non-string values.
7. `label_mode`: `species` emits `species=label` with `viewpoint=None`,
   `viewpoint` the inverse, `compound` delegates to `parse_class_label`
   including sentinel suppression.
8. Preprocessing at tensor level against an explicit torchvision reference:
   bicubic vs bilinear, custom mean/std, square-crop geometry including
   clipping at image borders, non-square and partially out-of-frame bboxes,
   and theta ordering relative to the crop.
9. Scalar config validation: `img_size`, interpolation name, mean/std lengths,
   `threshold`, boolean fields.
10. Registry lookup and `/pipeline/` classify-slot acceptance.
11. **Integration fixture**: a tiny timm arch with a DeepFaune-shaped
    checkpoint (`args` + `base_model.`-prefixed `state_dict` + an unrelated
    pickled `transform`), loaded through `ModelHandler` using the real config
    surface, driven through both `/classify/` and `/pipeline/`, asserting a
    plain species label arrives as top-level `iaClass`.

Real ViT-L weights are never built in tests; a tiny arch plus mocks covers all
of it.

## Verification evidence

Gathered against the real 1.2 GB checkpoint (SHA-256
`4f937643...63d0`, matching the SHA-256 Hugging Face publishes for the LFS
object) before implementation:

- **Checkpoint structure is as assumed.** Top-level keys `args`, `state_dict`,
  `transform`; `args == {'backbone': 'vit_large_patch16_dinov3.lvd1689m',
  'num_classes': 40}`; 320 state-dict keys, every one prefixed `base_model.`;
  head `(40, 1024)`; fp32. The pickled `transform` is exactly
  `Resize((224, 224), bicubic)` + `ToTensor` + `Normalize(ImageNet)`, matching
  the preprocessing this design specifies.
- **The `global_pool` failure mode is real, on these weights.** Loading them
  into `global_pool="token"` and `global_pool="avg"` both succeed *strictly*,
  0 missing and 0 unexpected keys. On four fixed random crops the two
  disagree on top-1 for **3 of 4**, with probabilities up to **0.254** apart.
  Nothing raises. This is why the load is strict, why `global_pool` is
  explicit in config, and why unknown keys are rejected rather than ignored.
- **The timm bump is safe for existing checkpoints.**
  `efficientnetv2_rw_m`, `hrnet_w32`, `tf_efficientnet_b4_ns` and
  `densenet201` produce identical state-dict key/shape signatures under
  1.0.19 and 1.0.25.

## Review round 2

Codex reviewed the implementation. Changes made in response:

- **Out-of-frame bbox could be resurrected by square expansion.** On a 64x48
  image `bbox=(70, 4, 10, 40)` lies wholly off it, but expanding the short
  side moved x to `55..95`, which clipped back to `55..64` and classified an
  unrelated strip. The requested box is now checked for overlap *before*
  expansion.
- **Booleans and normalization stats are validated properly.**
  `bool("false")` is `True`, so a stringy config value silently selected
  sigmoid over softmax; real `bool`s are now required. Means must be finite
  and standard deviations finite and strictly positive (a zero std divides the
  image by zero and still yields a classifiable tensor). `nearest`
  interpolation was dropped so code and design agree.
- **Checkpoint metadata is now cross-checked.** Config alone cannot know that
  `img_size: 256` or swapped `mean`/`std` are wrong -- they load strictly and
  merely predict worse -- but DeepFaune pickles its own transform, so both are
  now rejected at load, with `verify_checkpoint_transform: false` as the
  deliberate opt-out. An optional `checkpoint_sha256` authenticates the file
  itself. **Label order remains unverifiable** and is documented as such: it
  is part of the model contract, not something config can infer.
- **Partial prefix stripping is rejected.** A checkpoint mixing bare and
  prefixed copies of a tensor would collapse onto one key, last-writer-wins,
  and still load strictly. Every key must now carry `strip_prefix`; once that
  holds, a post-strip collision is impossible, so that is the entire guard.
- **`species`/`viewpoint` are emitted only when meaningful.** Always emitting
  them, including JSON `null`, differed from `EfficientNetModel`, which omits
  them unless compound parsing is on. Tests now compare exact key sets against
  a real `EfficientNetModel` in both modes rather than asserting a subset.
- **Label-map keys must be canonical.** `int(0.9) == 0` and `int(True) == 1`
  would quietly reindex the map.
- **Vacuous tests replaced.** The router test grepped source (the import alone
  made it pass); the multi-label test used a threshold of `0.0`; the strict-load
  test would have passed under `strict=False`; the compound test proved
  semantics rather than delegation. Each now fails if the behaviour it names
  is removed.

Deliberately **not** done: extracting the ImageNet constants into a shared
module. They are duplicated across four wrappers, but deduplicating means
editing three other models' preprocessing, which this PR explicitly does not
touch.
