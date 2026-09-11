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
| `mean` / `std` | ImageNet | per-channel |
| `square_crop` | `False` | expand the short side of the bbox, then clip — DeepFaune's `cropSquareCVtoPIL` |
| `multi_label` | `False` | `False` = softmax + argmax; `True` = sigmoid + threshold |
| `threshold` | `0.5` | multi-label only |
| `parse_compound_labels` / `sentinel_prefixes` | as `efficientnetv2` | shared `parse_class_label` |

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
0.16.2, timm 1.0.20+ declares no torch floor, and the DINOv3 path uses only
`scaled_dot_product_attention` (torch 2.0+). To be confirmed in the built image.

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
2. `torch.load` is called with `map_location='cpu'` (mocked) — the 2.3 GiB trap.
3. `state_dict` extraction across the three layouts + `strip_prefix`.
4. `args['num_classes']` disagreeing with the label count raises.
5. Label-count vs head-width mismatch raises.
6. softmax/argmax vs sigmoid/threshold output shapes.
7. Square-crop geometry matches upstream `cropSquareCVtoPIL` (including
   clipping at image borders).
8. Registry lookup and pipeline-router classify-slot acceptance.

Real ViT-L weights are never built in tests; a tiny arch plus mocks covers all
of it.
