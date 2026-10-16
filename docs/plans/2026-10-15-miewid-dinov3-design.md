# Design: DINOv3 backbone for the `miewid` model type (MiewID v5 DINOv3)

## Context

MiewID v5 is being trained in two variants on the same data split:

| | shipped v5 (EfficientNet) | v5 DINOv3 (in training) |
|---|---|---|
| backbone | timm `efficientnetv2_rw_m` | HF transformers `facebook/dinov3-vitb16-pretrain-lvd1689m` (ViT-B/16, 4 registers) |
| neck | GeM over feature map → BatchNorm1d | concat(CLS, GeM over 28×28 patch grid) → BatchNorm1d |
| embedding dim | 2152 | 1536 |
| input | 440×440 | 448×448 (must be a multiple of 16) |
| checkpoint keys | `backbone.*` (timm), `pooling.p`, `bn.*` | `backbone.*` (HF, transformers 5.17 layout), `pooling.p`, `bn.*` |
| eval precision | fp16 autocast (WBIA) | fp16 autocast |

Neither checkpoint carries an ArcFace head. Source for the DINOv3 column: the
training session's answers (2026-10-16, epoch 9 of 30, architecture and I/O
declared final; weights not yet final).

At matched epoch 8 the two are close on mean mAP (69.80 EffNet vs 68.89
DINOv3), but they split cleanly by species: DINOv3 wins on faces, heads and
whole-body patterns, while EfficientNet wins on fine edge detail (fin notches,
fluke edges, callosities). Routing per species to the better model is worth
about +2.7 mean mAP. **Both models will be served side by side**, and Wildbook
will choose a `model_id` per species in its own config. ml-service doesn't need
to know about species; each variant is just another `model_id`.

Today `miewid` can only serve the DINOv3 variant by accident or not at all:

1. `MiewIdNet` hardcodes `efficientnetv2_rw_m` and `backbone.classifier`
   (`app/models/miewid.py:38-52`).
2. ml-service pins `transformers==4.52.4` and `torch==2.1.2`; 4.52 has no
   DINOv3. 4.56 has it and still accepts torch 2.1, but names encoder layers
   `layer.*` while the training run's 5.17 names them `model.layer.*`, so the
   checkpoint would not load strictly. transformers ≥ 5.4 requires torch ≥ 2.4.
3. `facebook/dinov3-*` repos on the HF Hub are licence-gated, so
   `AutoModel.from_pretrained` at runtime would need credentials, which is
   unacceptable for a service that already has every backbone weight in
   the checkpoint.
4. `IMAGE_SIZE = (440, 440)` is a class constant, and `/explain` rebuilds its
   transform from it (`app/routers/explain_router.py:65-69`), so a 448 model
   would silently get 440 inputs there.
5. PairX treats hooked outputs as spatial maps and only has CNN canonizers; the
   default `layer_key` `backbone.blocks.3` is EfficientNet-specific.

## Goal

One `model_type: "miewid"` that serves, per `model_id`, either the existing
backbones (HF msv2/msv3, standalone EfficientNet) or the DINOv3 ViT-B/16
variant. Embeddings from the DINOv3 variant must match the training repo's
`MiewIdNet` within an agreed tolerance on fixed reference chips.

## Non-goals

- Per-species routing (Wildbook config).
- PairX `/explain` for the DINOv3 variant (fails closed, see below).
- Any dependency bump. torch, torchvision, timm, transformers and albumentations
  pins and the fidelity contract in `requirements.txt` are unchanged.
- Other DINOv3 sizes (ViT-S/L/H). The vendored module is config-driven, so
  adding one later means adding a frozen config and a parity run, not new code.
- Changing EfficientNet precision (stays fp32; see Precision).

## Design

Approach **A** (chosen 2026-10-15; Codex recommended it independently): vendor a
minimal pure-PyTorch DINOv3 ViT, built from a frozen local config, with
parameter names identical to transformers 5.17 so the checkpoint loads strictly
with no key remap. Rejected alternatives: **B**, bumping transformers to
4.56/4.57 (needs a key remap, runs different code than training, and the global
bump re-opens msv2/msv3 `trust_remote_code` and the fidelity contract); and
**C**, timm 1.0.25's DINOv3 with HF→timm remapping (QKV fusion, Q/V bias
handling, and RoPE written as periods rather than inverse frequencies make
equivalence the hardest to establish).

### Config contract

A new optional field, `backbone`, selects the network. It is checked **before**
the legacy `version` logic:

```json
{
  "model_id": "miewid-msv5-dinov3",
  "model_type": "miewid",
  "backbone": "dinov3_vitb16",
  "checkpoint_path": "/datasets/miew_id.msv5_dinov3.bin",
  "version": 5
}
```

| `backbone` | behaviour |
|---|---|
| absent | unchanged, predicates copied verbatim from `miewid.py:85-169` with no coercion (table below) |
| `"efficientnetv2_rw_m"` | standalone EfficientNet (explicit form of today's v4+ path; requires `checkpoint_path`) |
| `"dinov3_vitb16"` | vendored DINOv3 network; requires `checkpoint_path` |
| anything else | `ValueError` at load, naming the supported values |

Legacy dispatch when `backbone` is absent (today's exact behaviour, preserved):

| `checkpoint_path` | `version` | path |
|---|---|---|
| truthy | `int`/`float` and `>= 4` | standalone EfficientNet, strict load |
| truthy | anything else (missing, `< 4`, any string) | HF overlay, `strict=False` |
| falsy/absent | any | HF hub, no checkpoint |

Inside both HF paths only `version == 3` (so missing, `3`, `3.0`) selects
`miewid-msv3`; everything else (`3.5`, `"3"`, `"5"`, and numeric `4` without a
checkpoint) selects `miewid-msv2`. These quirks are preserved, not fixed.

An existing EfficientNet entry with `version: 5` and no `backbone` stays
EfficientNet. `version` stays informational for the new values. The training
repo's `dinov3:<hf-id>` prefix is not accepted as a selector, so there is one
way to say it.

Image size, embedding dim and precision come from the backbone, not from the
config. Optional `imgsz` and `embedding_dim` fields, if present, are
**assertions**: a mismatch with the resolved value raises at load. `imgsz` is a
single `int` (the square side, as today, e.g. `448`); `embedding_dim` is an
`int`. Other types raise. `imgsz` applies to every path, including legacy
entries without `backbone` (resolved 440 for HF and EffNet, 448 for DINOv3).
`embedding_dim` applies to the standalone paths (2152 EffNet, 1536 DINOv3,
read from `bn.weight`); setting it on an HF path raises, because those models
do not expose it statically. Today `imgsz` only feeds `model_info`, so
this changes nothing for existing entries unless they set a wrong value; the
plan includes auditing prod entries for that.

### Code layout

```
app/models/miewid.py          MiewidModel (public, unchanged name), MiewIdNet (EffNet, unchanged),
                              DinoV3MiewIdNet (new), backbone dispatch
app/models/dinov3_vit.py      vendored DINOv3 ViT (new, ~250 lines) + frozen VITB16 config
```

`MiewidModel` stays the one class in `MODEL_REGISTRY` and the one class the
routers check with `isinstance` (`extract_router.py:88`,
`pipeline_router.py:121`, `explain_router.py:59`). Decoding, `get_chip_from_img`
cropping, albumentations preprocessing, `inference_lock` and numpy conversion
stay shared.

### Vendored backbone: `app/models/dinov3_vit.py`

Derived from `transformers==5.17.0`
`models/dinov3_vit/modeling_dinov3_vit.py` (Apache-2.0; licence header and
source version recorded at the top of the file). Kept: `Embeddings` (CLS,
register tokens, mask_token, patch conv), `RopePositionEmbedding`, `Attention`,
`LayerScale`, `MLP`, `Layer`, `Encoder`, `Model`. Module attribute names are
preserved exactly so `state_dict` keys match training
(`embeddings.*`, `model.layer.N.*`, `norm.*`; `rope_embeddings` holds only a
non-persistent `inv_freq` buffer).

Stripped: `PreTrainedModel`, gradient checkpointing, output-capturing
decorators, the `ACT2FN` activation registry (replaced by `nn.GELU()`, which is
what `"gelu"` resolves to), the coordinate `lru_cache`, drop-path, training-time RoPE augmentation (shift/jitter/rescale
only run when `self.training`), gated MLP, the `Backbone` head, and the
attention-function registry.

The retained forward-path functions are copied **line for line**, not
re-implemented; generic ViT RoPE code has the same shapes and different values.
Invariants the port must preserve:

- RoPE coordinates: patch-centre `(y, x)` from `arange(0.5, n) / n`, meshgrid
  `ij`, flattened, mapped to `[-1, 1]` (`get_patches_center_coordinates`).
  `inv_freq = 1 / theta ** arange(0, 1, 4/head_dim)`; angles
  `2π · coords[:, :, None] · inv_freq`, flattened over the axis, then `tile(2)`;
  rotation by `rotate_half` (half-vector split, `cat(-x2, x1)`). The coordinate
  `lru_cache` decorator is dropped (it is a cache, not a numeric change).
- RoPE computed in **fp32 with autocast disabled**
  (`torch.autocast(device_type, enabled=False)` replaces `maybe_autocast`), then
  cast to `pixel_values.dtype`. `pixel_values` is first cast to the patch-conv
  weight dtype (fp32), so cos/sin stay fp32 under autocast.
- RoPE applied to patch tokens only, skipping the 1 + 4 prefix tokens. Under
  autocast, `q_patches * cos` promotes the fp16 patch Q/K to fp32 before
  `torch.cat` with the fp16 prefix tokens; the attention op then recasts. This
  promotion is part of the training numerics and is preserved by copying the
  lines as-is.
- Attention: non-causal, scaling `head_dim ** -0.5` applied once, dropout 0 in
  eval, heads transposed back before the output projection. Eager softmax has
  no explicit fp32 upcast.
- Each block: pre-norm → attention → LayerScale → residual; pre-norm → MLP →
  LayerScale → residual. MLP is up-projection → exact (erf) GELU →
  down-projection. LayerScale is active in eval; its weights come from the
  checkpoint.
- Final `norm` applied to every token before the neck.
- Attention through `torch.nn.functional.scaled_dot_product_attention` only.
  Training ran `_attn_implementation = "sdpa"` (the transformers default on
  torch 2.6; never set explicitly). The eager path is not ported.
- TF32: training ran process defaults, `matmul.allow_tf32 = False` and
  `cudnn.allow_tf32 = True`. These are also torch 2.1.2's defaults, and
  ml-service sets neither, so no change; the parity gate records both flags.
- torch 2.1-compatible: `register_buffer(..., persistent=False)` instead of
  `nn.Buffer` (torch ≥ 2.5), and no `enable_gqa`.

The frozen ViT-B/16 config is a Python constant, not a downloaded file:
hidden 768, intermediate 3072, 12 layers, 12 heads, 4 register tokens, patch
16, rope_theta 100, layer_norm_eps 1e-5, query/value/proj/mlp bias on, key bias
off, layerscale on, gelu, no gated MLP. These values come from the 5.17 defaults
plus the training answers; a unit test compares the constant against the
training run's `backbone.config.to_dict()` dump, committed as a test fixture.

### Network: `DinoV3MiewIdNet`

```python
tok   = self.backbone(x)                          # (B, 5 + h*w, 768), after final norm
cls   = tok[:, 0]
patch = tok[:, 5:]                                # skip CLS + 4 registers
grid  = patch.transpose(1, 2).reshape(B, 768, h, w)
feat  = torch.cat([cls, self.pooling(grid).flatten(1)], dim=1)   # (B, 1536)
return self.bn(feat)                              # BatchNorm1d(1536)
```

`h = w = 448 / 16 = 28`. The forward checks the **input** spatial size equals
`(448, 448)` before the backbone and raises `ValueError` otherwise. A
token-count check alone is not enough: the patch conv floors, so 449×448 gives
784 patches, and 224×896 gives 784 patches that would be wrongly reshaped to
28×28. The token count `5 + 784` is asserted as well. GeM is the existing `GeM` class
(learnable `p`, loaded from `pooling.p`). The raw BN output is returned with no
L2 normalisation, as with EfficientNet.

### Loading

The rules below apply to the **standalone** paths only (DINOv3 and standalone
EfficientNet, whether selected by `backbone` or by legacy `version >= 4`). The
legacy HF-overlay path keeps `strict=False` and its current tolerance of
missing/unexpected keys, unchanged.

- `torch.load(path, map_location="cpu", weights_only=True)` → must satisfy
  `isinstance(sd, dict)` (the artifact is an `OrderedDict`) with tensor values. A wrapped resume checkpoint (`model_state_dict`,
  `optimizer`, …) is rejected with a message telling the operator to ship
  `model_best.bin` instead.
- Before `load_state_dict`, check the selection against the keys: DINOv3
  requires `backbone.embeddings.register_tokens` and `bn.weight` of shape
  `(1536,)`; EfficientNet requires `backbone.conv_stem.weight`. A mismatch
  raises a `ValueError` naming both the configured backbone and what the
  checkpoint looks like. The keys only ever validate, never switch, the
  architecture.
- `load_state_dict(strict=True)`. Unexpected `final.*` keys (an ArcFace head)
  are rejected, not stripped. Neither v5 artifact carries one, and the
  WBIA-converted v4 checkpoints that do are a separate, existing problem.
- The checkpoint's SHA-256 is computed at load and reported in `model_info`.
  The checkpoint cache is keyed by URL, so an overwritten URL can serve stale
  weights; the digest makes that visible.
- Sequence: build the network on CPU, load the state, `eval()`, move to device,
  drop the CPU state dict.

### Preprocessing and per-instance image size

`MiewidModel.IMAGE_SIZE` is replaced by `self.image_size`, set from the backbone:
`(440, 440)` for HF and EfficientNet, `(448, 448)` for DINOv3. `self.preprocess`
is built from it, using the same albumentations `Resize` → `Normalize()`
(ImageNet) → `ToTensorV2` chain and the same `get_chip_from_img`. No HF image
processor.

`explain_router.preprocess` reuses `model.preprocess` instead of rebuilding the
transform from the class constant. The PairX capability check (next section)
lives in `resolve_pairx_model` before it returns, so it runs before any image
task is scheduled. The class constant is kept as a
backward-compatible alias equal to the tuple `(440, 440)` (its current type and
value), with no remaining internal readers.

### Precision

DINOv3 forwards run under `torch.autocast("cuda", dtype=torch.float16)` on CUDA,
covering backbone and neck, inside the existing `inference_lock` and
`no_grad`. Weights stay fp32 (no `.half()`). The output is cast to float32
before numpy. This matches training eval (`eval_fn.py:18`) and costs nothing in
compatibility: the 1536-d space is new, so there is no existing index to stay
consistent with.

The finiteness check (see Responses) sits on the final BN embedding in the
shared `MiewidModel.extract_embeddings`, before numpy conversion, so `/extract`
and `/pipeline` are both covered.

On CPU (no autocast), DINOv3 runs fp32 and logs a warning once at load.
`model_info.precision` reports the effective mode (`"fp16_autocast"` or
`"fp32"`).

EfficientNet and HF variants stay fp32. Switching them to autocast would move
embeddings against every existing Wildbook index; that is a separate decision.

### PairX `/explain`

`MiewidModel` gets a `supports_pairx` attribute: `True` for HF and EfficientNet,
`False` for DINOv3. `/explain` checks it right after resolving the model id,
**before** fetching any image, and returns
`400 {"detail": "PairX is not supported for model '<id>' (backbone dinov3_vitb16)"}`.
This holds even when the caller passes an explicit `layer_key`, since a
DINOv3 module name would pass the existing `named_modules()` check and then
fail deep inside PairX.

Supporting PairX on ViTs (a module exposing the patch grid as
`(B, 768, 28, 28)` plus validated transformer relevance propagation) is
separate work.

### `model_info`

Adds to the existing keys: `backbone`, `image_size`, `embedding_dim`,
`precision`, `checkpoint_sha256`, `supports_pairx`. Existing keys
(`model_type`, `device`, `imgsz`, `version`, `checkpoint_path`,
`use_checkpoint`) keep their meaning. `imgsz` now reports the resolved size.

### Responses

Unchanged. `/extract` and `/pipeline` already return a flat embedding list whose
length follows the model; DINOv3 returns 1536 floats. Embeddings are checked
for finiteness after the forward (fp16 overflow would show up as `inf`); a
non-finite embedding raises the existing 500 path rather than returning NaNs to
Wildbook.

## Resource cost

ViT-B/16 is about 86M backbone params (~345 MB fp32 resident), similar to
`efficientnetv2_rw_m` (~53M). At 448, 789 tokens per image, so attention is
cheap. Peak VRAM with all prod models loaded plus one DINOv3 instance is
measured on the target GPU before deploy (plan task), not assumed.

## Test plan

Unit (CPU, no network, synthetic weights):

- Legacy dispatch matrix, no `backbone`: checkpoint present/absent ×
  version missing, `3`, `3.0`, `3.5`, `4`, `4.1`, `5`, `"3"`, `"5"`; each asserts
  the path (EffNet / HF overlay / HF hub) and the HF tag (msv2/msv3) from the
  legacy table.
- HF overlay still loads with `strict=False`: a checkpoint with a missing key
  and an extra key loads as it does today.
- Config assertions: `imgsz` matching (440 legacy EffNet, 440 HF, 448 DINOv3)
  loads; mismatching (448 on EffNet, 440 on DINOv3) and non-int (`"448"`,
  `[448, 448]`) raise. `embedding_dim` matching (2152 EffNet incl. legacy
  entry without `backbone`, 1536 DINOv3) loads; mismatching raises; set on an
  HF path raises. `MiewidModel.IMAGE_SIZE == (440, 440)`.
- Explicit dispatch: `"efficientnetv2_rw_m"`; `"dinov3_vitb16"`; unknown value →
  `ValueError`; either explicit backbone without `checkpoint_path` →
  `ValueError`.
- Vendored module: `state_dict` key set and shapes equal the training key
  manifest (fixture from the training session); frozen config equals the
  training config dump.
- Numerics, not just shapes: a golden-output test. With a fixed seeded
  synthetic state dict and fixed input, `DinoV3MiewIdNet` output is compared
  against outputs produced once by the real transformers 5.17 `DINOv3ViTModel`
  plus the training neck (generated in the training venv, committed as a small
  fixture), fp32 on CPU, `atol=1e-5`. This catches RoPE, attention, LayerScale
  and norm-order errors that key/shape tests cannot.
- Forward: (1, 3, 448, 448) → (1, 1536); registers excluded (perturbing
  register-token outputs leaves the GeM half unchanged); 440×440, 449×448 and
  224×896 → `ValueError`.
- Precision: on CUDA (skipped without a GPU), a DINOv3 forward runs under fp16
  autocast (assert via a patched `torch.autocast` or by checking an
  intermediate dtype), weights stay fp32, and the returned numpy array is
  float32. On CPU, DINOv3 runs fp32 and reports `precision: "fp32"`. EffNet
  and HF forwards never enter autocast.
- Finiteness: a model whose forward returns NaN, +Inf or −Inf makes `/extract`
  and `/pipeline` return the 500 path, not a NaN embedding.
- Loading: resume-format checkpoint rejected; DINOv3 config + EffNet checkpoint
  (and vice versa) rejected with the named mismatch; `final.*` rejected.
- Image size: DINOv3 instance preprocesses to 448, EffNet to 440, both loaded
  together; `/explain` preprocessing uses the instance transform.
- `/explain` with a DINOv3 model id → 400 before any image fetch (assert the
  fetch mock is never called), with and without `layer_key`.
- `model_info` fields for both variants.
- Existing EffNet/HF and PairX concurrency tests unchanged and passing.

Parity gate (release-blocking, run in the production image on the production
GPU):

1. **Architecture parity.** Identical saved preprocessed tensors through the
   training `MiewIdNet` (transformers 5.17, training venv) and
   `DinoV3MiewIdNet`; compare final tokens, the concat feature, and BN output.
2. **End-to-end parity.** Identical image bytes, bbox and theta, including
   EXIF-rotated images, θ ≠ 0, boxes overhanging the frame, and full-frame;
   compare chips, input tensors and embeddings.
3. Acceptance requires **all** of the following, because cosine and rank
   checks alone accept a uniformly scaled embedding:
   - per-stage raw error limits on final tokens, the 1536-d concat feature and
     the BN output: max abs error and relative L2 error
     (`‖a − b‖ / ‖b‖`);
   - cosine distance per embedding;
   - top-k retrieval agreement on a fixed gallery from at least three
     DINOv3-routed species.

   Thresholds are agreed with the training owner before release. Proposed
   defaults under matched fp16 autocast: relative L2 ≤ 1e-2 at every stage,
   cosine distance ≤ 1e-3, identical top-5 on ≥ 99% of queries. A **negative
   control** (the service embedding multiplied by 1.05) must fail the gate,
   which proves the raw-error check is live.

The gate runs against the final checkpoint. Development uses a frozen dev
snapshot that must never ship or gate a release:
`miew_id.msv5_dinov3.DEV-ep10.bin` (epoch 10, group-avg 0.7261, 342,749,463
bytes, sha256 `1c30dec4cf20e82da5cb9a9e2bf53efe4a7080c75ec3db3416952f592b356735`).

### Reference chips and the image frame

19 chips (`reference_chips.csv`): θ ≠ 0 (4), bbox overhanging the frame (3),
full-frame (3), plain DINOv3-routed (3), plain EffNet-routed (2, as contrast),
EXIF-tagged (4, including trout with orientation 6 and θ = 1.31). Ten
DINOv3-routed species are covered.

Training decodes with `cv2.imread(path)` then BGR→RGB
(`wbia_miew_id/datasets/helpers.py:8` at `316688c`). OpenCV's default `imread`
**applies** EXIF orientation (verified on OpenCV 4.10: a 400×300-stored JPEG
with orientation 6 reads as 300×400; only `IMREAD_IGNORE_ORIENTATION` returns
the stored frame). `decode_image_rgb` produces the same upright frame, so
training and service crop from the same pixels and **no decode change is
needed**. The training session's note that cv2 ignores EXIF is incorrect.
Following it would reintroduce the Whiskerbook bug that `decode_image_rgb`
fixed (`tests/test_exif_orientation_decode.py`).

The trout row (recorded 4000×3000, decoded 3000×4000) and the lion row
(recorded 683×800, file 598×700) carry bboxes whose source frame or resolution
disagrees with the file. They stay in the gate as-is: both sides apply the
same bbox to the same decoded pixels, which is what parity needs. Whether
those bboxes crop the intended animal is a training-data question, not a
parity one.

The end-to-end stage of the gate feeds the reference side through the training
repo's own `load_image` + `get_chip_from_img` + `get_test_transforms` and the
service side through `MiewidModel.extract_embeddings`, both from the same file
bytes. It never reimplements either decoder.

## Training-session inputs

Received 2026-10-16 (`docs/claudeQuestions2.txt`; artifacts in
`miewid-dinov3-dev/`):

- Backbone config dump (`dinov3_backbone_config.json`), HF revision
  `5931719e67bbdb9737e363e781fb0c67687896bc`. It matches the frozen config
  above; `config.image_size` is 224 while we run 448, which is fine because
  DINOv3 uses RoPE, not learned absolute position embeddings.
- `state_dict_keys.tsv`: 217 keys, 85,666,562 params, all fp32
  (`backbone.*` 211, `pooling.p` 1, `bn.*` 5). `k_proj` has no bias.
- Attention `sdpa`, RTX A6000, TF32 at defaults (see above).
- Frozen dev checkpoint ep10 and the 19 reference chips (see above).

These become test fixtures under `tests/fixtures/dinov3/` (config dump, key
manifest) and the parity-gate manifest under `scripts/preflight/`.

Still pending, at end of training:

1. Final artifact name, immutable URL, and SHA-256.
2. Reference embeddings for the 19 chips from the final weights, under
   autocast, on the A6000.
3. Running the golden-output generator script (supplied with the plan) in the
   training venv: seeded synthetic weights through the real 5.17
   `DINOv3ViTModel` + neck, fp32 CPU. This one does not depend on training and
   can run as soon as the script exists.
