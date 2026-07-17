# wbia-orientation: restore theta for whale sharks

Design for a new ml-service model type that runs `wbia-plugin-orientation`
checkpoints as the **oriented-bbox regressors they are**, returning **theta**,
and wires that theta into `/pipeline/`.

Related: issue #33 (fabricated viewpoint labels), PR #34 (stop fabricating).

**Revision 6** — after five design-review rounds (r1: 3C/6M; r2: 2C/3M; r3: 2C/3M;
r4: 1C/1M; r5: 1C/2M). Newest changes marked ▲▲▲▲▲.

## Problem

Sharkbook lost annotation rotation when it moved to the v2 ml-service in June
2026. Measured on its production DB (`FEATURE."REVISION"` monthly histogram,
`IACLASS='whaleshark'`):

| month | theta_zero | theta_real | % zero |
|---|---|---|---|
| 2025-02 → 2026-05 (16 mo) | ~230 | ~9,200 | **~2–4%** |
| 2026-06 | 132 | 96 | **58%** |
| 2026-07 | **186** | **4** | **98%** |

~315 whale shark annotations have `theta = 0` that should carry a real rotation.

**Cause.** `theta` in `/pipeline/` comes only from the *detector*
(`pipeline_router.py:189,222`). Whale sharks detect with `whaleshark_v0`, a
**lightnet** model, and `lightnet_model.py` never emits `thetas` — so
`pipeline_router:198` defaults it to `0.0`. Under WBIA, theta came from
`orientation.whaleshark.v3` via `wbia-plugin-orientation`. ml-service loaded that
checkpoint under `densenet-orientation`, which treats it as a 5-class classifier
(issue #33), so it produced a garbage viewpoint label and **never produced theta
at all**.

**Why this outranks #33.** A wrong viewpoint corrupts the candidate *filter*. A
missing theta means the crop was never rotated, so **MiewID embedded a tilted
animal** — the embedding is wrong at the source, and every match touching those
annotations is degraded, including matches initiated from healthy ones. #33 is
recoverable by nulling a column; this is only recoverable by re-extracting
embeddings.

## What the checkpoint actually is

The 5 outputs are `[xc, yc, xt, yt, w]`, **normalized to [0,1]** —
`wbia_orientation/_plugin.py:326-340`:

> `output (torch tensor): tensor of shape (bs, 5) ... coords are between 0 and 1`

They are in [0,1] because the head is **sigmoid**, not softmax
(`models/orientation_net.py:64`) — which is why softmaxing them yields ~0.2
"probabilities".

▲ **All three deployed checkpoints verified structurally identical**: `whaleshark.v3`,
`leopard_shark.v0`, `salamander_fire_adult.v2` each expose
`model.classifier.weight` of `(5, 2048)`, carry HRNet `stage2/stage3` keys, and
declare **no `classes`**. ▲ **No species YAML overrides** `MODEL.IMSIZE`,
`MODEL.CORE_NAME`, `TEST.HFLIP` or `TEST.VFLIP` (grepped across
`wbia_orientation/config/*.yaml`), so `config/default.py` values apply. NB
leopard shark and salamander are not in the plugin's `CONFIGS` map at all —
newer models trained on the same pipeline.

## Reference algorithm (authoritative)

Sources: `_plugin.py:189-296` (inference), `dataset/animal_wbia.py:17-37`
(preprocess), `models/orientation_net.py:63-95` (head + TTA),
`utils/utils.py:88-116` (un-flip), `core/evaluate.py:9-20` (theta).

0. ▲▲ **Decode with `imageio`**, matching `animal_wbia.py:18` (`imageio.imread`),
   reading from bytes rather than a path. This is **not** interchangeable with
   ml-service's existing decoders — the paths diverge on non-RGB inputs:

   | | grayscale | RGBA |
   |---|---|---|
   | `imageio.imread` (reference) | `(H,W)` 2-D | `(H,W,4)` |
   | `cv2.imdecode(IMREAD_COLOR)` (`densenet_orientation.py:168`) | forced `(H,W,3)` | forced `(H,W,3)` |
   | `PIL.convert('RGB')` (`miewid.py:186`) | `(H,W,3)` | `(H,W,3)` |

   ▲▲▲ **Non-RGB policy — revision 3 was incoherent and is corrected.** It called
   grayscale/RGBA "fidelity strata", but **the reference cannot produce a theta from
   either** — its `Normalize(3-channel)` rejects them. Measured:

   ```
   grayscale (H,W)  -> RuntimeError: output with shape [1,224,224] doesn't match broadcast shape [3,...]
   RGBA (H,W,4)     -> RuntimeError: size of tensor a (4) must match tensor b (3)
   RGB (H,W,3)      -> tensor (1,3,224,224)   OK
   ```

   There is no reference angle to compare against, so that stratum was asking the
   gate to diff against a crash.

   **Policy: canonicalize to RGB before crop** — grayscale replicated to 3 channels,
   RGBA alpha dropped. This is an **explicit, documented deviation**: the reference's
   only "behaviour" here is an exception, which is not a specification, and failing a
   whole detection because a frame is grayscale (common for IR/night captures) is
   worse than handling it.

   **The wrapper is validated, not assumed**: convert the fixture to RGB *outside*
   and run the **reference** on it; run the **port** on the original; require the
   same circular-error thresholds. That makes canonicalization a tested transform
   rather than a silent one. Pure-fidelity strata stay **RGB-only**, where a
   reference exists.

1. Crop the **axis-aligned** bbox: `image[y1:y1+h, x1:x1+w]`.
   ▲ **Empty/degenerate-crop fallback** (`animal_wbia.py:25-28`): if
   `min(image.shape) < 1`, reload the **full image** and replace the bbox with
   `[0, 0, width, height]`. This changes the model's input and must be ported.

   ▲▲▲ **`effective_bbox` — revision 3 was not end-to-end sound.** The fallback
   silently *changes which region the theta describes*. If orientation falls back to
   the full image but classify/extract still receive the original degenerate bbox,
   the theta belongs to a different region than the crop it rotates — a new
   silently-wrong value, i.e. the #33 failure mode again.

   ▲▲▲▲ **"valid → use as-is" was still unsafe: NumPy slicing does not clamp.**
   `image[y1:y1+h, x1:x1+w]` resolves negative indices **from the far edge**, so an
   out-of-image bbox does not fail — it silently crops the wrong region. Measured on
   a 400px-wide image with the far-right 20 columns marked:

   | bbox | reference crop | far-edge content? | `slice().indices()` |
   |---|---|---|---|
   | `x1=-20, w=50` | width **0** (empty) | no | `[380:30]` |
   | **`x1=-20, w=500`** | width **20** | **YES** | `[380:400]` |
   | `x1=380, w=50` (overrun) | width **20** (truncated) | yes | `[380:400]` |

   A negative origin with a wide box crops the **far edge**; an overrun silently
   **truncates**. In both, theta would describe a region the detector never pointed
   at, while classify/extract — which clamp differently — crop somewhere else.
   `effective_bbox` alone does not fix this; the *resolution rule* must match.

   **Bbox policy — resolve the ACTUAL reference slice, then reuse it everywhere:**
   1. Validate finite numeric values; **fail the whole request (500)** on
      malformed/non-finite/NaN, before any consumer runs.
   2. Apply one explicit integerization rule (documented; no implicit float→int).
   3. Derive the real slice with `slice(start, stop).indices(dim)` — reproducing
      exactly what NumPy did, including the far-edge and truncation cases above.
   4. **Non-empty** → that in-bounds slice **is** `effective_bbox`.
      **Empty** → reference full-frame fallback, `effective_bbox = [0,0,W,H]`.
   5. Feed `effective_bbox` to **both** classify and extract, so all three consumers
      see the identical region.
   6. ▲▲▲▲▲ **Emit `effective_bbox` as the result's `bbox`.** Fixing only the internal
      consumers is not enough: `/pipeline/` emits `'bbox': bbox_coords`
      (`pipeline_router.py:334`) and Wildbook persists exactly that alongside theta via
      `MlServiceProcessor.featureParams(bbox, theta, viewpoint)`. Leaving the detector's
      original bbox in the result would store a bbox naming one region beside a theta
      describing another — the same mismatch, one layer further out. For
      `WbiaOrientationModel`, the result's `bbox` **is** `effective_bbox`; the raw
      detector bbox is retained only as a separate audit field (`detector_bbox`).

   ▲▲▲▲▲ **Integerization rule: `int()`, truncate-toward-zero, applied once** to each of
   `[x, y, w, h]` before deriving `start`/`stop`. This is not a new invention — it is
   what `pipeline_router.py:218` already does (`bbox_list = [int(x), int(y), int(width),
   int(height)]`). Declaring it matters because truncate/floor/round give materially
   different slices for negative fractional coordinates (`int(-0.5) == 0`, but
   `floor(-0.5) == -1`, which then resolves from the far edge). Test fractional
   **negative and positive** inputs.

   ▲▲▲▲▲ **Pre-existing inconsistency, noted:** today the emitted bbox
   (`bbox_coords`, raw, possibly float) differs from the consumed bbox (`bbox_list`,
   int-truncated) — so Wildbook already persists a bbox the embedding did not use. It
   is sub-pixel and harmless in practice, but it is the same bug class; this design
   closes it for the orientation path by emitting a single authoritative bbox.

   Tests: negative origin (narrow **and** wide), overrun, fully out-of-image,
   non-finite. NB `pipeline_router:225` today skips zero/negative-dimension bboxes
   *before* orientation; that skip must move behind this policy or the fallback can
   never fire.
2. Resize to `MODEL.IMSIZE = [224, 224]` with
   `skimage.transform.resize(..., order=3, anti_aliasing=True)` (bicubic).
3. `ToTensor()` then `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`.
   NB `skimage.resize` returns float64 in [0,1], so `ToTensor` does **not**
   rescale by 255.
4. ▲ **Cast to float32 immediately before inference** — `_plugin.py:272` calls
   `model(images.float(), ...)` precisely because skimage yields float64.
5. `out = sigmoid(hrnet(x))` → `(1,5)` in [0,1].
6. **TTA (on by default: `TEST.HFLIP=True`, `TEST.VFLIP=True`)**
   - `out_h = sigmoid(hrnet(flip(x, dim=3)))`; un-flip with `image_h_w=[1.0,1.0]`:
     `out_h[0] = 1 - out_h[0]`, `out_h[2] = 1 - out_h[2]`
   - `out_v = sigmoid(hrnet(flip(x, dim=2)))`; un-flip:
     `out_v[1] = 1 - out_v[1]`, `out_v[3] = 1 - out_v[3]`
   - `out = (out + out_h + out_v) / 3`
   - Recursion terminates: the inner `self.forward(flipped)` passes no flip args,
     so both default to `False`.
7. `theta = arctan2(out[3] - out[1], out[2] - out[0]) + radians(90)`

Theta is computed on the **normalized** coords, *before* `resize_oa_box`
(`orientation_post_proc` calls `compute_theta` first, then mutates `output`). A
theta-only port may skip the box resize — ▲ but any returned `coords` must be
labelled **normalized**, never image-space.

## Design

### 1. New model type `wbia-orientation`

New `app/models/wbia_orientation.py` → `WbiaOrientationModel`, registered in
`MODEL_REGISTRY`. Distinct from `densenet-orientation` (genuine viewpoint
classifiers; currently none — see #34).

- **Load**: `timm.create_model('hrnet_w32', pretrained=False, num_classes=5)`,
  strip `model.`/`module.` prefixes, `load_state_dict` **strict**. ▲ A strict load
  of each of the three checkpoints runs in the host preflight (not CI — the weights
  cannot be committed), not as an assumption.

  ▲▲ **timm-vs-reference equivalence: PROVEN, not assumed.** Revision 1 argued
  "strict load succeeds, so the architecture matches" — that is invalid reasoning
  (strict load establishes key/shape compatibility, not identical forward
  computation), and the reference builds
  `wbia_orientation.models.cls_hrnet.HighResolutionNet`, not timm. So it was
  measured. Both models were constructed, the **same** `orientation.whaleshark.v3`
  weights loaded into each (`missing=0, unexpected=0` both), and the same input run
  through both in eval/no-grad:

  ▲▲▲ **Extended to 3 checkpoints x 6 inputs = 18 comparisons** (revision 2's single
  checkpoint/input did not justify the word "PROVEN"). Every checkpoint strict-loads
  into both; inputs span noise (2 seeds), zeros, ones, ImageNet-scaled, and a batch
  of 3:

  ```
  WORST over 3 checkpoints x 6 inputs : 5.960e-08
  ```

  `5.96e-08` is exactly **1 ULP of float32** (2^-24) — floating-point op-ordering
  noise, not an architectural difference. Note the reference's `get_cls_net` builds a
  1000-class head that `OrientationNet` replaces with `nn.Linear(2048, 5)`; the
  comparison reproduces that.

  ▲▲▲ **Where it runs.** The checkpoints are hundreds of MB and cannot be committed,
  so this cannot be a normal CI test — and a *skipped* test is not a safeguard. It
  therefore runs as part of the release-blocking **host preflight** (below), where
  the weights exist — not as a CI test. If it ever fails, vendor `cls_hrnet`.
- **No label_map, no classes.** Must never be usable in the classify slot.
- ▲▲▲▲ **API contract** (revision 4 promised `effective_bbox` and batching in prose
  but declared neither):

  ```python
  predict(image_bytes, bbox) -> {
      "model_id": str,
      "theta": float,                    # radians; finite, else the request fails
      "coords_normalized": [xc, yc, xt, yt, w],   # [0,1]; NEVER image-space
      "effective_bbox": [x, y, w, h],    # the ACTUAL slice used (see bbox policy)
  }

  predict_batch(image_bytes, bboxes) -> [result, ...]
      # exactly ONE ordered result per input bbox, or ATOMIC failure.
      # Callers must assert len(results) == len(bboxes) before any consumer runs.
  ```

  `predict_batch` is the router's entry point (orientation batches across an image's
  bboxes, incl. TTA passes). Ordered one-per-bbox is load-bearing: a silently
  misaligned result would attach one bbox's theta to another's crop.

  ▲ No `label`, no `probability`, no `class_id` — nothing a viewpoint consumer
  could mistake for a classification (the #33 failure mode).
- ▲ **Accepts no input theta.** The reference consumes the *axis-aligned* bbox;
  feeding it a detector-rotated crop would double-rotate. The signature takes
  image bytes + bbox only, and rejects a non-zero theta if one is passed.

### 2. `pipeline_router` wiring

- Keep `orientation_model_id`; dispatch on model type — `DenseNetOrientationModel`
  (label, legacy) or `WbiaOrientationModel` (theta).
- ▲ **Fail-closed, at REQUEST level.** Orientation currently soft-fails and
  continues. For `WbiaOrientationModel` that is unacceptable: a failure, malformed
  result or NaN would fall back to the detector's `0.0` and embed an unrotated crop
  — *the exact bug being repaired*. A **predicted `0.0` is valid** and must be
  distinguished from missing/failed.

  ▲▲ **Scope (revision 2 was contradictory).** "5xx and emit no result for that
  bbox" is incoherent: a 5xx body cannot also carry successful siblings, and
  partial success would silently drop detections. **If orientation fails for ANY
  bbox requiring it, the whole request fails: one request-level `500`, no results,
  and neither classify nor extract runs for any bbox.** Dropping a detection
  silently is how annotations go missing; failing loudly is recoverable.
  `500` is correct for a malformed/non-finite result or internal inference failure;
  reserve `503` for genuinely transient dependency/resource failures.
  Test: two bboxes, one returns NaN → whole request 500, and assert classify/extract
  were never invoked.
- ▲ **Ordering and batching.** Theta must precede both consumers, so
  `asyncio.gather` over (orientation ∥ classify ∥ extract) is wrong. New shape:
  `detect → orientation (batched across all bboxes, incl. TTA passes) → per-bbox
  (classify ∥ extract)`. Do **not** spawn competing per-bbox GPU orientation
  threads. Latency budget: measure p50/p95 on the T4 at representative detection
  counts before rollout; adds 3 forward passes per bbox.
- Set the result's `theta` from orientation, overriding the detector's; emit
  `theta_source: "orientation"|"detector"` for auditability.

### 3. Wildbook (Java) — no change required ▲ *verified, not assumed*

`MlServiceProcessor.persistDetections` iterates `results` and reads each entry's
theta, persisting it unmodified:

```java
for (int i = 0; i < results.length(); i++) {
    JSONObject result = results.getJSONObject(i);   // :366
    double theta = result.getDouble("theta");       // :368
    ...
    JSONObject featureParams = featureParams(bbox, theta, viewpoint);  // :475
```

theta is not reassigned between :368 and :475. ▲ Still gated on an end-to-end
contract test asserting a non-zero orientation theta reaches `FEATURE.PARAMETERS`
— reading code is not proof.

▲ **Dedup interaction (found while verifying the above):**
`findExistingAnnotation(ma, bbox, theta)` (:417) dedups on bbox **and** theta. So
re-running detection over an affected asset would **not** match its `theta=0` row
and would create a **duplicate**. The backfill must update in place — never
re-detect.

### 4. Config and rollout ▲

Re-add the three checkpoints under new `-theta` ids (so nothing can resolve the
old classifier semantics):

```json
{ "model_id": "whaleshark_v3-theta", "model_type": "wbia-orientation",
  "checkpoint_path": "/datasets/orientation.whaleshark.v3.pth", "imsize": [224, 224] }
```

**Risk**: `main.py` eager-loads every entry and re-raises, so a bad entry downs a
service shared by 5 installs. The registry is also a single shared file (see the
`MODEL_CONFIG` mount issue).

▲▲ **Revision 2 claimed "other installs are unaffected by config they do not
name". That is materially FALSE and is struck.** `main.py` loads *every* entry in
the shared `model_config.json` at startup regardless of which install references
it — so an install that never names `whaleshark_v3-theta` still loads it, and still
dies if its checkpoint is missing or its architecture is incompatible on that box.
**Naming isolation is not startup isolation.** This is not hypothetical: on
2026-07-16 a config/code mismatch on exactly this path took the shared service down
for all installs.

Mitigations, in order:
1. **Startup preflight on the actual host**, with its real `/datasets` mount and
   GPU — not merely a local load. Local success does not predict the box.
2. Deploy code first (inert until config names the new type), then config.
3. Stage-load all three checkpoints against the new code before rollout.

Per-install model configuration is the real fix — it is the only thing that makes
startup risk proportional to which install needs a model. It is an architectural
change to the shared registry, out of scope here, and **should be filed**
alongside the `MODEL_CONFIG` mount fix.

## Dependency pins ▲▲

Revision 2 said "add and pin `scikit-image`" without naming versions — not a
specification. Interpolation and anti-alias behaviour are version-sensitive, and
so is `imageio` decoding. **Pin the exact versions the fidelity gate was executed
against**, record them here, and re-run the gate on any bump:

▲▲▲ Resolved and recorded (placeholders were not pins):

▲▲▲▲▲▲ **Corrected to the ACTUAL production stack.** Revision 5 recorded the
versions from a convenience venv (`--system-site-packages`, so it inherited
pillow 12.1.1 / numpy 1.26.4 / timm 1.0.25) — which are NOT what `requirements.txt`
pins. The gate had therefore proven fidelity for a stack production does not
install: the same "measured the convenient artifact" error this whole change
exists to correct. Both gates were **re-run against the real pins** and produce
identical numbers (timm 1.0.19 equivalence 5.960e-08; fidelity theta 7.726e-07,
coords 1.192e-07).

Only `scikit-image` and `imageio` are NEW. `Pillow` was already pinned and stays
put — bumping it on a service shared by five installs is out of scope here — but
it is part of this contract, because imageio's decoding is not reproducible from
the imageio pin alone.

```
scikit-image==0.26.0      # NEW: resize(order=3, anti_aliasing=True)
imageio==2.37.3           # NEW: decode
Pillow==10.1.0            # pre-existing; imageio's JPEG/PNG backend
numpy==1.26.3             # pre-existing
timm==1.0.19              # pre-existing; equivalence re-verified on THIS version
torch==2.10.0+cu128
```

The gate, its input manifest, and the reference runner live in
`scripts/preflight/` and are release-blocking on the host. Re-run on ANY bump to
these lines.

Both scikit-image and imageio are **new dependencies on a service shared by 5
installs** — which is itself why the host preflight above is mandatory. Re-run the
fidelity gate on any bump to any line here, including the Pillow backend.

## Validation ▲ — the gate is reference-vs-port, not legacy agreement

Revision 1 proposed "agreement against ≥200 legacy stored thetas". That is **not a
fidelity oracle**: it has no threshold, ignores angular wraparound, and compares
to historical annotation state rather than to the authoritative implementation.

**Primary gate (fidelity).** The plugin source and all three checkpoints are
available locally, so the reference can be executed directly — no WBIA test DB
needed. Run reference and port over production image bytes + bboxes and compare
per sample with circular error (handles wraparound):

```
d = abs(atan2(sin(theta_ref - theta_port), cos(theta_ref - theta_port)))
```

▲▲ **CI/release-blocking, with a checked-in manifest.** Revision 2's "≥200
stratified" was too vague to mean anything, and `1e-3 rad` tolerates a 0.057°
outlier — not "near exact" for a port of the same math onto the same weights on
the same device.

- **Thresholds: `max ≤ 1e-5 rad`, `mean ≤ 1e-6 rad`.** Relax only with measured,
  documented platform variance — never to make a failing port pass.
▲▲▲▲ **Input manifest, live comparison — NOT frozen expected values.** The weights
cannot live in CI, so the gate runs at host preflight and executes **the pinned
reference and the port live, comparing them to each other**. Freezing expected
thetas would only pin one implementation's output and rot on any dependency bump;
frozen values are an optional diagnostic baseline, never the oracle.

The checked-in manifest carries **inputs and identity**, not answers:
- fixture **byte hashes** + exact bboxes (immutable fixture identity)
- **checkpoint SHA-256s**
- reference-source **revision/hash** (which `wbia-plugin-orientation` commit)
- stratum tags with **enforced per-stratum minimums** (fail if unmet)

The preflight emits an artifact recording reference outputs, port outputs, and the
full environment versions.

**Compared per sample**, each with its own acceptance criterion (revision 5 compared
coordinates but only set thresholds for theta, which made that half unenforceable):

| compared | criterion |
|---|---|
| theta | circular error `abs(atan2(sin Δ, cos Δ))`: **max ≤ 1e-5 rad, mean ≤ 1e-6 rad** |
| `coords_normalized` | elementwise absolute error: **max ≤ 1e-6, mean ≤ 1e-7** |
| `effective_bbox` | **exact equality** (integers; no tolerance) |
| `predict_batch` | **exact count and order** vs input bboxes |

Coordinate tolerance cannot be exact equality: measured model variance is ~1 ULP of
float32 (5.96e-08) on outputs in [0,1], so 1e-6 max leaves ~17x headroom over
observed noise while still failing any real preprocessing divergence. Theta alone can
pass while coordinates are wrong, so both are required.

- **Coverage: all three checkpoints** (whaleshark_v3, leopard_shark_v0,
  salamander_fire_adult_v2), with per-stratum minimums, over strata that can each
  independently break the port:
  | stratum | why |
  |---|---|
  | codec / channel form: RGB JPEG, **grayscale**, **RGBA PNG** | decode paths diverge here (see step 0) |
  | crop **downscale** (crop ≫ 224) | skimage's anti-alias prefilter only engages downscaling |
  | crop **upscale** (crop ≪ 224) | different interpolation regime |
  | extreme aspect ratios | resize distortion feeds theta |
  | bbox edge / degenerate / out-of-image | exercises the reload-full-image fallback |
  | multi-detection images | exercises orientation batching |
- Reference and port must run **same device, same dtype, eval + no_grad, autocast
  disabled** — otherwise the comparison measures the harness, not the port.

**Secondary (model quality, not a gate).** Compare port theta against the ~9,200
legacy real thetas, using the same circular metric. Disagreement here is
informative (drift, or the legacy value was human-edited) but must not block.

**Unit tests.** sigmoid not softmax; un-flip index arithmetic (`1-x` on `0,2` for
h and `1,3` for v); 3-way mean; `+90°`; float32 cast; degenerate-crop fallback;
hand-computed theta (`[0.5,0.5,1.0,0.5,0.1]` → `arctan2(0,0.5)+90° = 90°`); no
`label`/`probability` key ever emitted; rejected from the classify slot; NaN theta
→ fail-closed, not 0.0.

## Backfill pathway (separate PR)

For the ~315 affected annotations, theta must be recomputed **and embeddings
re-extracted** — the existing embeddings came from unrotated crops.

1. Identify: `FEATURE."REVISION" >= 2026-06-01` AND `theta = 0` AND
   `IACLASS IN ('whaleshark', <leopard shark class>)`.
2. Recompute theta via the orientation model.
3. **Update `FEATURE."PARAMETERS"` theta in place** — ▲ never re-detect (dedup is
   theta-sensitive; re-detection would duplicate the annotation).
4. Re-extract the embedding via `/extract/` **with the corrected theta**.
5. Bump `ANNOTATION."VERSION"` so OpenSearch reindexes.

Shape it like `appadmin/catchUpEmbeddings.jsp`. Order matters: theta must be
written before re-extraction, or the new embedding repeats the original error.
