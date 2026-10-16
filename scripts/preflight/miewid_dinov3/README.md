# MiewID DINOv3 parity gate

Release-blocking. Run on the **final** checkpoint, never the DEV snapshot.
Spec: `docs/plans/2026-10-15-miewid-dinov3-design.md` ("Parity gate").

## Why

`app/models/dinov3_vit.py` is a vendored port. Its failure mode is
embeddings that are slightly wrong, which retrieval cannot show you until
matches go missing. The only trustworthy oracle is the training code itself,
run live on the same bytes.

## 1. Reference (training host, training venv)

First check the two ASSUMPTIONS in `reference_runner.py`'s docstring against
the DINOv3 experiment tree. Then:

```bash
python scripts/preflight/miewid_dinov3/reference_runner.py \
  --repo /path/to/audit_v4_1/dinov3-exp --repo-revision <commit or tarball md5> \
  --checkpoint /path/to/miew_id.msv5_dinov3.bin \
  --chips reference_chips.csv --path-column host_path --out /tmp/dinov3-ref
```

Copy `/tmp/dinov3-ref/` (it includes `reference_chips.csv`) and the 19 chip
images to the ml-service host.

## 2. Service (ml-service host, INSIDE the production image, from the repo root)

```bash
docker build -t ml-service -f docker/dockerfile .
docker run --rm --gpus all --entrypoint "" -v "$PWD:/src" -w /src \
  -v "$MODELS_DIR:/datasets:ro" -v /path/to/dinov3-ref:/ref:ro \
  -v /path/to/chip-images:/chips:ro -v "$PWD/preflight-artifacts:/artifacts" \
  ml-service python3 -m scripts.preflight.miewid_dinov3.run_gate \
    --manifest scripts/preflight/miewid_dinov3/manifest.json \
    --reference /ref \
    --rewrite-prefix /nas/=/chips/nas/ --rewrite-prefix /data0/=/chips/data0/ \
    --artifact /artifacts/dinov3-gate.json
```

Fill `manifest.json` `checkpoint.sha256` first; the gate refuses a placeholder.
It also aborts before inference if any row, image file or the checkpoint
differs from what the reference used.

## Reading the result

- Two bbox modes, each gated on its own: `router` (`int()` of each
  coordinate, which truncates toward zero, exactly what `/extract` passes in
  production) and `float` (the CSV values, as training passes them). Both
  sides use the same bbox within a mode, so a failure is implementation
  error, never rounding.
- `pass` needs every stage within limits in both modes **and** every ×1.05
  negative control failing. Cosine and top-k are blind to scale, so the
  control proves the raw-error check is live.
- `thresholds_status: "proposed"` means the limits have not yet been agreed
  with the training owner. A pass under proposed limits is not a release
  pass, so the gate exits 2 unless `--allow-proposed` is given.
- Chip failures with `shape_equal: false` are crop/rotation disagreements;
  small `max_abs` differences with equal shapes are JPEG-decoder (cv2 vs
  Pillow) differences.
- The trout and lion rows carry bboxes whose source frame or resolution
  disagrees with the file; whether they crop the intended animal is a
  training-data question. They still compare (same bbox, same decoded pixels).
- The three `bbox_overhang` rows (bowhead_whale 56466, lion 36100,
  bottlenose_dolphin 36) are listed in `manifest.json` `known_divergence`.
  Decoding matches but cropping does not: the service's `get_chip_from_img`
  pads overhanging boxes (`crop_rect`), the training helper clamps and slices.
  This predates the DINOv3 work and affects EfficientNet too; which chipper is
  correct is an open item. These rows are computed and reported in every e2e
  stage under the artifact's `known_divergence` section (chip, input,
  embedding, cosine; with the row count excluded and the reason) but are
  excluded from the gated e2e stages, e2e top-5 and the e2e negative control.
  The arch stage (identical reference tensors in) is gated on all 19 rows.
  Every `known_divergence` entry must match a row of `expected_chips`.
- `manifest.json` `expected_chips` is the ordered 19-row gallery. The gate
  aborts before inference unless the reference CSV matches it exactly, so a
  missing, extra, duplicate or reordered row cannot pass.
- Exit codes: 0 pass; 1 a check failed; 2 every check passed but
  `thresholds_status` is not `"agreed"` (pass `--allow-proposed` to get 0 for a
  dry run; it is still not a release pass).

## Before deploying this branch

Any MiewID entry that now raises in `MiewidModel.load` aborts the whole service
at startup. Audit the production config first (no weights, no network):

```bash
python3 -m scripts.preflight.miewid_dinov3.audit_config /path/to/model_config.json
```

Exit 1 names every entry that would abort startup and why.
