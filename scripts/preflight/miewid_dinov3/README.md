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
  pass.
- Chip failures with `shape_equal: false` are crop/rotation disagreements;
  small `max_abs` differences with equal shapes are JPEG-decoder (cv2 vs
  Pillow) differences.
- The trout and lion chips carry bboxes whose source frame or resolution
  disagrees with the file. Both sides apply the same bbox to the same decoded
  pixels, so they still compare.
