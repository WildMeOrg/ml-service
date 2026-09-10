# wbia-orientation host preflight

Release-blocking. **Run on the ml-service host**, with its real `/datasets` mount
before `wbia-orientation` config is deployed. This gate validates CPU inference
with 224×224 inputs and both flips enabled. It does not certify GPU kernels or
custom `imsize`/TTA deployment settings.

Not CI: the checkpoints are hundreds of MB and cannot be committed — and a
*skipped* test is not a safeguard.

## Why it exists

`app/models/wbia_orientation.py` ports `wbia-plugin-orientation`. Its failure
mode is being **silently wrong by an angle**, which downstream is
indistinguishable from correct output — the same class of bug as #33, whose
fabricated viewpoint labels went unnoticed for two months. The only trustworthy
oracle is the reference implementation itself, executed live on the same bytes.

## What it checks

1. **Architecture equivalence** — reference `cls_hrnet` vs `timm hrnet_w32` on the
   same weights. Justifies not vendoring. Measured: 3 checkpoints × 6 inputs,
   worst `5.960e-08` (1 ULP of float32).
2. **Fidelity** — reference vs port over `manifest.json`'s fixtures. Measured on
   whaleshark_v3: theta `7.726e-07` rad, coords `1.192e-07`.
3. **Strict load** of each deployed checkpoint.

Compared per sample: circular theta error, elementwise `coords_normalized`,
`effective_bbox` exact equality, and `predict_batch` count/order. Theta alone can
pass while coordinates are wrong.

## Running it — INSIDE THE BUILT IMAGE

Run it in the container, **not** on the host's Python. The image is the stack that
ships; a host venv is not. Earlier revisions of this gate ran on Python 3.12 with
torch 2.10 while the image is **Python 3.10 with torch 2.1.2** — so the gate
"passed" for a stack that does not exist in production, and only `docker compose
build` caught it (`scikit-image==0.26.0` requires Python >=3.11 and cannot install
in the image at all).

```bash
# Tag the image explicitly.  Compose otherwise chooses a project-derived tag
# (for example `docker-ml-service`), while the run command below uses
# `ml-service`.
docker build -t ml-service -f docker/dockerfile .

docker build -t ml-service-preflight -f scripts/preflight/Dockerfile .
mkdir -p preflight-artifacts

docker run --rm \
  -v "$MODELS_DIR:/datasets:ro" \
  -v "$PWD/fixtures:/fixtures:ro" \
  -v /path/to/wbia-plugin-orientation:/reference:ro \
  -v "$PWD/preflight-artifacts:/artifacts" \
  ml-service-preflight \
  python3 scripts/preflight/run_gate.py \
      --manifest scripts/preflight/manifest.json \
      --fixtures /fixtures \
      --reference-root /reference \
      --artifact /artifacts/preflight-artifact.json
```

`--reference-root` is the checkout root containing `wbia_orientation/`. It
wins over `reference_source.path` in the manifest; the default is `/reference`.
The reference loads lazily, so `--help` works without a mounted checkout.
Populate the manifest and fixture directory before building the preflight image.

The preflight image adds pinned `yacs` and `utool` without changing the runtime
image. Its install is constrained by the runtime's `pip freeze`, so conflicting
dependencies fail the build instead of silently upgrading the inference stack.
To use an existing runtime tag, pass `--build-arg RUNTIME_IMAGE=your-runtime-tag`.

Emits an artifact recording reference outputs, port outputs, and the full
environment. **Re-run on any bump to `scikit-image`, `imageio`, `Pillow`, `timm`,
`torch`, or `numpy`** — `imageio`'s decoding is not reproducible from its own pin
alone, which is why Pillow is part of the contract.

`reference_runner.py` loads the plugin's config and `cls_hrnet` via `importlib`,
bypassing the `wbia` package import (which needs a full WBIA install). `cls_hrnet` imports
`utool` at module load, so the preflight image installs it even though pretrained
weight downloading is disabled.

## Environment the gate was last validated against

Matches `requirements.txt`, not a convenience venv — an earlier run validated
`pillow 12.1.1 / numpy 1.26.4 / timm 1.0.25` while requirements pinned
`10.1.0 / 1.26.3 / 1.0.19`, i.e. it proved fidelity for a stack production does
not install.

The stack that actually ships is **requirements.txt + dockerfile + the base
image's Python** — not requirements.txt alone:

```
python 3.10              # base image: nvidia/cuda:12.1.1-runtime-ubuntu22.04
torch==2.1.2             # docker/dockerfile (NOT in requirements.txt)
torchvision==0.16.2      # docker/dockerfile
Pillow==10.1.0           numpy==1.26.3        timm==1.0.19
scikit-image==0.25.2     imageio==2.37.3
```

`scikit-image` is capped at 0.25.2 because 0.26+ requires Python >=3.11. Pinning a
newer version from a dev machine's Python breaks the image build outright.

## Canonicalization fixtures

Use 8-bit grayscale (`L`) and `RGBA` originals for `canonicalization_wrapper`.
The port receives the original bytes; the reference receives an independent
RGB PNG copy. This copy uses imageio decoding followed by Pillow conversion:
grayscale channels are replicated and alpha is dropped, without compositing.
EXIF handling follows the same decoder as inference. Ordinary fidelity strata
require RGB and receive the original bytes unchanged. Other wrapper modes,
including 16-bit grayscale, are rejected rather than silently converted lossily.

## Acceptance and batch fixtures

Each fixture supplies either `bbox: [x, y, w, h]` or
`bboxes: [[x, y, w, h], ...]`. A `multi_detection` fixture must have at least two
distinct effective crops. Its reference predictions must be separated by more
than twice the maximum theta or coordinate tolerance for every pair, so swapping
predictions can actually be detected. The reference evaluates each crop separately;
the port receives all boxes in a single `predict_batch` call. Fractional bbox values
are truncated toward zero before the reference's native NumPy slice is taken.

All five coordinates and theta must be finite; effective bboxes must match exactly,
as must batch count and order. Both maximum and mean errors are enforced for each
checkpoint. Theta mean is the mean absolute circular error; coordinate mean is
across all five components of every comparison, not a mean of per-row maxima.
Equality with a threshold passes. Existing thresholds are unchanged; measure the
real-checkpoint baseline before proposing any tolerance changes.

Each declared stratum needs a positive `min_samples`, met separately for every
checkpoint by unique image-byte/effective-crop sets. Duplicate files and extra rows
in a batch do not inflate coverage. Empty manifests, malformed results, failed
loads and inference exceptions fail the gate. An atomic JSON artifact records
failures as well as successful comparisons, all four metrics, coverage, actual
checkpoint/fixture hashes, environment and observed model configuration per checkpoint.
It also preserves the manifest's claimed reference revision and hashes the exact source
bytes executed for all four standalone reference modules. These hashes remain tied to
the loaded code even if files change later in the process. Artifacts are written with
mode 0644 so a container-root run produces host-readable release evidence. The artifact's parent
directory must exist and be writable. Earlier numerical measurements above are
historical context; rerun the populated gate with real weights in the built image.
