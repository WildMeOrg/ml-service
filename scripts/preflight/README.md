# wbia-orientation host preflight

Release-blocking. **Run on the ml-service host**, with its real `/datasets` mount
and GPU, before `wbia-orientation` config is deployed.

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

docker run --rm --gpus all \
  -v "$MODELS_DIR:/datasets:ro" \
  -v "$PWD/scripts/preflight:/app/scripts/preflight:ro" \
  -v "$PWD/fixtures:/fixtures:ro" \
  -v /path/to/wbia-plugin-orientation:/reference:ro \
  ml-service \
  python3 scripts/preflight/run_gate.py \
      --manifest scripts/preflight/manifest.json \
      --fixtures /fixtures
```

`reference_runner.py` needs the `wbia-plugin-orientation` checkout mounted (it
loads the plugin's config and `cls_hrnet` via importlib) plus `yacs`. Neither is a
runtime dependency of ml-service, which is another reason this is a preflight and
not CI.

Emits an artifact recording reference outputs, port outputs, and the full
environment. **Re-run on any bump to `scikit-image`, `imageio`, `Pillow`, `timm`,
`torch`, or `numpy`** — `imageio`'s decoding is not reproducible from its own pin
alone, which is why Pillow is part of the contract.

`reference_runner.py` loads the plugin's config and `cls_hrnet` via `importlib`,
bypassing the `wbia` package import (which needs a full WBIA install). `utool` is
only required for pretrained-weight download, which we skip.

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
