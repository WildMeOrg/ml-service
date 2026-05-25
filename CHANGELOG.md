## v1.0.0 — Wildbook ML Service GA

First stable release of the Wildbook ML Service, a FastAPI replacement for the
inference half of WBIA. ml-service is the path forward for serving CV models to
Wildbook: a single GPU-backed HTTP service that handles detection, classification,
orientation, embedding extraction, explainability, and part-body assignment.

This release marks the point at which Wildbook can use ml-service as a drop-in
replacement for WBIA detection/labeling without changing its HTTP client code.

### Supported model types

| Type | Architecture | Use case |
|------|--------------|----------|
| `yolo-ultralytics` | YOLOv11 | Object detection |
| `megadetector` | MegaDetector (PyTorch-Wildlife) | Animal/person/vehicle detection |
| `lightnet` | PyDarknet YOLO v2/v3 | Legacy WBIA species detectors |
| `efficientnetv2` | EfficientNet-B4 (timm) | Species/viewpoint classification |
| `densenet-classifier` | DenseNet (torchvision) | Ensemble species/viewpoint classification |
| `densenet-orientation` | DenseNet-201 | Orientation classification |
| `miewid` | MiewID (EfficientNetV2-RW-M + ArcFace) | Re-ID embeddings |

### Endpoints

- `POST /predict/` — detection (YOLO / MegaDetector / LightNet)
- `POST /classify/` — classification (EfficientNet / DenseNet) with optional
  compound `species:viewpoint` label parsing
- `POST /extract/` — MiewID embeddings
- `POST /explain/` — PAIR-X pairwise explainability (lines + color maps)
- `POST /pipeline/` — detect → classify → extract, with optional orientation
- `POST /assign/` — geometric part-body assignment with species-specific
  scikit-learn classifiers
- `GET  /health` — GPU, CUDA, and loaded-model status

All image inputs (`image_uri`) accept HTTPS URLs, local paths, and `data:` URIs
(strict validation, no response or log echo).

### WBIA compatibility layer

Wildbook-facing endpoints under `/api/engine/` mimic WBIA's async job queue:

- `POST /api/engine/detect/cnn/` (and `.../yolo/`, `.../lightnet/`)
- `GET  /api/engine/job/status/?jobid=<id>`
- `GET  /api/engine/job/result/?jobid=<id>`
- `GET  /api/engine/job/`

Responses are wrapped in WBIA's `{"status": {...}, "response": ...}` envelope.
Annotation fields (`xtl`/`ytl`/`left`/`top`/`width`/`height`/`theta`/`confidence`/
`class`/`species`/`viewpoint`) match WBIA's format. The job store is bounded to
10,000 entries with LRU eviction of completed jobs.

### Highlights since the fork from `jmcdonald27/api-service`

#### Model coverage

- MegaDetector integration (v2 refactor)
- MiewID embedding extraction; MiewID v4.1 standalone as the default
- EfficientNet image classification with compound `species:viewpoint` label
  parsing
- DenseNet orientation support (DenseNet-201 + HRNet-W32 checkpoints)
- DenseNet classifier with ensemble + compound-label support
- LightNet detection for WBIA legacy species models (vendored to fix upstream
  submodule breakage)
- Assigner endpoint for part-body matching

#### Wildbook integration

- WBIA-compatible async job queue under `/api/engine/`
- `/pipeline/` and `/extract/` aligned with the Wildbook v2 response contract
- Optional orientation step inside `/pipeline/`
- Detection `theta` propagated through classify / orient / extract
- Support for negative thetas
- `data:` URI input across all endpoints, with strict validation

#### Stability and operations

- Docker hardening, healthcheck, autoheal, IP whitelist support
- Configurable per-environment via `docker/.env` and `app/model_config.json`
- MiewID preprocessing switched to albumentations for embedding parity with
  training (PR #25)
- Canonical `get_chip_from_img` used for MiewID embedding extraction, with
  guards for zero-size and out-of-bounds bboxes
- PAIR-X fixes: removed duplicate channel swap that inverted red/blue, chip used
  as display image when a bbox is supplied, theta-only requests handled,
  visualization sized to the bbox
- VRAM-growth fix after PAIR-X explain
- Stress test suite

### Breaking changes vs. pre-1.0 callers

- `/pipeline/` and `/extract/` response shapes were aligned with the Wildbook v2
  contract. Pre-1.0 callers consuming the older shape need to update.
- `/explain/` standardized on the PAIR-X "colors only" payload, then extended to
  `lines_and_colors` / `only_lines` / `only_colors` visualization modes.

### Requirements

- Python 3.10+ (3.12 recommended)
- NVIDIA GPU with CUDA 12.1+ drivers for GPU inference
- Docker + Docker Compose v2 for the recommended deployment

See [README.md](README.md) for configuration, deployment, and the full API
reference.
