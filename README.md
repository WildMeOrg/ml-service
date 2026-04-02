# Wildbook ML Service

A flexible FastAPI service for serving computer vision models used by the [Wildbook](https://wildbook.org/) platform. Supports detection, classification, orientation estimation, embedding extraction, explainability, and part-body assignment across multiple model architectures.

## Supported Model Types

| Type | Architecture | Use Case |
|------|-------------|----------|
| `yolo-ultralytics` | YOLOv11 (Ultralytics) | Object detection |
| `megadetector` | MegaDetector (PytorchWildlife) | Animal/person/vehicle detection |
| `lightnet` | PyDarknet YOLO v2/v3 | Species-specific detection (WBIA legacy models) |
| `efficientnetv2` | EfficientNet-B4 (timm) | Species/viewpoint classification |
| `densenet-orientation` | DenseNet-201 (torchvision) | Orientation classification |
| `miewid` | MiewID transformer | Embedding extraction for re-identification |

## API Endpoints

### Detection

```
POST /predict/
```

Runs object detection and returns bounding boxes. Works with any detection model type (YOLO, MegaDetector, LightNet).

**Request**:
```json
{
    "model_id": "msv3",
    "image_uri": "https://example.com/image.jpg",
    "model_params": {
        "conf": 0.6,
        "imgsz": 640
    }
}
```

**Response**:
```json
{
    "bboxes": [[68.0, 134.6, 71.5, 130.7]],
    "scores": [0.9054],
    "thetas": [0.0],
    "class_names": ["dog"],
    "class_ids": [16]
}
```

- `bboxes`: `[x, y, width, height]` in pixels (top-left origin)
- `thetas`: Rotation angle in radians (0.0 for axis-aligned boxes)
- `scores`: Confidence scores (0.0 to 1.0)

### Classification

```
POST /classify/
```

Runs image classification. Works with EfficientNet and DenseNet orientation models.

**Request**:
```json
{
    "model_id": "efficientnet-classifier",
    "image_uri": "https://example.com/image.jpg",
    "bbox": [100, 100, 300, 200],
    "theta": 0.0
}
```

- `bbox` (optional): Crop region `[x, y, width, height]` before classifying
- `theta` (optional): Rotation angle in radians

**Response**:
```json
{
    "model_id": "efficientnet-classifier",
    "predictions": [
        {"index": 0, "label": "back", "probability": 0.811}
    ],
    "all_probabilities": [0.811, 0.0, 0.0003, 0.007, 0.457, 0.00003],
    "threshold": 0.5,
    "bbox": [100, 100, 300, 200],
    "theta": 0.0
}
```

When `parse_compound_labels` is enabled in the model config, predictions include parsed species and viewpoint:

```json
{
    "predictions": [
        {
            "label": "chelonia_mydas:left",
            "species": "chelonia_mydas",
            "viewpoint": "left",
            "probability": 0.92
        }
    ]
}
```

### Embedding Extraction

```
POST /extract/
```

Extracts feature embeddings for re-identification using MiewID models.

**Request**:
```json
{
    "model_id": "miewid-msv4.1",
    "image_uri": "https://example.com/image.jpg",
    "bbox": [50, 50, 200, 200],
    "theta": 0.0
}
```

**Response**:
```json
{
    "model_id": "miewid-msv4.1",
    "embeddings": [0.1234, -0.5678, 0.9012],
    "embeddings_shape": [1, 512],
    "bbox": [50, 50, 200, 200],
    "theta": 0.0
}
```

### Explainability

```
POST /explain/
```

Generates visual explanations of what features two images share, using PAIR-X.

**Request**:
```json
{
    "image1_uris": ["image_a.jpg"],
    "bb1": [[100, 100, 300, 200]],
    "theta1": [0.0],
    "image2_uris": ["image_b.jpg"],
    "bb2": [[0, 0, 0, 0]],
    "theta2": [0.0],
    "model_id": "miewid-msv3",
    "algorithm": "pairx",
    "visualization_type": "lines_and_colors",
    "layer_key": "backbone.blocks.3",
    "k_lines": 20,
    "k_colors": 5,
    "crop_bbox": false
}
```

- `bb` of `[0,0,0,0]` means no crop (use full image)
- `visualization_type`: `lines_and_colors`, `only_lines`, or `only_colors`
- `layer_key`: Earlier layers (e.g. `backbone.blocks.1`) focus on specific points; later layers focus on broad areas
- Returns a list of numpy arrays (images)

### Pipeline (Detect + Classify + Extract)

```
POST /pipeline/
```

Runs detection, then classifies and extracts embeddings for each detected region above a confidence threshold.

**Request**:
```json
{
    "predict_model_id": "msv3",
    "classify_model_id": "efficientnet-classifier",
    "extract_model_id": "miewid-msv4.1",
    "image_uri": "https://example.com/image.jpg",
    "bbox_score_threshold": 0.5,
    "predict_model_params": {"conf": 0.6}
}
```

**Response**:
```json
{
    "image_uri": "https://example.com/image.jpg",
    "models_used": {
        "predict_model_id": "msv3",
        "classify_model_id": "efficientnet-classifier",
        "extract_model_id": "miewid-msv4.1"
    },
    "total_predictions": 15,
    "filtered_predictions": 3,
    "pipeline_results": [
        {
            "bbox": [68.0, 134.6, 71.5, 130.7],
            "bbox_score": 0.9054,
            "detection_class": "dog",
            "detection_class_id": 16,
            "classification": {
                "class": "back",
                "probability": 0.811,
                "class_id": 0
            },
            "embedding": [0.1234, -0.5678],
            "embedding_shape": [1, 512]
        }
    ]
}
```

### Part-Body Assignment

```
POST /assign/
```

Matches "part" annotations (e.g. `lion+head`) to "body" annotations using geometric features and species-specific scikit-learn classifiers.

**Request**:
```json
{
    "species": "lion",
    "annotations": [
        {"aid": 1, "bbox": [100, 50, 200, 300], "theta": 0.0, "viewpoint": "left", "is_part": false},
        {"aid": 2, "bbox": [120, 60, 80, 80], "theta": 0.0, "viewpoint": "left", "is_part": true}
    ],
    "image_width": 1024,
    "image_height": 768,
    "cutoff_score": 0.5
}
```

**Response**:
```json
{
    "assigned_pairs": [
        {"part_aid": 2, "body_aid": 1, "score": 0.87}
    ],
    "unassigned_aids": []
}
```

The assignment algorithm computes geometric features (IoU, distances, containment, aspect ratios, viewpoint matches) for every (part, body) pair, scores them with the species classifier, then greedily assigns highest-scoring pairs.

Supported species include wild dog (default fallback), lion, zebra (Grevy's, plains), sea turtles, hyena, and others.

### Health Check

```
GET /health
```

Returns service health including GPU status, CUDA availability, and loaded model count.

---

## WBIA Compatibility Layer

For backward compatibility with [Wildbook](https://github.com/WildMeOrg/Wildbook), ml-service provides endpoints that mimic WBIA's async job queue pattern. Wildbook can point to ml-service as a drop-in replacement for WBIA's detection/labeling pipeline without changing its HTTP client code.

### Flow

1. **Submit job**: `POST /api/engine/detect/cnn/` returns a `jobid`
2. **Poll status**: `GET /api/engine/job/status/?jobid=X` returns `{"jobstatus": "completed"}`
3. **Fetch result**: `GET /api/engine/job/result/?jobid=X` returns detection results

All responses are wrapped in WBIA's standard envelope:
```json
{
    "status": {"success": true, "code": "", "message": "", "cache": -1},
    "response": "<data>"
}
```

### Submit Detection Job

```
POST /api/engine/detect/cnn/
POST /api/engine/detect/cnn/yolo/
POST /api/engine/detect/cnn/lightnet/
```

**Request**:
```json
{
    "image_uuid_list": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
    "model_tag": "detect-hyaena",
    "labeler_model_tag": "labeler-hyaena",
    "use_labeler_species": true,
    "sensitivity": 0.3,
    "nms_thresh": 0.4,
    "assigner_algo": null,
    "callback_url": null
}
```

| Parameter | Description |
|-----------|-------------|
| `image_uuid_list` | List of image file paths or URLs |
| `model_tag` | Detection model ID (must be loaded in model config) |
| `labeler_model_tag` | Optional classification model for viewpoint/species labeling |
| `viewpoint_model_tag` | Alias for `labeler_model_tag` |
| `use_labeler_species` | If true, override detection species with labeler's species prediction |
| `sensitivity` | Minimum detection confidence threshold |
| `nms_thresh` | NMS threshold |
| `assigner_algo` | If set, run part-body assignment after detection |
| `callback_url` | Accepted but not used (Wildbook polls instead) |

**Response** (immediate): job ID string wrapped in WBIA envelope.

### Poll Job Status

```
GET /api/engine/job/status/?jobid=<jobid>
```

**Response**:
```json
{
    "status": {"success": true, "code": "", "message": "", "cache": -1},
    "response": {"jobstatus": "completed"}
}
```

Job statuses: `received`, `working`, `completed`, `exception`, `unknown`.

### Fetch Job Result

```
GET /api/engine/job/result/?jobid=<jobid>
```

**Response**:
```json
{
    "status": {"success": true, "code": "", "message": "", "cache": -1},
    "response": {
        "json_result": {
            "image_uuid_list": ["/path/to/image1.jpg"],
            "results_list": [
                [
                    {
                        "id": 1,
                        "uuid": "a1b2c3d4-...",
                        "xtl": 120,
                        "ytl": 45,
                        "left": 120,
                        "top": 45,
                        "width": 200,
                        "height": 150,
                        "theta": 0.0,
                        "confidence": 0.92,
                        "class": "hyaena",
                        "species": "hyaena",
                        "viewpoint": "left",
                        "quality": null,
                        "multiple": false,
                        "interest": false
                    }
                ]
            ],
            "score_list": [0.0],
            "has_assignments": false
        }
    }
}
```

Each entry in `results_list` is a list of annotation dicts for the corresponding image. Annotation fields match WBIA's format:

| Field | Description |
|-------|-------------|
| `xtl`, `ytl` / `left`, `top` | Top-left corner (pixels) |
| `width`, `height` | Bounding box dimensions |
| `theta` | Rotation angle (radians) |
| `confidence` | Detection score |
| `class`, `species` | Detected/labeled species |
| `viewpoint` | Viewpoint label (set by labeler, null if no labeler) |
| `quality`, `multiple`, `interest` | WBIA-compatible flags (defaults) |

### Pipeline Behavior

When `labeler_model_tag` is provided, the endpoint runs a combined pipeline per image:

1. **Detect** with `model_tag` -- get bounding boxes
2. **Label** with `labeler_model_tag` -- classify each detection for viewpoint (and optionally species)
3. **Assign** (if `assigner_algo` set) -- match parts to bodies

Images are loaded once and passed through all pipeline steps as bytes.

### List Jobs

```
GET /api/engine/job/
```

Returns a list of all job IDs. The job store is bounded to 10,000 entries with LRU eviction of completed jobs.

---

## Model Configuration

Models are configured in `app/model_config.json`:

```json
{
    "models": [
        {
            "model_id": "msv3",
            "model_type": "yolo-ultralytics",
            "model_path": "/path/to/detect.yolov11.msv3.pt",
            "imgsz": 640,
            "conf": 0.5
        },
        {
            "model_id": "mdv6",
            "model_type": "megadetector",
            "model_path": "/path/to/mdv6-yolov10-e.pt",
            "imgsz": 1280,
            "conf": 0.1,
            "iou": 0.45
        },
        {
            "model_id": "detect-hyaena",
            "model_type": "lightnet",
            "config_path": "/path/to/detect.lightnet.hyaena.v0.py",
            "weight_path": "/path/to/detect.lightnet.hyaena.v0.weights",
            "conf": 0.1,
            "nms_thresh": 0.4
        },
        {
            "model_id": "efficientnet-classifier",
            "model_type": "efficientnetv2",
            "checkpoint_path": "/path/to/vplabeler-msv3.pt",
            "img_size": 512,
            "threshold": 0.5
        },
        {
            "model_id": "labeler-seaturtle",
            "model_type": "efficientnetv2",
            "checkpoint_path": "/path/to/classifier.seaturtle.v0.pth",
            "img_size": 512,
            "threshold": 0.5,
            "model_arch": "tf_efficientnet_b4_ns",
            "multi_label": true,
            "parse_compound_labels": true
        },
        {
            "model_id": "orientation-seaturtle",
            "model_type": "densenet-orientation",
            "checkpoint_path": "/path/to/orientation.seaturtle.v0.pth",
            "img_size": 224
        },
        {
            "model_id": "miewid-msv4.1",
            "model_type": "miewid",
            "checkpoint_path": "/path/to/miew_id.msv4_1_main.bin",
            "imgsz": 440
        }
    ]
}
```

### Configuration by Model Type

**YOLO Ultralytics** (`yolo-ultralytics`):
- `model_path`: Path or URL to `.pt` weights
- `imgsz`: Input image size (default: 640)
- `conf`: Confidence threshold (default: 0.5)
- `dilation_factors`: Optional `[x, y]` bbox dilation

**MegaDetector** (`megadetector`):
- `model_path`: Path or URL to `.pt` weights
- `imgsz`: Input image size (default: 1280)
- `conf`: Confidence threshold (default: 0.1)
- `iou`: IoU threshold for NMS (default: 0.45)

**LightNet** (`lightnet`):
- `config_path`: Path or URL to `.py` HyperParameters config
- `weight_path`: Path or URL to `.weights` binary
- `conf`: Confidence threshold (default: 0.1)
- `nms_thresh`: NMS threshold (default: 0.4)
- `batch_size`: Batch size for multi-image inference (default: 192)

**EfficientNet** (`efficientnetv2`):
- `checkpoint_path`: Path or URL to checkpoint
- `img_size`: Input image size (default: 512)
- `threshold`: Classification threshold (default: 0.5)
- `model_arch`: timm architecture name (default: `tf_efficientnet_b4_ns`)
- `label_map`: Optional dict of `{index: "label"}` (otherwise loaded from checkpoint)
- `n_classes`: Optional explicit class count
- `multi_label`: Use sigmoid + threshold (true) or softmax + argmax (false) (default: true)
- `parse_compound_labels`: Split labels on `:` into species/viewpoint fields (default: false)

**DenseNet Orientation** (`densenet-orientation`):
- `checkpoint_path`: Path or URL to checkpoint (format: `{"state": state_dict, "classes": [...]}`)
- `img_size`: Input image size (default: 224)
- `label_map`: Optional explicit label map (otherwise loaded from checkpoint `classes` key)

**MiewID** (`miewid`):
- `checkpoint_path`: Path or URL to model binary
- `imgsz`: Input image size (default: 440)

All path parameters accept URLs, which are downloaded and cached on startup.

## Setup

### Prerequisites

- Python 3.10+ (3.12 recommended)
- NVIDIA GPU with CUDA 12.1+ drivers (for GPU inference)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for Docker GPU access)
- Docker and Docker Compose v2 (for containerized deployment)

### Docker Deployment (Recommended)

#### 1. Configure environment

```bash
cd docker
cp _env .env
```

Edit `.env` and set `MODELS_DIR` to the directory containing your model weight files:

```bash
# Required: directory with .pt, .weights, .bin, .pth model files
MODELS_DIR=/data0/models

# Optional overrides
# GPU_ID=0          # which GPU (default: 0)
# DEVICE=cuda       # cuda or cpu (default: cuda)
# WORKERS=1         # uvicorn workers (default: 1, use 1 for GPU)
# ML_SERVICE_PORT=6050  # host port (default: 6050)
# DATA_DB_DIR=/data/db  # shared image path with WBIA/Wildbook
```

#### 2. Configure models

Edit `app/model_config.json` to list the models you want to load. Paths in the config should use `/models/` (the container mount point for `MODELS_DIR`):

```json
{
    "models": [
        {
            "model_id": "msv3",
            "model_type": "yolo-ultralytics",
            "model_path": "/models/detect.yolov11.msv3.pt",
            "imgsz": 640,
            "conf": 0.5
        }
    ]
}
```

Model weights can also be URLs — they will be downloaded on first startup.

#### 3. Build and start

```bash
cd docker
docker compose up --build -d
```

The service starts on port 6050 (or `ML_SERVICE_PORT` if set). Check health:

```bash
curl http://localhost:6050/health
```

Watch logs:

```bash
docker compose logs -f ml-service
```

#### 4. Stop

```bash
docker compose down
```

#### Docker volume layout

| Container path | Source | Purpose |
|----------------|--------|---------|
| `/models/` | `MODELS_DIR` | Model weight files (read-only) |
| `/datasets/` | `MODELS_DIR` | Alias for `/models/` (backward compat with existing configs) |
| `/app/app/model_config.json` | `app/model_config.json` | Model configuration (read-only) |
| `/data/db/` | `DATA_DB_DIR` | Shared image directory with WBIA/Wildbook (optional) |

#### Running without GPU

To run on CPU (e.g. for testing), set `DEVICE=cpu` in `.env` and remove the `deploy.resources.reservations.devices` block from `docker-compose.yml`, or use:

```bash
DEVICE=cpu docker compose up --build
```

#### Connecting to Wildbook

If Wildbook and ml-service run on the same host, create a shared Docker network so Wildbook can reach ml-service by container name:

```bash
docker network create shared_net
```

Then add to `docker-compose.yml`:

```yaml
services:
  ml-service:
    networks:
      - shared_net

networks:
  shared_net:
    external: true
```

Wildbook can then call `http://ml-service:6050/api/engine/detect/cnn/`.

### Local Development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start the server:

```bash
# Development (auto-reload on code changes)
python3 -m app.main --device cuda --host 0.0.0.0 --port 6050 --reload

# Production
python3 -m app.main --device cuda --host 0.0.0.0 --port 6050 --workers 1
```

Or with uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 6050
```

### Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cuda` | PyTorch device: `cuda`, `cpu`, or `mps` |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8888` | Listen port |
| `--workers` | `1` | Uvicorn worker count (use 1 for GPU to avoid VRAM contention) |
| `--reload` | off | Auto-reload on code changes (development only) |
