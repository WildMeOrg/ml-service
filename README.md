# FastAPI Model Serving

This project provides a flexible and extensible framework for serving computer vision models using FastAPI. It supports multiple model types including YOLO variants and MegaDetector, providing a unified interface for object detection.

## Features

- **Multiple Model Support**: Supports YOLO variants and MegaDetector models
- **Unified API**: Consistent interface for all models
- **Flexible Configuration**: Configure models via JSON config with support for remote model files
- **Concurrent Processing**: Handle multiple prediction requests efficiently
- **Detailed Model Information**: Get information about loaded models
- **Rotation Support**: Handle oriented bounding boxes for specialized detection tasks

## Supported Models

1. **YOLO Ultralytics** (`yolo-ultralytics`):

2. **MegaDetector** (`megadetector`):

## Model Architecture

The application uses a plugin-like architecture for model handlers:

1. **BaseModel**: Abstract base class that all model handlers must implement
2. **ModelHandler**: Manages multiple model instances and routes requests to the appropriate handler
3. **Model Handlers**: Implementations for specific model types (YOLOUltralyticsModel, MegaDetectorModel)

## Configuration

Models are configured in `app/model_config.json`. The configuration includes pre-configured models for various use cases.

### Configuration Options

#### Common Parameters
- `model_id`: Unique identifier for the model
- `model_type`: Type of the model (`yolo-ultralytics` or `megadetector`)
- `model_path`: URL or local path to the model file
- `conf`: Default confidence threshold (0.0 to 1.0)

#### YOLO-specific Parameters
- `imgsz`: Input image size (e.g., 640)
- `dilation_factors`: Optional list of [x, y] dilation factors for bounding boxes

#### MegaDetector-specific Parameters
- `version`: Model version identifier

### Example Configuration

```json
{
    "models": [
        {
            "model_id": "msv3",
            "model_type": "yolo-ultralytics",
            "model_path": "https://example.com/models/detect.yolov11.msv3.pt",
            "imgsz": 640,
            "conf": 0.5
        },
        {
            "model_id": "mdv6",
            "model_type": "megadetector",
            "model_path": "https://example.com/models/mdv6-yolov9-c.pt",
            "version": "MDV6-yolov9-c",
            "conf": 0.5
        }
    ]
}
```

## API Endpoints

### Run Prediction

```
POST /predict
```

**Request Body**:
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
    "bboxes": [
        [68.0103, 134.594, 71.4746, 130.7195],
        [118.8744, 93.5619, 177.9465, 115.3508]
    ],
    "scores": [0.9054, 0.9014],
    "thetas": [0.0, 0.0],
    "class_names": ["dog", "cat"],
    "class_ids": [16, 15]
}
```

### Response Format

- `bboxes`: List of bounding boxes in [x, y, w, h] format (top-left corner, width, height)
- `thetas`: List of rotation angles in radians (0.0 for axis-aligned boxes)
- `scores`: List of confidence scores (0.0 to 1.0)
- `class_names`: List of class names
- `class_ids`: List of class IDs

## Setup

### Option 1: Local Development

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Docker

1. Navigate to the docker directory:
   ```bash
   cd docker
   ```

2. Start the container:
   ```bash
   docker-compose up
   ```
   
   This will build and start a Docker container with the FastAPI server, exposing it on port 8000.

## Running the Application

Start the FastAPI server with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

For production use with multiple workers:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Variables

- `DEVICE`: Set to `cuda`, `mps`, or `cpu` (default: `cpu`)
- `LOG_LEVEL`: Logging level (default: `info`)
- `MODEL_CONFIG_PATH`: Path to model configuration (default: `app/model_config.json`)
