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

### Run Explain

```
POST /explain
```

**Request Body**:
```json
{
  "image1_uris": ["Images/img1.png", "Images/img1.png"],
  "bb1": [
    [100, 100, 300, 200], [100, 100, 300, 200]
  ],
  "theta1": [10.3, 10.3]
  "image2_uris": ["Images/img1.png", "Images/img2.png"],
  "bb2": [
    [0, 0, 0, 0]
  ],
  "theta2": [0.0]
  "model_id": "miewid-msv3",
  "crop_bbox": false,
  "visualization_type": "lines_and_colors",
  "layer_key": "backbone.blocks.3",
  "k_lines": 20,
  "k_colors": 5,
  "algorithm": "pairx"
}
```

**Considerations:**

image_uris can be either urls or file paths. If only one image1_uri is specified, it will be run with every image2_uri.
If there are more uris specified than bounding boxes (bb1 or bb2), the remaining images will be uncropped. 

A bounding box of [0, 0, 0, 0] will not crop the image. Thetas function the same way if not enough are supplied.
The first bounding box coordinate is the number of pixels to be cropped from the left side of the image. The second is 
the number to be cropped from the top of the image. The third is the width of the new image and the fourth is the height
of the new image. 

Currently the only supported model_id is miewid-msv3 and the only supported algorithm is pairx. These are also the default
if not specified.

If crop_bbox is true, the final visualization will include only the image cropped based on the bounding box. 
If it is false, the full image will be used. However, regardless of the value of crop_bbox only the cropped image will be used
in matching.

A visualization_type of lines_and_colors will return the whole visualization, only_lines will return only the two images with
lines matching similar points, and only_colors will return only a heatmap of relevant points.

layer_key determines how focused or general the generated visualization will be. Layer keys that are earlier in the model
(e.g. backbone.blocks.1) will focus on very specific points while later ones (e.g. backbone.blocks.5) will generate visualizations
that focus on broad areas of similarity. Using the default is generally recommended.

Larger values of k_lines often leads to erroneous matches (e.g. matching an animal's ear with another animal's paw), but 
it does not have a significant impact on the time to generate visualizations. 
The higher k_colors is, the longer it will take to generate visualizations. 

### Response format

Returns a list of images. Each image will be a numpy array.

## Image Classification

The service provides a dedicated endpoint for image classification using EfficientNet models. This endpoint classifies images into predefined categories with configurable confidence thresholds.

### Classify Image

```
POST /classify/
```

**Request Body**:
```json
{
    "model_id": "efficientnet-classifier",
    "image_uri": "https://example.com/image.jpg",
    "bbox": [0, 0, 1000, 1000],
    "theta": 0.0
}
```

**Parameters**:
- `model_id` (required): ID of the EfficientNet model to use for classification
- `image_uri` (required): URI of the image to process (URL or local file path)
- `bbox` (optional): Bounding box coordinates `[x, y, width, height]` to crop the image before classification. If not provided, uses the full image
- `theta` (optional): Rotation angle in radians (default: 0.0)

**Response**:
```json
{
    "model_id": "efficientnet-classifier",
    "predictions": [
        {
            "index": 0,
            "label": "back",
            "probability": 0.8111463785171509
        }
    ],
    "all_probabilities": [
        0.8111463785171509,
        5.336962090041197e-07,
        0.00027203475474379957,
        0.0071563138626515865,
        0.457361102104187,
        2.727882019826211e-05
    ],
    "threshold": 0.5,
    "bbox": [0, 0, 1000, 1000],
    "theta": 0.0,
    "image_uri": "https://example.com/image.jpg"
}
```

**Response Fields**:
- `model_id`: The model used for classification
- `predictions`: List of predictions above the threshold, sorted by probability (descending)
  - `index`: Class index in the model
  - `label`: Human-readable class label
  - `probability`: Confidence score (0.0 to 1.0)
- `all_probabilities`: Raw probabilities for all classes
- `threshold`: The confidence threshold used to filter predictions
- `bbox`: The bounding box used (if any)
- `theta`: The rotation angle applied
- `image_uri`: The original image URI

### Usage Examples

#### Basic Image Classification

Classify a full image:

```bash
curl -X POST "http://localhost:8000/classify/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "efficientnet-classifier",
    "image_uri": "https://example.com/image.jpg"
  }'
```

#### Classification with Bounding Box

Classify a specific region of the image:

```bash
curl -X POST "http://localhost:8000/classify/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "efficientnet-classifier",
    "image_uri": "https://example.com/image.jpg",
    "bbox": [100, 100, 300, 200]
  }'
```

#### Local File with Rotation

Classify a local file with rotation:

```bash
curl -X POST "http://localhost:8000/classify/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "efficientnet-classifier",
    "image_uri": "/path/to/local/image.jpg",
    "bbox": [50, 50, 200, 200],
    "theta": 0.785
  }'
```

#### Python Example

```python
import requests
from pprint import pprint

# Classify image
response = requests.post("http://localhost:8000/classify/", json={
    "model_id": "efficientnet-classifier",
    "image_uri": "https://example.com/image.jpg",
    "bbox": [0, 0, 1000, 1000]
})

if response.status_code == 200:
    result = response.json()
    
    print(f"Model: {result['model_id']}")
    print(f"Threshold: {result['threshold']}")
    print(f"Predictions above threshold: {len(result['predictions'])}")
    
    for pred in result['predictions']:
        print(f"  - {pred['label']}: {pred['probability']:.4f}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Model Configuration

EfficientNet models are configured in `model_config.json` with the following parameters:

```json
{
    "model_id": "efficientnet-classifier",
    "model_type": "efficientnetv2",
    "checkpoint_path": "/path/to/checkpoint.pt",
    "img_size": 512,
    "threshold": 0.5
}
```

**Configuration Parameters**:
- `checkpoint_path` (required): Path or URL to the model checkpoint
- `img_size`: Input image size for preprocessing (default: 512)
- `threshold`: Classification confidence threshold (default: 0.5)

### Supported Classes

The current model supports the following classes:
- `back` (index: 0)
- `down` (index: 1) 
- `front` (index: 2)
- `left` (index: 3)
- `right` (index: 4)
- `up` (index: 5)

### Error Handling

The endpoint returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid bbox format, file not found, non-EfficientNet model)
- `404`: Model not found
- `500`: Internal server error

**Common Error Responses**:

```json
{
    "detail": {
        "error": "Model 'invalid_model' not found.",
        "available_models": ["efficientnet-classifier", "miewid_v3"]
    }
}
```

```json
{
    "detail": "Bounding box must contain exactly 4 values: [x, y, width, height]"
}
```

```json
{
    "detail": "Model 'miewid_v3' is not an EfficientNet model. Only EfficientNet models support classification."
}
```

### Performance Considerations

- The service limits concurrent classifications to prevent out-of-memory errors
- Large images or high resolution inputs may take longer to process
- Consider using appropriate bounding boxes to focus on regions of interest
- URL-based images are downloaded and cached temporarily during processing

## Pipeline Processing

The service provides a comprehensive pipeline endpoint that orchestrates multiple machine learning operations in sequence. The pipeline runs prediction (bbox detection), then performs classification and embeddings extraction on each detected region above a specified confidence threshold.

### Run Pipeline

```
POST /pipeline/
```

**Request Body**:
```json
{
    "predict_model_id": "msv3",
    "classify_model_id": "efficientnet-classifier",
    "extract_model_id": "miewid_v3",
    "image_uri": "https://example.com/image.jpg",
    "bbox_score_threshold": 0.5,
    "predict_model_params": {
        "conf": 0.6,
        "imgsz": 640
    }
}
```

**Parameters**:
- `predict_model_id` (required): ID of the model to use for object detection/prediction
- `classify_model_id` (required): ID of the EfficientNet model to use for classification
- `extract_model_id` (required): ID of the MiewID model to use for embeddings extraction
- `image_uri` (required): URI of the image to process (URL or local file path)
- `bbox_score_threshold` (optional): Minimum bbox confidence score threshold to process (default: 0.5)
- `predict_model_params` (optional): Parameters to override prediction model configuration

**Response**:
```json
{
    "image_uri": "https://example.com/image.jpg",
    "models_used": {
        "predict_model_id": "msv3",
        "classify_model_id": "efficientnet-classifier", 
        "extract_model_id": "miewid_v3"
    },
    "bbox_score_threshold": 0.5,
    "total_predictions": 15,
    "filtered_predictions": 3,
    "pipeline_results": [
        {
            "bbox": [68.0103, 134.594, 71.4746, 130.7195],
            "bbox_score": 0.9054,
            "detection_class": "dog",
            "detection_class_id": 16,
            "classification": {
                "class": "back",
                "probability": 0.8111,
                "class_id": 0
            },
            "embedding": [0.1234, -0.5678, 0.9012, ...],
            "embedding_shape": [1, 512]
        }
    ],
    "original_predict": {...},
    "original_classify": [...],
    "original_extract": [...]
}
```

**Response Fields**:
- `image_uri`: The original image URI processed
- `models_used`: Dictionary containing the IDs of all models used in the pipeline
- `bbox_score_threshold`: The confidence threshold used to filter detections
- `total_predictions`: Total number of bboxes detected by the prediction model
- `filtered_predictions`: Number of bboxes above the score threshold that were processed
- `pipeline_results`: Array of results for each processed bbox containing:
  - `bbox`: Bounding box coordinates `[x, y, width, height]`
  - `bbox_score`: Confidence score from the detection model
  - `detection_class`: Class name from the detection model
  - `detection_class_id`: Class ID from the detection model
  - `classification`: Top classification result with class, probability, and class_id
  - `embedding`: Extracted embeddings as a flat array
  - `embedding_shape`: Shape of the embeddings array
- `original_predict`: Full prediction results from the detection model
- `original_classify`: Array of full classification results for each bbox
- `original_extract`: Array of full extraction results for each bbox

### Pipeline Workflow

The pipeline executes the following steps:

1. **Validation**: Validates that all specified models exist and are of the correct types
   - Prediction model: Any supported detection model (YOLO, MegaDetector)
   - Classification model: Must be an EfficientNet model
   - Extraction model: Must be a MiewID model

2. **Image Loading**: Downloads the image (if URL) or loads from local path

3. **Object Detection**: Runs the prediction model to detect bounding boxes

4. **Filtering**: Filters detected bboxes by the specified confidence threshold

5. **Parallel Processing**: For each filtered bbox, runs classification and extraction in parallel

6. **Result Aggregation**: Combines all results into a structured response

### Usage Examples

#### Basic Pipeline Processing

Process an image with default threshold:

```bash
curl -X POST "http://localhost:8000/pipeline/" \
  -H "Content-Type: application/json" \
  -d '{
    "predict_model_id": "msv3",
    "classify_model_id": "efficientnet-classifier",
    "extract_model_id": "miewid_v3",
    "image_uri": "https://example.com/image.jpg"
  }'
```

#### Custom Threshold and Parameters

Use a higher confidence threshold and custom prediction parameters:

```bash
curl -X POST "http://localhost:8000/pipeline/" \
  -H "Content-Type: application/json" \
  -d '{
    "predict_model_id": "msv3",
    "classify_model_id": "efficientnet-classifier",
    "extract_model_id": "miewid_v3",
    "image_uri": "https://example.com/image.jpg",
    "bbox_score_threshold": 0.7,
    "predict_model_params": {
      "conf": 0.8,
      "imgsz": 1024
    }
  }'
```

#### Local File Processing

Process a local image file:

```bash
curl -X POST "http://localhost:8000/pipeline/" \
  -H "Content-Type: application/json" \
  -d '{
    "predict_model_id": "mdv6",
    "classify_model_id": "efficientnet-classifier",
    "extract_model_id": "miewid_v3",
    "image_uri": "/path/to/local/image.jpg",
    "bbox_score_threshold": 0.6
  }'
```

#### Python Example

```python
import requests
import numpy as np
from pprint import pprint

# Run pipeline
response = requests.post("http://localhost:8000/pipeline/", json={
    "predict_model_id": "msv3",
    "classify_model_id": "efficientnet-classifier", 
    "extract_model_id": "miewid_v3",
    "image_uri": "https://example.com/image.jpg",
    "bbox_score_threshold": 0.5
})

if response.status_code == 200:
    result = response.json()
    
    print(f"Processed {result['total_predictions']} detections")
    print(f"Found {result['filtered_predictions']} above threshold")
    print(f"Pipeline results: {len(result['pipeline_results'])}")
    
    for i, bbox_result in enumerate(result['pipeline_results']):
        print(f"\nBbox {i+1}:")
        print(f"  Detection: {bbox_result['detection_class']} ({bbox_result['bbox_score']:.3f})")
        if bbox_result['classification']:
            print(f"  Classification: {bbox_result['classification']['class']} ({bbox_result['classification']['probability']:.3f})")
        print(f"  Embedding shape: {bbox_result['embedding_shape']}")
        
        # Convert embedding back to numpy array
        embedding = np.array(bbox_result['embedding'])
        embedding = embedding.reshape(bbox_result['embedding_shape'])
        print(f"  First 5 embedding values: {embedding.flatten()[:5]}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Error Handling

The endpoint returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid parameters, file not found, wrong model types)
- `404`: Model not found
- `500`: Internal server error

**Common Error Responses**:

```json
{
    "detail": {
        "error": "Prediction model 'invalid_model' not found.",
        "available_models": ["msv3", "mdv6", "efficientnet-classifier", "miewid_v3"]
    }
}
```

```json
{
    "detail": "Model 'msv3' is not an EfficientNet model. Only EfficientNet models support classification."
}
```

```json
{
    "detail": "File not found: /path/to/nonexistent/image.jpg"
}
```

### Performance Considerations

- **Concurrency**: Pipeline operations are limited to prevent out-of-memory errors
- **Parallel Processing**: Classification and extraction run in parallel for each bbox to optimize performance
- **Memory Usage**: Large images or many detections may require more memory
- **Threshold Tuning**: Higher bbox_score_threshold values reduce processing time by filtering out low-confidence detections
- **Model Selection**: Choose appropriate models based on your use case and available computational resources

### Model Requirements

- **Prediction Model**: Any supported detection model (YOLO variants, MegaDetector)
- **Classification Model**: Must be configured as `efficientnetv2` type in model_config.json
- **Extraction Model**: Must be configured as `miewid` type in model_config.json

## Embeddings Extraction

The service provides a dedicated endpoint for extracting embeddings from images using MiewID models. This is useful for feature extraction, similarity matching, and other machine learning tasks.

### Extract Embeddings

```
POST /extract/
```

**Request Body**:
```json
{
    "model_id": "miewid_v3",
    "image_uri": "https://example.com/image.jpg",
    "bbox": [50, 50, 200, 200],
    "theta": 0.0
}
```

**Parameters**:
- `model_id` (required): ID of the MiewID model to use for extraction
- `image_uri` (required): URI of the image to process (URL or local file path)
- `bbox` (optional): Bounding box coordinates `[x, y, width, height]` to crop the image before extraction. If not provided, uses the full image
- `theta` (optional): Rotation angle in radians (default: 0.0)

**Response**:
```json
{
    "model_id": "miewid_v3",
    "embeddings": [0.1234, -0.5678, 0.9012, ...],
    "embeddings_shape": [1, 512],
    "bbox": [50, 50, 200, 200],
    "theta": 0.0,
    "image_uri": "https://example.com/image.jpg"
}
```

**Response Fields**:
- `model_id`: The model used for extraction
- `embeddings`: The extracted feature embeddings as a flat list
- `embeddings_shape`: Shape of the embeddings array (e.g., `[1, 512]` for a 512-dimensional feature vector)
- `bbox`: The bounding box used (if any)
- `theta`: The rotation angle applied
- `image_uri`: The original image URI

### Usage Examples

#### Basic Embeddings Extraction

Extract embeddings from a full image:

```bash
curl -X POST "http://localhost:8000/extract/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "miewid_v3",
    "image_uri": "https://example.com/image.jpg"
  }'
```

#### Cropped Region Extraction

Extract embeddings from a specific region of the image:

```bash
curl -X POST "http://localhost:8000/extract/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "miewid_v3",
    "image_uri": "https://example.com/image.jpg",
    "bbox": [100, 100, 300, 200]
  }'
```

#### Local File with Rotation

Extract embeddings from a local file with rotation:

```bash
curl -X POST "http://localhost:8000/extract/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "miewid_v3",
    "image_uri": "/path/to/local/image.jpg",
    "bbox": [50, 50, 200, 200],
    "theta": 0.785
  }'
```

#### Python Example

```python
import requests
import numpy as np

# Extract embeddings
response = requests.post("http://localhost:8000/extract/", json={
    "model_id": "miewid_v3",
    "image_uri": "https://example.com/image.jpg",
    "bbox": [100, 100, 300, 200]
})

if response.status_code == 200:
    result = response.json()
    
    # Convert embeddings back to numpy array
    embeddings = np.array(result["embeddings"])
    embeddings = embeddings.reshape(result["embeddings_shape"])
    
    print(f"Extracted {embeddings.shape} embeddings")
    print(f"First 5 values: {embeddings.flatten()[:5]}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Error Handling

The endpoint returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid bbox format, file not found, non-MiewID model)
- `404`: Model not found
- `500`: Internal server error

**Common Error Responses**:

```json
{
    "detail": {
        "error": "Model 'invalid_model' not found.",
        "available_models": ["miewid_v3", "msv3", "mdv6"]
    }
}
```

```json
{
    "detail": "Bounding box must contain exactly 4 values: [x, y, width, height]"
}
```

```json
{
    "detail": "Model 'msv3' is not a MiewID model. Only MiewID models support embeddings extraction."
}
```

### Performance Considerations

- The service limits concurrent extractions to prevent out-of-memory errors
- Large images or complex models may take longer to process
- Consider using appropriate bounding boxes to focus on regions of interest
- URL-based images are downloaded and cached temporarily during processing

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

## Image Classification

The service provides a dedicated endpoint for image classification using EfficientNet models. This endpoint classifies images into predefined categories with configurable confidence thresholds.

### Classify Image

```
POST /classify/
```

**Request Body**:
```json
{
    "model_id": "efficientnet-classifier",
    "image_uri": "https://example.com/image.jpg",
    "bbox": [0, 0, 1000, 1000],
    "theta": 0.0
}
```

**Parameters**:
- `model_id` (required): ID of the EfficientNet model to use for classification
- `image_uri` (required): URI of the image to process (URL or local file path)
- `bbox` (optional): Bounding box coordinates `[x, y, width, height]` to crop the image before classification. If not provided, uses the full image
- `theta` (optional): Rotation angle in radians (default: 0.0)

**Response**:
```json
{
    "model_id": "efficientnet-classifier",
    "predictions": [
        {
            "index": 0,
            "label": "back",
            "probability": 0.8111463785171509
        }
    ],
    "all_probabilities": [
        0.8111463785171509,
        5.336962090041197e-07,
        0.00027203475474379957,
        0.0071563138626515865,
        0.457361102104187,
        2.727882019826211e-05
    ],
    "threshold": 0.5,
    "bbox": [0, 0, 1000, 1000],
    "theta": 0.0,
    "image_uri": "https://example.com/image.jpg"
}
```

**Response Fields**:
- `model_id`: The model used for classification
- `predictions`: List of predictions above the threshold, sorted by probability (descending)
  - `index`: Class index in the model
  - `label`: Human-readable class label
  - `probability`: Confidence score (0.0 to 1.0)
- `all_probabilities`: Raw probabilities for all classes
- `threshold`: The confidence threshold used to filter predictions
- `bbox`: The bounding box used (if any)
- `theta`: The rotation angle applied
- `image_uri`: The original image URI

### Usage Examples

#### Basic Image Classification

Classify a full image:

```bash
curl -X POST "http://localhost:8000/classify/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "efficientnet-classifier",
    "image_uri": "https://example.com/image.jpg"
  }'
```

#### Classification with Bounding Box

Classify a specific region of the image:

```bash
curl -X POST "http://localhost:8000/classify/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "efficientnet-classifier",
    "image_uri": "https://example.com/image.jpg",
    "bbox": [100, 100, 300, 200]
  }'
```

#### Local File with Rotation

Classify a local file with rotation:

```bash
curl -X POST "http://localhost:8000/classify/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "efficientnet-classifier",
    "image_uri": "/path/to/local/image.jpg",
    "bbox": [50, 50, 200, 200],
    "theta": 0.785
  }'
```

#### Python Example

```python
import requests
from pprint import pprint

# Classify image
response = requests.post("http://localhost:8000/classify/", json={
    "model_id": "efficientnet-classifier",
    "image_uri": "https://example.com/image.jpg",
    "bbox": [0, 0, 1000, 1000]
})

if response.status_code == 200:
    result = response.json()
    
    print(f"Model: {result['model_id']}")
    print(f"Threshold: {result['threshold']}")
    print(f"Predictions above threshold: {len(result['predictions'])}")
    
    for pred in result['predictions']:
        print(f"  - {pred['label']}: {pred['probability']:.4f}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Model Configuration

EfficientNet models are configured in `model_config.json` with the following parameters:

```json
{
    "model_id": "efficientnet-classifier",
    "model_type": "efficientnetv2",
    "checkpoint_path": "/path/to/checkpoint.pt",
    "img_size": 512,
    "threshold": 0.5
}
```

**Configuration Parameters**:
- `checkpoint_path` (required): Path or URL to the model checkpoint
- `img_size`: Input image size for preprocessing (default: 512)
- `threshold`: Classification confidence threshold (default: 0.5)

### Supported Classes

The current model supports the following classes:
- `back` (index: 0)
- `down` (index: 1) 
- `front` (index: 2)
- `left` (index: 3)
- `right` (index: 4)
- `up` (index: 5)

### Error Handling

The endpoint returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid bbox format, file not found, non-EfficientNet model)
- `404`: Model not found
- `500`: Internal server error

**Common Error Responses**:

```json
{
    "detail": {
        "error": "Model 'invalid_model' not found.",
        "available_models": ["efficientnet-classifier", "miewid_v3"]
    }
}
```

```json
{
    "detail": "Bounding box must contain exactly 4 values: [x, y, width, height]"
}
```

```json
{
    "detail": "Model 'miewid_v3' is not an EfficientNet model. Only EfficientNet models support classification."
}
```

### Performance Considerations

- The service limits concurrent classifications to prevent out-of-memory errors
- Large images or high resolution inputs may take longer to process
- Consider using appropriate bounding boxes to focus on regions of interest
- URL-based images are downloaded and cached temporarily during processing
