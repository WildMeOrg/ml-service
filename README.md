# FastAPI YOLO Inference

This project serves YOLO model inference results using FastAPI.

## Usage

Once the application is running, you can make a POST request to the `/predict` endpoint with an image file to get predictions.

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" -F "url=https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR73V4nXNazAtkt4peyPOjzhJsFRXBtLVQHDRiYS0r3TA&s&ec=72940544"
```

### Example Response

The response will be in JSON format with prediction results:

```json
{
  "bboxes": [
    [68.0103, 134.594, 71.4746, 130.7195],
    [118.8744, 93.5619, 177.9465, 115.3508],
    [204.6536, 30.8491, 72.5705, 36.8663]
  ],
  "scores": [0.9054, 0.9014, 0.8432],
  "thetas": [0.0, 0.0, 0.0],
  "class_names": ["dog", "bicycle", "car"],
  "class_ids": [16, 1, 2]
}
```

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To run the FastAPI application with the updated module, use the following command:

```bash
python -m app.main --host 0.0.0.0 --port 8000 --reload --workers 4 --device mps
```

4. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```
