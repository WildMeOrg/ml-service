from io import BytesIO
from PIL import Image
from app.schemas.model_response import ModelResponse  # Import your response schema

def run_inference(image_bytes: bytes, model, device: str, imgsz: int, conf: float) -> ModelResponse:
    """Run object detection inference on an image.
    
    This function processes an image and runs inference using the provided YOLO model.
    It supports both oriented bounding boxes (OBB) and standard bounding boxes.
    
    Args:
        image_bytes (bytes): The input image as bytes.
        model: The loaded YOLO model instance.
        device (str): The device to run inference on ('cpu', 'cuda', etc.).
        imgsz (int): The size to which the image will be resized before inference.
        conf (float): Confidence threshold for detections (0-1).
        
    Returns:
        ModelResponse: A Pydantic model containing the detection results including:
            - bboxes: List of bounding boxes in [x, y, w, h] format
            - scores: List of confidence scores for each detection
            - thetas: List of rotation angles for each detection (for OBB)
            - class_ids: List of class IDs for each detection
            - class_names: List of class names for each detection
            
    Note:
        For standard bounding boxes, thetas will be a list of zeros.
        For oriented bounding boxes (OBB), they will contain the rotation angles.
    """
    img = Image.open(BytesIO(image_bytes))
    results = model.predict(img, save=False, imgsz=imgsz, conf=conf, device=device, verbose=False)
    results = results[0]
    names = model.names  # Get class names from the model

    if hasattr(results, 'obb') and results.obb is not None:
        # OBB mode
        xywhr = results.obb.xywhr.cpu().numpy()
        bboxes = [[x - w / 2, y - h / 2, w, h] for x, y, w, h, r in xywhr]
        thetas = [float(r) for x, y, w, h, r in xywhr]
        scores = results.obb.conf.tolist()
        class_ids = [int(cls) for cls in results.obb.cls.tolist()]
        class_names = [names[class_id] for class_id in class_ids]

    elif hasattr(results, 'boxes') and results.boxes is not None:
        # Axis-aligned box mode
        xywh = results.boxes.xywh.cpu().numpy()
        bboxes = [[x - w / 2, y - h / 2, w, h] for x, y, w, h in xywh]
        thetas = [0.0] * len(bboxes)
        scores = results.boxes.conf.tolist()
        class_ids = [int(cls) for cls in results.boxes.cls.tolist()]
        class_names = [names[class_id] for class_id in class_ids]

    else:
        return ModelResponse([], [], [], [], [])

    return ModelResponse(
        bboxes=bboxes,
        scores=scores,
        thetas=thetas,
        class_ids=class_ids,
        class_names=class_names
    )
