from io import BytesIO
from PIL import Image
from app.schemas.model_response import ModelResponse  # Import your response schema

def run_inference(image_bytes: bytes, model, device: str, imgsz: int, conf: float) -> ModelResponse:
    """
    Run object detection inference on an image, with postprocessing to dilate bounding boxes.

    This function processes an image and runs inference using the provided YOLO model.
    It supports both oriented bounding boxes (OBB) and standard axis-aligned bounding boxes.
    After detection, each bounding box is "dilated" — its width and height are increased 
    by 15% while maintaining the same center coordinates.

    Args:
        image_bytes (bytes): The input image as bytes.
        model: The loaded YOLO model instance.
        device (str): The device to run inference on ('cpu', 'cuda', etc.).
        imgsz (int): The size to which the image will be resized before inference.
        conf (float): Confidence threshold for detections (0-1).

    Returns:
        ModelResponse: A Pydantic model containing the detection results including:
            - bboxes: List of dilated bounding boxes in [x_min, y_min, w, h] format
            - scores: List of confidence scores for each detection
            - thetas: List of rotation angles for each detection (0.0 for axis-aligned)
            - class_ids: List of class IDs for each detection
            - class_names: List of class names for each detection

    Note:
        - For OBB results, thetas contain the predicted rotation angles.
        - For standard bounding boxes, thetas are all 0.0.
        - Bounding box dilation increases size by 15% in both width and height.
    """
    img = Image.open(BytesIO(image_bytes))
    results = model.predict(img, save=False, imgsz=imgsz, conf=conf, device=device, verbose=False)
    results = results[0]
    names = model.names

    dilation_factor = 0.15

    if hasattr(results, 'obb') and results.obb is not None:
        xywhr = results.obb.xywhr.cpu().numpy()
        bboxes = []
        for x, y, w, h, r in xywhr:
            w *= (1 + dilation_factor)
            h *= (1 + dilation_factor)
            bboxes.append([x - w / 2, y - h / 2, w, h])
        thetas = [float(r) for x, y, w, h, r in xywhr]
        scores = results.obb.conf.tolist()
        class_ids = [int(cls) for cls in results.obb.cls.tolist()]
        class_names = [names[class_id] for class_id in class_ids]

    elif hasattr(results, 'boxes') and results.boxes is not None:
        xywh = results.boxes.xywh
