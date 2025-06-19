from io import BytesIO
from PIL import Image
from app.schemas.model_response import ModelResponse  # Import your response schema

def run_inference(image_bytes: bytes, model, device: str, imgsz: int, conf: float) -> ModelResponse:
    """
    Run object detection inference on an image, with optional postprocessing to dilate bounding boxes.

    Only oriented bounding boxes (OBB) are dilated by 15%. Axis-aligned bounding boxes are returned as-is.
    """
    img = Image.open(BytesIO(image_bytes))
    results = model.predict(img, save=False, imgsz=imgsz, conf=conf, device=device, verbose=False)
    results = results[0]
    names = model.names

    dilation_factor = 0.15  # 15% dilation only for OBB

    if hasattr(results, 'obb') and results.obb is not None:
        xywhr = results.obb.xywhr.cpu().numpy()
        bboxes = []
        thetas = []
        for x, y, w, h, r in xywhr:
            w_dilated = w * (1 + dilation_factor)
            h_dilated = h * (1 + dilation_factor)
            bboxes.append([x - w_dilated / 2, y - h_dilated / 2, w_dilated, h_dilated])
            thetas.append(float(r))
        scores = results.obb.conf.tolist()
        class_ids = [int(cls) for cls in results.obb.cls.tolist()]
        class_names = [names[class_id] for class_id in class_ids]

    elif hasattr(results, 'boxes') and results.boxes is not None:
        xywh = results.boxes.xywh.cpu().numpy()
        bboxes = []
        for x, y, w, h in xywh:
            bboxes.append([x - w / 2, y - h / 2, w, h])  # No dilation
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