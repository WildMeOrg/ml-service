from io import BytesIO
from PIL import Image
from app.schemas.model_response import ModelResponse  # Import your response schema

def run_inference(image_bytes: bytes, model, device: str, imgsz: int, conf: float):
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
