from io import BytesIO
from PIL import Image

def run_inference(image_bytes: bytes, model, device: str, imgsz: int, conf: float):
    img = Image.open(BytesIO(image_bytes))
    results = model.predict(img, save=False, imgsz=imgsz, conf=conf, device=device, verbose=False)
    results = results[0]

    # Check for OBB or bounding boxes
    if hasattr(results, 'obb') and results.obb is not None:
        # Extract [x_center, y_center, w, h, r]
        xywhr = results.obb.xywhr.cpu().numpy()
        # Convert to [x_topleft, y_topleft, w, h, r]
        converted = [[x - w / 2, y - h / 2, w, h, r] for x, y, w, h, r in xywhr]
        return converted

    elif hasattr(results, 'boxes') and results.boxes is not None:
        # Extract [x_center, y_center, w, h], add r = 0.0
        xywh = results.boxes.xywh.cpu().numpy()
        # Convert to [x_topleft, y_topleft, w, h, 0.0]
        converted = [[x - w / 2, y - h / 2, w, h, 0.0] for x, y, w, h in xywh]
        return converted

    else:
        return []
