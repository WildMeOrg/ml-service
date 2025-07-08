from io import BytesIO
from PIL import Image
from app.schemas.model_response import ModelResponse  # Your custom schema

def run_inference(
    image_bytes: bytes,
    model,
    device: str,
    imgsz: int,
    conf: float,
    dilation_factors: list = [0.15, 0.1]  # [long_edge_dil, short_edge_dil]
) -> ModelResponse:
    """
    Run object detection inference on an image, applying OBB-specific dilation.
    Dilation is asymmetrical: the long edge gets one factor, the short edge another.
    Axis-aligned boxes are returned as-is.
    """
    img = Image.open(BytesIO(image_bytes))
    results = model.predict(img, save=False, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    names = model.names

    long_dil, short_dil = dilation_factors
    bboxes = []
    thetas = []
    scores = []
    class_ids = []
    class_names = []

    if hasattr(results, 'obb') and results.obb is not None:
        xywhr = results.obb.xywhr.cpu().numpy()

        for x, y, w, h, r in xywhr:
            # Determine long and short side
            if w >= h:
                w_dilated = w * (1 + long_dil)
                h_dilated = h * (1 + short_dil)
            else:
                w_dilated = w * (1 + short_dil)
                h_dilated = h * (1 + long_dil)

            # Centered bbox: x, y = center
            bboxes.append([x - w_dilated / 2, y - h_dilated / 2, w_dilated, h_dilated])
            thetas.append(float(r))

        scores = results.obb.conf.tolist()
        class_ids = [int(cls) for cls in results.obb.cls.tolist()]
        class_names = [names[class_id] for class_id in class_ids]

    elif hasattr(results, 'boxes') and results.boxes is not None:
        xywh = results.boxes.xywh.cpu().numpy()

        for x, y, w, h in xywh:
            bboxes.append([x - w / 2, y - h / 2, w, h])  # No dilation for axis-aligned
            thetas.append(0.0)

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
