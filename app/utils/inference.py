from io import BytesIO
from PIL import Image

def run_inference(image_bytes: bytes, model, device: str, imgsz: int, conf: float):
    img = Image.open(BytesIO(image_bytes))
    results = model.predict(img, save=False, imgsz=imgsz, conf=conf, device=device, verbose=False)
    return results
