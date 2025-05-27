from ultralytics import YOLO
from .inference import run_inference
import logging

logger = logging.getLogger(__name__)

class YOLOHandler:
    def __init__(self):
        """Initialize the YOLOHandler with an empty models dictionary."""
        self.models = {}

    def load_model(self, model_id: str, model_path: str, device: str, imgsz: int = 640, conf: float = 0.25):
        """
        Load a YOLO model with specified parameters.

        Args:
            model_id (str): Unique identifier for the model.
            model_path (str): Path to the model file.
            device (str): Device to load the model on (e.g., 'cpu', 'cuda').
            imgsz (int, optional): Image size for inference. Defaults to 640.
            conf (float, optional): Confidence threshold for predictions. Defaults to 0.25.
        """
        logger.info("Loading model with ID: %s, path: %s, device: %s, image size: %d, confidence: %.2f", model_id, model_path, device, imgsz, conf)
        model = YOLO(model_path)
        model.to(device)
        self.models[model_id] = {
            "model": model,
            "imgsz": imgsz,
            "conf": conf
        }

    def predict(self, model_id: str, image_bytes: bytes):
        """
        Predict objects in the given image using the specified model.

        Args:
            model_id (str): Identifier of the model to use for prediction.
            image_bytes (bytes): Image data in bytes.

        Returns:
            results: Prediction results from the model.
        """
        if model_id not in self.models:
            raise ValueError(f"Model with ID {model_id} not loaded.")
        model_info = self.models[model_id]
        model = model_info["model"]
        imgsz = model_info["imgsz"]
        conf = model_info["conf"]
        
        return run_inference(image_bytes, model, model.device, imgsz, conf)
