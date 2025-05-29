from ultralytics import YOLO
from .inference import run_inference
import logging

logger = logging.getLogger(__name__)

class YOLOHandler:
    """Handler class for managing YOLO model instances and their inference.
    
    This class provides a convenient interface for loading multiple YOLO models
    and running predictions on them. It handles model lifecycle and configuration.
    
    Attributes:
        models (dict): Dictionary storing loaded models and their configurations.
                     Keys are model_ids and values are dictionaries containing
                     the model instance and its configuration.
    """
    
    def __init__(self):
        """Initialize the YOLOHandler with an empty models dictionary."""
        self.models = {}

    def load_model(self, model_id: str, model_path: str, device: str, imgsz: int = 640, conf: float = 0.25) -> None:
        """Load a YOLO model with specified parameters.

        This method loads a YOLO model from the given path, moves it to the specified
        device, and stores it along with its configuration in the models dictionary.

        Args:
            model_id (str): Unique identifier for the model. Used as a key to retrieve the model later.
            model_path (str): Filesystem path to the YOLO model file (.pt, .onnx, etc.).
            device (str): Device to load the model on. Can be 'cpu', 'cuda', 'mps', etc.
            imgsz (int, optional): Default image size for inference. Defaults to 640.
            conf (float, optional): Default confidence threshold for detections (0-1). Defaults to 0.25.
            
        Raises:
            FileNotFoundError: If the model file does not exist at the specified path.
            RuntimeError: If there is an error loading the model.
        """
        logger.info("Loading model with ID: %s, path: %s, device: %s, image size: %d, confidence: %.2f", model_id, model_path, device, imgsz, conf)
        model = YOLO(model_path)
        model.to(device)
        self.models[model_id] = {
            "model": model,
            "imgsz": imgsz,
            "conf": conf
        }

    def predict(self, model_id: str, image_bytes: bytes) -> 'ModelResponse':
        """Run object detection on an image using the specified model.

        This method takes image data in bytes, processes it using the specified model,
        and returns the detection results.

        Args:
            model_id (str): Identifier of the model to use for prediction. Must have been
                         previously loaded using load_model().
            image_bytes (bytes): Raw image data in bytes. The image should be in a format
                              that PIL can open (JPEG, PNG, etc.).

        Returns:
            ModelResponse: A Pydantic model containing the detection results.

        Raises:
            ValueError: If the specified model_id has not been loaded.
            RuntimeError: If there is an error during prediction.
        """
        if model_id not in self.models:
            raise ValueError(f"Model with ID {model_id} not loaded.")
        model_info = self.models[model_id]
        model = model_info["model"]
        imgsz = model_info["imgsz"]
        conf = model_info["conf"]
        
        return run_inference(image_bytes, model, model.device, imgsz, conf)
