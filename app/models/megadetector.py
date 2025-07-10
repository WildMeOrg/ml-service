import os
import urllib.request
import logging
import numpy as np
from io import BytesIO
from typing import Any, Dict, Optional, Union, Tuple
import cv2
from PIL import Image

from PytorchWildlife.models import detection as pw_detection
from .base_model import BaseModel

logger = logging.getLogger(__name__)


def cache_model_file(url: str, weights_dir: str = "weights") -> str:
    """
    Downloads the file from the given URL into the specified weights directory if not already present.

    Args:
        url (str): The URL to download the model from.
        weights_dir (str): The directory to save the model file into. Defaults to 'weights'.

    Returns:
        str: The full path to the cached file.
    """
    try:
        os.makedirs(weights_dir, exist_ok=True)
        filename = os.path.join(weights_dir, os.path.basename(url))
        
        if not os.path.isfile(filename):
            logger.info(f"Downloading model to '{filename}'...")
            urllib.request.urlretrieve(url, filename)
            logger.info("Download complete.")
        else:
            logger.info(f"Using cached model at '{filename}'.")
        
        return filename
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

class MegaDetectorModel(BaseModel):
    """MegaDetector model implementation."""
    
    def __init__(self):
        self.model = None
        self.model_info = {}
    
    def load(self, model_path: str, device: str, **kwargs) -> None:
        """Load the MegaDetector model from the specified path or URL.
        
        Args:
            model_path: URL or local path to the model weights file
            device: Device to load the model on (e.g., 'cpu', 'cuda', 'mps')
            **kwargs: Additional parameters including:
                - conf: Default confidence threshold
        """
        logger.info(f"Loading MegaDetector model from {model_path} on device {device}")
        
        # Cache the model file if it's a URL
        if model_path.startswith(('http://', 'https://')):
            try:
                model_path = cache_model_file(model_path)
            except Exception as e:
                logger.error(f"Failed to download model: {str(e)}")
                raise
        
        # Get model version from kwargs (passed from config)
        model_version = kwargs.get('version')
        if not model_version:
            raise ValueError("Model version must be provided in the model configuration")
        
        # Load the model
        try:
            self.model = pw_detection.MegaDetectorV6(
                version=model_version,
                weights=model_path,
                device=device
            )
            
            # Store model info
            self.model_info = {
                'model_type': 'MegaDetector',
                'version': model_version,
                'device': device,
                'weights_path': model_path,
                'confidence_threshold': kwargs.get('conf', 0.5)
            }
            
            logger.info(f"Successfully loaded {model_version} on {device}")
            
        except Exception as e:
            logger.error(f"Error loading MegaDetector model: {str(e)}")
            raise

    def _bytes_to_numpy(self, image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to a NumPy array in BGR format.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            np.ndarray: Image as a NumPy array in (H, W, C) BGR format, dtype=uint8
        """
        try:
            # Convert bytes to PIL Image
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            
            # Convert to NumPy array and BGR format (for OpenCV compatibility)
            img_np = np.array(img, dtype=np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            return img_bgr
            
        except Exception as e:
            logger.error(f"Error converting image bytes to numpy array: {str(e)}")
            raise

    def predict(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Run object detection on the provided image.
        
        Args:
            image_bytes: Image data as bytes
            **kwargs: Additional inference parameters
                - conf: Confidence threshold (overrides default if provided)
                
        Returns:
            Dictionary containing detection results with keys:
                - bboxes: List of [x, y, w, h] bounding boxes (top-left corner, width, height)
                - thetas: List of rotation angles (always 0.0 for MegaDetector)
                - scores: List of confidence scores
                - class_ids: List of class IDs
                - class_names: List of class names
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        try:
            # Convert image bytes to numpy array
            img_np = self._bytes_to_numpy(image_bytes)
            
            # Get confidence threshold (use provided or default from model info)
            conf_threshold = kwargs.get('conf', self.model_info.get('confidence_threshold', 0.5))
            
            # Run prediction
            detection_result = self.model.single_image_detection(img_np)
            
            # Process detections
            detections = detection_result['detections']
            
            # Initialize result lists
            bboxes = []
            thetas = []
            scores = []
            class_ids = []
            class_names = []
            
            # Process each detection
            if hasattr(detections, 'xyxy'):
                boxes_xyxy = detections.xyxy
                confidences = detections.confidence
                
                for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
                    # Skip detections below confidence threshold
                    if conf_threshold is not None and confidences[i] < conf_threshold:
                        continue
                        
                    # Convert from [x1, y1, x2, y2] to [x, y, w, h] format
                    w = x2 - x1
                    h = y2 - y1
                    bboxes.append([float(x1), float(y1), float(w), float(h)])
                    thetas.append(0.0)  # MegaDetector doesn't support rotation
                    scores.append(float(confidences[i]))
                    
                    # Get class info
                    class_id = int(detections.class_id[i]) if hasattr(detections, 'class_id') else 0
                    class_ids.append(class_id)
                    
                    # Get class name (e.g., "animal 0.87" -> "animal")
                    if 'labels' in detection_result and i < len(detection_result['labels']):
                        label = detection_result['labels'][i]
                        class_name = label.split()[0]  # Get the first word (class name)
                    else:
                        class_name = str(class_id)
                    class_names.append(class_name)
            
            return {
                'bboxes': bboxes,
                'thetas': thetas,
                'scores': scores,
                'class_ids': class_ids,
                'class_names': class_names
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self.model_info
