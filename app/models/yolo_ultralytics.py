from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from ultralytics import YOLO
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class YOLOUltralyticsModel(BaseModel):
    """YOLO model implementation using Ultralytics."""
    
    def __init__(self):
        self.model = None
        self.model_info = {}
    
    def load(self, model_path: str, device: str, **kwargs) -> None:
        """Load the YOLO model from the specified path.
        
        Args:
            model_path: Path to the YOLO model file (.pt, .onnx, etc.)
            device: Device to load the model on (e.g., 'cpu', 'cuda', 'mps')
            **kwargs: Additional parameters including:
                - imgsz: Default image size for inference
                - conf: Default confidence threshold
        """
        logger.info(f"Loading YOLO model from {model_path} on device {device}")
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Store model info
        self.model_info = {
            'model_type': 'yolo-ultralytics',
            'model_path': model_path,
            'device': device,
            'imgsz': kwargs.get('imgsz', 640),
            'conf': kwargs.get('conf', 0.25),
            'dilation_factors': kwargs.get('dilation_factors', [0.0, 0.0])
        }
    
    def predict(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Run object detection on the provided image.
        
        Args:
            image_bytes: Image data as bytes
            **kwargs: Additional inference parameters that can override defaults:
                - imgsz: Image size for this inference
                - conf: Confidence threshold
                - dilation_factors: Dilation factors for OBB [long_edge_dil, short_edge_dil]
                
        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
            
        # Get inference parameters, using model defaults if not provided
        imgsz = kwargs.get('imgsz', self.model_info['imgsz'])
        conf = kwargs.get('conf', self.model_info['conf'])
        device = self.model_info['device']
        dilation_factors = kwargs.get('dilation_factors', self.model_info['dilation_factors'])
        
        # Run prediction
        img = Image.open(BytesIO(image_bytes))
        results = self.model.predict(img, save=False, imgsz=imgsz, conf=conf, 
                                   device=device, verbose=False)[0]
        
        # Process results
        return self._process_results(results, dilation_factors)
    
    def _process_results(self, results, dilation_factors):
        """Process YOLO results into a standardized format."""
        long_dil, short_dil = dilation_factors
        bboxes = []
        thetas = []
        scores = []
        class_ids = []
        class_names = []
        
        # Handle OBB (Oriented Bounding Box) results
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
                bboxes.append([float(x - w_dilated / 2), float(y - h_dilated / 2), 
                         float(w_dilated), float(h_dilated)])
                thetas.append(float(r))

            scores = results.obb.conf.tolist()
            class_ids = [int(cls) for cls in results.obb.cls.tolist()]
            class_names = [results.names[class_id] for class_id in class_ids]
            
        # Handle standard bounding box results
        elif hasattr(results, 'boxes') and results.boxes is not None:
            xywh = results.boxes.xywh.cpu().numpy()
            for x, y, w, h in xywh:
                # Apply the same dilation logic as OBB
                if w >= h:
                    w_dilated = w * (1 + long_dil)
                    h_dilated = h * (1 + short_dil)
                else:
                    w_dilated = w * (1 + short_dil)
                    h_dilated = h * (1 + long_dil)
                
                bboxes.append([float(x - w_dilated/2), float(y - h_dilated/2), 
                             float(w_dilated), float(h_dilated)])
                thetas.append(0.0)  # No rotation for standard bboxes
                
            scores = results.boxes.conf.tolist()
            class_ids = [int(cls) for cls in results.boxes.cls.tolist()]
            class_names = [results.names[class_id] for class_id in class_ids]
        
        # Convert any remaining NumPy types to Python native types
        scores = [float(score) for score in scores]
        
        return {
            'bboxes': bboxes,
            'thetas': thetas,
            'scores': scores,
            'class_ids': class_ids,
            'class_names': class_names
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self.model_info
