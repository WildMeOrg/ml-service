import math
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from ultralytics import YOLO
from .base_model import BaseModel
from app.utils.helpers import decode_image_rgb
import logging

logger = logging.getLogger(__name__)

# How an oriented box's (w, h, theta) representation is normalised before it
# leaves the handler. ultralytics 8.3.x regularizes every OBB to
# theta in [0, pi/2), swapping w/h when it wraps -- a discontinuity at 0 where a
# -1 degree tilt becomes an 89 degree box with swapped edges while +1 degree
# stays +1 degree. get_chip_from_img honours theta, so the two representations
# of the SAME rectangle produce chips a quarter turn apart.
#   raw          -- pass ultralytics' representation through (default; the
#                   behaviour every existing OBB model was deployed with)
#   min_rotation -- pick the representation with the smallest |theta|, i.e.
#                   theta in (-pi/4, pi/4], continuous through 0. Right for
#                   near-upright subjects (faces); wrong for long-axis subjects
#                   whose heading the chip must keep (aerial whales).
OBB_THETA_MODES = ("raw", "min_rotation")


def _min_rotation(w, h, r):
    """Return (w, h, theta) for the same rectangle with theta in (-pi/4, pi/4]."""
    if r > math.pi / 4:
        return h, w, r - math.pi / 2
    if r <= -math.pi / 4:
        return h, w, r + math.pi / 2
    return w, h, r


def _validate_obb_theta(mode):
    if mode not in OBB_THETA_MODES:
        raise ValueError(
            f"Unknown obb_theta {mode!r}; expected one of {OBB_THETA_MODES}")
    return mode


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
                - dilation_factors: [long_edge_dil, short_edge_dil] box padding
                - obb_theta: "raw" (default) or "min_rotation"; see OBB_THETA_MODES.
                  Validated here so a misconfigured model fails at startup.
        """
        obb_theta = _validate_obb_theta(kwargs.get('obb_theta', 'raw') or 'raw')
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
            'dilation_factors': kwargs.get('dilation_factors', [0.0, 0.0]),
            'obb_theta': obb_theta,
        }
    
    def predict(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Run object detection on the provided image.
        
        Args:
            image_bytes: Image data as bytes
            **kwargs: Additional inference parameters that can override defaults:
                - imgsz: Image size for this inference
                - conf: Confidence threshold
                - dilation_factors: Dilation factors for OBB [long_edge_dil, short_edge_dil]
                - obb_theta: OBB representation mode, see OBB_THETA_MODES
                
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
        obb_theta = kwargs.get('obb_theta', self.model_info.get('obb_theta', 'raw'))
        
        # Run prediction. Decode through decode_image_rgb so the EXIF
        # Orientation tag is applied: ultralytics does not transpose PIL
        # inputs itself, and boxes must be in the upright (displayed) frame.
        img = Image.fromarray(decode_image_rgb(image_bytes))
        results = self.model.predict(img, save=False, imgsz=imgsz, conf=conf, 
                                   device=device, verbose=False)[0]
        
        # Process results
        return self._process_results(results, dilation_factors, obb_theta=obb_theta)
    
    def _process_results(self, results, dilation_factors, obb_theta='raw'):
        """Process YOLO results into a standardized format.

        `obb_theta` applies to oriented boxes only and runs BEFORE dilation, so
        the long/short dilation factors follow the normalised edges.
        """
        _validate_obb_theta(obb_theta)
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
                if obb_theta == 'min_rotation':
                    w, h, r = _min_rotation(w, h, r)
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
