import torch
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from io import BytesIO
from PIL import Image
import torchvision.transforms as tf

from .base_model import BaseModel
from ..utils.checkpoint_utils import get_checkpoint_path

logger = logging.getLogger(__name__)


class LightNetModel(BaseModel):
    """LightNet (PyDarknet YOLO v2/v3) detection model.

    Compatible with WBIA lightnet species-specific detection models.
    Uses .py config files and .weights binary weight files.
    """

    _ln = None  # Lazy-loaded lightnet module

    @classmethod
    def _import_lightnet(cls):
        """Lazy-import lightnet with brambox compatibility shim."""
        if cls._ln is None:
            from ._brambox_compat import ensure_brambox_compat
            ensure_brambox_compat()
            import lightnet
            cls._ln = lightnet
        return cls._ln

    def __init__(self):
        self.params = None
        self.device = None
        self.model_id = None
        self.conf = 0.1
        self.nms_thresh = 0.4
        self.batch_size = 192

    def load(self, model_path: str = "", device: str = 'cpu', model_id: str = "",
             config_path: str = None, weight_path: str = None,
             conf: float = 0.1, nms_thresh: float = 0.4, batch_size: int = 192,
             **kwargs) -> None:
        """Load a LightNet detection model.

        Args:
            model_path: Not used (kept for compatibility)
            device: Device to load the model on
            model_id: Unique identifier for the model
            config_path: Path or URL to the lightnet .py config file
            weight_path: Path or URL to the .weights file
            conf: Confidence threshold
            nms_thresh: NMS threshold
            batch_size: Batch size for multi-image inference
            **kwargs: Additional parameters
        """
        try:
            self.device = torch.device(device)
            self.model_id = model_id
            self.conf = conf
            self.nms_thresh = nms_thresh
            self.batch_size = batch_size

            if config_path is None:
                raise ValueError("config_path is required for LightNet models")
            if weight_path is None:
                raise ValueError("weight_path is required for LightNet models")

            # Download config and weights if URLs
            actual_config_path = get_checkpoint_path(config_path)
            actual_weight_path = get_checkpoint_path(weight_path)

            # Load via lightnet HyperParameters
            ln = self._import_lightnet()
            self.params = ln.engine.HyperParameters.from_file(actual_config_path)
            self.params.load(actual_weight_path)
            self.params.device = self.device

            # Set postprocessing thresholds
            self.params.network.postprocess[0].conf_thresh = self.conf
            self.params.network.postprocess[1].nms_thresh = self.nms_thresh

            self.params.network.eval()

            try:
                self.params.network.to(self.device)
            except Exception:
                logger.warning(f"Failed to move LightNet to {device}, falling back to CPU")
                self.device = torch.device('cpu')
                self.params.network.to(self.device)

            # Extract class labels from config
            self.class_labels = self.params.class_label_map if hasattr(self.params, 'class_label_map') else []

            logger.info(f"Loaded LightNet model '{model_id}' with {len(self.class_labels)} classes")

        except Exception as e:
            logger.error(f"Error loading LightNet model: {str(e)}")
            raise

    def predict(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Run detection on the provided image.

        Args:
            image_bytes: Image data as bytes
            **kwargs: Additional parameters

        Returns:
            Dictionary containing detection results in standard format
        """
        if self.params is None:
            raise ValueError("Model not loaded. Call load() first.")

        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_h, img_w = img.shape[:2]
            img_size = (img_w, img_h)

            # Letterbox preprocessing
            ln = self._import_lightnet()
            img_lb = ln.data.transform.Letterbox.apply(img, dimension=self.params.input_dimension)
            img_tensor = tf.ToTensor()(img_lb).unsqueeze(0)

            try:
                img_tensor = img_tensor.to(self.device)
            except Exception:
                pass

            # Run detection
            with torch.no_grad():
                out = self.params.network(img_tensor)

            # Reverse letterbox to original image coordinates
            result = ln.data.transform.ReverseLetterbox.apply(
                [out[0]], self.params.input_dimension, img_size
            )
            result = result[0]

            # Convert to standard detection format
            bboxes = []
            scores = []
            class_names = []

            for detection in list(result):
                xtl = int(np.around(float(detection.x_top_left)))
                ytl = int(np.around(float(detection.y_top_left)))
                xbr = int(np.around(float(detection.x_top_left + detection.width)))
                ybr = int(np.around(float(detection.y_top_left + detection.height)))
                width = xbr - xtl
                height = ybr - ytl

                bboxes.append([xtl, ytl, width, height])
                scores.append(float(detection.confidence))
                class_names.append(detection.class_label)

            return {
                'model_id': self.model_id,
                'bboxes': bboxes,
                'scores': scores,
                'class_names': class_names,
                'num_detections': len(bboxes),
                'image_size': {'width': img_w, 'height': img_h}
            }

        except Exception as e:
            logger.error(f"Error during LightNet prediction: {str(e)}")
            raise

    @property
    def model(self):
        """Access the underlying network for compatibility with shutdown handler."""
        return self.params.network if self.params else None

    @model.setter
    def model(self, value):
        """Allow setting model to None for cleanup."""
        if value is None and self.params is not None:
            self.params.network = None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_type': 'lightnet',
            'confidence_threshold': self.conf,
            'nms_threshold': self.nms_thresh,
            'class_labels': self.class_labels,
            'num_classes': len(self.class_labels),
            'device': str(self.device) if self.device else None
        }
