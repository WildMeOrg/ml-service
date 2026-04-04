import torch
import torchvision
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from collections import OrderedDict
from torch import nn
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, Normalize
from .base_model import BaseModel
from ..utils.checkpoint_utils import get_checkpoint_path

logger = logging.getLogger(__name__)


class DenseNetOrientationModel(BaseModel):
    """DenseNet-201 model for orientation classification.

    Compatible with WBIA orientation model checkpoints.
    Checkpoint format: {'state': state_dict, 'classes': class_list}
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.img_size = 224
        self.label_map = None
        self.transforms = None
        self.model_id = None

    def load(self, model_path: str = "", device: str = 'cpu', model_id: str = "",
             checkpoint_path: str = None, img_size: int = 224, label_map: dict = None,
             **kwargs) -> None:
        """Load the DenseNet-201 orientation model.

        Args:
            model_path: Not used (kept for compatibility)
            device: Device to load the model on
            model_id: Unique identifier for the model
            checkpoint_path: Path or URL to the model checkpoint
            img_size: Input image size (default 224)
            label_map: Optional explicit label map; otherwise loaded from checkpoint
            **kwargs: Additional parameters
        """
        try:
            self.device = torch.device(device)
            self.model_id = model_id
            self.img_size = img_size

            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for DenseNet orientation models")

            actual_checkpoint_path = get_checkpoint_path(checkpoint_path)

            # Load checkpoint
            checkpoint = torch.load(actual_checkpoint_path, map_location=self.device, weights_only=False)

            # Extract classes and state
            if isinstance(checkpoint, dict) and 'classes' in checkpoint:
                classes = checkpoint['classes']
                state_dict = checkpoint['state']
            else:
                raise ValueError("Checkpoint must contain 'state' and 'classes' keys")

            # Determine label map
            if label_map is not None:
                self.label_map = {int(k): v for k, v in label_map.items()}
            else:
                self.label_map = {i: c for i, c in enumerate(classes)}

            num_classes = len(self.label_map)

            # Create DenseNet-201 model
            self.model = torchvision.models.densenet201()
            num_ftrs = self.model.classifier.in_features  # 1920
            self.model.classifier = nn.Linear(num_ftrs, num_classes)

            # Strip 'module.' prefix from DataParallel-wrapped models
            clean_state = OrderedDict()
            for k, v in state_dict.items():
                clean_state[k.replace('module.', '')] = v

            self.model.load_state_dict(clean_state)

            # Add softmax for inference (matches WBIA behavior)
            self.model.classifier = nn.Sequential(
                self.model.classifier,
                nn.Softmax(dim=1)
            )

            self.model.to(self.device)
            self.model.eval()

            # Setup transforms
            self.transforms = Compose([
                Resize(self.img_size, self.img_size),
                Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                ToTensorV2()
            ])

            logger.info(f"Loaded DenseNet orientation model '{model_id}' with {num_classes} classes")

        except Exception as e:
            logger.error(f"Error loading DenseNet orientation model: {str(e)}")
            raise

    def predict(self, image_bytes: bytes, bbox: Optional[List[int]] = None,
                theta: float = 0.0, **kwargs) -> Dict[str, Any]:
        """Run orientation classification on the image.

        Args:
            image_bytes: Image data as bytes
            bbox: Optional bounding box [x, y, width, height]
            theta: Rotation angle in radians

        Returns:
            Dictionary with orientation predictions
        """
        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply bounding box
            if bbox is not None:
                x, y, w, h = bbox
                img_h, img_w = image.shape[:2]
                # Clamp to image bounds
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(img_w, int(x + w))
                y2 = min(img_h, int(y + h))
                if x2 > x1 and y2 > y1:
                    image = image[y1:y2, x1:x2]
                else:
                    logger.warning(f"Invalid crop bbox [{x},{y},{w},{h}] for image {img_w}x{img_h}, using full image")

            # Apply rotation
            if theta != 0.0:
                angle_degrees = np.degrees(theta)
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
                image = cv2.warpAffine(image, rotation_matrix, (w, h))

            # Transform
            augmented = self.transforms(image=image)
            tensor = augmented['image'].unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                probs = self.model(tensor).squeeze(0).cpu().numpy()

            # Build results — return all classes with probabilities
            results = []
            for i, prob in enumerate(probs):
                results.append({
                    'label': self.label_map[i],
                    'index': i,
                    'probability': float(prob)
                })

            results.sort(key=lambda x: x['probability'], reverse=True)

            return {
                'model_id': self.model_id,
                'predictions': results,
                'all_probabilities': probs.tolist(),
                'bbox': bbox,
                'theta': theta
            }

        except Exception as e:
            logger.error(f"Error during DenseNet orientation prediction: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_type': 'densenet-orientation',
            'model_architecture': 'densenet201',
            'image_size': self.img_size,
            'num_classes': len(self.label_map),
            'label_map': self.label_map,
            'device': str(self.device) if self.device else None
        }
