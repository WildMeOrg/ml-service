import torch
import timm
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from collections import OrderedDict
from torch import nn
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, Normalize
import io
from PIL import Image
from .base_model import BaseModel
from ..utils.checkpoint_utils import get_checkpoint_path

logger = logging.getLogger(__name__)

DEFAULT_LABEL_MAP = {0: 'back', 1: 'down', 2: 'front', 3: 'left', 4: 'right', 5: 'up'}

class ImgClassifier(nn.Module):
    def __init__(self, model_arch: str, n_class: int, pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        return self.model(x)

class EfficientNetModel(BaseModel):
    """EfficientNet model for image classification.

    Supports configurable label maps, model architectures, and compound label parsing
    for compatibility with WBIA species:viewpoint classification models.
    """

    def __init__(self):
        """Initialize the EfficientNet model."""
        self.model = None
        self.device = None
        self.model_arch = 'tf_efficientnet_b4_ns'
        self.img_size = 512
        self.threshold = 0.5
        self.label_map = None
        self.transforms = None
        self.model_id = None
        self.multi_label = True
        self.parse_compound_labels = False

    def load(self, model_path: str = "", device: str = 'cpu', model_id: str = "",
             checkpoint_path: str = None, img_size: int = 512, threshold: float = 0.5,
             model_arch: str = None, label_map: dict = None, n_classes: int = None,
             multi_label: bool = True, parse_compound_labels: bool = False, **kwargs) -> None:
        """Load the EfficientNet model.

        Args:
            model_path: Not used for EfficientNet (kept for compatibility)
            device: Device to load the model on
            model_id: Unique identifier for the model
            checkpoint_path: Path or URL to the model checkpoint
            img_size: Input image size for preprocessing
            threshold: Classification threshold
            model_arch: timm model architecture name (default: tf_efficientnet_b4_ns)
            label_map: Dict mapping class index to label string. If not provided,
                       will attempt to load from checkpoint 'classes' key.
            n_classes: Number of classes (inferred from label_map if not provided)
            multi_label: If True, use sigmoid+threshold. If False, use softmax+argmax.
            parse_compound_labels: If True, split labels on ':' to return separate
                                   species and viewpoint fields.
            **kwargs: Additional parameters
        """
        try:
            self.device = torch.device(device)
            self.model_id = model_id
            self.img_size = img_size
            self.threshold = threshold
            self.multi_label = multi_label
            self.parse_compound_labels = parse_compound_labels

            if model_arch is not None:
                self.model_arch = model_arch

            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for EfficientNet models")

            # Get the actual checkpoint path (download if URL)
            actual_checkpoint_path = get_checkpoint_path(checkpoint_path)

            # Load checkpoint to inspect for classes
            checkpoint = torch.load(actual_checkpoint_path, map_location=self.device, weights_only=False)

            # Determine label map: config > checkpoint > default
            if label_map is not None:
                # Config provides explicit label_map (keys may be strings from JSON)
                self.label_map = {int(k): v for k, v in label_map.items()}
            elif isinstance(checkpoint, dict) and 'classes' in checkpoint:
                # WBIA checkpoint format: {'state': state_dict, 'classes': [...]}
                classes = checkpoint['classes']
                self.label_map = {i: c for i, c in enumerate(classes)}
            else:
                self.label_map = DEFAULT_LABEL_MAP

            # Detect mismatch between label count and actual classifier output size.
            # Some WBIA checkpoints have stale 'classes' metadata that doesn't match
            # the trained classifier dimensions.
            if n_classes is not None:
                num_classes = n_classes
            else:
                num_classes = len(self.label_map)
                # Check actual classifier output dimension from state dict
                sd = checkpoint.get('state', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                classifier_keys = [k for k in sd.keys() if 'classifier' in k and 'weight' in k]
                if classifier_keys:
                    actual_n = sd[classifier_keys[-1]].shape[0]
                    if actual_n != num_classes:
                        logger.warning(
                            f"Label map has {num_classes} classes but classifier weights have "
                            f"{actual_n} outputs. Truncating label map to match weights."
                        )
                        num_classes = actual_n
                        self.label_map = {i: v for i, v in self.label_map.items() if i < actual_n}

            # Create model
            self.model = ImgClassifier(
                model_arch=self.model_arch,
                n_class=num_classes,
                pretrained=False
            )

            # Load state dict — handle both raw state_dict and WBIA format
            if isinstance(checkpoint, dict) and 'state' in checkpoint:
                state_dict = checkpoint['state']
            else:
                state_dict = checkpoint

            # Strip 'module.' prefix from DataParallel-wrapped models
            clean_state = OrderedDict()
            for k, v in state_dict.items():
                clean_state[k.replace('module.', '')] = v

            self.model.load_state_dict(clean_state, strict=False)

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            # Setup transforms
            self._setup_transforms()

            logger.info(f"Loaded EfficientNet model '{model_id}' with {num_classes} classes, "
                       f"multi_label={multi_label}, parse_compound={parse_compound_labels}")

        except Exception as e:
            logger.error(f"Error loading EfficientNet model: {str(e)}")
            raise

    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        self.transforms = Compose([
            Resize(self.img_size, self.img_size),
            Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2()
        ])

    def _preprocess_image(self, image_bytes: bytes, bbox: Optional[Tuple[int, int, int, int]] = None,
                         theta: float = 0.0) -> torch.Tensor:
        """Preprocess image for classification.

        Args:
            image_bytes: Raw image bytes
            bbox: Optional bounding box [x, y, width, height]
            theta: Rotation angle in radians

        Returns:
            Preprocessed image tensor
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply bounding box if provided
        if bbox is not None:
            x, y, w, h = bbox
            image = image[y:y+h, x:x+w]

        # Apply rotation if provided
        if theta != 0.0:
            # Convert radians to degrees
            angle_degrees = np.degrees(theta)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h))

        # Apply transforms
        augmented = self.transforms(image=image)
        tensor = augmented['image'].unsqueeze(0)  # Add batch dimension

        return tensor

    def predict(self, image_bytes: bytes, bbox: Optional[List[int]] = None,
                theta: float = 0.0, **kwargs) -> Dict[str, Any]:
        """Run classification inference on the image.

        Args:
            image_bytes: Image data as bytes
            bbox: Optional bounding box coordinates [x, y, width, height]
            theta: Rotation angle in radians
            **kwargs: Additional parameters

        Returns:
            Dictionary containing classification results
        """
        try:
            # Convert bbox to tuple if provided
            bbox_tuple = tuple(bbox) if bbox is not None else None

            # Preprocess image
            image_tensor = self._preprocess_image(image_bytes, bbox_tuple, theta)
            image_tensor = image_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                logits = self.model(image_tensor)

                if self.multi_label:
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    preds = (probs > self.threshold).astype(int)
                    predicted_indices = np.where(preds == 1)[0]
                else:
                    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                    predicted_indices = [np.argmax(probs)]

                # Build results
                results = []
                for i in predicted_indices:
                    label = self.label_map[int(i)]
                    entry = {
                        'label': label,
                        'index': int(i),
                        'probability': float(probs[i])
                    }
                    if self.parse_compound_labels and ':' in label:
                        parts = label.split(':', 1)
                        entry['species'] = parts[0]
                        entry['viewpoint'] = parts[1]
                    results.append(entry)

                results.sort(key=lambda x: x['probability'], reverse=True)

                return {
                    'model_id': self.model_id,
                    'predictions': results,
                    'all_probabilities': probs.tolist(),
                    'threshold': self.threshold,
                    'bbox': bbox,
                    'theta': theta
                }

        except Exception as e:
            logger.error(f"Error during EfficientNet prediction: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_type': 'efficientnetv2',
            'model_architecture': self.model_arch,
            'image_size': self.img_size,
            'threshold': self.threshold,
            'num_classes': len(self.label_map),
            'label_map': self.label_map,
            'multi_label': self.multi_label,
            'parse_compound_labels': self.parse_compound_labels,
            'device': str(self.device) if self.device else None
        }
