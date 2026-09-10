import torch
import torchvision
import timm
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from collections import OrderedDict
from collections.abc import Mapping
from torch import nn
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, Normalize
from .base_model import BaseModel
from ..utils.checkpoint_utils import get_checkpoint_path

logger = logging.getLogger(__name__)

# NOTE: there is deliberately no default class list here.
#
# A previous version fell back to ['down','front','left','right','up'] whenever a
# checkpoint declared no classes and happened to have 5 outputs. That guess had no
# basis in the model, and every checkpoint deployed under this type turned out to be
# a wbia-plugin-orientation ORIENTED-BBOX REGRESSOR whose 5 outputs are the
# coordinates [xc, yc, xt, yt, w] used to derive theta -- not class logits. Softmaxing
# them produced near-uniform "probabilities" (~0.2 against a 1/5 baseline) and
# effectively random viewpoints, which Wildbook then stored: its
# Annotation.isValidViewpoint guard could not reject them because down/front/left/
# right/up are all legitimate viewpoints. Whale-shark matching filters candidates on
# viewpoint, so this sent queries at the wrong flank. See issue #33.
#
# A label map must now come from the checkpoint's 'classes' or an explicit config
# label_map. Real classifiers in this ecosystem carry 'classes'; the regressors do
# not -- so requiring them is also what distinguishes the two, since num_classes==5
# matches a 5-class classifier and a 5-coordinate regressor identically.


def _validated_labels(label_map: dict, model_id: str) -> dict:
    """Every resolved label must be a non-empty string.

    A non-string label would be emitted verbatim as a prediction and, for
    viewpoints, silently fail Wildbook's isValidViewpoint check downstream.
    """
    for idx, label in sorted(label_map.items()):
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"Orientation model '{model_id}': label for index {idx} must be a "
                f"non-empty string; got {label!r}."
            )
    return label_map


def _detect_architecture(state_dict: dict) -> str:
    """Detect model architecture from state dict keys."""
    keys = set(state_dict.keys())
    # Check DenseNet first — both DenseNet and HRNet have 'transition' keys,
    # but only DenseNet has 'denseblock' or 'denselayer'.
    if any('denseblock' in k or 'denselayer' in k for k in keys):
        return 'densenet201'
    # HRNet has 'stage2'/'stage3'/'stage4' which DenseNet does not
    if any('stage2' in k or 'stage3' in k or 'stage4' in k for k in keys):
        return 'hrnet_w32'
    # Fallback: check classifier output features
    for k in keys:
        if 'classifier.weight' in k:
            n_features = state_dict[k].shape[1]
            if n_features == 1920:
                return 'densenet201'
            if n_features == 2048:
                return 'hrnet_w32'
    return 'densenet201'  # fallback


class DenseNetOrientationModel(BaseModel):
    """Orientation classification model supporting DenseNet-201 and HRNet-W32.

    Compatible with WBIA orientation model checkpoints in two formats:
    - Wrapped: {'state': state_dict, 'classes': class_list}
    - Raw: bare state_dict (classes MUST then come from an explicit config
      label_map; they are never guessed -- see issue #33)
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.img_size = 224
        self.label_map = None
        self.model_id = None
        self.architecture = None

    def load(self, model_path: str = "", device: str = 'cpu', model_id: str = "",
             checkpoint_path: str = None, img_size: int = 224, label_map: dict = None,
             **kwargs) -> None:
        """Load the orientation model.

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
                raise ValueError("checkpoint_path is required for orientation models")

            actual_checkpoint_path = get_checkpoint_path(checkpoint_path)

            # Load checkpoint
            checkpoint = torch.load(actual_checkpoint_path, map_location=self.device, weights_only=False)

            # Handle both wrapped {'state': ..., 'classes': ...} and raw state_dict formats
            if isinstance(checkpoint, dict) and 'state' in checkpoint:
                state_dict = checkpoint['state']
                classes = checkpoint.get('classes')
            else:
                state_dict = checkpoint
                classes = None

            # Strip 'model.' and 'module.' prefixes from DataParallel-wrapped models
            clean_state = OrderedDict()
            for k, v in state_dict.items():
                clean_key = k.replace('module.', '').replace('model.', '')
                clean_state[clean_key] = v

            # Detect architecture from state dict
            self.architecture = _detect_architecture(clean_state)

            # Determine number of classes from classifier weights
            classifier_key = 'classifier.weight'
            if classifier_key in clean_state:
                num_classes = clean_state[classifier_key].shape[0]
            else:
                raise ValueError(f"Cannot determine num_classes: '{classifier_key}' not found in checkpoint")

            # Determine label map: config > checkpoint. Never guessed -- see the
            # note at the top of this module and issue #33.
            if label_map is not None:
                if not isinstance(label_map, Mapping):
                    raise ValueError(
                        f"Orientation model '{model_id}': label_map must be a mapping "
                        f"of index -> label; got {type(label_map).__name__}."
                    )
                coerced = {}
                for k, v in label_map.items():
                    try:
                        ik = int(k)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"Orientation model '{model_id}': label_map key {k!r} is "
                            f"not an integer index."
                        )
                    # JSON allows "0" and "00" as distinct keys; both coerce to 0
                    # and one would silently overwrite the other.
                    if ik in coerced:
                        raise ValueError(
                            f"Orientation model '{model_id}': label_map has duplicate "
                            f"index {ik} after integer coercion (e.g. \"0\" and \"00\")."
                        )
                    coerced[ik] = v
                expected = set(range(num_classes))
                if set(coerced.keys()) != expected:
                    raise ValueError(
                        f"Orientation model '{model_id}': config label_map keys must be "
                        f"exactly {sorted(expected)}; got {sorted(coerced.keys())}."
                    )
                self.label_map = _validated_labels(coerced, model_id)
            elif classes is not None:
                # A bare string is iterable: enumerate("abc") would silently yield
                # 'a','b','c' for a 3-output head. Require a real sequence.
                if isinstance(classes, str) or not isinstance(classes, (list, tuple)):
                    raise ValueError(
                        f"Orientation model '{model_id}': checkpoint 'classes' must be "
                        f"a list or tuple of labels; got {type(classes).__name__}."
                    )
                if len(classes) != num_classes:
                    raise ValueError(
                        f"Orientation model '{model_id}': checkpoint 'classes' has "
                        f"{len(classes)} entries but the classifier head has "
                        f"num_classes={num_classes} -- stale metadata."
                    )
                self.label_map = _validated_labels(
                    {i: c for i, c in enumerate(classes)}, model_id)
            else:
                raise ValueError(
                    f"Orientation model '{model_id}': checkpoint declares no 'classes' "
                    f"and no label_map was configured, so the meaning of its "
                    f"{num_classes} outputs is unknown. This model type will NOT guess "
                    f"-- guessed labels are indistinguishable from real predictions "
                    f"downstream (issue #33). Either add an explicit label_map to this "
                    f"model's model_config.json entry, or remove the entry: a "
                    f"wbia-plugin-orientation checkpoint is an oriented-bbox regressor "
                    f"(outputs [xc, yc, xt, yt, w] for deriving theta), not a "
                    f"classifier, and no label_map can make it one."
                )

            # Create model based on detected architecture
            if self.architecture == 'hrnet_w32':
                self.model = timm.create_model('hrnet_w32', pretrained=False, num_classes=num_classes)
            else:
                self.model = torchvision.models.densenet201()
                num_ftrs = self.model.classifier.in_features  # 1920
                self.model.classifier = nn.Linear(num_ftrs, num_classes)

            self.model.load_state_dict(clean_state)

            # Add softmax for inference (matches WBIA behavior)
            self.model.classifier = nn.Sequential(
                self.model.classifier,
                nn.Softmax(dim=1)
            )

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Loaded {self.architecture} orientation model '{model_id}' with {num_classes} classes")

        except Exception as e:
            logger.error(f"Error loading orientation model: {str(e)}")
            raise

    @staticmethod
    def _preprocess_tensor(
        image_bytes: bytes,
        bbox: Optional[List[int]],
        theta: float,
        img_size: int,
        device,
    ) -> "torch.Tensor":
        """Decode, crop, rotate, and normalise image bytes into a model-ready
        float tensor of shape (1, 3, img_size, img_size) on *device*.

        Extracted as a staticmethod so that sibling classifiers (e.g.
        DenseNetClassifierModel) can reuse identical preprocessing without
        inheriting from this class.
        """
        transforms = Compose([
            Resize(img_size, img_size),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2()
        ])

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if bbox is not None:
            x, y, w, h = bbox
            img_h, img_w = image.shape[:2]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, int(x + w))
            y2 = min(img_h, int(y + h))
            if x2 > x1 and y2 > y1:
                image = image[y1:y2, x1:x2]
            else:
                logger.warning(
                    f"Invalid crop bbox [{x},{y},{w},{h}] for image "
                    f"{img_w}x{img_h}, using full image"
                )

        if theta != 0.0:
            angle_degrees = np.degrees(theta)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h))

        augmented = transforms(image=image)
        return augmented["image"].unsqueeze(0).to(device)

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
            tensor = self._preprocess_tensor(
                image_bytes, bbox, theta, self.img_size, self.device
            )

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
            logger.error(f"Error during orientation prediction: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_type': 'densenet-orientation',
            'model_architecture': self.architecture or 'unknown',
            'image_size': self.img_size,
            'num_classes': len(self.label_map),
            'label_map': self.label_map,
            'device': str(self.device) if self.device else None
        }
