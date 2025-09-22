import torch
import timm
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from torch import nn
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, Normalize
import io
from PIL import Image
from .base_model import BaseModel
from ..utils.checkpoint_utils import get_checkpoint_path

logger = logging.getLogger(__name__)

LABEL_MAP = {0: 'back', 1: 'down', 2: 'front', 3: 'left', 4: 'right', 5: 'up'}

class ImgClassifier(nn.Module):
    def __init__(self, model_arch: str, n_class: int, pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        return self.model(x)

class EfficientNetModel(BaseModel):
    """EfficientNet model for image classification."""
    
    def __init__(self):
        """Initialize the EfficientNet model."""
        self.model = None
        self.device = None
        self.model_arch = 'tf_efficientnet_b4_ns'  # Fixed architecture
        self.img_size = 512  # Default, will be overridden by config
        self.threshold = 0.5  # Default, will be overridden by config
        self.label_map = LABEL_MAP
        self.transforms = None
        self.model_id = None
        
    def load(self, model_path: str = "", device: str = 'cpu', model_id: str = "", 
             checkpoint_path: str = None, img_size: int = 512, threshold: float = 0.5, **kwargs) -> None:
        """Load the EfficientNet model.
        
        Args:
            model_path: Not used for EfficientNet (kept for compatibility)
            device: Device to load the model on
            model_id: Unique identifier for the model
            checkpoint_path: Path or URL to the model checkpoint
            img_size: Input image size for preprocessing
            threshold: Classification threshold
            **kwargs: Additional parameters
        """
        try:
            self.device = torch.device(device)
            self.model_id = model_id
            self.img_size = img_size
            self.threshold = threshold
            
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for EfficientNet models")
            
            # Get the actual checkpoint path (download if URL)
            actual_checkpoint_path = get_checkpoint_path(checkpoint_path)
            
            # Create model
            self.model = ImgClassifier(
                model_arch=self.model_arch,
                n_class=len(self.label_map),
                pretrained=False
            )
            
            # Load checkpoint
            checkpoint = torch.load(actual_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=False)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self._setup_transforms()
            
            logger.info(f"Successfully loaded EfficientNet model '{model_id}' from {checkpoint_path}")
            
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
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                
                # Apply threshold
                preds = (probs > self.threshold).astype(int)
                
                # Get predictions above threshold
                predicted_indices = np.where(preds == 1)[0]
                predicted_labels = [self.label_map[i] for i in predicted_indices]
                
                # Create results with probabilities and sort by probability (descending)
                results = []
                for i in predicted_indices:
                    results.append({
                        'label': self.label_map[i],
                        'index': int(i),
                        'probability': float(probs[i])
                    })
                
                # Sort by probability descending
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
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': 'efficientnetv2',
            'model_architecture': self.model_arch,
            'image_size': self.img_size,
            'threshold': self.threshold,
            'num_classes': len(self.label_map),
            'label_map': self.label_map,
            'device': str(self.device) if self.device else None
        }
