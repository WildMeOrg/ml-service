from .base_model import BaseModel
import logging
from transformers import AutoModel
from fastapi import HTTPException
from typing import Dict, Any, List, Optional, Tuple
import timm
import torch
import torch.nn as nn
import albumentations
from albumentations.pytorch import ToTensorV2
# torchvision import removed: preprocessing now uses albumentations to match
# wbia-plugin-miew-id's training/inference pipeline. See the comment on
# self.preprocess below for details.
from PIL import Image
import numpy as np
import io
from app.utils.checkpoint_utils import get_checkpoint_path
from app.utils.helpers import get_chip_from_img

logger = logging.getLogger(__name__)


class GeM(nn.Module):
    """Generalized Mean Pooling."""
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class MiewIdNet(nn.Module):
    """Standalone MiewID model architecture matching wbia-plugin-miew-id training code."""
    def __init__(self, model_name='efficientnetv2_rw_m'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False)
        final_in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling = GeM()
        self.bn = nn.BatchNorm1d(final_in_features)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pooling(x).view(x.size(0), -1)
        return self.bn(x)


class MiewidModel(BaseModel):
    # Input shape the MiewID v3/v4 architecture expects. Also the shape used
    # at training time by wbia-plugin-miew-id (see
    # wbia_miew_id/datasets/transforms.py:get_valid_transforms). Kept as a
    # class-level constant so the preprocessing config and any future
    # model_info reporting stay in sync.
    IMAGE_SIZE = (440, 440)

    def __init__(self):
        self.model = None
        self.model_info = {}
        self.device = "cuda"
        self.preprocess = None
        self.use_checkpoint = False

    def load(self, model_path: str = "", device: str = "cuda", **kwargs) -> None:
        self.device = device
        checkpoint_path = kwargs.get('checkpoint_path')
        
        # Check if we should use checkpoint loading
        if checkpoint_path:
            self.use_checkpoint = True
            local_checkpoint_path = get_checkpoint_path(checkpoint_path)
            # Remove checkpoint_path from kwargs to avoid duplicate parameter error
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'checkpoint_path'}
            version = kwargs.get('version', 3)
            if isinstance(version, (int, float)) and version >= 4:
                self._load_standalone(local_checkpoint_path, device)
            else:
                self._load_from_checkpoint(local_checkpoint_path, device, **filtered_kwargs)
        else:
            self.use_checkpoint = False
            self._load_from_huggingface(device, **kwargs)

        # Initialize preprocessing transforms.
        # Matches wbia-plugin-miew-id's get_valid_transforms exactly so
        # embeddings extracted here agree numerically with embeddings
        # extracted by the legacy WBIA MiewID pipeline for the same input
        # image+bbox. albumentations 1.3.1's Normalize defaults are ImageNet
        # mean/std with max_pixel_value=255.0, equivalent to torchvision's
        # ToTensor + Normalize(mean,std) chain modulo the bilinear resize
        # implementation (PIL vs cv2.INTER_LINEAR) — and the model was
        # trained against cv2 bilinear, so albumentations is the correct
        # match. Bumping to torchvision was a ~3° angular drift per
        # embedding, large enough to reshuffle top-N matches.
        self.preprocess = albumentations.Compose([
            albumentations.Resize(*self.IMAGE_SIZE),
            albumentations.Normalize(),
            ToTensorV2(),
        ])

        self.model_info = {
            'model_type': 'miewid',
            'device': device,
            'imgsz': kwargs.get('imgsz', 440),
            'version': kwargs.get('version', 3),
            'checkpoint_path': checkpoint_path,
            'use_checkpoint': self.use_checkpoint
        }

    def _load_from_huggingface(self, device: str, **kwargs) -> None:
        """Load model from HuggingFace Hub."""
        if kwargs.get('version', 3) == 3:
            model_tag = f"conservationxlabs/miewid-msv3"
        else:
            model_tag = f"conservationxlabs/miewid-msv2"
            
        self.model = AutoModel.from_pretrained(model_tag, trust_remote_code=True)
        self.model.eval()
        self.model.to(device)
        logger.info(f"Loaded MiewID model from HuggingFace: {model_tag}")

    def _apply_checkpoint(self, model: nn.Module, checkpoint_path: str, device: str, strict: bool = True) -> None:
        """Load checkpoint weights into a model and move to device."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint, strict=strict)
            model.eval()
            model.to(device)
            try:
                model.device = torch.device(device)
            except AttributeError:
                pass  # HuggingFace models have read-only .device property
            self.model = model
            logger.info(f"Loaded MiewID model from checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model checkpoint: {str(e)}"
            )

    def _load_from_checkpoint(self, checkpoint_path: str, device: str, **kwargs) -> None:
        """Load HuggingFace base model and overlay checkpoint weights."""
        if kwargs.get('version', 3) == 3:
            model_tag = "conservationxlabs/miewid-msv3"
        else:
            model_tag = "conservationxlabs/miewid-msv2"
        model = AutoModel.from_pretrained(model_tag, trust_remote_code=True)
        self._apply_checkpoint(model, checkpoint_path, device, strict=False)

    def _load_standalone(self, checkpoint_path: str, device: str) -> None:
        """Load model from checkpoint using standalone timm-based architecture."""
        model = MiewIdNet()
        self._apply_checkpoint(model, checkpoint_path, device, strict=True)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self.model_info
    
    def predict(self, **kwargs):
        raise HTTPException(status_code=400, detail=f"MiewID should not be used for prediction")

    def extract_embeddings(self, image_bytes: bytes, bbox: Optional[Tuple[int, int, int, int]] = None, theta: float = 0.0) -> np.ndarray:
        """
        Extract embeddings from an image using optional bounding box and rotation.

        Args:
            image_bytes: Image data as bytes
            bbox: Optional tuple of (x, y, width, height). If None, uses full image
            theta: Rotation angle in radians

        Returns:
            Numpy array containing the embeddings
        """
        try:
            # Load image as RGB HWC numpy. We use get_chip_from_img (the same
            # helper wbia-plugin-miew-id uses at training time) so the chip
            # fed to the model matches the training-time convention exactly.
            # The previous PIL crop-then-rotate(-theta) produced a different
            # chip whenever theta != 0, silently degrading embeddings for
            # every rotated annotation.
            image_np = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))

            if bbox is not None:
                processed_np = get_chip_from_img(image_np, list(bbox), float(theta))
            elif theta != 0.0:
                # No bbox but rotation requested: rotate the full image using
                # the same canonical helper by passing a full-frame bbox.
                h, w = image_np.shape[:2]
                processed_np = get_chip_from_img(image_np, [0, 0, w, h], float(theta))
            else:
                processed_np = image_np

            augmented = self.preprocess(image=processed_np)
            input_tensor = augmented["image"]
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                embeddings = self.model(input_batch)
            
            # Convert to numpy and return
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting embeddings: {str(e)}")
