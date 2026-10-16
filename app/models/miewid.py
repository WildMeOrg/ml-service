from .base_model import BaseModel
import contextlib
import logging
import pickle
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
import re
import threading
from app.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel, VITB16
from app.utils.checkpoint_utils import get_checkpoint_path, sha256_file
from app.utils.helpers import get_chip_from_img, decode_image_rgb

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


# DINOv3 MiewID runs at 448: ViT-B/16 needs a multiple of 16, so the
# EfficientNet's 440 is not an option (the patch grid would not tile it).
DINOV3_IMAGE_SIZE = (448, 448)


class DinoV3MiewIdNet(nn.Module):
    """MiewID with a DINOv3 ViT backbone.

    Matches the training repo's MiewIdNet `dinov3:` branch
    (wbia_miew_id/models/model.py, MiewID v5 DINOv3 run):
    concat(CLS, GeM over the patch grid with CLS and registers skipped)
    -> BatchNorm1d. No projection, no L2 normalisation.
    """

    def __init__(self, config: DINOv3ViTConfig = VITB16,
                 input_size: Tuple[int, int] = DINOV3_IMAGE_SIZE):
        super().__init__()
        if input_size[0] % config.patch_size or input_size[1] % config.patch_size:
            raise ValueError(
                f"DINOv3 input size {input_size} must be a multiple of {config.patch_size}")
        self.backbone = DINOv3ViTModel(config)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.n_skip = 1 + config.num_register_tokens
        self.grid = (input_size[0] // config.patch_size, input_size[1] // config.patch_size)
        self.final_in_features = config.hidden_size * 2
        self.pooling = GeM()
        self.bn = nn.BatchNorm1d(self.final_in_features)

    def forward_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Check the input, not just the token count: the patch conv floors,
        # so 449x448 and 224x896 both yield 784 patches, and the second would
        # be reshaped onto the wrong 28x28 grid without any error.
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(
                f"DINOv3 MiewID expects {self.input_size[0]}x{self.input_size[1]} input, "
                f"got {x.shape[-2]}x{x.shape[-1]}")
        tokens = self.backbone(x)
        b, n, c = tokens.shape
        h, w = self.grid
        if n != self.n_skip + h * w:
            raise ValueError(f"expected {self.n_skip + h * w} tokens, got {n}")
        cls = tokens[:, 0]
        patch = tokens[:, self.n_skip:]
        grid = patch.transpose(1, 2).reshape(b, c, h, w)
        concat = torch.cat([cls, self.pooling(grid).view(b, -1)], dim=1)
        return {"tokens": tokens, "concat": concat, "embedding": self.bn(concat)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_stages(x)["embedding"]


BACKBONE_EFFNET = "efficientnetv2_rw_m"
BACKBONE_DINOV3_VITB16 = "dinov3_vitb16"
SUPPORTED_BACKBONES = (BACKBONE_EFFNET, BACKBONE_DINOV3_VITB16)

# Top-level keys of a training resume checkpoint (checkpoint_latest.pth).
_RESUME_KEYS = {"model_state_dict", "optimizer", "scheduler", "epoch", "best_score"}


def describe_checkpoint(sd: Dict[str, Any]) -> str:
    """Which standalone MiewID backbone a state dict was saved from."""
    if "backbone.embeddings.register_tokens" in sd:
        return BACKBONE_DINOV3_VITB16
    if "backbone.conv_stem.weight" in sd:
        return BACKBONE_EFFNET
    return "unrecognised (first keys: %s)" % ", ".join(list(sd)[:3])


def load_standalone_state_dict(path: str, backbone: str) -> Dict[str, torch.Tensor]:
    """Load and validate a standalone MiewID checkpoint for `backbone`.

    Standalone paths only (EfficientNet v4+ and DINOv3). The legacy HF
    overlay keeps its strict=False tolerance and never comes through here.
    """
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as e:
        raise ValueError(
            f"{path} is not a plain tensor state dict (torch.load weights_only "
            f"refused it: {e}). A training resume checkpoint is not servable; "
            f"ship model_best.bin.") from e
    if not isinstance(sd, dict):
        raise ValueError(f"{path}: expected a state dict, got {type(sd).__name__}")
    resume_keys = _RESUME_KEYS & set(sd)
    if resume_keys:
        raise ValueError(
            f"{path} is a training resume checkpoint (keys {sorted(resume_keys)}); "
            f"ship model_best.bin instead.")
    non_tensor = [k for k, v in sd.items() if not isinstance(v, torch.Tensor)]
    if non_tensor:
        raise ValueError(f"{path}: non-tensor values under {non_tensor[:3]}")
    head = [k for k in sd if k.startswith("final.")]
    if head:
        raise ValueError(
            f"{path} carries an ArcFace head ({len(head)} 'final.*' keys). Serving "
            f"checkpoints must not include it; re-export without 'final.*'.")
    looks_like = describe_checkpoint(sd)
    if looks_like != backbone:
        raise ValueError(
            f"checkpoint {path} does not match configured backbone '{backbone}': "
            f"it looks like {looks_like}.")
    if "bn.weight" not in sd:
        raise ValueError(f"{path}: missing 'bn.weight'")
    if backbone == BACKBONE_DINOV3_VITB16:
        want = (2 * VITB16.hidden_size,)
        if tuple(sd["bn.weight"].shape) != want:
            raise ValueError(
                f"{path}: bn.weight has shape {tuple(sd['bn.weight'].shape)}, "
                f"expected {want} for {backbone}")
    return sd


_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


def build_miewid_transform(image_size: Tuple[int, int]) -> albumentations.Compose:
    """wbia-plugin-miew-id's get_valid/test_transforms at `image_size`.

    albumentations 1.3.1 Normalize defaults are ImageNet mean/std with
    max_pixel_value=255. The models were trained against cv2 bilinear
    resize, which is what albumentations uses; torchvision's PIL resize
    measured ~3 degrees of angular drift per embedding.
    """
    return albumentations.Compose([
        albumentations.Resize(*image_size),
        albumentations.Normalize(),
        ToTensorV2(),
    ])


def resolve_precision(backbone_name: Optional[str], device: str) -> str:
    """fp16 autocast for DINOv3 on CUDA (as training evaluated); fp32 otherwise.

    EfficientNet and HF stay fp32: changing them would move embeddings
    against every existing Wildbook index.
    """
    if backbone_name == BACKBONE_DINOV3_VITB16 and torch.device(device).type == "cuda":
        return "fp16_autocast"
    return "fp32"


def _parse_load_options(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the config fields this design adds, before anything loads."""
    backbone = kwargs.get("backbone")
    checkpoint_path = kwargs.get("checkpoint_path")
    if backbone is not None and backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported MiewID backbone {backbone!r}; supported: {list(SUPPORTED_BACKBONES)}")
    if backbone is not None and not checkpoint_path:
        raise ValueError(f"MiewID backbone {backbone!r} requires checkpoint_path")

    sha = kwargs.get("checkpoint_sha256")
    if sha is not None:
        if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
            raise ValueError(f"checkpoint_sha256 must be 64 hex characters, got {sha!r}")
        if not checkpoint_path:
            raise ValueError("checkpoint_sha256 is set but checkpoint_path is not")
        sha = sha.lower()

    # JSON null means "not set", consistently with "backbone": null.
    for name in ("imgsz", "embedding_dim"):
        value = kwargs.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{name} must be an int, got {value!r}")

    return {
        "backbone": backbone,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha,
        "imgsz": kwargs.get("imgsz"),
        "embedding_dim": kwargs.get("embedding_dim"),
    }


LOAD_PATH_DINOV3 = "dinov3"
LOAD_PATH_EFFNET = "efficientnet"
LOAD_PATH_HF_OVERLAY = "hf_overlay"
LOAD_PATH_HF_HUB = "hf_hub"


def resolve_load_path(opts: Dict[str, Any], version: Any) -> str:
    """Which loader `MiewidModel.load` takes. `backbone` first; without it,
    today's legacy dispatch exactly (spec, legacy table). Pure: no I/O."""
    backbone = opts["backbone"]
    checkpoint_path = opts["checkpoint_path"]
    if backbone == BACKBONE_DINOV3_VITB16:
        return LOAD_PATH_DINOV3
    if backbone == BACKBONE_EFFNET or (
            checkpoint_path and isinstance(version, (int, float)) and version >= 4):
        return LOAD_PATH_EFFNET
    if checkpoint_path:
        return LOAD_PATH_HF_OVERLAY
    return LOAD_PATH_HF_HUB


class MiewidModel(BaseModel):
    # Backward-compatible alias for the EfficientNet/HF input size. Nothing
    # in this service reads it any more: every instance carries its own
    # `image_size` (448 for DINOv3), and /explain uses `model.preprocess`.
    IMAGE_SIZE = (440, 440)
    # Class-level defaults: MagicMock(spec=MiewidModel) stubs expose them, and
    # tests that build instances with MiewidModel.__new__ (skipping __init__)
    # still get a working fp32 extract_embeddings.
    supports_pairx = True
    backbone_name: Optional[str] = None
    precision = "fp32"
    embedding_dim: Optional[int] = None
    checkpoint_sha256: Optional[str] = None
    image_size = IMAGE_SIZE

    def __init__(self):
        self.model = None
        self.model_info = {}
        self.device = "cuda"
        self.use_checkpoint = False
        # Serializes forwards through this instance across routers.
        #
        # PAIR-X (/explain) captures its feature map with a forward hook
        # registered on a submodule of this very object (pairx/core.py:23-43),
        # and that hook fires for EVERY forward through the submodule, from any
        # thread, until it is removed. A /extract or /pipeline forward landing
        # in that window -- same model id, same instance, under no_grad --
        # overwrites the captured tensor with one that has no grad_fn, and
        # PAIR-X's backward then dies with "element 0 of tensors does not
        # require grad".
        #
        # A plain Lock is sufficient: run_pairx_locked acquires once and
        # run_pairx's OOM recursion stays inside that single acquisition,
        # never reacquiring. Nothing takes this lock twice on one thread.
        self.inference_lock = threading.Lock()
        self.image_size = self.IMAGE_SIZE
        self.preprocess = build_miewid_transform(self.image_size)
        self.backbone_name = None
        self.embedding_dim = None
        self.precision = "fp32"
        self.checkpoint_sha256 = None
        self.supports_pairx = True

    def load(self, model_path: str = "", device: str = "cuda", **kwargs) -> None:
        opts = _parse_load_options(kwargs)
        self.device = device
        # Reset everything a previous load() resolved, so reusing an instance
        # (DINOv3 -> EfficientNet -> HF) never inherits 448, supports_pairx
        # False, an old embedding_dim or an old digest.
        self.image_size = self.IMAGE_SIZE
        self.backbone_name = None
        self.embedding_dim = None
        self.checkpoint_sha256 = None
        self.supports_pairx = True
        checkpoint_path = opts["checkpoint_path"]
        backbone = opts["backbone"]
        version = kwargs.get('version', 3)
        # Remove checkpoint_path from kwargs to avoid duplicate parameter error
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'checkpoint_path'}

        local_checkpoint_path = None
        if checkpoint_path:
            self.use_checkpoint = True
            local_checkpoint_path = get_checkpoint_path(checkpoint_path)
            self.checkpoint_sha256 = sha256_file(local_checkpoint_path)
            expected = opts["checkpoint_sha256"]
            if expected is not None and self.checkpoint_sha256 != expected:
                raise ValueError(
                    f"checkpoint_sha256 mismatch for {checkpoint_path} (local file "
                    f"{local_checkpoint_path}): expected {expected}, got {self.checkpoint_sha256}")
        else:
            self.use_checkpoint = False

        # `backbone` first; without it, today's dispatch exactly (spec, legacy table).
        load_path = resolve_load_path(opts, version)
        if load_path == LOAD_PATH_DINOV3:
            self._load_dinov3(local_checkpoint_path, device)
        elif load_path == LOAD_PATH_EFFNET:
            self._load_standalone(local_checkpoint_path, device)
        elif load_path == LOAD_PATH_HF_OVERLAY:
            self._load_from_checkpoint(local_checkpoint_path, device, **filtered_kwargs)
        else:
            self._load_from_huggingface(device, **kwargs)

        self.precision = resolve_precision(self.backbone_name, device)
        if self.backbone_name == BACKBONE_DINOV3_VITB16 and self.precision == "fp32":
            logger.warning(
                "DINOv3 MiewID on %s runs fp32; training evaluated under CUDA fp16 "
                "autocast, so embeddings will differ slightly from the reference.", device)
        self._check_assertions(opts)

        # Matches wbia-plugin-miew-id's get_valid_transforms at this
        # instance's size, so embeddings agree with the training pipeline.
        self.preprocess = build_miewid_transform(self.image_size)

        self.model_info = {
            'model_type': 'miewid',
            'device': device,
            'imgsz': self.image_size[0],
            'version': kwargs.get('version', 3),
            'checkpoint_path': checkpoint_path,
            'use_checkpoint': self.use_checkpoint,
            'backbone': self.backbone_name,
            'image_size': list(self.image_size),
            'embedding_dim': self.embedding_dim,
            'precision': self.precision,
            'checkpoint_sha256': self.checkpoint_sha256,
            'supports_pairx': self.supports_pairx,
        }

    def _check_assertions(self, opts: Dict[str, Any]) -> None:
        """`imgsz` / `embedding_dim` in config are assertions, never settings."""
        imgsz = opts["imgsz"]
        if imgsz is not None and imgsz != self.image_size[0]:
            raise ValueError(
                f"imgsz {imgsz} does not match backbone {self.backbone_name}, "
                f"which runs at {self.image_size[0]}")
        embedding_dim = opts["embedding_dim"]
        if embedding_dim is not None:
            if self.embedding_dim is None:
                raise ValueError(
                    f"embedding_dim cannot be asserted for {self.backbone_name}: "
                    f"HuggingFace MiewID models do not expose it statically")
            if embedding_dim != self.embedding_dim:
                raise ValueError(
                    f"embedding_dim {embedding_dim} does not match the checkpoint's "
                    f"{self.embedding_dim} ({self.backbone_name})")

    def _load_from_huggingface(self, device: str, **kwargs) -> None:
        """Load model from HuggingFace Hub."""
        if kwargs.get('version', 3) == 3:
            model_tag = f"conservationxlabs/miewid-msv3"
        else:
            model_tag = f"conservationxlabs/miewid-msv2"
            
        self.backbone_name = f"hf:{model_tag}"
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
        self.backbone_name = f"hf:{model_tag}"
        model = AutoModel.from_pretrained(model_tag, trust_remote_code=True)
        self._apply_checkpoint(model, checkpoint_path, device, strict=False)

    def _install(self, model: nn.Module, device: str) -> None:
        model.eval()
        model.to(device)
        # PairX reads model.device directly (pairx/core.py:288), and plain
        # nn.Modules have no such property. _apply_checkpoint set this for
        # the standalone path before; keep doing it.
        try:
            model.device = torch.device(device)
        except AttributeError:
            pass  # HuggingFace models have a read-only .device property
        self.model = model

    def _load_standalone(self, checkpoint_path: str, device: str) -> None:
        """Standalone timm EfficientNet MiewID (v4+), strict load."""
        sd = load_standalone_state_dict(checkpoint_path, BACKBONE_EFFNET)
        model = MiewIdNet()
        model.load_state_dict(sd, strict=True)
        self.embedding_dim = int(sd["bn.weight"].shape[0])
        del sd
        self._install(model, device)
        self.backbone_name = BACKBONE_EFFNET
        logger.info(f"Loaded MiewID EfficientNet from checkpoint: {checkpoint_path}")

    def _load_dinov3(self, checkpoint_path: str, device: str) -> None:
        """DINOv3 ViT-B/16 MiewID (v5 DINOv3), strict load, built from the
        frozen local config: nothing is downloaded (the HF repo is gated)."""
        sd = load_standalone_state_dict(checkpoint_path, BACKBONE_DINOV3_VITB16)
        model = DinoV3MiewIdNet()
        model.load_state_dict(sd, strict=True)
        del sd
        self._install(model, device)
        self.backbone_name = BACKBONE_DINOV3_VITB16
        self.embedding_dim = model.final_in_features
        self.image_size = DINOV3_IMAGE_SIZE
        self.supports_pairx = False
        logger.info(f"Loaded MiewID DINOv3 from checkpoint: {checkpoint_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self.model_info
    
    def predict(self, **kwargs):
        raise HTTPException(status_code=400, detail=f"MiewID should not be used for prediction")

    def _autocast(self):
        """fp16 autocast for DINOv3 on CUDA only (resolve_precision)."""
        if self.precision == "fp16_autocast":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

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
            # decode_image_rgb applies the EXIF Orientation tag so the crop is
            # taken from the same upright frame the bbox was drawn in.
            image_np = decode_image_rgb(image_bytes)

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
            
            # Extract embeddings. The lock is held only across the forward,
            # not the decode/preprocess above it, so a slow image never blocks
            # another endpoint's inference. Autocast covers backbone AND neck
            # for DINOv3 (as training evaluated) and is a no-op otherwise.
            with self.inference_lock, torch.no_grad(), self._autocast():
                embeddings = self.model(input_batch)
            embeddings = embeddings.float()

            # An fp16 overflow shows up as inf/NaN. Never hand that to
            # Wildbook as an embedding: the except below turns it into the
            # existing 500, which /extract and /pipeline both propagate.
            if not torch.isfinite(embeddings).all():
                raise ValueError(
                    f"non-finite embedding from {self.backbone_name} "
                    f"(precision {self.precision})")
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting embeddings: {str(e)}")
