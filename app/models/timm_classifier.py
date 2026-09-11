"""Config-driven classifier for any single-head timm architecture.

Sibling to EfficientNetModel, but nothing about the architecture, pooling,
preprocessing or label semantics is baked in: a new classifier is a
`model_config.json` entry, not a new module.

Written for DeepFaune v1.5 (`vit_large_patch16_dinov3.lvd1689m`, 40 European
species, 224px, `global_pool="token"`), whose defining hazard shapes this
module: `global_pool="token"` and `"avg"` yield state dicts with *identical*
keys, so loading those weights under the wrong pooling succeeds strictly --
0 missing, 0 unexpected -- and then classifies off the wrong feature. Measured
on the real checkpoint: top-1 differs on 3 of 4 crops, probabilities up to
0.254 apart, nothing raised. Hence: pooling is explicit config, the state-dict
load is strict, and unknown config keys are rejected rather than ignored,
because `app/main.py` forwards every config key into `load(**kwargs)` and a
typo would otherwise be swallowed into that same silent fallback.

Design: docs/plans/2026-09-11-timm-classifier-design.md
"""
import io
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import timm
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, transforms

from app.models.base_model import BaseModel
from app.utils.checkpoint_utils import get_checkpoint_path
from app.utils.label_parsing import parse_class_label

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INTERPOLATIONS = {
    "bicubic": InterpolationMode.BICUBIC,
    "bilinear": InterpolationMode.BILINEAR,
    "nearest": InterpolationMode.NEAREST,
}
STATE_DICT_KEYS = ("state_dict", "state", "model")
LABEL_MODES = ("species", "viewpoint", "compound")


class TimmClassifierModel(BaseModel):
    """Generic timm classifier with config-driven pooling, preprocessing and
    label semantics."""

    def __init__(self):
        self.model = None
        self.device = None
        self.model_id = ""
        self.model_arch = None
        self.global_pool = None
        self.img_size = 224
        self.interpolation = "bicubic"
        self.mean = list(IMAGENET_MEAN)
        self.std = list(IMAGENET_STD)
        self.square_crop = False
        self.multi_label = False
        self.threshold = 0.5
        self.label_map: Dict[int, str] = {}
        self.label_mode = "species"
        self.sentinel_prefixes: Optional[List[str]] = None
        self.transforms = None

    # ------------------------------------------------------------------ load

    def load(self, model_path: str = "", device: str = "cpu", model_id: str = "",
             checkpoint_path: Optional[str] = None,
             model_arch: Optional[str] = None,
             img_size: int = 224,
             global_pool: Optional[str] = None,
             labels: Optional[Sequence[str]] = None,
             label_map: Optional[Dict] = None,
             label_mode: str = "species",
             sentinel_prefixes: Optional[List[str]] = None,
             state_dict_key: Optional[str] = None,
             strip_prefix: Optional[str] = None,
             interpolation: str = "bicubic",
             mean: Optional[Sequence[float]] = None,
             std: Optional[Sequence[float]] = None,
             square_crop: bool = False,
             multi_label: bool = False,
             threshold: float = 0.5,
             **kwargs) -> None:
        # `model_path` is always forwarded by ModelHandler.load_model and is
        # meaningless here; every *other* leftover key is an operator typo.
        # Swallowing one would reintroduce the silent-fallback failure this
        # module exists to prevent. `_`-prefixed keys are comments, as in the
        # `_note` convention model_config.json already uses.
        unknown = sorted(k for k in kwargs if not k.startswith("_"))
        if unknown:
            raise ValueError(
                f"Unknown config key(s) for timm-classifier '{model_id}': "
                f"{', '.join(unknown)}. Check for a typo -- an ignored key "
                f"here silently changes model behaviour."
            )

        if not checkpoint_path:
            raise ValueError("checkpoint_path is required for timm-classifier models")
        if not model_arch:
            raise ValueError("model_arch is required for timm-classifier models")
        if label_mode not in LABEL_MODES:
            raise ValueError(
                f"label_mode must be one of {LABEL_MODES}, got {label_mode!r}")
        if interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"interpolation must be one of {tuple(INTERPOLATIONS)}, "
                f"got {interpolation!r}")
        if not isinstance(img_size, int) or isinstance(img_size, bool) or img_size <= 0:
            raise ValueError(f"img_size must be a positive int, got {img_size!r}")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) \
                or not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {threshold!r}")
        self.mean = self._validate_channel_stat(mean, IMAGENET_MEAN, "mean")
        self.std = self._validate_channel_stat(std, IMAGENET_STD, "std")

        self.model_id = model_id
        self.device = torch.device(device)
        self.model_arch = model_arch
        self.global_pool = global_pool
        self.img_size = img_size
        self.interpolation = interpolation
        self.square_crop = bool(square_crop)
        self.multi_label = bool(multi_label)
        self.threshold = float(threshold)
        self.label_mode = label_mode
        self.sentinel_prefixes = sentinel_prefixes
        self.label_map = self._build_label_map(labels, label_map)
        num_classes = len(self.label_map)

        actual_path = get_checkpoint_path(checkpoint_path)
        # map_location='cpu', never the target device: torch.load materialises
        # every tensor *before* load_state_dict copies into the already
        # allocated parameters. Straight to CUDA, a 1.2 GB checkpoint peaks at
        # 2315 MiB of VRAM against a 1157 MiB steady state.
        checkpoint = torch.load(actual_path, map_location="cpu", weights_only=False)
        state = self._extract_state_dict(checkpoint, state_dict_key)

        ckpt_classes = self._checkpoint_num_classes(checkpoint)
        if ckpt_classes is not None and ckpt_classes != num_classes:
            raise ValueError(
                f"Checkpoint declares num_classes={ckpt_classes} but "
                f"{num_classes} labels were configured for '{model_id}'")

        if strip_prefix:
            state = {k[len(strip_prefix):] if k.startswith(strip_prefix) else k: v
                     for k, v in state.items()}

        create_kwargs = {"pretrained": False, "num_classes": num_classes}
        if global_pool is not None:
            create_kwargs["global_pool"] = global_pool
        self.model = timm.create_model(model_arch, **create_kwargs)
        # strict=True is deliberate. A lenient load is how a wrong-shaped or
        # wrong-prefixed checkpoint ships as a silently wrong classifier.
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

        self._setup_transforms()
        logger.info(
            "Loaded timm-classifier '%s': arch=%s global_pool=%s img_size=%d "
            "classes=%d label_mode=%s square_crop=%s",
            model_id, model_arch, global_pool, img_size, num_classes,
            label_mode, self.square_crop)

    @staticmethod
    def _validate_channel_stat(value, default, name) -> List[float]:
        if value is None:
            return list(default)
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) \
                or len(value) != 3:
            raise ValueError(f"{name} must be a sequence of 3 floats, got {value!r}")
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            raise ValueError(f"{name} must contain numbers, got {value!r}")

    @staticmethod
    def _build_label_map(labels, label_map) -> Dict[int, str]:
        if (labels is None) == (label_map is None):
            raise ValueError(
                "Provide exactly one of labels / label_map "
                "(a list of class names, or an {index: name} mapping)")

        if labels is not None:
            if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
                raise ValueError(f"labels must be a list of strings, got {labels!r}")
            resolved = dict(enumerate(labels))
        else:
            if not isinstance(label_map, dict):
                raise ValueError(f"label_map must be a mapping, got {label_map!r}")
            resolved = {}
            for raw_key, value in label_map.items():
                try:
                    key = int(raw_key)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"label_map keys must be integer indices, got {raw_key!r}")
                if key in resolved:
                    # "1" and "01" both coerce to 1; one label would win silently.
                    raise ValueError(
                        f"label_map has duplicate index {key} after coercion "
                        f"(collision on key {raw_key!r})")
                resolved[key] = value

        if not resolved:
            raise ValueError("labels / label_map must not be empty")
        expected = set(range(len(resolved)))
        if set(resolved) != expected:
            raise ValueError(
                f"label indices must be exactly 0..{len(resolved) - 1}, got "
                f"{sorted(resolved)}")
        for index, value in resolved.items():
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"label at index {index} must be a non-empty string, "
                    f"got {value!r}")
        return resolved

    @staticmethod
    def _extract_state_dict(checkpoint, state_dict_key) -> Dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dict-like object")
        if state_dict_key is not None:
            if state_dict_key not in checkpoint:
                raise ValueError(
                    f"Configured state_dict_key '{state_dict_key}' is not in the "
                    f"checkpoint (keys: {sorted(checkpoint)})")
            return checkpoint[state_dict_key]
        for key in STATE_DICT_KEYS:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        # Raw fallback: only if this really is a tensor state dict, otherwise a
        # metadata blob would reach load_state_dict as an opaque error.
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
        raise ValueError(
            f"Could not find a state dict in the checkpoint: no {STATE_DICT_KEYS} "
            f"key and the top level is not a tensor state dict "
            f"(keys: {sorted(checkpoint)})")

    @staticmethod
    def _checkpoint_num_classes(checkpoint) -> Optional[int]:
        args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
        if isinstance(args, dict) and isinstance(args.get("num_classes"), int):
            return args["num_classes"]
        return None

    def _setup_transforms(self) -> None:
        self.transforms = transforms.Compose([
            transforms.Resize(size=(self.img_size, self.img_size),
                              interpolation=INTERPOLATIONS[self.interpolation],
                              max_size=None, antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(self.mean),
                                 std=torch.tensor(self.std)),
        ])

    # ------------------------------------------------------------- inference

    def _preprocess_image(self, image_bytes: bytes,
                          bbox: Optional[Tuple[int, int, int, int]] = None,
                          theta: float = 0.0) -> torch.Tensor:
        """Decode -> crop -> rotate -> resize -> normalise.

        Crop-then-rotate matches EfficientNetModel's ordering, so theta means
        the same thing across classify models.
        """
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")

        if bbox is not None:
            x, y, w, h = (int(v) for v in bbox)
            if w <= 0 or h <= 0:
                raise ValueError(f"bbox must have positive width and height, got {bbox}")
            x1, y1, x2, y2 = x, y, x + w, y + h
            if self.square_crop:
                # Upstream DeepFaune cropSquareCVtoPIL: grow the short side by
                # int(diff/2) each way, then clip -- so a crop near a border
                # stays a rectangle rather than running off the image.
                if w > h:
                    expand = int((w - h) / 2)
                    y1, y2 = y1 - expand, y2 + expand
                elif h > w:
                    expand = int((h - w) / 2)
                    x1, x2 = x1 - expand, x2 + expand
            width, height = image.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(x2, width), min(y2, height)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"bbox {bbox} does not overlap the image")
            image = image.crop((x1, y1, x2, y2))

        if theta:
            image = image.rotate(float(np.degrees(theta)))

        return self.transforms(image).unsqueeze(0)

    def _label_fields(self, label: str) -> Tuple[Optional[str], Optional[str]]:
        """Split a label into (species, viewpoint) per configured semantics.

        `/pipeline/` promotes predictions[0]['species'] to the annotation's
        iaClass, so a plain species classifier must say so explicitly; the
        compound parser reads a colon-free label as a viewpoint.
        """
        if self.label_mode == "species":
            return label, None
        if self.label_mode == "viewpoint":
            return None, label
        return parse_class_label(label, compound_labels=True,
                                 sentinel_prefixes=self.sentinel_prefixes)

    def predict(self, image_bytes: bytes, bbox: Optional[List[int]] = None,
                theta: float = 0.0, **kwargs) -> Dict[str, Any]:
        bbox_tuple = tuple(bbox) if bbox is not None else None
        tensor = self._preprocess_image(image_bytes, bbox_tuple, theta).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            if self.multi_label:
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                indices = np.where(probs > self.threshold)[0]
            else:
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                indices = [int(np.argmax(probs))]

        results = []
        for i in indices:
            label = self.label_map[int(i)]
            species, viewpoint = self._label_fields(label)
            results.append({
                "label": label,
                "index": int(i),
                "probability": float(probs[i]),
                "species": species,
                "viewpoint": viewpoint,
            })
        results.sort(key=lambda r: r["probability"], reverse=True)

        return {
            "model_id": self.model_id,
            "predictions": results,
            "all_probabilities": probs.tolist(),
            "threshold": self.threshold,
            "bbox": bbox,
            "theta": theta,
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "timm-classifier",
            "model_architecture": self.model_arch,
            "global_pool": self.global_pool,
            "image_size": self.img_size,
            "interpolation": self.interpolation,
            "square_crop": self.square_crop,
            "threshold": self.threshold,
            "num_classes": len(self.label_map),
            "label_map": self.label_map,
            "label_mode": self.label_mode,
            "multi_label": self.multi_label,
            "device": str(self.device) if self.device else None,
        }
