"""DenseNet classifier with ensemble support and compound-label parsing.

Sibling to DenseNetOrientationModel — same checkpoint format and arch
detection, but oriented at the classify slot (top-level iaClass +
viewpoint promotion via shared parser) rather than the orientation slot.

Design: docs/plans/2026-05-16-densenet-classifier-design.md
"""
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from app.models.base_model import BaseModel
from app.utils.checkpoint_utils import get_checkpoint_path
from app.utils.label_parsing import parse_class_label

logger = logging.getLogger(__name__)


class DenseNetClassifierModel(BaseModel):
    """Ensemble-averaging DenseNet classifier for compound or
    pure-viewpoint label sets."""

    def __init__(self):
        super().__init__()
        self.models: List[nn.Module] = []
        self.label_map: Dict[int, str] = {}
        self.compound_labels: bool = False
        self.sentinel_prefixes: List[str] = ["species"]
        self.architecture: str = "densenet201"
        self.img_size: int = 224
        self.device: str = "cpu"

    def load(self, model_path: str = "", device: str = "cpu",
             model_id: str = "",
             checkpoint_path: Optional[str] = None,
             checkpoint_paths: Optional[List[str]] = None,
             img_size: int = 224,
             label_map: Optional[Dict] = None,
             compound_labels: bool = False,
             sentinel_prefixes: Optional[List[str]] = None,
             ensemble_indices: Optional[List[int]] = None,
             **kwargs) -> None:
        self.device = device
        self.img_size = img_size
        self.compound_labels = bool(compound_labels)
        if sentinel_prefixes is not None:
            self.sentinel_prefixes = list(sentinel_prefixes)

        # --- Resolve checkpoint paths ---
        if checkpoint_path and checkpoint_paths:
            raise ValueError(
                "Provide exactly one of checkpoint_path / checkpoint_paths"
            )
        if not checkpoint_path and not checkpoint_paths:
            raise ValueError(
                "Either checkpoint_path or checkpoint_paths is required"
            )
        paths = checkpoint_paths or [checkpoint_path]

        # --- Apply ensemble_indices subset selection ---
        if ensemble_indices is not None:
            for i in ensemble_indices:
                if not (0 <= i < len(paths)):
                    raise ValueError(
                        f"ensemble_indices contains out-of-range index "
                        f"{i} for checkpoint_paths of length {len(paths)}"
                    )
            paths = [paths[i] for i in ensemble_indices]

        # --- Load all checkpoints ---
        checkpoints = []
        for p in paths:
            actual = get_checkpoint_path(p)
            ck = torch.load(actual, map_location=device, weights_only=False)
            checkpoints.append(ck)

        # --- Determine num_classes and arch from the first checkpoint ---
        first_state = _state_dict_of(checkpoints[0])
        num_classes, self.architecture = _detect_arch_and_num_classes(first_state)

        # --- Validate every member matches num_classes ---
        for i, ck in enumerate(checkpoints[1:], start=1):
            n_i, _ = _detect_arch_and_num_classes(_state_dict_of(ck))
            if n_i != num_classes:
                raise ValueError(
                    f"Ensemble checkpoint {i} has num_classes={n_i}, "
                    f"first checkpoint has num_classes={num_classes}. "
                    f"All members must share num_classes."
                )

        # --- Resolve label map ---
        if label_map is not None:
            coerced = {int(k): v for k, v in label_map.items()}
            expected = set(range(num_classes))
            actual_keys = set(coerced.keys())
            if actual_keys != expected:
                raise ValueError(
                    f"label_map keys must be exactly {sorted(expected)}; "
                    f"got {sorted(actual_keys)}"
                )
            self.label_map = coerced
        else:
            classes_lists = [ck.get("classes") for ck in checkpoints]
            if any(c is None for c in classes_lists):
                raise ValueError(
                    "Every checkpoint must carry a 'classes' list when no "
                    "explicit label_map is provided"
                )
            first_classes = classes_lists[0]
            if len(first_classes) != num_classes:
                raise ValueError(
                    f"Checkpoint 'classes' list has length "
                    f"{len(first_classes)} but classifier head has "
                    f"num_classes={num_classes} — stale metadata"
                )
            for i, c in enumerate(classes_lists[1:], start=1):
                if list(c) != list(first_classes):
                    raise ValueError(
                        f"Ensemble checkpoint {i} 'classes' differs from "
                        f"first checkpoint. Averaging by index requires "
                        f"identical class order across all members."
                    )
            self.label_map = {i: c for i, c in enumerate(first_classes)}

        # --- Validate compound_labels vs label content ---
        any_colon = any(":" in lbl for lbl in self.label_map.values())
        if any_colon and not self.compound_labels:
            raise ValueError(
                "label_map contains ':' but compound_labels=False. "
                "Set compound_labels: true to enable species:viewpoint "
                "parsing, or fix the labels."
            )
        if self.compound_labels and not any_colon:
            logger.warning(
                f"Model '{model_id}': compound_labels=true but no label "
                f"contains ':' — every emitted prediction will have "
                f"species=None, viewpoint=label."
            )

        # --- Build model instances and load weights ---
        # NOTE: NO nn.Softmax wrap during load. Softmax is applied once
        # per ensemble member inside predict(). Wrapping here would cause
        # double-softmax once predict() also applies softmax.
        self.models = []
        for ck in checkpoints:
            backbone = _build_backbone(self.architecture, num_classes)
            state = _state_dict_of(ck)
            cleaned = _strip_module_prefix(state)
            backbone.load_state_dict(cleaned, strict=False)
            backbone.to(device).eval()
            self.models.append(backbone)

        logger.info(
            f"Loaded DenseNetClassifierModel '{model_id}': "
            f"{len(self.models)}-member ensemble, num_classes={num_classes}, "
            f"compound_labels={self.compound_labels}"
        )

    def predict(self, image_bytes: bytes,
                bbox: Optional[List[int]] = None,
                theta: float = 0.0) -> Dict[str, Any]:
        inputs = self._preprocess(image_bytes, bbox, theta)
        summed = None
        with torch.no_grad():
            for m in self.models:
                probs = torch.softmax(m(inputs), dim=-1)
                summed = probs if summed is None else summed + probs
        avg = summed / len(self.models)
        return self._format_output(avg)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "densenet-classifier",
            "device": self.device,
            "img_size": self.img_size,
            "num_classes": len(self.label_map),
            "label_map": self.label_map,
            "compound_labels": self.compound_labels,
            "ensemble_size": len(self.models),
            "architecture": self.architecture,
        }

    def _preprocess(self, image_bytes, bbox, theta):
        # Reuse DenseNetOrientationModel's preprocessing as the single
        # source of truth. Import locally to avoid circular dependency.
        from app.models.densenet_orientation import DenseNetOrientationModel
        return DenseNetOrientationModel._preprocess_tensor(
            image_bytes, bbox, theta, self.img_size, self.device
        )

    def _format_output(self, avg: torch.Tensor) -> Dict[str, Any]:
        k = min(3, avg.shape[-1])
        top_probs, top_idxs = torch.topk(avg, k, dim=-1)
        top_probs = top_probs[0].tolist()
        top_idxs = top_idxs[0].tolist()

        predictions = []
        for prob, idx in zip(top_probs, top_idxs):
            label = self.label_map[int(idx)]
            species, viewpoint = parse_class_label(
                label, self.compound_labels, self.sentinel_prefixes
            )
            predictions.append({
                "label": label,
                "probability": float(prob),
                "index": int(idx),
                "species": species,
                "viewpoint": viewpoint,
            })

        top = predictions[0]
        return {
            "class": top["label"],
            "probability": top["probability"],
            "class_id": top["index"],
            "predictions": predictions,
        }


def _state_dict_of(checkpoint):
    if isinstance(checkpoint, dict) and "state" in checkpoint:
        return checkpoint["state"]
    return checkpoint


def _strip_module_prefix(state_dict):
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v
    return cleaned


def _detect_arch_and_num_classes(state):
    cleaned = _strip_module_prefix(state)
    keys = set(cleaned.keys())
    # Check DenseNet first — denseblock/denselayer keys are unique to DenseNet
    if any("denseblock" in k or "denselayer" in k for k in keys):
        arch = "densenet201"
    # HRNet has stage2/stage3/stage4 keys which DenseNet does not
    elif any("stage2" in k or "stage3" in k or "stage4" in k for k in keys):
        arch = "hrnet_w32"
    else:
        # Fallback: infer from classifier feature dim
        for k in keys:
            if "classifier.weight" in k:
                n_features = cleaned[k].shape[1]
                if n_features == 2048:
                    arch = "hrnet_w32"
                    break
                if n_features == 1920:
                    arch = "densenet201"
                    break
        else:
            arch = "densenet201"

    classifier_keys = [k for k in cleaned if "classifier.weight" in k]
    if not classifier_keys:
        raise ValueError("No classifier.weight key in checkpoint state-dict")
    n = cleaned[classifier_keys[-1]].shape[0]
    return n, arch


def _build_backbone(architecture, num_classes):
    if architecture == "hrnet_w32":
        import timm
        return timm.create_model("hrnet_w32", pretrained=False, num_classes=num_classes)
    from torchvision import models
    model = models.densenet201(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model
