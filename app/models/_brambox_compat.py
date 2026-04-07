"""Brambox v3 compatibility shim for LightNet.

LightNet (WBIA fork) imports from `brambox.boxes`, which was removed in
brambox v4+. This module injects a minimal `brambox.boxes` subpackage at
runtime so LightNet can import without requiring an ancient brambox version.

Call `ensure_brambox_compat()` before `import lightnet`.
"""

import importlib
import sys
import types


class Detection:
    """Detection object matching the brambox v3 API used by LightNet."""

    def __init__(self):
        self.x_top_left = 0.0
        self.y_top_left = 0.0
        self.width = 0.0
        self.height = 0.0
        self.confidence = 0.0
        self.class_label = ''

    def serialize(self, return_dict=False):
        if return_dict:
            return {
                'x_top_left': self.x_top_left,
                'y_top_left': self.y_top_left,
                'width': self.width,
                'height': self.height,
                'confidence': self.confidence,
                'class_label': self.class_label,
            }
        return [
            self.x_top_left, self.y_top_left,
            self.width, self.height,
            self.confidence, self.class_label,
        ]

    def __repr__(self):
        return (
            f"Detection({self.class_label}: "
            f"[{self.x_top_left:.1f}, {self.y_top_left:.1f}, "
            f"{self.width:.1f}, {self.height:.1f}] "
            f"conf={self.confidence:.3f})"
        )


class CropModifier:
    """Minimal stub — not needed for inference."""

    def __init__(self, max_value=float('Inf'), intersection_threshold=0.001):
        self.max_value = max_value
        self.intersection_threshold = intersection_threshold


def modify(annotations, modifiers):
    """No-op stub — annotation modification not needed for inference."""
    return annotations


def parse(*args, **kwargs):
    """Stub for brambox.boxes.parse — only needed for training datasets."""
    raise NotImplementedError(
        "brambox.boxes.parse requires brambox v3 which is not available. "
        "This shim only supports LightNet inference, not training."
    )


def ensure_brambox_compat():
    """Inject brambox.boxes into sys.modules if not already present."""
    if 'brambox.boxes' in sys.modules:
        return

    # Ensure brambox itself is importable (v4+ is fine, we just need the namespace)
    try:
        importlib.import_module('brambox')
    except ImportError:
        # Create a minimal brambox package too
        brambox_mod = types.ModuleType('brambox')
        brambox_mod.__path__ = []
        sys.modules['brambox'] = brambox_mod

    # Create brambox.boxes module with Detection, CropModifier, modify
    boxes_mod = types.ModuleType('brambox.boxes')
    boxes_mod.__path__ = []
    boxes_mod.Detection = Detection
    boxes_mod.CropModifier = CropModifier
    boxes_mod.modify = modify
    boxes_mod.parse = parse
    sys.modules['brambox.boxes'] = boxes_mod

    # Create brambox.boxes.detections.detection for the wildcard import
    det_pkg = types.ModuleType('brambox.boxes.detections')
    det_pkg.__path__ = []
    sys.modules['brambox.boxes.detections'] = det_pkg

    det_mod = types.ModuleType('brambox.boxes.detections.detection')
    det_mod.Detection = Detection
    det_mod.__all__ = ['Detection']
    sys.modules['brambox.boxes.detections.detection'] = det_mod
