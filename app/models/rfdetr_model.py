"""RF-DETR detector handler (Roboflow ``rfdetr`` package).

Serves RF-DETR checkpoints -- the Community Fish Detector
(https://github.com/WildHackers/community-fish-detector) is the motivating one --
as a ``/predict/`` and ``/pipeline/`` detector. Boxes are axis-aligned and theta is
always 0.0, so an ``orientation_model_id`` chains after it exactly as after
``yolo-ultralytics``.

Config keys (``model_config.json``)::

    model_type:   "rfdetr"
    model_path:   path or URL to the ``.pth`` checkpoint as saved by rfdetr training (the
                  package loads it itself); ``checkpoint_path`` is accepted as an alias
    variant:      "nano" | "small" | "medium" | "base" | "large" (default "medium").
                  Must match the checkpoint's architecture; rfdetr fails the load on
                  a mismatch (strict state-dict load), which aborts startup, deliberately.
    resolution:   input resolution the checkpoint was trained at (optional; rfdetr's
                  per-variant default otherwise). Fixed at construction, not per request.
    conf:         default confidence threshold (default 0.5); overridable per request
                  through ``model_params`` / ``predict_model_params`` ``conf``.
    class_names:  REQUIRED map from the checkpoint's class ids to labels, either a
                  list (index = id) or a dict with integer-like keys ({"0": "fish"}).

Why ``class_names`` is required: fine-tuned RF-DETR checkpoints carry no label
names, and the package then reports the COCO list -- which is wrong for a
fine-tune and does not even contain id 0, the id a single-class fine-tune emits.
There is deliberately no default: a detection whose id is not in the map fails the
request instead of being labelled "person".

Dependency note: ``rfdetr==1.5.2`` is the newest release that fits the service's
pins (transformers 4.x, torch 2.1.2, Python 3.10); 1.6+ requires transformers
>= 5.1 and torch >= 2.2. ``rfdetr`` is imported lazily so the module imports
without it (tests stub it, and services without an rfdetr model never need it).
"""
import logging
from typing import Any, Dict, Optional

from PIL import Image

from .base_model import BaseModel
from app.utils.checkpoint_utils import get_checkpoint_path
from app.utils.helpers import decode_image_rgb

logger = logging.getLogger(__name__)

RFDETR_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


def _class_id(key: Any, model_id: str) -> int:
    """A class id is a non-negative integer: an int, or a string of ASCII digits (JSON dict
    keys are strings). Floats, bools and anything else are rejected rather than truncated."""
    if isinstance(key, bool):
        idx = None
    elif isinstance(key, int):
        idx = key
    elif isinstance(key, str) and key.isascii() and key.isdecimal():
        idx = int(key)
    else:
        idx = None
    if idx is None or idx < 0:
        raise ValueError(
            f"rfdetr '{model_id}': class_names key {key!r} is not a non-negative integer class id")
    return idx


def normalise_class_names(class_names: Any, model_id: str = "") -> Dict[int, str]:
    """Accept a list (index = class id) or a dict keyed by class id; return {id: name}.
    Duplicate ids (e.g. "0" and "00"), non-integral keys and blank names are errors."""
    if isinstance(class_names, dict):
        items = class_names.items()
    elif isinstance(class_names, (list, tuple)):
        items = enumerate(class_names)
    else:
        raise ValueError(
            f"rfdetr '{model_id}': class_names must be a list or a dict, got {type(class_names).__name__}")
    out: Dict[int, str] = {}
    for key, name in items:
        idx = _class_id(key, model_id)
        if idx in out:
            raise ValueError(
                f"rfdetr '{model_id}': class_names has duplicate class id {idx} (key {key!r})")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"rfdetr '{model_id}': class_names[{key!r}] must be a non-blank string")
        out[idx] = name.strip()
    if not out:
        raise ValueError(f"rfdetr '{model_id}': class_names must not be empty")
    return out


class RFDETRModel(BaseModel):
    """RF-DETR detector served through the ``rfdetr`` package."""

    def __init__(self):
        self.model = None
        self.model_id: str = ""
        self.model_info: Dict[str, Any] = {}
        self.class_names: Dict[int, str] = {}

    def load(self, model_path: str = "", device: str = "cpu", model_id: str = "",
             variant: str = "medium", resolution: Optional[int] = None,
             conf: float = 0.5, class_names: Any = None,
             checkpoint_path: Optional[str] = None, **kwargs) -> None:
        if kwargs:
            raise ValueError(f"rfdetr '{model_id}': unknown config key(s) {sorted(kwargs)}")
        if variant not in RFDETR_VARIANTS:
            raise ValueError(
                f"rfdetr '{model_id}': unknown variant {variant!r}; expected one of "
                f"{sorted(RFDETR_VARIANTS)}")
        if class_names is None:
            raise ValueError(
                f"rfdetr '{model_id}': class_names is required (fine-tuned checkpoints carry "
                f"no label names and the package's COCO fallback is wrong). Give a list such "
                f"as [\"fish\"] or a dict such as {{\"0\": \"fish\"}}.")
        self.class_names = normalise_class_names(class_names, model_id)
        weights = get_checkpoint_path(checkpoint_path or model_path)
        if not weights:
            raise ValueError(f"rfdetr '{model_id}': model_path (or checkpoint_path) is required.")

        import rfdetr as _rfdetr  # optional dependency; see module docstring
        model_class = getattr(_rfdetr, RFDETR_VARIANTS[variant])
        ctor: Dict[str, Any] = {"pretrain_weights": weights, "device": device}
        if resolution is not None:
            ctor["resolution"] = int(resolution)
        logger.info(f"Loading RF-DETR {variant} from {weights} on device {device}"
                    f"{'' if resolution is None else f' at resolution {int(resolution)}'}")
        self.model = model_class(**ctor)
        self.model_id = model_id
        self.model_info = {
            "model_type": "rfdetr",
            "model_path": weights,
            "device": device,
            "variant": variant,
            "resolution": None if resolution is None else int(resolution),
            "conf": float(conf),
            "class_names": {str(k): v for k, v in sorted(self.class_names.items())},
        }

    def predict(self, image_bytes: bytes, **kwargs) -> Dict[str, Any]:
        """Run detection. ``conf`` may be overridden per request; every other key the
        routers copy in from the model config (variant, resolution, class_names, ...)
        is fixed at load time and ignored here."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        conf = float(kwargs.get("conf", self.model_info["conf"]))

        # Decode through decode_image_rgb so the EXIF Orientation tag is applied:
        # boxes must be in the upright (displayed) frame, as for every PIL-decoding model.
        img = Image.fromarray(decode_image_rgb(image_bytes))
        det = self.model.predict(img, threshold=conf)

        xyxy = det.xyxy if det.xyxy is not None else []
        n = len(xyxy)
        confidence = det.confidence if det.confidence is not None else [None] * n
        class_id = det.class_id if det.class_id is not None else [None] * n

        bboxes, thetas, scores, class_ids, class_names = [], [], [], [], []
        for (x1, y1, x2, y2), score, cid in zip(xyxy, confidence, class_id):
            if score is None or cid is None:
                raise ValueError(
                    f"rfdetr '{self.model_id}': prediction without a confidence or class id")
            cid = int(cid)
            if cid not in self.class_names:
                raise ValueError(
                    f"rfdetr '{self.model_id}': emitted class id {cid}, which class_names "
                    f"{self.model_info['class_names']} does not map; refusing to fabricate a label")
            bboxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            thetas.append(0.0)  # axis-aligned detector
            scores.append(float(score))
            class_ids.append(cid)
            class_names.append(self.class_names[cid])

        return {
            "bboxes": bboxes,
            "thetas": thetas,
            "scores": scores,
            "class_ids": class_ids,
            "class_names": class_names,
        }

    def get_model_info(self) -> Dict[str, Any]:
        return self.model_info
