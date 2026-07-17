"""wbia-plugin-orientation checkpoints run as the regressors they are.

These checkpoints (orientation.whaleshark.v3, orientation.leopard_shark.v0,
orientation.salamander_fire_adult.v2) are ORIENTED-BOUNDING-BOX REGRESSORS. Their
5 outputs are `[xc, yc, xt, yt, w]`, normalized to [0,1] by a sigmoid head, and
exist to derive **theta** -- the crop's angle of rotation. They are NOT
classifiers.

ml-service previously loaded them under `densenet-orientation`, which softmaxed
those coordinates into fabricated viewpoint labels (issue #33) and never produced
theta at all. Sharkbook whale sharks consequently lost rotation entirely in June
2026: theta went from ~2-4% zero to 98% zero, so MiewID embedded tilted animals.
This model type restores what WBIA actually did.

Design: docs/plans/2026-07-16-wbia-orientation-theta-design.md
Reference: wbia-plugin-orientation --
  _plugin.py:189-296 (inference), dataset/animal_wbia.py:17-37 (preprocess),
  models/orientation_net.py:63-95 (sigmoid head + TTA),
  utils/utils.py:88-116 (un-flip), core/evaluate.py:9-20 (theta).

Fidelity is load-bearing: this port's failure mode is being SILENTLY wrong by an
angle, which is indistinguishable from correct output downstream. Every
preprocessing choice below mirrors the reference exactly, and the host preflight
compares this implementation against the reference live.
"""
import io
import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import imageio.v2 as imageio
import numpy as np
import timm
import torch
from skimage.transform import resize as sk_resize
from torchvision import transforms

from app.models.base_model import BaseModel
from app.utils.checkpoint_utils import get_checkpoint_path

logger = logging.getLogger(__name__)

# wbia_orientation/config/default.py:41 -- MODEL.IMSIZE. No species YAML overrides
# it (verified across wbia_orientation/config/*.yaml).
DEFAULT_IMSIZE = (224, 224)

# The head emits exactly [xc, yc, xt, yt, w].
NUM_OUTPUTS = 5

# dataset/animal_wbia.py:35 -- ImageNet statistics.
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class OrientationInferenceError(RuntimeError):
    """Orientation could not produce a trustworthy theta.

    Raised rather than returning a default: a fallback theta of 0.0 is exactly
    the failure this model type exists to repair, and 0.0 is also a *valid*
    prediction, so a sentinel cannot be distinguished from a real answer. The
    router must fail the request rather than embed an unrotated crop.
    """


def _canonicalize_rgb(image: np.ndarray) -> np.ndarray:
    """Coerce a decoded image to (H, W, 3).

    DELIBERATE DEVIATION from the reference, documented in the design. The
    reference feeds imageio's output straight into a 3-channel Normalize, so it
    raises on grayscale (H,W) and RGBA (H,W,4) -- it has no behaviour to port
    there, and an exception is not a specification. Failing a whole detection
    because a frame is grayscale (routine for IR/night capture) is worse than
    handling it. The preflight validates this wrapper by converting fixtures to
    RGB externally and requiring the reference to agree with us.
    """
    if image.ndim == 2:                      # grayscale -> replicate
        return np.stack([image] * 3, axis=-1)
    if image.ndim == 3:
        if image.shape[2] == 3:
            return image                     # already reference-shaped: untouched
        if image.shape[2] == 4:              # RGBA -> drop alpha
            return image[:, :, :3]
        if image.shape[2] == 1:
            return np.repeat(image, 3, axis=2)
    raise OrientationInferenceError(
        f"Unsupported image shape {image.shape}; expected 2-D grayscale or "
        f"(H, W, {{1,3,4}})."
    )


def resolve_bbox(bbox: Sequence[float], width: int, height: int) -> List[int]:
    """Resolve a detector bbox to the slice the reference would ACTUALLY take.

    The reference crops with raw NumPy slicing (`image[y1:y1+h, x1:x1+w]`), which
    does NOT clamp: a negative origin resolves from the FAR EDGE. Measured on a
    400px-wide image:

        x1=-20, w=50   -> width 0   (empty)
        x1=-20, w=500  -> width 20  taken from the image's RIGHT edge
        x1=380, w=50   -> width 20  (silently truncated)

    So an out-of-image bbox does not fail; it silently describes a region the
    detector never pointed at. `slice(start, stop).indices(dim)` reproduces
    exactly what NumPy did, and the resolved slice becomes `effective_bbox` --
    which classify, extract, and the persisted result must all share, or theta
    ends up describing a different region than the crop it rotates.

    Returns [x, y, w, h] of the real in-bounds slice; w/h may be 0, which the
    caller treats as the reference's empty-crop fallback.
    """
    if len(bbox) != 4:
        raise OrientationInferenceError(
            f"bbox must be [x, y, w, h]; got {len(bbox)} value(s)."
        )
    for v in bbox:
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise OrientationInferenceError(
                f"bbox contains a non-finite or non-numeric value: {bbox!r}"
            )

    # Integerize ONCE, truncate-toward-zero, matching pipeline_router.py:218
    # (`[int(x), int(y), int(width), int(height)]`). The rule matters: int(-0.5)
    # is 0 but floor(-0.5) is -1, and -1 resolves from the far edge.
    x, y, w, h = (int(v) for v in bbox)

    x_start, x_stop, _ = slice(x, x + w).indices(width)
    y_start, y_stop, _ = slice(y, y + h).indices(height)
    return [x_start, y_start, max(0, x_stop - x_start), max(0, y_stop - y_start)]


def compute_theta(coords: Sequence[float]) -> float:
    """theta from normalized [xc, yc, xt, yt, w]. Mirrors core/evaluate.py:9-20.

    arctan2(yt - yc, xt - xc), plus 90 degrees to align arctan2's notation with
    the annotation convention. Computed on NORMALIZED coords, before any resize
    back to image space -- orientation_post_proc calls compute_theta first.
    """
    theta = math.atan2(coords[3] - coords[1], coords[2] - coords[0])
    return theta + math.radians(90)


class WbiaOrientationModel(BaseModel):
    """HRNet-W32 oriented-bbox regressor returning theta."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_id: str = ""
        self.device: str = "cpu"
        self.imsize = DEFAULT_IMSIZE
        self.hflip: bool = True
        self.vflip: bool = True

    def load(self, model_path: str = "", device: str = "cpu", model_id: str = "",
             checkpoint_path: Optional[str] = None,
             imsize: Optional[Sequence[int]] = None,
             hflip: bool = True, vflip: bool = True,
             **kwargs) -> None:
        if kwargs:
            raise ValueError(
                f"wbia-orientation '{model_id}': unknown config key(s) "
                f"{sorted(kwargs)}. Expected: checkpoint_path, imsize, hflip, vflip."
            )
        if not checkpoint_path:
            raise ValueError(
                f"wbia-orientation '{model_id}': checkpoint_path is required."
            )
        self.model_id = model_id
        self.device = device
        self.imsize = tuple(imsize) if imsize else DEFAULT_IMSIZE
        # TEST.HFLIP / TEST.VFLIP default True in config/default.py:117-118 and no
        # species YAML overrides them. Configurable, but on by default to match.
        self.hflip, self.vflip = bool(hflip), bool(vflip)

        actual = get_checkpoint_path(checkpoint_path)
        raw = torch.load(actual, map_location=device, weights_only=False)
        state = raw.get("state", raw) if isinstance(raw, dict) and "state" in raw else raw
        clean = OrderedDict(
            (k.replace("module.", "").replace("model.", ""), v) for k, v in state.items()
        )

        # timm's hrnet_w32 is equivalent to the reference's cls_hrnet: measured
        # bit-identical (worst 5.96e-08 = 1 ULP of float32) across all 3
        # checkpoints x 6 inputs. Re-checked by the host preflight; vendor
        # cls_hrnet if that ever regresses. strict=True so a mismatched
        # checkpoint fails loudly here rather than silently mispredicting.
        model = timm.create_model("hrnet_w32", pretrained=False,
                                  num_classes=NUM_OUTPUTS)
        model.load_state_dict(clean, strict=True)
        model.to(device).eval()
        self.model = model

        logger.info(
            f"Loaded WbiaOrientationModel '{model_id}': imsize={self.imsize}, "
            f"hflip={self.hflip}, vflip={self.vflip}"
        )

    # -- inference ---------------------------------------------------------

    def _preprocess(self, image: np.ndarray, eff: List[int]) -> torch.Tensor:
        crop = image[eff[1]:eff[1] + eff[3], eff[0]:eff[0] + eff[2]]
        # dataset/animal_wbia.py:25-28 -- a degenerate crop falls back to the FULL
        # image. The caller has already rewritten effective_bbox to match, so
        # theta and the crop describe the same region.
        if min(crop.shape[:2]) < 1:
            crop = image
        # skimage bicubic + anti-alias. NOT interchangeable with cv2.INTER_CUBIC:
        # skimage applies a Gaussian prefilter when downscaling, and crops here
        # routinely exceed 224. For a model whose output IS an angle, a resize
        # mismatch is angular error (cf. miewid.py's ~3-degree drift note).
        resized = sk_resize(crop, self.imsize, order=3, anti_aliasing=True)
        # sk_resize returns float64 in [0,1], so ToTensor does NOT rescale by 255.
        # float32 cast mirrors _plugin.py:272's `model(images.float(), ...)`.
        return _TRANSFORM(resized).unsqueeze(0).float()

    def _forward_tta(self, x: torch.Tensor) -> torch.Tensor:
        """sigmoid head + hflip/vflip TTA. Mirrors orientation_net.py:63-95."""
        with torch.no_grad():
            out = torch.sigmoid(self.model(x))
            if not (self.hflip or self.vflip):
                return out
            acc, n = out.clone(), 1
            if self.hflip:
                oh = torch.sigmoid(self.model(torch.flip(x, [3])))
                # utils.py:88-102 with image_h_w=[1.0,1.0]: x-coords mirror, the
                # rest are unchanged.
                oh[:, 0] = 1.0 - oh[:, 0]
                oh[:, 2] = 1.0 - oh[:, 2]
                acc, n = acc + oh, n + 1
            if self.vflip:
                ov = torch.sigmoid(self.model(torch.flip(x, [2])))
                # utils.py:104-116 -- y-coords mirror.
                ov[:, 1] = 1.0 - ov[:, 1]
                ov[:, 3] = 1.0 - ov[:, 3]
                acc, n = acc + ov, n + 1
            return acc / n

    def predict(self, image_bytes: bytes,
                bbox: Optional[Sequence[float]] = None,
                theta: float = 0.0,
                **kwargs) -> Dict[str, Any]:
        # The reference regressor consumes the AXIS-ALIGNED bbox and infers the
        # rotation itself. Handing it a crop that a detector already rotated would
        # double-rotate. Harmless for lightnet (always 0.0) but wrong for an OBB
        # detector, so reject it rather than silently accept.
        if theta:
            raise OrientationInferenceError(
                f"wbia-orientation '{self.model_id}' takes an axis-aligned bbox and "
                f"derives theta itself; it must not be given a pre-rotated crop "
                f"(got theta={theta})."
            )
        return self.predict_batch(image_bytes, [bbox] if bbox is not None else [None])[0]

    def predict_batch(self, image_bytes: bytes,
                      bboxes: Sequence[Optional[Sequence[float]]]) -> List[Dict[str, Any]]:
        """Exactly one ordered result per input bbox, or atomic failure.

        Ordering is load-bearing: a misaligned result would attach one bbox's
        theta to another bbox's crop. Callers must assert
        len(results) == len(bboxes) before running any consumer.
        """
        if self.model is None:
            raise OrientationInferenceError(
                f"wbia-orientation '{self.model_id}': model is not loaded."
            )
        try:
            decoded = imageio.imread(io.BytesIO(image_bytes))
        except Exception as e:
            raise OrientationInferenceError(f"could not decode image: {e}") from e
        image = _canonicalize_rgb(np.asarray(decoded))
        height, width = image.shape[:2]

        results: List[Dict[str, Any]] = []
        for bbox in bboxes:
            src = [0, 0, width, height] if bbox is None else list(bbox)
            eff = resolve_bbox(src, width, height)
            if min(eff[2], eff[3]) < 1:
                eff = [0, 0, width, height]   # reference full-frame fallback

            x = self._preprocess(image, eff).to(self.device)
            coords = self._forward_tta(x)[0].tolist()
            th = compute_theta(coords)
            if not math.isfinite(th):
                # Fail rather than default: 0.0 is a legitimate prediction, so a
                # sentinel would be indistinguishable from a real answer.
                raise OrientationInferenceError(
                    f"wbia-orientation '{self.model_id}': non-finite theta from "
                    f"coords={coords}"
                )
            results.append({
                "model_id": self.model_id,
                "theta": float(th),
                "coords_normalized": [float(c) for c in coords],
                "effective_bbox": eff,
            })
        if len(results) != len(bboxes):   # defensive: the contract is ordered 1:1
            raise OrientationInferenceError(
                f"wbia-orientation '{self.model_id}': produced {len(results)} "
                f"result(s) for {len(bboxes)} bbox(es)."
            )
        return results

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "wbia-orientation",
            "device": str(self.device),
            "imsize": list(self.imsize),
            "num_outputs": NUM_OUTPUTS,
            "hflip": self.hflip,
            "vflip": self.vflip,
            "outputs": "theta (radians) + normalized [xc, yc, xt, yt, w]",
        }
