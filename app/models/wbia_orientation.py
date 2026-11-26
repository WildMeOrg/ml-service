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
import numbers
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import imageio.v2 as imageio
import numpy as np
import timm
import torch
from shapely import affinity
from shapely.geometry import Polygon, box as shapely_box
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

# Per-detection plausibility gate (issue #59). When the regressor is unsure it
# predicts a near-zero half-width and/or pins the centre at the crop edge, and
# oriented_box() faithfully turns that into a long thin sliver. Every sliver in
# the GrouperSpotter sweep had long/short >= 13; the healthy population is
# <= ~3. Overridable per model in model_config, since an eel legitimately
# exceeds 6 (lower min_short_frac too in that case: a diagonal 15:1 animal's
# short side is ~9% of its detector box).
DEFAULT_MAX_ASPECT = 6.0          # long/short; None disables
DEFAULT_MIN_SHORT_FRAC = 0.15     # short side vs min(detector box side); 0 disables
DEFAULT_MIN_OVERLAP_FRAC = 0.3    # oriented area inside the detector box; 0 disables

# Status values carried on every result row. The router requires one of these.
STATUS_OK = "ok"
STATUS_REJECTED = "rejected_implausible"

# dataset/animal_wbia.py:35 -- ImageNet statistics.
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def is_finite_number(v: Any) -> bool:
    """A real, finite number. numbers.Real, not (int, float): detectors hand
    us np.float32/np.int64, which are Real but NOT instances of the builtins.
    bool is excluded -- it is an int subclass, and a coordinate or angle of
    True is a bug, not a 1."""
    return (isinstance(v, numbers.Real) and not isinstance(v, bool)
            and math.isfinite(float(v)))


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
    """Resolve a detector bbox to the crop WBIA's orientation plugin ACTUALLY took.

    Two WBIA stages, in order:

      1. Wildbook's lightnet detections went through the `localizations` table,
         which clips every box to the frame (wbia/core_images.py:1539-1557,
         `boundary=True` by default; detect_cnn_lightnet cannot override it).
         The plugin then read those stored, clipped boxes (_plugin.py:309).
      2. The plugin crops with raw NumPy slicing (`image[y1:y1+h, x1:x1+w]`,
         animal_wbia.py:21-23).

    Porting (2) without (1) is wrong in a way that never fails: NumPy resolves
    a negative origin from the FAR EDGE, so a whale shark box hanging 28 px over
    the top of a 2860x1464 photo ([22, -28, 2839, 1492]) cropped the BOTTOM 28
    rows, the regressor described that strip, and the persisted box was a sliver
    (Sharkbook, 2026-11-25). With the clip it is [22, 0, 2838, 1464].

    Integerize first (truncate toward zero, matching the router's detector-box
    rule; lightnet already emits integer corners), then clip both endpoints of
    each axis. Clipping the floats first could turn a 1 px crop into 0 px and
    trigger the full-frame fallback.

    The result becomes `effective_bbox`: the orienter's input crop, and the
    router's fallback box for a rejected axis-aligned detection. Returns
    [x, y, w, h] inside the frame; w/h may be 0
    (box entirely outside), which the caller treats as the reference's
    empty-crop full-frame fallback.
    """
    if len(bbox) != 4:
        raise OrientationInferenceError(
            f"bbox must be [x, y, w, h]; got {len(bbox)} value(s)."
        )
    for v in bbox:
        if not is_finite_number(v):
            raise OrientationInferenceError(
                f"bbox contains a non-finite or non-numeric value: {bbox!r}"
            )

    # Integerize ONCE, truncate-toward-zero, matching the router's
    # `[int(x), int(y), int(width), int(height)]`.
    x, y, w, h = (int(v) for v in bbox)

    # core_images.py:1549-1552: clamp both endpoints of each axis to the frame.
    x1, x2 = min(max(0, x), width), min(max(0, x + w), width)
    y1, y2 = min(max(0, y), height), min(max(0, y + h), height)
    return [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]


def _validate_coords(coords: Sequence[float], model_id: str) -> None:
    """All five outputs must be finite before theta is derived.

    Checking only the derived theta is insufficient: theta reads indices 0-3, so
    a NaN in `w` (index 4) yields a *finite* theta and would be emitted inside
    coords_normalized. Any non-finite output means the model result is malformed
    and the request must fail (design: malformed/non-finite -> fail-closed).
    """
    if len(coords) != NUM_OUTPUTS:
        raise OrientationInferenceError(
            f"wbia-orientation '{model_id}': expected {NUM_OUTPUTS} outputs, "
            f"got {len(coords)}."
        )
    bad = [i for i, c in enumerate(coords) if not math.isfinite(float(c))]
    if bad:
        raise OrientationInferenceError(
            f"wbia-orientation '{model_id}': non-finite model output at "
            f"index/indices {bad}: {coords}"
        )


def compute_theta(coords: Sequence[float]) -> float:
    """theta from normalized [xc, yc, xt, yt, w]. Mirrors core/evaluate.py:9-20.

    arctan2(yt - yc, xt - xc), plus 90 degrees to align arctan2's notation with
    the annotation convention. Computed on NORMALIZED coords, before any resize
    back to image space -- orientation_post_proc calls compute_theta first.
    """
    theta = math.atan2(coords[3] - coords[1], coords[2] - coords[0])
    return theta + math.radians(90)


def oriented_box(coords: Sequence[float], effective_bbox: Sequence[int]):
    """Persisted (bbox, theta) for the regressor's prediction.

    This reproduces WBIA's OWN construction rather than inventing one. The
    authoritative reference is `wbia/core_annots.py:2467`:

        predicted_verts  = get_object_aligned_box(xc, yc, xt, yt, w)
        calculated_theta = arctan2(yt - yc, xt - xc) + deg2rad(90)
        predicted_rot    = rotation_around_mat3x3(calculated_theta * -1.0, xc, yc)
        aligned_verts    = transform_points_with_homography(predicted_rot, verts)
        predicted_bbox   = bboxes_from_vert_list([aligned_verts])[0]

    i.e. build the four corners of the object-aligned box, DE-ROTATE them by
    -theta about the centre, and persist the axis-aligned bound of the result.
    Width and height fall out of that construction -- there is no convention to
    choose and, importantly, NO SORTING: the reference never guarantees that
    width is the shorter side, so forcing that would introduce a fresh error.

    Two things this gets right that a hand-rolled "long side as width" does not:

      * theta carries the reference's +90. That is not a bug in `compute_theta`
        (an earlier revision of this function removed it as if it were): it is
        the convention every catalog built through WBIA was stored in. Dropping
        it rotates the MiewID crop by a quarter turn, so new embeddings stop
        matching the existing catalog even though the drawn box looks correct.
      * theta is computed in IMAGE space, after the anisotropic rescale. An
        angle is not preserved by non-uniform scaling -- a -45deg axis in a
        400x100 crop is -45deg normalized but -75.96deg in image space -- so
        adding the +90 to the normalized angle would be wrong.

    Rescaling into image space mirrors `resize_oa_box`: the perpendicular
    endpoint is reconstructed in NORMALIZED space, all three points scale
    anisotropically, and the half-width is re-measured afterwards. (The
    reference truncates with int() for plotting; we keep float precision.)

    Returns ([x, y, width, height], theta), or (None, None) for a degenerate
    prediction the caller must not persist.
    """
    xc, yc, xt, yt, w = (float(c) for c in coords)
    x1, y1, bw, bh = (float(v) for v in effective_bbox)

    dx, dy = xt - xc, yt - yc
    norm = math.hypot(dx, dy)
    if norm == 0.0 or bw <= 0.0 or bh <= 0.0:
        # Centre and side point coincide: no axis, so no angle. The caller
        # fails closed rather than persisting a fabricated orientation.
        return None, None
    ux, uy = dx / norm, dy / norm
    # add_dict_perpendicular_vector([xc,yc], [xt,yt], w) -> p1 + w * (-uy, ux)
    xw_end, yw_end = xt + w * -uy, yt + w * ux

    # --- into image space (orientation_post_proc: resize_oa_box, then shift) ---
    cx, cy = xc * bw + x1, yc * bh + y1
    tx, ty = xt * bw + x1, yt * bh + y1
    wx, wy = xw_end * bw + x1, yw_end * bh + y1
    half_w = math.hypot(wx - tx, wy - ty)          # re-measured, as resize_oa_box does
    half_l = math.hypot(tx - cx, ty - cy)
    if half_w <= 0.0 or half_l <= 0.0:
        return None, None

    # --- get_object_aligned_box (data_manipulation.py:47), in image space ------
    ax, ay = (tx - cx) / half_l, (ty - cy) / half_l   # unit vector centre -> side point
    px, py = -ay, ax                                  # its perpendicular
    tx2, ty2 = tx + 2.0 * half_l * -ax, ty + 2.0 * half_l * -ay   # tip reflected through centre
    verts = [
        (tx + half_w * px, ty + half_w * py),
        (tx - half_w * px, ty - half_w * py),
        (tx2 + half_w * px, ty2 + half_w * py),
        (tx2 - half_w * px, ty2 - half_w * py),
    ]

    # --- theta, then de-rotate the corners about the centre -------------------
    theta = math.atan2(ty - cy, tx - cx) + math.radians(90)
    ct, st = math.cos(-theta), math.sin(-theta)
    rot = [((vx - cx) * ct - (vy - cy) * st + cx,
            (vx - cx) * st + (vy - cy) * ct + cy) for vx, vy in verts]

    xs = [p[0] for p in rot]
    ys = [p[1] for p in rot]
    bx, by = min(xs), min(ys)
    bw_out, bh_out = max(xs) - bx, max(ys) - by
    if bw_out <= 0.0 or bh_out <= 0.0:
        return None, None
    return [bx, by, bw_out, bh_out], theta


def _object_polygon(oriented_bbox: Sequence[float], theta: float) -> Polygon:
    """The object-aligned rectangle oriented_box() describes.

    oriented_box() de-rotates the corners by -theta about the centre and keeps
    the axis-aligned bound, so rotating that bound by +theta about its centre
    gives the corners back (the WBIA annotation-polygon convention, same matrix
    as assigner._bbox_to_polygon_verts)."""
    x, y, w, h = (float(v) for v in oriented_bbox)
    return affinity.rotate(shapely_box(x, y, x + w, y + h), theta,
                           origin="center", use_radians=True)


def plausibility_failures(oriented_bbox: Sequence[float], theta: float,
                          effective_bbox: Sequence[int],
                          max_aspect: Optional[float] = DEFAULT_MAX_ASPECT,
                          min_short_frac: float = DEFAULT_MIN_SHORT_FRAC,
                          min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC) -> List[str]:
    """Why this oriented box should not be trusted; [] when it is plausible.

    Compares the box with the detector crop it came from. Every failing check
    is reported, so the log says everything that went wrong:

      sub_pixel   a side rounds to < 1 px. Unconditional: the router would
                  otherwise fail the WHOLE request on it.
      aspect      long/short > max_aspect.
      short_side  short side < min_short_frac * the detector box's short side.
      low_overlap less than min_overlap_frac of the oriented box's area lies
                  inside the detector box. (Its centre is always inside: the
                  sigmoid keeps every coordinate in [0, 1].)
    """
    w, h = float(oriented_bbox[2]), float(oriented_bbox[3])
    short, long_ = min(w, h), max(w, h)
    reasons: List[str] = []
    if int(round(w)) < 1 or int(round(h)) < 1:
        reasons.append("sub_pixel")
    if max_aspect is not None and long_ > max_aspect * short:
        reasons.append("aspect")
    if short < min_short_frac * min(effective_bbox[2], effective_bbox[3]):
        reasons.append("short_side")
    if min_overlap_frac > 0:
        poly = _object_polygon(oriented_bbox, theta)
        ex, ey, ew, eh = effective_bbox
        inside = poly.intersection(shapely_box(ex, ey, ex + ew, ey + eh)).area
        if inside < min_overlap_frac * poly.area:
            reasons.append("low_overlap")
    return reasons


def validate_plausibility_limits(model_id: str,
                                 max_aspect: Any = DEFAULT_MAX_ASPECT,
                                 min_short_frac: Any = DEFAULT_MIN_SHORT_FRAC,
                                 min_overlap_frac: Any = DEFAULT_MIN_OVERLAP_FRAC
                                 ) -> Dict[str, Optional[float]]:
    """Check the gate's model_config values at load, so a typo fails loudly
    instead of silently disabling (or tightening) the gate."""
    def _num(key, v):
        if not is_finite_number(v):
            raise ValueError(f"wbia-orientation '{model_id}': {key} must be a "
                             f"finite number, got {v!r}.")
        return float(v)

    if max_aspect is not None:
        max_aspect = _num("max_aspect", max_aspect)
        if max_aspect < 1.0:
            raise ValueError(f"wbia-orientation '{model_id}': max_aspect must be "
                             f">= 1 (or null to disable), got {max_aspect!r}.")
    fracs = {}
    for key, v in (("min_short_frac", min_short_frac),
                   ("min_overlap_frac", min_overlap_frac)):
        f = _num(key, v)
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"wbia-orientation '{model_id}': {key} must be in "
                             f"[0, 1] (0 disables), got {v!r}.")
        fracs[key] = f
    return {"max_aspect": max_aspect, **fracs}


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
        self.limits: Dict[str, Optional[float]] = validate_plausibility_limits("")

    def load(self, model_path: str = "", device: str = "cpu", model_id: str = "",
             checkpoint_path: Optional[str] = None,
             imsize: Optional[Sequence[int]] = None,
             hflip: bool = True, vflip: bool = True,
             max_aspect: Any = DEFAULT_MAX_ASPECT,
             min_short_frac: Any = DEFAULT_MIN_SHORT_FRAC,
             min_overlap_frac: Any = DEFAULT_MIN_OVERLAP_FRAC,
             **kwargs) -> None:
        if kwargs:
            raise ValueError(
                f"wbia-orientation '{model_id}': unknown config key(s) "
                f"{sorted(kwargs)}. Expected: checkpoint_path, imsize, hflip, vflip, "
                f"max_aspect, min_short_frac, min_overlap_frac."
            )
        # Before the checkpoint: a bad limit is a config error, fail fast.
        self.limits = validate_plausibility_limits(
            model_id, max_aspect=max_aspect, min_short_frac=min_short_frac,
            min_overlap_frac=min_overlap_frac)
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
        # bbox is required. Defaulting a missing bbox to the full frame would
        # silently change the region theta describes -- safe-looking, and exactly
        # the class of silent default this model type exists to remove. A caller
        # wanting full-frame must say so: bbox=[0, 0, width, height].
        if bbox is None:
            raise OrientationInferenceError(
                f"wbia-orientation '{self.model_id}': bbox is required; pass "
                f"[0, 0, width, height] explicitly for full-frame."
            )
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
        return self.predict_batch(image_bytes, [bbox])[0]

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

        if not bboxes:
            return []

        # Resolve every bbox first, in input order, so a malformed one fails the
        # whole call before any inference runs.
        effs: List[List[int]] = []
        for bbox in bboxes:
            if bbox is None:
                raise OrientationInferenceError(
                    f"wbia-orientation '{self.model_id}': bbox is required; pass "
                    f"[0, 0, width, height] explicitly for full-frame."
                )
            eff = resolve_bbox(list(bbox), width, height)
            if min(eff[2], eff[3]) < 1:
                eff = [0, 0, width, height]   # reference full-frame fallback
            effs.append(eff)

        # Batch: one stacked tensor -> 3 forwards TOTAL (base/hflip/vflip) for the
        # whole image, rather than 3 per bbox. Order is preserved by construction
        # (crops are stacked in input order and split back by index), which is
        # load-bearing: a misaligned row would attach one bbox's theta to
        # another's crop.
        batch = torch.cat([self._preprocess(image, e) for e in effs], dim=0)
        out = self._forward_tta(batch.to(self.device))
        if out.shape != (len(effs), NUM_OUTPUTS):
            raise OrientationInferenceError(
                f"wbia-orientation '{self.model_id}': model returned "
                f"{tuple(out.shape)}, expected {(len(effs), NUM_OUTPUTS)}."
            )

        results: List[Dict[str, Any]] = []
        for eff, row in zip(effs, out):
            coords = row.tolist()
            _validate_coords(coords, self.model_id)   # ALL five, not just theta
            th = compute_theta(coords)
            if not math.isfinite(th):
                # Fail rather than default: 0.0 is a legitimate prediction, so a
                # sentinel would be indistinguishable from a real answer.
                raise OrientationInferenceError(
                    f"wbia-orientation '{self.model_id}': non-finite theta from "
                    f"coords={coords}"
                )
            # The object-aligned box these coordinates describe, in image
            # space, plus its LONG-axis angle -- the pair the annotation
            # convention wants. `theta` above stays reference-faithful so the
            # host preflight keeps comparing like with like.
            oa_bbox, oa_theta = oriented_box(coords, eff)
            if oa_bbox is None or oa_theta is None:
                # Same stance as a non-finite theta: an unusable prediction is
                # raised, not defaulted. A fabricated angle (or a fabricated
                # 0.0) is indistinguishable from a real one downstream, which
                # is the one failure mode this module exists to prevent.
                raise OrientationInferenceError(
                    f"wbia-orientation '{self.model_id}': coords describe no "
                    f"object axis (coords={coords}, bbox={eff})"
                )
            row = {
                "model_id": self.model_id,
                "theta": float(th),
                "oriented_bbox": oa_bbox,
                "theta_oriented": oa_theta,
                "coords_normalized": [float(c) for c in coords],
                "effective_bbox": eff,
                "orientation_status": STATUS_OK,
            }
            # Per detection, never per request (issue #59): an implausible
            # box is flagged, not raised, so one bad crop cannot fail the good
            # detections in the same image. The router decides the fallback.
            reasons = plausibility_failures(oa_bbox, oa_theta, eff, **self.limits)
            if reasons:
                row["orientation_status"] = STATUS_REJECTED
                row["orientation_reject_reasons"] = reasons
                logger.warning(
                    f"wbia-orientation '{self.model_id}': rejected implausible "
                    f"oriented box {reasons}: effective_bbox={eff}, "
                    f"coords={row['coords_normalized']}, oriented_bbox={oa_bbox}, "
                    f"theta_oriented={oa_theta}")
            results.append(row)
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
            "plausibility": dict(self.limits),
        }
