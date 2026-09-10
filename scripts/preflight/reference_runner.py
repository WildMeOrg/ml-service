"""Execute the wbia-plugin-orientation REFERENCE inference standalone.

Loads the plugin's own config + cls_hrnet via importlib (bypassing the `wbia`
package import), reproduces OrientationNet (classifier->5, sigmoid, hflip/vflip
TTA) and AnimalWbiaDataset preprocessing exactly, and returns theta.

This is the fidelity oracle: the port must agree with THIS, not with a stored
value or a human's expectation.
"""
import importlib.util, math, sys
import hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np, torch, torch.nn as nn, imageio.v2 as imageio, io
from collections import OrderedDict
from skimage.transform import resize as sk_resize
from torchvision import transforms

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

def load_reference(reference_root):
    """Load only the standalone modules, isolated and cached per checkout root."""
    return _load_reference(str(Path(reference_root).resolve()))


@lru_cache(maxsize=None)
def _load_reference(reference_root):
    root = Path(reference_root) / "wbia_orientation"
    modules = {"cfg": "config/default.py", "hrnet": "models/cls_hrnet.py",
               "utils": "utils/utils.py", "eval": "core/evaluate.py"}
    missing = [str(root / path) for path in modules.values() if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing reference modules under {reference_root}: {missing}. "
            "Mount the wbia-plugin-orientation checkout and set --reference-root "
            "to its root (containing wbia_orientation)."
        )
    prefix = "wd_" + hashlib.sha256(reference_root.encode()).hexdigest()
    names = {key: f"{prefix}_{key}" for key in modules}
    try:
        return {key: _load(names[key], str(root / path)) for key, path in modules.items()}
    except Exception:
        for name in names.values():
            sys.modules.pop(name, None)
        raise


_T = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

class Reference:
    def __init__(self, ckpt, reference_root="/reference"):
        source = load_reference(reference_root)
        cfg = source["cfg"]._C.clone()
        self._utils, self._eval = source["utils"], source["eval"]
        m = source["hrnet"].HighResolutionNet(cfg)
        m.classifier = nn.Linear(m.classifier.in_features, 5)   # OrientationNet does this
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        st = raw.get("state", raw) if isinstance(raw, dict) and "state" in raw else raw
        m.load_state_dict(OrderedDict((k.replace("module.","").replace("model.",""), v)
                                      for k, v in st.items()), strict=True)
        m.eval(); self.m = m
        self.imsize = tuple(cfg.MODEL.IMSIZE)
        self.hflip, self.vflip = cfg.TEST.HFLIP, cfg.TEST.VFLIP

    def theta(self, image_bytes, bbox):
        # --- AnimalWbiaDataset.__getitem__ ---
        image = imageio.imread(io.BytesIO(image_bytes))
        x1, y1, w, h = bbox
        crop = image[y1:y1+h, x1:x1+w]
        if min(crop.shape) < 1:                       # animal_wbia.py:25-28
            crop = image
        crop = sk_resize(crop, self.imsize, order=3, anti_aliasing=True)
        x = _T(crop).unsqueeze(0).float()             # _plugin.py:272 .float()
        # --- OrientationNet.forward ---
        with torch.no_grad():
            out = torch.sigmoid(self.m(x))
            if self.hflip:
                oh = torch.sigmoid(self.m(torch.flip(x, [3]))).numpy()
                oh = self._utils.hflip_back(oh, [1.0, 1.0]); oh = torch.from_numpy(oh.copy())
            if self.vflip:
                ov = torch.sigmoid(self.m(torch.flip(x, [2]))).numpy()
                ov = self._utils.vflip_back(ov, [1.0, 1.0]); ov = torch.from_numpy(ov.copy())
            if self.hflip and self.vflip:
                out = (out + oh + ov) / 3
        coords = out.numpy()
        return float(self._eval.compute_theta(coords)[0]), coords[0].tolist()
