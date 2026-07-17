"""Execute the wbia-plugin-orientation REFERENCE inference standalone.

Loads the plugin's own config + cls_hrnet via importlib (bypassing the `wbia`
package import), reproduces OrientationNet (classifier->5, sigmoid, hflip/vflip
TTA) and AnimalWbiaDataset preprocessing exactly, and returns theta.

This is the fidelity oracle: the port must agree with THIS, not with a stored
value or a human's expectation.
"""
import importlib.util, math, sys
import numpy as np, torch, torch.nn as nn, imageio.v2 as imageio, io
from collections import OrderedDict
from skimage.transform import resize as sk_resize
from torchvision import transforms

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

_R = "/mnt/c/wbia-plugin-orientation/wbia_orientation/"
_cfg = _load("wd_cfg", _R + "config/default.py")._C.clone()
_cls = _load("wd_hrnet", _R + "models/cls_hrnet.py")
_utils = _load("wd_utils", _R + "utils/utils.py")
_eval = _load("wd_eval", _R + "core/evaluate.py")

_T = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

class Reference:
    def __init__(self, ckpt):
        m = _cls.HighResolutionNet(_cfg)
        m.classifier = nn.Linear(m.classifier.in_features, 5)   # OrientationNet does this
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        st = raw.get("state", raw) if isinstance(raw, dict) and "state" in raw else raw
        m.load_state_dict(OrderedDict((k.replace("module.","").replace("model.",""), v)
                                      for k, v in st.items()), strict=True)
        m.eval(); self.m = m
        self.imsize = tuple(_cfg.MODEL.IMSIZE)
        self.hflip, self.vflip = _cfg.TEST.HFLIP, _cfg.TEST.VFLIP

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
                oh = _utils.hflip_back(oh, [1.0, 1.0]); oh = torch.from_numpy(oh.copy())
            if self.vflip:
                ov = torch.sigmoid(self.m(torch.flip(x, [2]))).numpy()
                ov = _utils.vflip_back(ov, [1.0, 1.0]); ov = torch.from_numpy(ov.copy())
            if self.hflip and self.vflip:
                out = (out + oh + ov) / 3
        coords = out.numpy()
        return float(_eval.compute_theta(coords)[0]), coords[0].tolist()
