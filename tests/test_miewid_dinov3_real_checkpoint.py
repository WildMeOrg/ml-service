"""Load the real FINAL MiewID v5 DINOv3 checkpoint (and the shipped v5
EfficientNet when present) through the production config path. Skipped where
the files are absent (CI). Override the path and expected digest together with
MIEWID_DINOV3_CKPT / MIEWID_DINOV3_SHA256."""
import io
import os

import numpy as np
import pytest
from PIL import Image

from app.models.miewid import MiewidModel

DEV = os.environ.get(
    "MIEWID_DINOV3_CKPT",
    "/mnt/c/claude-skills/models/reference/miewid-dinov3-dev/miew_id.msv5_dinov3.FINAL.bin")
DEV_SHA = os.environ.get(
    "MIEWID_DINOV3_SHA256",
    "05686c4b18a57d14073668e879ba915846263f990feb7c2ac45866845c7ec437")
EFFNET = os.environ.get(
    "MIEWID_EFFNET_V5_CKPT",
    "/mnt/c/claude-skills/models/reference/miewIDv5/_model/miew_id.msv5_main.bin")

pytestmark = pytest.mark.skipif(not os.path.exists(DEV), reason="DINOv3 checkpoint not present")


def _jpeg(seed):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def test_dev_checkpoint_loads_and_embeds():
    m = MiewidModel()
    m.load(device="cpu", model_id="dino", backbone="dinov3_vitb16",
           checkpoint_path=DEV, checkpoint_sha256=DEV_SHA, version=5,
           imgsz=448, embedding_dim=1536)
    a1 = m.extract_embeddings(_jpeg(0), bbox=(10, 20, 200, 150), theta=0.2)
    a2 = m.extract_embeddings(_jpeg(0), bbox=(10, 20, 200, 150), theta=0.2)
    b = m.extract_embeddings(_jpeg(1), bbox=(10, 20, 200, 150), theta=0.2)
    assert a1.shape == (1, 1536) and a1.dtype == np.float32
    assert np.array_equal(a1, a2), "same input must embed identically"
    cos = float((a1 @ b.T).item() / (np.linalg.norm(a1) * np.linalg.norm(b)))
    assert cos < 0.9999, "different images must not collapse to one embedding"


@pytest.mark.skipif(not os.path.exists(EFFNET), reason="v5 EfficientNet not present")
def test_dinov3_and_effnet_coexist():
    dino, eff = MiewidModel(), MiewidModel()
    dino.load(device="cpu", model_id="dino", backbone="dinov3_vitb16", checkpoint_path=DEV)
    eff.load(device="cpu", model_id="eff", checkpoint_path=EFFNET, version=5)
    img = _jpeg(2)
    assert dino.extract_embeddings(img).shape == (1, 1536)
    assert eff.extract_embeddings(img).shape == (1, 2152)
    assert dino.image_size == (448, 448) and eff.image_size == (440, 440)
    assert dino.supports_pairx is False and eff.supports_pairx is True
