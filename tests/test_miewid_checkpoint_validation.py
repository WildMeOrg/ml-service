"""Standalone MiewID checkpoints are validated before load_state_dict.

The keys only ever VALIDATE the configured backbone, never switch it: a
DINOv3 entry pointed at an EfficientNet file (or the reverse) must fail with
both names in the message, not load into the wrong network.
"""
import hashlib

import numpy as np
import pytest
import torch

from app.models.miewid import (
    BACKBONE_DINOV3_VITB16, BACKBONE_EFFNET, load_standalone_state_dict,
)
from app.utils.checkpoint_utils import sha256_file


def _save(tmp_path, obj, name="ckpt.bin"):
    path = tmp_path / name
    torch.save(obj, path)
    return str(path)


def _dinov3_sd(bn_dim=1536):
    return {"backbone.embeddings.register_tokens": torch.zeros(1, 4, 768),
            "pooling.p": torch.tensor([3.0]),
            "bn.weight": torch.ones(bn_dim)}


def _effnet_sd():
    return {"backbone.conv_stem.weight": torch.zeros(24, 3, 3, 3),
            "pooling.p": torch.tensor([3.0]),
            "bn.weight": torch.ones(2152)}


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 3_000_001)
    assert sha256_file(str(path)) == hashlib.sha256(b"x" * 3_000_001).hexdigest()


def test_valid_dinov3_and_effnet_load(tmp_path):
    assert "bn.weight" in load_standalone_state_dict(_save(tmp_path, _dinov3_sd()), BACKBONE_DINOV3_VITB16)
    assert "bn.weight" in load_standalone_state_dict(_save(tmp_path, _effnet_sd(), "e.bin"), BACKBONE_EFFNET)


def test_dinov3_config_with_effnet_checkpoint_names_both(tmp_path):
    with pytest.raises(ValueError) as e:
        load_standalone_state_dict(_save(tmp_path, _effnet_sd()), BACKBONE_DINOV3_VITB16)
    assert BACKBONE_DINOV3_VITB16 in str(e.value) and BACKBONE_EFFNET in str(e.value)


def test_effnet_config_with_dinov3_checkpoint_names_both(tmp_path):
    with pytest.raises(ValueError) as e:
        load_standalone_state_dict(_save(tmp_path, _dinov3_sd()), BACKBONE_EFFNET)
    assert BACKBONE_DINOV3_VITB16 in str(e.value) and BACKBONE_EFFNET in str(e.value)


def test_resume_checkpoint_rejected(tmp_path):
    resume = {"model_state_dict": _dinov3_sd(), "epoch": 10, "best_score": 0.7}
    with pytest.raises(ValueError, match="model_best.bin"):
        load_standalone_state_dict(_save(tmp_path, resume), BACKBONE_DINOV3_VITB16)


def test_resume_checkpoint_with_numpy_scalars_rejected(tmp_path):
    # The real checkpoint_latest.pth carries numpy scalars, which
    # weights_only=True refuses to unpickle.
    resume = {"model_state_dict": _dinov3_sd(), "best_score": np.float64(0.7)}
    with pytest.raises(ValueError, match="model_best.bin"):
        load_standalone_state_dict(_save(tmp_path, resume), BACKBONE_DINOV3_VITB16)


def test_non_dict_rejected(tmp_path):
    with pytest.raises(ValueError, match="state dict"):
        load_standalone_state_dict(_save(tmp_path, torch.zeros(3)), BACKBONE_DINOV3_VITB16)


def test_final_head_rejected(tmp_path):
    sd = _effnet_sd()
    sd["final.weight"] = torch.zeros(10, 2152)
    with pytest.raises(ValueError, match="ArcFace head"):
        load_standalone_state_dict(_save(tmp_path, sd), BACKBONE_EFFNET)


def test_dinov3_wrong_bn_width_rejected(tmp_path):
    with pytest.raises(ValueError, match="1536"):
        load_standalone_state_dict(_save(tmp_path, _dinov3_sd(bn_dim=2152)), BACKBONE_DINOV3_VITB16)
