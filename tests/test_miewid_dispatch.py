"""MiewidModel.load dispatch.

Entries without `backbone` must keep TODAY's exact behaviour, quirks
included (spec, "Config contract"). Prod runs 40+ MiewID entries whose
configs are not in this repo, so any drift here changes production silently.
"""
import pytest
import torch
import torch.nn as nn

from app.models import miewid as miewid_mod
from app.models.miewid import (
    BACKBONE_DINOV3_VITB16, BACKBONE_EFFNET, DinoV3MiewIdNet, MiewidModel,
    resolve_precision,
)
from app.utils.checkpoint_utils import sha256_file

_ABSENT = object()


@pytest.fixture
def recorder(monkeypatch):
    calls = {"tags": [], "paths": []}

    def fake_from_pretrained(tag, trust_remote_code=True):
        calls["tags"].append(tag)
        return nn.Linear(2, 2)

    def fake_standalone(self, path, device):
        calls["paths"].append("standalone")
        self.model = nn.Identity()
        self.backbone_name = BACKBONE_EFFNET
        self.embedding_dim = 2152

    def fake_dinov3(self, path, device):
        calls["paths"].append("dinov3")
        self.model = nn.Identity()
        self.backbone_name = BACKBONE_DINOV3_VITB16
        self.embedding_dim = 1536
        self.image_size = (448, 448)
        self.supports_pairx = False

    monkeypatch.setattr(miewid_mod.AutoModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(MiewidModel, "_load_standalone", fake_standalone)
    monkeypatch.setattr(MiewidModel, "_load_dinov3", fake_dinov3)
    return calls


@pytest.fixture
def empty_ckpt(tmp_path):
    path = tmp_path / "overlay.bin"
    torch.save({}, path)
    return str(path)


def _load(version=_ABSENT, **kw):
    if version is not _ABSENT:
        kw["version"] = version
    m = MiewidModel()
    m.load(device="cpu", model_id="t", **kw)
    return m


def _path(calls):
    if calls["paths"]:
        return calls["paths"][-1]
    return "hf"


# (checkpoint?, version) -> (path, hf tag or None)
LEGACY_MATRIX = [
    (True, _ABSENT, "overlay", "conservationxlabs/miewid-msv3"),
    (True, 3, "overlay", "conservationxlabs/miewid-msv3"),
    (True, 3.0, "overlay", "conservationxlabs/miewid-msv3"),
    (True, 3.5, "overlay", "conservationxlabs/miewid-msv2"),
    (True, 4, "standalone", None),
    (True, 4.1, "standalone", None),
    (True, 5, "standalone", None),
    (True, "3", "overlay", "conservationxlabs/miewid-msv2"),
    (True, "5", "overlay", "conservationxlabs/miewid-msv2"),
    (False, _ABSENT, "hub", "conservationxlabs/miewid-msv3"),
    (False, 3, "hub", "conservationxlabs/miewid-msv3"),
    (False, 3.0, "hub", "conservationxlabs/miewid-msv3"),
    (False, 3.5, "hub", "conservationxlabs/miewid-msv2"),
    (False, 4, "hub", "conservationxlabs/miewid-msv2"),
    (False, 4.1, "hub", "conservationxlabs/miewid-msv2"),
    (False, 5, "hub", "conservationxlabs/miewid-msv2"),
    (False, "3", "hub", "conservationxlabs/miewid-msv2"),
    (False, "5", "hub", "conservationxlabs/miewid-msv2"),
]


@pytest.mark.parametrize("has_ckpt,version,path,tag", LEGACY_MATRIX)
def test_legacy_dispatch_is_unchanged(recorder, empty_ckpt, has_ckpt, version, path, tag):
    kw = {"checkpoint_path": empty_ckpt} if has_ckpt else {}
    m = _load(version, **kw)
    if path == "standalone":
        assert recorder["paths"] == ["standalone"] and recorder["tags"] == []
    else:
        assert recorder["paths"] == []
        assert recorder["tags"] == [tag]
    assert m.image_size == (440, 440)
    assert m.precision == "fp32"


def test_backbone_null_is_legacy(recorder, empty_ckpt):
    _load(4, checkpoint_path=empty_ckpt, backbone=None)
    assert recorder["paths"] == ["standalone"]


def test_explicit_backbones(recorder, empty_ckpt):
    _load(checkpoint_path=empty_ckpt, backbone=BACKBONE_EFFNET)
    _load(checkpoint_path=empty_ckpt, backbone=BACKBONE_DINOV3_VITB16, version=5)
    assert recorder["paths"] == ["standalone", "dinov3"]


def test_effnet_version5_without_backbone_stays_effnet(recorder, empty_ckpt):
    _load(5, checkpoint_path=empty_ckpt)
    assert recorder["paths"] == ["standalone"]


@pytest.mark.parametrize("bad", ["dinov3", "vit_base", "", 7])
def test_unknown_backbone_rejected(recorder, empty_ckpt, bad):
    with pytest.raises(ValueError, match="backbone"):
        _load(checkpoint_path=empty_ckpt, backbone=bad)


@pytest.mark.parametrize("backbone", [BACKBONE_EFFNET, BACKBONE_DINOV3_VITB16])
def test_explicit_backbone_requires_checkpoint(recorder, backbone):
    with pytest.raises(ValueError, match="checkpoint_path"):
        _load(backbone=backbone)


def test_hf_overlay_keeps_strict_false(monkeypatch, tmp_path):
    """A partial checkpoint with a stray key loads exactly as it did before."""
    target = nn.Linear(2, 2)
    monkeypatch.setattr(miewid_mod.AutoModel, "from_pretrained",
                        lambda tag, trust_remote_code=True: target)
    weight = torch.full((2, 2), 7.0)
    path = tmp_path / "partial.bin"
    torch.save({"weight": weight, "not.a.key": torch.zeros(1)}, path)   # bias missing
    _load(3, checkpoint_path=str(path))
    assert torch.equal(target.weight, weight)


# --- checkpoint_sha256 ------------------------------------------------------

def test_sha_matches_case_insensitively(recorder, empty_ckpt):
    digest = sha256_file(empty_ckpt)
    for value in (digest, digest.upper()):
        m = _load(4, checkpoint_path=empty_ckpt, checkpoint_sha256=value)
        assert m.model_info["checkpoint_sha256"] == digest


def test_sha_mismatch_names_both_digests(recorder, empty_ckpt):
    wrong = "0" * 64
    with pytest.raises(ValueError) as e:
        _load(4, checkpoint_path=empty_ckpt, checkpoint_sha256=wrong)
    assert wrong in str(e.value) and sha256_file(empty_ckpt) in str(e.value)
    assert recorder["paths"] == [], "must fail before loading weights"


@pytest.mark.parametrize("bad", ["abc", "0" * 63, "g" * 64, 123])
def test_sha_malformed_rejected(recorder, empty_ckpt, bad):
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        _load(4, checkpoint_path=empty_ckpt, checkpoint_sha256=bad)


def test_sha_without_checkpoint_rejected(recorder):
    with pytest.raises(ValueError, match="checkpoint_path"):
        _load(checkpoint_sha256="0" * 64)


def test_sha_works_on_hf_overlay(recorder, empty_ckpt):
    m = _load(3, checkpoint_path=empty_ckpt, checkpoint_sha256=sha256_file(empty_ckpt))
    assert recorder["tags"] == ["conservationxlabs/miewid-msv3"]
    assert m.model_info["checkpoint_sha256"] == sha256_file(empty_ckpt)


# --- imgsz / embedding_dim assertions ---------------------------------------

def test_matching_assertions_load(recorder, empty_ckpt):
    _load(4, checkpoint_path=empty_ckpt, imgsz=440, embedding_dim=2152)
    _load(3, imgsz=440)
    _load(checkpoint_path=empty_ckpt, backbone=BACKBONE_DINOV3_VITB16, imgsz=448, embedding_dim=1536)


@pytest.mark.parametrize("kw", [
    {"version": 4, "imgsz": 448},
    {"backbone": BACKBONE_DINOV3_VITB16, "imgsz": 440},
    {"version": 4, "embedding_dim": 1536},
    {"backbone": BACKBONE_DINOV3_VITB16, "embedding_dim": 2152},
])
def test_mismatching_assertions_rejected(recorder, empty_ckpt, kw):
    with pytest.raises(ValueError):
        _load(checkpoint_path=empty_ckpt, **kw)


@pytest.mark.parametrize("bad", ["448", [448, 448], 448.0, True])
def test_imgsz_must_be_int(recorder, empty_ckpt, bad):
    with pytest.raises(ValueError, match="imgsz"):
        _load(4, checkpoint_path=empty_ckpt, imgsz=bad)


def test_null_assertions_are_unset(recorder, empty_ckpt):
    """JSON null means "not set", consistently with "backbone": null."""
    m = _load(4, checkpoint_path=empty_ckpt, imgsz=None, embedding_dim=None)
    assert m.image_size == (440, 440) and m.embedding_dim == 2152
    _load(3, embedding_dim=None)   # allowed on HF: null is not an assertion


def test_embedding_dim_on_hf_path_rejected(recorder):
    with pytest.raises(ValueError, match="embedding_dim"):
        _load(3, embedding_dim=2152)


# --- precision and model_info -----------------------------------------------

def test_precision_resolution():
    assert resolve_precision(BACKBONE_DINOV3_VITB16, "cuda") == "fp16_autocast"
    assert resolve_precision(BACKBONE_DINOV3_VITB16, "cuda:1") == "fp16_autocast"
    assert resolve_precision(BACKBONE_DINOV3_VITB16, "cpu") == "fp32"
    assert resolve_precision(BACKBONE_EFFNET, "cuda") == "fp32"
    assert resolve_precision("hf:conservationxlabs/miewid-msv3", "cuda:0") == "fp32"


def test_model_info_fields(recorder, empty_ckpt):
    info = _load(checkpoint_path=empty_ckpt, backbone=BACKBONE_DINOV3_VITB16).model_info
    assert info["backbone"] == BACKBONE_DINOV3_VITB16
    assert info["image_size"] == [448, 448]
    assert info["imgsz"] == 448
    assert info["embedding_dim"] == 1536
    assert info["precision"] == "fp32"          # device cpu
    assert info["supports_pairx"] is False
    assert info["checkpoint_sha256"] == sha256_file(empty_ckpt)
    hf = _load(3).model_info
    assert hf["backbone"] == "hf:conservationxlabs/miewid-msv3"
    assert hf["embedding_dim"] is None and hf["supports_pairx"] is True


# --- real loaders (no recorder) ---------------------------------------------

@pytest.fixture(scope="module")
def dinov3_ckpt(tmp_path_factory):
    """A full-size synthetic DINOv3 MiewID checkpoint (~340 MB)."""
    path = tmp_path_factory.mktemp("dinov3") / "synthetic.bin"
    torch.save(DinoV3MiewIdNet().state_dict(), path)
    return str(path)


def test_real_dinov3_load(dinov3_ckpt):
    m = _load(checkpoint_path=dinov3_ckpt, backbone=BACKBONE_DINOV3_VITB16, version=5)
    assert isinstance(m.model, DinoV3MiewIdNet)
    assert not m.model.training
    assert m.image_size == (448, 448)
    assert m.embedding_dim == 1536
    assert m.supports_pairx is False
    assert m.checkpoint_sha256 == sha256_file(dinov3_ckpt)
    assert m.model.device == torch.device("cpu")


def test_install_sets_model_device_for_pairx():
    """PairX reads model.device (pairx/core.py:288); plain nn.Modules lack it."""
    m = MiewidModel()
    m._install(nn.Linear(2, 2), "cpu")
    assert m.model.device == torch.device("cpu")


def test_reused_instance_does_not_inherit_state(recorder, empty_ckpt):
    m = MiewidModel()
    m.load(device="cpu", model_id="t", checkpoint_path=empty_ckpt, backbone=BACKBONE_DINOV3_VITB16)
    assert (m.image_size, m.supports_pairx, m.embedding_dim) == ((448, 448), False, 1536)
    m.load(device="cpu", model_id="t", checkpoint_path=empty_ckpt, version=4)
    assert (m.image_size, m.supports_pairx, m.embedding_dim) == ((440, 440), True, 2152)
    assert tuple(m.preprocess(image=torch.zeros(9, 9, 3).numpy().astype("uint8"))["image"].shape) == (3, 440, 440)
    m.load(device="cpu", model_id="t", version=3)
    assert m.embedding_dim is None and m.checkpoint_sha256 is None
    assert m.backbone_name == "hf:conservationxlabs/miewid-msv3"


def test_legacy_v4_with_arcface_head_names_the_head(tmp_path):
    sd = {"backbone.conv_stem.weight": torch.zeros(1), "bn.weight": torch.ones(2152),
          "final.weight": torch.zeros(3, 2152)}
    path = tmp_path / "wbia.bin"
    torch.save(sd, path)
    with pytest.raises(ValueError, match="ArcFace head"):
        _load(4.1, checkpoint_path=str(path))
