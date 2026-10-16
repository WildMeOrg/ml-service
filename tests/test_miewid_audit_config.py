"""audit_config: flags MiewID entries that would abort startup. No weights, no network."""
import json

import pytest

from scripts.preflight.miewid_dinov3 import audit_config

SHA = "a" * 64


def _write(tmp_path, *models):
    p = tmp_path / "model_config.json"
    p.write_text(json.dumps({"models": list(models)}))
    return str(p)


def _run(tmp_path, capsys, *models):
    code = audit_config.main([_write(tmp_path, *models)])
    return code, capsys.readouterr().out


CLEAN = [
    {"model_id": "legacy-hf", "model_type": "miewid", "imgsz": 440},
    {"model_id": "legacy-overlay", "model_type": "miewid", "imgsz": 440,
     "checkpoint_path": "/d/a.bin", "version": 3},
    {"model_id": "legacy-eff", "model_type": "miewid", "imgsz": 440,
     "checkpoint_path": "/d/b.bin", "version": 5},
    {"model_id": "explicit-eff", "model_type": "miewid", "imgsz": 440, "backbone": "efficientnetv2_rw_m",
     "checkpoint_path": "/d/b.bin", "embedding_dim": 2152},
    {"model_id": "dino", "model_type": "miewid", "imgsz": 448, "backbone": "dinov3_vitb16",
     "checkpoint_path": "/d/c.bin", "checkpoint_sha256": SHA, "embedding_dim": 1536, "version": 5},
    {"model_id": "det", "model_type": "yolo-ultralytics", "imgsz": "garbage"},
]


def test_clean_config_passes(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, *CLEAN)
    assert code == 0, out


BAD = {
    "string imgsz": {"imgsz": "440"},
    "float imgsz": {"imgsz": 440.0},
    "bool imgsz": {"imgsz": True},
    "448 on legacy effnet": {"imgsz": 448, "checkpoint_path": "/d/b.bin", "version": 5},
    "440 on dinov3": {"imgsz": 440, "backbone": "dinov3_vitb16", "checkpoint_path": "/d/c.bin"},
    "embedding_dim on HF hub": {"embedding_dim": 2152},
    "embedding_dim on HF overlay": {"embedding_dim": 2152, "checkpoint_path": "/d/a.bin", "version": 3},
    "wrong dinov3 embedding_dim": {"embedding_dim": 2152, "backbone": "dinov3_vitb16",
                                   "checkpoint_path": "/d/c.bin"},
    "malformed sha": {"checkpoint_path": "/d/a.bin", "checkpoint_sha256": "xyz"},
    "sha without checkpoint": {"checkpoint_sha256": SHA},
    "unknown backbone": {"backbone": "resnet50", "checkpoint_path": "/d/a.bin"},
}


@pytest.mark.parametrize("name", list(BAD))
def test_failure_modes_name_the_model(tmp_path, capsys, name):
    entry = {"model_id": "victim", "model_type": "miewid", **BAD[name]}
    code, out = _run(tmp_path, capsys, CLEAN[0], entry)
    assert code == 1
    assert "victim" in out
    assert "legacy-hf" not in out.split("summary")[0]


def test_effnet_embedding_dim_is_reported_not_failed(tmp_path, capsys):
    entry = {"model_id": "eff", "model_type": "miewid", "imgsz": 440,
             "checkpoint_path": "/d/b.bin", "version": 5, "embedding_dim": 2152}
    code, out = _run(tmp_path, capsys, entry)
    assert code == 0
    assert "verify against checkpoint" in out


def test_resolve_load_path_matches_legacy_rules():
    from app.models.miewid import resolve_load_path
    o = lambda **k: {"backbone": None, "checkpoint_path": None, **k}
    assert resolve_load_path(o(), 3) == "hf_hub"
    assert resolve_load_path(o(checkpoint_path="x"), 3) == "hf_overlay"
    assert resolve_load_path(o(checkpoint_path="x"), "5") == "hf_overlay"
    assert resolve_load_path(o(checkpoint_path="x"), 4) == "efficientnet"
    assert resolve_load_path(o(backbone="dinov3_vitb16", checkpoint_path="x"), 3) == "dinov3"
