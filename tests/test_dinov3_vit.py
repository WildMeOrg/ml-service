"""The vendored DINOv3 ViT must be the training backbone, not a lookalike.

Shape and key tests alone would pass a port with the wrong RoPE frequencies,
a missing LayerScale or a misplaced norm, so the golden test pins numerics
against the real transformers 5.17.0 implementation the MiewID v5 DINOv3
training run used (tests/fixtures/dinov3/make_golden.py).
"""
import dataclasses
import json
import pathlib

import torch

from app.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel, VITB16

FIX = pathlib.Path(__file__).parent / "fixtures" / "dinov3"


def test_vitb16_constant_matches_training_config_dump():
    dump = json.loads((FIX / "vitb16_config.json").read_text())
    for field in dataclasses.fields(DINOv3ViTConfig):
        assert getattr(VITB16, field.name) == dump[field.name], field.name
    # Options the port does not implement must be set to the values that
    # make them inert, or the port is not this model.
    assert dump["model_type"] == "dinov3_vit"
    assert dump["hidden_act"] == "gelu"
    assert dump["use_gated_mlp"] is False
    assert dump["drop_path_rate"] == 0.0
    assert dump["attention_dropout"] == 0.0


def test_backbone_state_dict_matches_training_checkpoint_manifest():
    rows = [line.split("\t") for line in
            (FIX / "vitb16_state_keys.tsv").read_text().splitlines()[1:]]
    expected = {key[len("backbone."):]: shape
                for key, shape, _dtype in rows if key.startswith("backbone.")}
    got = {k: str(tuple(v.shape)) for k, v in DINOv3ViTModel(VITB16).state_dict().items()}
    assert got == expected


def test_rope_inv_freq_is_not_persisted():
    assert "rope_embeddings.inv_freq" not in DINOv3ViTModel(VITB16).state_dict()


def test_tiny_backbone_matches_transformers_5_17_golden():
    golden = torch.load(FIX / "golden_tiny.pt", weights_only=True)
    model = DINOv3ViTModel(DINOv3ViTConfig(**golden["config"])).eval()
    backbone_state = {k[len("backbone."):]: v for k, v in golden["state_dict"].items()
                      if k.startswith("backbone.")}
    model.load_state_dict(backbone_state, strict=True)
    with torch.no_grad():
        tokens = model(golden["input"])
    torch.testing.assert_close(tokens, golden["tokens"], atol=1e-5, rtol=0)
