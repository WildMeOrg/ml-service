# MiewID DINOv3 Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the `miewid` model type serve the MiewID v5 DINOv3 ViT-B/16 variant beside the existing EfficientNet/HF variants, each as its own `model_id`, with embeddings that match the training code.

**Architecture:** A vendored, inference-only DINOv3 ViT (`app/models/dinov3_vit.py`, copied from transformers 5.17.0 with identical parameter names) feeds a new `DinoV3MiewIdNet` neck (CLS + GeM over the patch grid → BatchNorm1d). `MiewidModel` stays the single public class; a new `backbone` config field selects the network, while entries without it keep today's dispatch exactly. Image size, precision, and PairX capability become per-instance.

**Tech Stack:** Python 3.10 (prod container) / 3.12 (dev host), PyTorch 2.1.2 (prod) / 2.10 (dev), albumentations, FastAPI, pytest. No dependency changes.

**Spec:** `docs/plans/2026-10-15-miewid-dinov3-design.md`. Read it before starting any task.

## Global Constraints

- No dependency changes: `requirements.txt` and `docker/dockerfile` pins (torch 2.1.2, torchvision 0.16.2, timm 1.0.25, transformers 4.52.4, albumentations 1.3.1) are untouched.
- Prod runs Python 3.10 + torch 2.1.2. No `nn.Buffer` (torch ≥ 2.5), no `enable_gqa`, no syntax newer than 3.10.
- The vendored backbone's `state_dict` keys must equal transformers 5.17's: `embeddings.*`, `model.layer.N.*`, `norm.*`; `rope_embeddings.inv_freq` is non-persistent.
- DINOv3 input is exactly 448×448; embedding is 1536-d. EfficientNet and HF stay 440×440. EfficientNet is 2152-d.
- DINOv3 forwards run under `torch.autocast("cuda", dtype=torch.float16)` on CUDA and fp32 on CPU. EfficientNet and HF never enter autocast.
- Legacy dispatch (no `backbone`) is preserved verbatim, including the quirks: checkpoint + `int`/`float` `version >= 4` → standalone EfficientNet; checkpoint + anything else → HF overlay with `strict=False`; no checkpoint → HF hub. Only `version == 3` selects msv3, everything else selects msv2.
- Decode stays `decode_image_rgb` (EXIF applied). Do not switch to an EXIF-ignoring decoder; see the spec's "Reference chips and the image frame".
- Dev tests run with `python3 -m pytest` on the dev host (Python 3.12, torch 2.10). Task 7 re-runs the new tests inside the prod image.
- Dev artifacts live outside the repo at `/mnt/c/claude-skills/models/reference/miewid-dinov3-dev/`. Never commit the checkpoint.
- Commit trailers on every commit:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Co-Authored-By: Codex <noreply@openai.com>`.

## Review Focus

Inputs the spec implies but no other test pins; each has a test in the named task.

1. **RGBA, grayscale and 16-bit PNG inputs to a DINOv3 model** should embed like RGB, not crash or produce NaNs (Task 5, `test_dinov3_accepts_non_rgb_inputs`).
2. **`device: "cuda:1"`-style strings** should still select fp16 autocast for DINOv3 (Task 4, `test_precision_resolution`).
3. **An explicit `"backbone": null`** in JSON config should behave exactly like an absent field (Task 4, `test_backbone_null_is_legacy`).
4. **`EXPLAIN_DEFAULT_MODEL_ID` pointing at a DINOv3 model**, with the caller omitting `model_id`, should be a clean 400, not a 500 deep in PairX (Task 6, `test_default_model_id_pointing_at_dinov3_is_400`).
5. **A WBIA-converted EfficientNet checkpoint carrying `final.*` (ArcFace head) on the legacy `version >= 4` path** should fail with a message naming the head, not an opaque strict-load error (Task 3, `test_final_head_rejected`, and Task 4, `test_legacy_v4_with_arcface_head_names_the_head`).

---

### Task 1: Vendored DINOv3 ViT backbone

**Files:**
- Create: `app/models/dinov3_vit.py`
- Create: `tests/fixtures/dinov3/vitb16_config.json` (copied)
- Create: `tests/fixtures/dinov3/vitb16_state_keys.tsv` (copied)
- Create: `tests/fixtures/dinov3/make_golden.py`
- Create: `tests/fixtures/dinov3/golden_tiny.pt` (generated)
- Test: `tests/test_dinov3_vit.py`

**Interfaces:**
- Produces: `DINOv3ViTConfig` (frozen dataclass; fields `hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_register_tokens, patch_size=16, num_channels=3, layer_norm_eps=1e-5, rope_theta=100.0, query_bias=True, key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True, layerscale_value=1.0`); `VITB16: DINOv3ViTConfig`; `DINOv3ViTModel(config)` whose `forward(pixel_values: Tensor[B,3,H,W]) -> Tensor[B, 1 + num_register_tokens + (H/p)*(W/p), hidden_size]` returns every token after the final norm.

- [ ] **Step 1: Copy the training-session fixtures into the repo**

```bash
mkdir -p tests/fixtures/dinov3
cp /mnt/c/claude-skills/models/reference/miewid-dinov3-dev/dinov3_backbone_config.json tests/fixtures/dinov3/vitb16_config.json
cp /mnt/c/claude-skills/models/reference/miewid-dinov3-dev/state_dict_keys.tsv tests/fixtures/dinov3/vitb16_state_keys.tsv
```

- [ ] **Step 2: Write the golden-output generator**

`tests/fixtures/dinov3/make_golden.py`. It runs against the **real** transformers 5.17.0, so the golden pins the vendored port to the training code's numerics:

```python
"""Generate golden_tiny.pt for tests/test_dinov3_vit.py and
tests/test_miewid_dinov3_net.py.

A tiny DINOv3 config with every parameter randomised is pushed through the
REAL transformers 5.17.0 DINOv3ViTModel (sdpa attention, as the MiewID v5
DINOv3 training run used) and then through the training repo's MiewID neck,
copied from its MiewIdNet `dinov3:` branch:

    tok = backbone(pixel_values=x).last_hidden_state
    cls = tok[:, 0]; patch = tok[:, n_skip:]
    grid = patch.transpose(1, 2).reshape(B, C, h, w)
    feat = cat([cls, GeM(grid).view(B, -1)], 1); emb = bn(feat)

fp32 on CPU. Run in a throwaway venv that has transformers==5.17.0:

    python3.12 -m venv --system-site-packages "$VENV"
    "$VENV/bin/pip" install transformers==5.17.0
    "$VENV/bin/python" tests/fixtures/dinov3/make_golden.py tests/fixtures/dinov3/golden_tiny.pt
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import DINOv3ViTConfig, DINOv3ViTModel

assert transformers.__version__ == "5.17.0", transformers.__version__

TINY = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_register_tokens=4, patch_size=16,
    num_channels=3, layer_norm_eps=1e-5, rope_theta=100.0,
    query_bias=True, key_bias=False, value_bias=True, proj_bias=True,
    mlp_bias=True, layerscale_value=1.0,
)
INPUT_SIZE = (64, 64)


def gem(x, p, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def main(out_path):
    torch.manual_seed(0)
    cfg = DINOv3ViTConfig(**TINY, hidden_act="gelu", use_gated_mlp=False,
                          drop_path_rate=0.0, image_size=INPUT_SIZE[0])
    cfg._attn_implementation = "sdpa"
    backbone = DINOv3ViTModel(cfg).eval()
    assert backbone.config._attn_implementation == "sdpa"
    with torch.no_grad():
        # Randomise everything, including biases, layer scales and tokens,
        # so a port that drops or reorders any of them fails the golden.
        for _, p in backbone.named_parameters():
            p.copy_(torch.randn_like(p) * 0.2)
    p_gem = torch.tensor([3.7])
    dim = 2 * TINY["hidden_size"]
    bn = nn.BatchNorm1d(dim).eval()
    with torch.no_grad():
        bn.weight.copy_(torch.rand(dim) + 0.5)
        bn.bias.copy_(torch.randn(dim) * 0.1)
        bn.running_mean.copy_(torch.randn(dim) * 0.1)
        bn.running_var.copy_(torch.rand(dim) + 0.5)

    x = torch.randn(2, 3, *INPUT_SIZE)
    with torch.no_grad():
        tok = backbone(pixel_values=x).last_hidden_state
        n_skip = 1 + TINY["num_register_tokens"]
        b, _, c = tok.shape
        h, w = INPUT_SIZE[0] // TINY["patch_size"], INPUT_SIZE[1] // TINY["patch_size"]
        cls = tok[:, 0]
        patch = tok[:, n_skip:]
        grid = patch.transpose(1, 2).reshape(b, c, h, w)
        feat = torch.cat([cls, gem(grid, p_gem).view(b, -1)], dim=1)
        emb = bn(feat)

    state = {"backbone." + k: v.clone() for k, v in backbone.state_dict().items()}
    state["pooling.p"] = p_gem.clone()
    for k, v in bn.state_dict().items():
        state["bn." + k] = v.clone()
    torch.save({
        "config": TINY,
        "input_size": list(INPUT_SIZE),
        "state_dict": state,
        "input": x,
        "tokens": tok,
        "concat": feat,
        "embedding": emb,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }, out_path)
    print(f"wrote {out_path}: tokens {tuple(tok.shape)}, embedding {tuple(emb.shape)}")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 3: Generate the golden fixture**

```bash
VENV=$(mktemp -d)/golden-venv
python3.12 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -q transformers==5.17.0
"$VENV/bin/python" tests/fixtures/dinov3/make_golden.py tests/fixtures/dinov3/golden_tiny.pt
ls -la tests/fixtures/dinov3/golden_tiny.pt
```
Expected: `wrote ...: tokens (2, 21, 64), embedding (2, 128)`, and a file under 1 MB. If `pip` cannot resolve transformers 5.17.0 against the dev host's torch, stop and report; do not substitute another transformers version.

- [ ] **Step 4: Write the failing tests**

`tests/test_dinov3_vit.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_dinov3_vit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.dinov3_vit'`.

- [ ] **Step 6: Write the vendored module**

`app/models/dinov3_vit.py`:

```python
# Copyright 2025 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only DINOv3 ViT backbone, vendored from transformers 5.17.0.

Source: transformers==5.17.0, src/transformers/models/dinov3_vit/
modeling_dinov3_vit.py. Vendored rather than imported because ml-service pins
transformers 4.52.4 (no DINOv3) and torch 2.1.2, and the transformers releases
that run the MiewID v5 DINOv3 checkpoint unmodified (>= 5.4) require torch
>= 2.4. Rationale: docs/plans/2026-10-15-miewid-dinov3-design.md.

Parameter names are identical to 5.17, so the training checkpoint loads
strictly with no key remap: embeddings.*, model.layer.N.*, norm.*.
rope_embeddings holds only the non-persistent inv_freq buffer.

Copied line for line: get_patches_center_coordinates, rotate_half,
apply_rotary_pos_emb, the RoPE forward (fp32 with autocast disabled), the
attention projections and head layout, LayerScale, the MLP, and the block and
model forward order. Generic ViT RoPE has the same shapes and different
values, so do not "simplify" these.

Removed, because each is inert at eval for this config or is HF plumbing:
PreTrainedModel, gradient checkpointing, output capturing, the ACT2FN registry
(hidden_act "gelu" -> nn.GELU, exact erf), the coordinate lru_cache, drop
path, training-time RoPE augmentation, the gated MLP, mask-token
substitution (mask_token stays as a parameter so the checkpoint loads), and
the eager attention path (training ran sdpa; see sdpa_attention_forward with
attention_mask=None, is_causal=False, dropout 0).

torch 2.1: register_buffer(persistent=False) instead of nn.Buffer; no
enable_gqa.
"""
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DINOv3ViTConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_register_tokens: int
    patch_size: int = 16
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    rope_theta: float = 100.0
    query_bias: bool = True
    key_bias: bool = False
    value_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    layerscale_value: float = 1.0


# facebook/dinov3-vitb16-pretrain-lvd1689m @ 5931719e67bbdb9737e363e781fb0c67687896bc,
# checked field by field against the training run's config.to_dict() dump in
# tests/fixtures/dinov3/vitb16_config.json.
VITB16 = DINOv3ViTConfig(
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_register_tokens=4,
)


def get_patches_center_coordinates(
    num_patches_h: int, num_patches_w: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """(y, x) patch centres normalised to [-1, +1], shape (h * w, 2)."""
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device)
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device)
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    coords = 2.0 * coords - 1.0
    return coords


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE on patch tokens only; CLS and register tokens pass through.

    Under autocast the fp32 cos/sin promote the fp16 patch Q/K to fp32 here,
    and the cat then promotes the prefix tokens. That promotion is part of the
    training numerics; keep it.
    """
    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches

    q_prefix_tokens, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
    k_prefix_tokens, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)

    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    q = torch.cat((q_prefix_tokens, q_patches), dim=-2)
    k = torch.cat((k_prefix_tokens, k_patches), dim=-2)
    return q, k


class DINOv3ViTEmbeddings(nn.Module):
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # Unused at inference; kept so the checkpoint loads strictly.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(
            torch.zeros(1, config.num_register_tokens, config.hidden_size))
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_size,
            kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.weight.dtype
        patch_embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        return torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)


class DINOv3ViTRopePositionEmbedding(nn.Module):
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        device = pixel_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):  # force float32
            patch_coords = get_patches_center_coordinates(
                num_patches_h, num_patches_w, dtype=torch.float32, device=device)
            # (h*w, 2, head_dim/4) -> (h*w, head_dim/2) -> (h*w, head_dim)
            angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2)
            angles = angles.tile(2)
            cos = torch.cos(angles)
            sin = torch.sin(angles)

        dtype = pixel_values.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class DINOv3ViTAttention(nn.Module):
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.key_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.value_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.query_bias)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.proj_bias)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # transformers' sdpa_attention_forward with attention_mask=None,
        # is_causal=False and dropout 0.0 (eval).
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=None, dropout_p=0.0, scale=self.scaling, is_causal=False)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        return self.o_proj(attn_output)


class DINOv3ViTLayerScale(nn.Module):
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


class DINOv3ViTMLP(nn.Module):
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class DINOv3ViTLayer(nn.Module):
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DINOv3ViTAttention(config)
        self.layer_scale1 = DINOv3ViTLayerScale(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = DINOv3ViTMLP(config)
        self.layer_scale2 = DINOv3ViTLayerScale(config)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, position_embeddings=position_embeddings)
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class DINOv3ViTEncoder(nn.Module):
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.layer = nn.ModuleList([DINOv3ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
        return hidden_states


class DINOv3ViTModel(nn.Module):
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.config = config
        self.embeddings = DINOv3ViTEmbeddings(config)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(config)
        self.model = DINOv3ViTEncoder(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Every token after the final norm (transformers' last_hidden_state)."""
        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.embeddings(pixel_values)
        position_embeddings = self.rope_embeddings(pixel_values)
        hidden_states = self.model(hidden_states, position_embeddings)
        return self.norm(hidden_states)
```

Note: in 5.17 `drop_path` is `nn.Identity()` when `drop_path_rate == 0`, so `hidden_states + residual` is the same op sequence as `self.drop_path(hidden_states) + residual`.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_dinov3_vit.py -v`
Expected: 4 passed. If the golden test fails, diff against `modeling_dinov3_vit.py` from the transformers 5.17.0 wheel, line by line. Do not loosen `atol`.

- [ ] **Step 8: Commit**

```bash
git add app/models/dinov3_vit.py tests/test_dinov3_vit.py tests/fixtures/dinov3/
git commit -m "feat(miewid): vendor an inference-only DINOv3 ViT from transformers 5.17

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Co-Authored-By: Codex <noreply@openai.com>"
```

---

### Task 2: `DinoV3MiewIdNet` neck

**Files:**
- Modify: `app/models/miewid.py` (add import, constants and class after `MiewIdNet`, around line 53)
- Test: `tests/test_miewid_dinov3_net.py`

**Interfaces:**
- Consumes: `DINOv3ViTConfig`, `DINOv3ViTModel`, `VITB16` from Task 1; existing `GeM` in `miewid.py`.
- Produces: `DinoV3MiewIdNet(config: DINOv3ViTConfig = VITB16, input_size: Tuple[int, int] = (448, 448))` with `.forward(x) -> Tensor[B, 2*hidden]`, `.forward_stages(x) -> {"tokens", "concat", "embedding"}`, and attributes `.input_size`, `.n_skip`, `.grid`; module constant `DINOV3_IMAGE_SIZE = (448, 448)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_miewid_dinov3_net.py`:

```python
"""The DINOv3 MiewID neck: concat(CLS, GeM over the patch grid) -> BN.

Pinned against the training repo's neck via the golden fixture, plus the two
ways a lookalike would silently go wrong: register tokens leaking into GeM,
and an input that yields the right token count from the wrong grid.
"""
import pathlib

import pytest
import torch

from app.models.dinov3_vit import DINOv3ViTConfig, VITB16
from app.models.miewid import DINOV3_IMAGE_SIZE, DinoV3MiewIdNet

FIX = pathlib.Path(__file__).parent / "fixtures" / "dinov3"


def _tiny_net():
    golden = torch.load(FIX / "golden_tiny.pt", weights_only=True)
    net = DinoV3MiewIdNet(DINOv3ViTConfig(**golden["config"]),
                          input_size=tuple(golden["input_size"])).eval()
    net.load_state_dict(golden["state_dict"], strict=True)
    return net, golden


def test_tiny_net_matches_training_neck_golden():
    net, golden = _tiny_net()
    with torch.no_grad():
        stages = net.forward_stages(golden["input"])
    torch.testing.assert_close(stages["tokens"], golden["tokens"], atol=1e-5, rtol=0)
    torch.testing.assert_close(stages["concat"], golden["concat"], atol=1e-5, rtol=0)
    torch.testing.assert_close(stages["embedding"], golden["embedding"], atol=1e-5, rtol=0)


def test_full_size_output_is_1536():
    net = DinoV3MiewIdNet().eval()
    assert DINOV3_IMAGE_SIZE == (448, 448)
    with torch.no_grad():
        out = net(torch.zeros(1, 3, 448, 448))
    assert out.shape == (1, 1536)


def test_register_tokens_do_not_reach_the_embedding():
    net, _ = _tiny_net()
    tokens = torch.randn(1, net.n_skip + 16, 64)
    net.backbone.forward = lambda x: tokens.clone()
    with torch.no_grad():
        before = net(torch.zeros(1, 3, 64, 64))
        tokens[:, 1:net.n_skip] += 100.0   # registers only
        after = net(torch.zeros(1, 3, 64, 64))
    assert torch.equal(before, after)


@pytest.mark.parametrize("hw", [(440, 440), (449, 448), (224, 896)])
def test_wrong_input_size_is_rejected_before_the_backbone(hw):
    net = DinoV3MiewIdNet().eval()
    net.backbone.forward = lambda x: pytest.fail("backbone must not run")
    with pytest.raises(ValueError, match="448x448"):
        net(torch.zeros(1, 3, *hw))


def test_wrong_token_count_is_rejected():
    net, _ = _tiny_net()
    net.backbone.forward = lambda x: torch.zeros(1, net.n_skip + 15, 64)
    with pytest.raises(ValueError, match="tokens"):
        net(torch.zeros(1, 3, 64, 64))


def test_input_size_must_be_a_multiple_of_the_patch():
    with pytest.raises(ValueError, match="multiple of 16"):
        DinoV3MiewIdNet(VITB16, input_size=(440, 440))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_miewid_dinov3_net.py -v`
Expected: FAIL with `ImportError: cannot import name 'DINOV3_IMAGE_SIZE'`.

- [ ] **Step 3: Implement**

In `app/models/miewid.py`, add to the imports:

```python
from app.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel, VITB16
```

and after `class MiewIdNet` (ends around line 53), add:

```python
# DINOv3 MiewID runs at 448: ViT-B/16 needs a multiple of 16, so the
# EfficientNet's 440 is not an option (the patch grid would not tile it).
DINOV3_IMAGE_SIZE = (448, 448)


class DinoV3MiewIdNet(nn.Module):
    """MiewID with a DINOv3 ViT backbone.

    Matches the training repo's MiewIdNet `dinov3:` branch
    (wbia_miew_id/models/model.py, MiewID v5 DINOv3 run):
    concat(CLS, GeM over the patch grid with CLS and registers skipped)
    -> BatchNorm1d. No projection, no L2 normalisation.
    """

    def __init__(self, config: DINOv3ViTConfig = VITB16,
                 input_size: Tuple[int, int] = DINOV3_IMAGE_SIZE):
        super().__init__()
        if input_size[0] % config.patch_size or input_size[1] % config.patch_size:
            raise ValueError(
                f"DINOv3 input size {input_size} must be a multiple of {config.patch_size}")
        self.backbone = DINOv3ViTModel(config)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.n_skip = 1 + config.num_register_tokens
        self.grid = (input_size[0] // config.patch_size, input_size[1] // config.patch_size)
        self.final_in_features = config.hidden_size * 2
        self.pooling = GeM()
        self.bn = nn.BatchNorm1d(self.final_in_features)

    def forward_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Check the input, not just the token count: the patch conv floors,
        # so 449x448 and 224x896 both yield 784 patches, and the second would
        # be reshaped onto the wrong 28x28 grid without any error.
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(
                f"DINOv3 MiewID expects {self.input_size[0]}x{self.input_size[1]} input, "
                f"got {x.shape[-2]}x{x.shape[-1]}")
        tokens = self.backbone(x)
        b, n, c = tokens.shape
        h, w = self.grid
        if n != self.n_skip + h * w:
            raise ValueError(f"expected {self.n_skip + h * w} tokens, got {n}")
        cls = tokens[:, 0]
        patch = tokens[:, self.n_skip:]
        grid = patch.transpose(1, 2).reshape(b, c, h, w)
        concat = torch.cat([cls, self.pooling(grid).view(b, -1)], dim=1)
        return {"tokens": tokens, "concat": concat, "embedding": self.bn(concat)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_stages(x)["embedding"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_miewid_dinov3_net.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add app/models/miewid.py tests/test_miewid_dinov3_net.py
git commit -m "feat(miewid): add the DINOv3 MiewID neck (CLS + GeM over patches -> BN)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Co-Authored-By: Codex <noreply@openai.com>"
```

---

### Task 3: Checkpoint validation and SHA-256 helper

**Files:**
- Modify: `app/utils/checkpoint_utils.py` (add `sha256_file` after `is_url`, line ~25)
- Modify: `app/models/miewid.py` (add backbone constants, `describe_checkpoint`, `load_standalone_state_dict`)
- Test: `tests/test_miewid_checkpoint_validation.py`

**Interfaces:**
- Consumes: `VITB16` (Task 1).
- Produces: `sha256_file(path: str, chunk_size: int = 1 << 20) -> str` (lowercase hex); in `miewid.py`: `BACKBONE_EFFNET = "efficientnetv2_rw_m"`, `BACKBONE_DINOV3_VITB16 = "dinov3_vitb16"`, `SUPPORTED_BACKBONES`, `describe_checkpoint(sd) -> str`, `load_standalone_state_dict(path: str, backbone: str) -> Dict[str, torch.Tensor]` (raises `ValueError` on every rejection).

- [ ] **Step 1: Write the failing tests**

`tests/test_miewid_checkpoint_validation.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_miewid_checkpoint_validation.py -v`
Expected: FAIL with `ImportError: cannot import name 'BACKBONE_DINOV3_VITB16'`.

- [ ] **Step 3: Implement**

In `app/utils/checkpoint_utils.py`, after `is_url` (hashlib is already imported):

```python
def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Streamed SHA-256 of a local file, lowercase hex.

    The download cache is keyed by URL, not content, so this is the only
    way to notice a checkpoint replaced at the same URL.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
```

In `app/models/miewid.py`, add `import pickle` to the imports, and after `DinoV3MiewIdNet`:

```python
BACKBONE_EFFNET = "efficientnetv2_rw_m"
BACKBONE_DINOV3_VITB16 = "dinov3_vitb16"
SUPPORTED_BACKBONES = (BACKBONE_EFFNET, BACKBONE_DINOV3_VITB16)

# Top-level keys of a training resume checkpoint (checkpoint_latest.pth).
_RESUME_KEYS = {"model_state_dict", "optimizer", "scheduler", "epoch", "best_score"}


def describe_checkpoint(sd: Dict[str, Any]) -> str:
    """Which standalone MiewID backbone a state dict was saved from."""
    if "backbone.embeddings.register_tokens" in sd:
        return BACKBONE_DINOV3_VITB16
    if "backbone.conv_stem.weight" in sd:
        return BACKBONE_EFFNET
    return "unrecognised (first keys: %s)" % ", ".join(list(sd)[:3])


def load_standalone_state_dict(path: str, backbone: str) -> Dict[str, torch.Tensor]:
    """Load and validate a standalone MiewID checkpoint for `backbone`.

    Standalone paths only (EfficientNet v4+ and DINOv3). The legacy HF
    overlay keeps its strict=False tolerance and never comes through here.
    """
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as e:
        raise ValueError(
            f"{path} is not a plain tensor state dict (torch.load weights_only "
            f"refused it: {e}). A training resume checkpoint is not servable; "
            f"ship model_best.bin.") from e
    if not isinstance(sd, dict):
        raise ValueError(f"{path}: expected a state dict, got {type(sd).__name__}")
    resume_keys = _RESUME_KEYS & set(sd)
    if resume_keys:
        raise ValueError(
            f"{path} is a training resume checkpoint (keys {sorted(resume_keys)}); "
            f"ship model_best.bin instead.")
    non_tensor = [k for k, v in sd.items() if not isinstance(v, torch.Tensor)]
    if non_tensor:
        raise ValueError(f"{path}: non-tensor values under {non_tensor[:3]}")
    head = [k for k in sd if k.startswith("final.")]
    if head:
        raise ValueError(
            f"{path} carries an ArcFace head ({len(head)} 'final.*' keys). Serving "
            f"checkpoints must not include it; re-export without 'final.*'.")
    looks_like = describe_checkpoint(sd)
    if looks_like != backbone:
        raise ValueError(
            f"checkpoint {path} does not match configured backbone '{backbone}': "
            f"it looks like {looks_like}.")
    if "bn.weight" not in sd:
        raise ValueError(f"{path}: missing 'bn.weight'")
    if backbone == BACKBONE_DINOV3_VITB16:
        want = (2 * VITB16.hidden_size,)
        if tuple(sd["bn.weight"].shape) != want:
            raise ValueError(
                f"{path}: bn.weight has shape {tuple(sd['bn.weight'].shape)}, "
                f"expected {want} for {backbone}")
    return sd
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_miewid_checkpoint_validation.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add app/utils/checkpoint_utils.py app/models/miewid.py tests/test_miewid_checkpoint_validation.py
git commit -m "feat(miewid): validate standalone checkpoints; add sha256_file

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Co-Authored-By: Codex <noreply@openai.com>"
```

---

### Task 4: `MiewidModel.load`: dispatch, assertions, digest, `model_info`, docs

**Files:**
- Modify: `app/models/miewid.py`: `MiewidModel` class attributes, `__init__`, `load`, `_load_from_huggingface`, `_load_from_checkpoint`, `_load_standalone`; new `_load_dinov3`, `_parse_load_options`, `_check_assertions`, `resolve_precision`, `build_miewid_transform`
- Modify: `README.md` (the `**MiewID** (\`miewid\`):` parameter list, around line 636)
- Test: `tests/test_miewid_dispatch.py`

**Interfaces:**
- Consumes: `DinoV3MiewIdNet`, `DINOV3_IMAGE_SIZE` (Task 2); `load_standalone_state_dict`, backbone constants, `sha256_file` (Task 3).
- Produces: on `MiewidModel`: class attrs `IMAGE_SIZE = (440, 440)`, `supports_pairx = True`, `backbone_name = None`; instance attrs set by `load()`: `image_size: Tuple[int,int]`, `backbone_name: str` (`"efficientnetv2_rw_m"`, `"dinov3_vitb16"`, or `"hf:<repo>"`), `embedding_dim: Optional[int]`, `precision: str` (`"fp32"` or `"fp16_autocast"`), `checkpoint_sha256: Optional[str]`, `supports_pairx: bool`, `preprocess`. Module functions: `resolve_precision(backbone_name: Optional[str], device: str) -> str`, `build_miewid_transform(image_size: Tuple[int, int]) -> albumentations.Compose`.

- [ ] **Step 1: Write the failing tests**

`tests/test_miewid_dispatch.py`:

```python
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


def test_legacy_v4_with_arcface_head_names_the_head(tmp_path):
    sd = {"backbone.conv_stem.weight": torch.zeros(1), "bn.weight": torch.ones(2152),
          "final.weight": torch.zeros(3, 2152)}
    path = tmp_path / "wbia.bin"
    torch.save(sd, path)
    with pytest.raises(ValueError, match="ArcFace head"):
        _load(4.1, checkpoint_path=str(path))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_miewid_dispatch.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_precision'`.

- [ ] **Step 3: Implement**

In `app/models/miewid.py`, add `import re` to the imports and extend the existing import line to
`from app.utils.checkpoint_utils import get_checkpoint_path, sha256_file`.

Add these module functions after `load_standalone_state_dict`:

```python
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


def build_miewid_transform(image_size: Tuple[int, int]) -> albumentations.Compose:
    """wbia-plugin-miew-id's get_valid/test_transforms at `image_size`.

    albumentations 1.3.1 Normalize defaults are ImageNet mean/std with
    max_pixel_value=255. The models were trained against cv2 bilinear
    resize, which is what albumentations uses; torchvision's PIL resize
    measured ~3 degrees of angular drift per embedding.
    """
    return albumentations.Compose([
        albumentations.Resize(*image_size),
        albumentations.Normalize(),
        ToTensorV2(),
    ])


def resolve_precision(backbone_name: Optional[str], device: str) -> str:
    """fp16 autocast for DINOv3 on CUDA (as training evaluated); fp32 otherwise.

    EfficientNet and HF stay fp32: changing them would move embeddings
    against every existing Wildbook index.
    """
    if backbone_name == BACKBONE_DINOV3_VITB16 and torch.device(device).type == "cuda":
        return "fp16_autocast"
    return "fp32"


def _parse_load_options(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the config fields this design adds, before anything loads."""
    backbone = kwargs.get("backbone")
    checkpoint_path = kwargs.get("checkpoint_path")
    if backbone is not None and backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported MiewID backbone {backbone!r}; supported: {list(SUPPORTED_BACKBONES)}")
    if backbone is not None and not checkpoint_path:
        raise ValueError(f"MiewID backbone {backbone!r} requires checkpoint_path")

    sha = kwargs.get("checkpoint_sha256")
    if sha is not None:
        if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
            raise ValueError(f"checkpoint_sha256 must be 64 hex characters, got {sha!r}")
        if not checkpoint_path:
            raise ValueError("checkpoint_sha256 is set but checkpoint_path is not")
        sha = sha.lower()

    for name in ("imgsz", "embedding_dim"):
        value = kwargs.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{name} must be an int, got {value!r}")

    return {
        "backbone": backbone,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha,
        "imgsz": kwargs.get("imgsz"),
        "embedding_dim": kwargs.get("embedding_dim"),
    }
```

In `class MiewidModel`, replace the `IMAGE_SIZE` comment block and constant with:

```python
    # Backward-compatible alias for the EfficientNet/HF input size. Nothing
    # in this service reads it any more: every instance carries its own
    # `image_size` (448 for DINOv3), and /explain uses `model.preprocess`.
    IMAGE_SIZE = (440, 440)
    # Class-level defaults so MagicMock(spec=MiewidModel) stubs expose them.
    supports_pairx = True
    backbone_name: Optional[str] = None
```

At the end of `__init__` (after `self.inference_lock = threading.Lock()`), add:

```python
        self.image_size = self.IMAGE_SIZE
        self.preprocess = build_miewid_transform(self.image_size)
        self.backbone_name = None
        self.embedding_dim = None
        self.precision = "fp32"
        self.checkpoint_sha256 = None
        self.supports_pairx = True
```
and delete the earlier `self.preprocess = None` line in `__init__`.

Replace `load` entirely:

```python
    def load(self, model_path: str = "", device: str = "cuda", **kwargs) -> None:
        opts = _parse_load_options(kwargs)
        self.device = device
        checkpoint_path = opts["checkpoint_path"]
        backbone = opts["backbone"]
        version = kwargs.get('version', 3)
        # Remove checkpoint_path from kwargs to avoid duplicate parameter error
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'checkpoint_path'}

        local_checkpoint_path = None
        if checkpoint_path:
            self.use_checkpoint = True
            local_checkpoint_path = get_checkpoint_path(checkpoint_path)
            self.checkpoint_sha256 = sha256_file(local_checkpoint_path)
            expected = opts["checkpoint_sha256"]
            if expected is not None and self.checkpoint_sha256 != expected:
                raise ValueError(
                    f"checkpoint_sha256 mismatch for {checkpoint_path} (local file "
                    f"{local_checkpoint_path}): expected {expected}, got {self.checkpoint_sha256}")
        else:
            self.use_checkpoint = False

        # `backbone` first; without it, today's dispatch exactly (spec, legacy table).
        if backbone == BACKBONE_DINOV3_VITB16:
            self._load_dinov3(local_checkpoint_path, device)
        elif backbone == BACKBONE_EFFNET or (
                checkpoint_path and isinstance(version, (int, float)) and version >= 4):
            self._load_standalone(local_checkpoint_path, device)
        elif checkpoint_path:
            self._load_from_checkpoint(local_checkpoint_path, device, **filtered_kwargs)
        else:
            self._load_from_huggingface(device, **kwargs)

        self.precision = resolve_precision(self.backbone_name, device)
        if self.backbone_name == BACKBONE_DINOV3_VITB16 and self.precision == "fp32":
            logger.warning(
                "DINOv3 MiewID on %s runs fp32; training evaluated under CUDA fp16 "
                "autocast, so embeddings will differ slightly from the reference.", device)
        self._check_assertions(opts)

        # Matches wbia-plugin-miew-id's get_valid_transforms at this
        # instance's size, so embeddings agree with the training pipeline.
        self.preprocess = build_miewid_transform(self.image_size)

        self.model_info = {
            'model_type': 'miewid',
            'device': device,
            'imgsz': self.image_size[0],
            'version': kwargs.get('version', 3),
            'checkpoint_path': checkpoint_path,
            'use_checkpoint': self.use_checkpoint,
            'backbone': self.backbone_name,
            'image_size': list(self.image_size),
            'embedding_dim': self.embedding_dim,
            'precision': self.precision,
            'checkpoint_sha256': self.checkpoint_sha256,
            'supports_pairx': self.supports_pairx,
        }

    def _check_assertions(self, opts: Dict[str, Any]) -> None:
        """`imgsz` / `embedding_dim` in config are assertions, never settings."""
        imgsz = opts["imgsz"]
        if imgsz is not None and imgsz != self.image_size[0]:
            raise ValueError(
                f"imgsz {imgsz} does not match backbone {self.backbone_name}, "
                f"which runs at {self.image_size[0]}")
        embedding_dim = opts["embedding_dim"]
        if embedding_dim is not None:
            if self.embedding_dim is None:
                raise ValueError(
                    f"embedding_dim cannot be asserted for {self.backbone_name}: "
                    f"HuggingFace MiewID models do not expose it statically")
            if embedding_dim != self.embedding_dim:
                raise ValueError(
                    f"embedding_dim {embedding_dim} does not match the checkpoint's "
                    f"{self.embedding_dim} ({self.backbone_name})")
```

In `_load_from_huggingface`, after `model_tag` is chosen, add `self.backbone_name = f"hf:{model_tag}"`. Do the same in `_load_from_checkpoint`. Leave the tag selection (`kwargs.get('version', 3) == 3`) untouched.

Replace `_load_standalone`, and add `_install` and `_load_dinov3`:

```python
    def _install(self, model: nn.Module, device: str) -> None:
        model.eval()
        model.to(device)
        self.model = model

    def _load_standalone(self, checkpoint_path: str, device: str) -> None:
        """Standalone timm EfficientNet MiewID (v4+), strict load."""
        sd = load_standalone_state_dict(checkpoint_path, BACKBONE_EFFNET)
        model = MiewIdNet()
        model.load_state_dict(sd, strict=True)
        self.embedding_dim = int(sd["bn.weight"].shape[0])
        del sd
        self._install(model, device)
        self.backbone_name = BACKBONE_EFFNET
        logger.info(f"Loaded MiewID EfficientNet from checkpoint: {checkpoint_path}")

    def _load_dinov3(self, checkpoint_path: str, device: str) -> None:
        """DINOv3 ViT-B/16 MiewID (v5 DINOv3), strict load, built from the
        frozen local config: nothing is downloaded (the HF repo is gated)."""
        sd = load_standalone_state_dict(checkpoint_path, BACKBONE_DINOV3_VITB16)
        model = DinoV3MiewIdNet()
        model.load_state_dict(sd, strict=True)
        del sd
        self._install(model, device)
        self.backbone_name = BACKBONE_DINOV3_VITB16
        self.embedding_dim = model.final_in_features
        self.image_size = DINOV3_IMAGE_SIZE
        self.supports_pairx = False
        logger.info(f"Loaded MiewID DINOv3 from checkpoint: {checkpoint_path}")
```

`_apply_checkpoint` stays as is; only the HF overlay uses it now.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_miewid_dispatch.py -v`
Expected: all pass (the module-scoped fixture writes one ~340 MB temp file; allow ~1 minute).

- [ ] **Step 5: Run the existing MiewID-adjacent suites for regressions**

Run: `python3 -m pytest tests/ -q -k "miewid or explain or pipeline or exif or pairx"`
Expected: no new failures compared with `git stash; python3 -m pytest tests/ -q -k "..."; git stash pop` on this branch's parent commit. If a test fails, record the test name and output and fix the cause, not the test, unless the test asserts the class-level `IMAGE_SIZE` being read by consumers (it now is not).

- [ ] **Step 6: Document the config fields**

In `README.md`, replace the MiewID parameter block (around line 636):

```markdown
**MiewID** (`miewid`):
- `checkpoint_path`: Path or URL to model binary
- `backbone` (optional): `"dinov3_vitb16"` for the MiewID v5 DINOv3 variant (448×448 input, 1536-d embedding, fp16 autocast on CUDA, no PairX), or `"efficientnetv2_rw_m"` to name the standalone EfficientNet explicitly. Both require `checkpoint_path`. Omit it for existing entries: dispatch is then by `version` exactly as before.
- `checkpoint_sha256` (optional): expected SHA-256 of the checkpoint file. A mismatch fails startup. Recommended for every URL checkpoint, because the download cache is keyed by URL, so a file replaced at the same URL is otherwise served stale. Required in practice for DINOv3 release entries.
- `imgsz` (optional, int): asserted against the backbone's input size (440 for EfficientNet/HF, 448 for DINOv3); a mismatch fails startup. It does not change the input size.
- `embedding_dim` (optional, int): asserted against the checkpoint (2152 EfficientNet v5, 1536 DINOv3). Not allowed on the HF (msv2/msv3) paths.

DINOv3 entry:

```json
{
  "model_id": "miewid-msv5-dinov3",
  "model_type": "miewid",
  "backbone": "dinov3_vitb16",
  "checkpoint_path": "https://…/miew_id.msv5_dinov3.bin",
  "checkpoint_sha256": "<sha256 from the training session>",
  "version": 5
}
```

DINOv3 and EfficientNet embeddings are separate spaces (1536-d vs 2152-d) and must never share an index or a distance threshold.
```

- [ ] **Step 7: Commit**

```bash
git add app/models/miewid.py tests/test_miewid_dispatch.py README.md
git commit -m "feat(miewid): backbone dispatch, checkpoint_sha256 and config assertions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Co-Authored-By: Codex <noreply@openai.com>"
```

---

### Task 5: Inference: per-instance transform, autocast, finiteness

**Files:**
- Modify: `app/models/miewid.py` (`extract_embeddings`; add `_autocast`; add `import contextlib`)
- Test: `tests/test_miewid_inference.py`

**Interfaces:**
- Consumes: `build_miewid_transform`, `resolve_precision`, instance attrs from Task 4; `DinoV3MiewIdNet` (Task 2).
- Produces: `MiewidModel._autocast() -> ContextManager`; `extract_embeddings` returns a `float32` `np.ndarray[1, D]` and raises `HTTPException(500)` on a non-finite embedding.

- [ ] **Step 1: Write the failing tests**

`tests/test_miewid_inference.py`:

```python
"""MiewID inference: per-instance input size, DINOv3-only autocast, and no
non-finite embedding ever reaching Wildbook."""
import base64
import io
import pathlib
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from app.models import miewid as miewid_mod
from app.models.dinov3_vit import DINOv3ViTConfig
from app.models.miewid import (
    BACKBONE_DINOV3_VITB16, DinoV3MiewIdNet, MiewidModel, build_miewid_transform,
)

FIX = pathlib.Path(__file__).parent / "fixtures" / "dinov3"


def _png(arr):
    # Mode is inferred from the array: (H,W,4) uint8 -> RGBA, (H,W) uint8 -> L,
    # (H,W) uint16 -> I;16. (fromarray's `mode=` argument is deprecated.)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _rgb_png(h=80, w=60):
    rng = np.random.default_rng(0)
    return _png(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _tiny_dinov3_model(device="cpu", precision="fp32"):
    """A MiewidModel wired to the tiny golden DINOv3 net (64x64 input)."""
    golden = torch.load(FIX / "golden_tiny.pt", weights_only=True)
    net = DinoV3MiewIdNet(DINOv3ViTConfig(**golden["config"]),
                          input_size=tuple(golden["input_size"])).eval()
    net.load_state_dict(golden["state_dict"], strict=True)
    m = MiewidModel()
    m.device = device
    m.model = net.to(device)
    m.backbone_name = BACKBONE_DINOV3_VITB16
    m.image_size = tuple(golden["input_size"])
    m.preprocess = build_miewid_transform(m.image_size)
    m.precision = precision
    return m


class _Const(torch.nn.Module):
    def __init__(self, value, dim=8):
        super().__init__()
        self.value, self.dim = value, dim

    def forward(self, x):
        return torch.full((x.shape[0], self.dim), self.value)


def test_bare_instance_defaults():
    m = MiewidModel()
    assert MiewidModel.IMAGE_SIZE == (440, 440)
    assert m.image_size == (440, 440)
    assert m.precision == "fp32"
    out = m.preprocess(image=np.zeros((30, 20, 3), np.uint8))["image"]
    assert tuple(out.shape) == (3, 440, 440)


def test_dinov3_instance_preprocesses_to_its_own_size():
    m = MiewidModel()
    m.image_size = (448, 448)
    m.preprocess = build_miewid_transform(m.image_size)
    assert tuple(m.preprocess(image=np.zeros((30, 20, 3), np.uint8))["image"].shape) == (3, 448, 448)


def test_tiny_dinov3_extract_returns_float32():
    m = _tiny_dinov3_model()
    emb = m.extract_embeddings(_rgb_png(), bbox=(5, 5, 40, 50), theta=0.3)
    assert emb.shape == (1, 128) and emb.dtype == np.float32
    assert np.isfinite(emb).all()


@pytest.mark.parametrize("make", [
    lambda: _png(np.zeros((40, 40, 4), np.uint8)),
    lambda: _png(np.zeros((40, 40), np.uint8)),
    lambda: _png(np.arange(1600, dtype=np.uint16).reshape(40, 40) * 40),
])
def test_dinov3_accepts_non_rgb_inputs(make):
    emb = _tiny_dinov3_model().extract_embeddings(make())
    assert emb.shape == (1, 128) and np.isfinite(emb).all()


def test_fp32_models_never_enter_autocast(monkeypatch):
    spy = MagicMock()
    monkeypatch.setattr(miewid_mod.torch, "autocast", spy)
    m = MiewidModel()
    m.device = "cpu"
    m.model = _Const(1.0)
    m.extract_embeddings(_rgb_png())
    spy.assert_not_called()


def test_dinov3_cuda_precision_selects_fp16_autocast(monkeypatch):
    spy = MagicMock()
    monkeypatch.setattr(miewid_mod.torch, "autocast", spy)
    m = MiewidModel()
    m.precision = "fp16_autocast"
    m._autocast()
    spy.assert_called_once_with(device_type="cuda", dtype=torch.float16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dinov3_runs_fp16_internally_on_cuda_and_returns_float32():
    m = _tiny_dinov3_model(device="cuda", precision="fp16_autocast")
    seen = {}
    m.model.backbone.model.layer[0].mlp.up_proj.register_forward_hook(
        lambda mod, inp, out: seen.setdefault("dtype", out.dtype))
    emb = m.extract_embeddings(_rgb_png())
    assert seen["dtype"] == torch.float16
    assert next(m.model.parameters()).dtype == torch.float32
    assert emb.dtype == np.float32


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_embedding_is_a_500(bad):
    m = MiewidModel()
    m.device = "cpu"
    m.model = _Const(bad)
    with pytest.raises(HTTPException) as e:
        m.extract_embeddings(_rgb_png())
    assert e.value.status_code == 500


def _nan_model():
    m = MiewidModel()
    m.device = "cpu"
    m.model = _Const(float("nan"))
    return m


def _app(router, models, infos):
    app = FastAPI()
    app.include_router(router)
    handler = MagicMock()
    handler.get_model.side_effect = models.get
    handler.get_model_info.side_effect = infos.get
    handler.list_models.return_value = {k: {} for k in models}
    app.state.model_handler = handler
    return TestClient(app, raise_server_exceptions=False)


DATA_URI = "data:image/png;base64," + base64.b64encode(_rgb_png()).decode()


def test_extract_endpoint_returns_500_not_nan():
    from app.routers import extract_router
    client = _app(extract_router.router, {"e": _nan_model()}, {"e": {"config": {"version": 5}}})
    r = client.post("/extract/", json={"model_id": "e", "image_uri": DATA_URI})
    assert r.status_code == 500, r.text
    assert "NaN" not in r.text


def test_pipeline_endpoint_returns_500_not_nan():
    from app.models.lightnet_model import LightNetModel
    from app.routers import pipeline_router
    detector = MagicMock(spec=LightNetModel)
    detector.predict.return_value = {"bboxes": [[1, 1, 20, 20]], "scores": [0.9],
                                     "class_names": ["x"], "num_detections": 1}
    client = _app(pipeline_router.router, {"p": detector, "e": _nan_model()},
                  {"p": {"config": {}}, "e": {"config": {"version": 5}}})
    r = client.post("/pipeline/", json={"predict_model_id": "p", "extract_model_id": "e",
                                        "image_uri": DATA_URI})
    assert r.status_code == 500, r.text
    assert "NaN" not in r.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_miewid_inference.py -v`
Expected: FAIL. `test_non_finite_embedding_is_a_500` returns NaN instead of raising; `test_dinov3_cuda_precision_selects_fp16_autocast` fails with `AttributeError: '_autocast'`.

- [ ] **Step 3: Implement**

Add `import contextlib` to `app/models/miewid.py` imports. Add to `MiewidModel`:

```python
    def _autocast(self):
        """fp16 autocast for DINOv3 on CUDA only (resolve_precision)."""
        if self.precision == "fp16_autocast":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()
```

In `extract_embeddings`, replace the forward/return block:

```python
            # Extract embeddings. The lock is held only across the forward,
            # not the decode/preprocess above it, so a slow image never blocks
            # another endpoint's inference.
            with self.inference_lock, torch.no_grad():
                embeddings = self.model(input_batch)
            
            # Convert to numpy and return
            return embeddings.cpu().numpy()
```

with:

```python
            # Extract embeddings. The lock is held only across the forward,
            # not the decode/preprocess above it, so a slow image never blocks
            # another endpoint's inference. Autocast covers backbone AND neck
            # for DINOv3 (as training evaluated) and is a no-op otherwise.
            with self.inference_lock, torch.no_grad(), self._autocast():
                embeddings = self.model(input_batch)
            embeddings = embeddings.float()

            # An fp16 overflow shows up as inf/NaN. Never hand that to
            # Wildbook as an embedding: the except below turns it into the
            # existing 500, which /extract and /pipeline both propagate.
            if not torch.isfinite(embeddings).all():
                raise ValueError(
                    f"non-finite embedding from {self.backbone_name} "
                    f"(precision {self.precision})")
            return embeddings.cpu().numpy()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_miewid_inference.py -v`
Expected: all pass (the CUDA test passes on a GPU host and is skipped otherwise; say which in your report).

- [ ] **Step 5: Commit**

```bash
git add app/models/miewid.py tests/test_miewid_inference.py
git commit -m "feat(miewid): DINOv3-only fp16 autocast; reject non-finite embeddings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Co-Authored-By: Codex <noreply@openai.com>"
```

---

### Task 6: `/explain`: per-instance transform and PairX fail-closed

**Files:**
- Modify: `app/routers/explain_router.py` (`preprocess`, lines ~50-72; `resolve_pairx_model`, lines ~162-182; drop the now-unused `albumentations`/`ToTensorV2` imports only if nothing else in the file uses them)
- Modify stubs: `tests/test_explain_router_model_resolution.py` (`_miewid`), `tests/test_explain_router_fetch.py` (`_miewid`), `tests/test_pairx_offset_and_color.py` (`_miewid_stub`), `tests/test_explain_router_model_lock.py` (the `entry = MagicMock(spec=MiewidModel)` around line 129)
- Test: `tests/test_explain_router_pairx_capability.py`

**Interfaces:**
- Consumes: `MiewidModel.supports_pairx`, `.backbone_name`, `.preprocess`, `build_miewid_transform` (Task 4).
- Produces: `resolve_pairx_model` raises `HTTPException(400)` for a model with `supports_pairx == False`; `explain_router.preprocess(image, model)` uses `model.preprocess`.

- [ ] **Step 1: Write the failing tests**

`tests/test_explain_router_pairx_capability.py`:

```python
"""PairX fails closed for DINOv3 MiewID models.

PairX hooks spatial CNN feature maps and has no ViT canonizer. A DINOv3
module name would pass the router's named_modules() check and then fail deep
inside PairX, so the capability check must run at model resolution, before
any image is fetched, whatever layer_key the caller sends.
"""
import inspect
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.miewid import MiewidModel, build_miewid_transform
from app.routers import explain_router


@pytest.fixture(autouse=True)
def _restore_default_model_id():
    original = explain_router._default_model_id
    yield
    explain_router._default_model_id = original


def _dinov3_stub():
    m = MagicMock(spec=MiewidModel)
    m.inference_lock = threading.RLock()
    m.supports_pairx = False
    m.backbone_name = "dinov3_vitb16"
    m.preprocess = build_miewid_transform((448, 448))
    return m


def _client(models):
    app = FastAPI()
    app.include_router(explain_router.router)
    handler = MagicMock()
    handler.get_model.side_effect = models.get
    handler.list_models.return_value = {k: {} for k in models}
    app.state.model_handler = handler
    app.state.device = "cpu"
    return TestClient(app, raise_server_exceptions=False)


def _body(**over):
    b = {"image1_uris": ["https://example.invalid/a.jpg"], "bb1": [[0, 0, 10, 10]],
         "image2_uris": ["https://example.invalid/b.jpg"], "bb2": [[0, 0, 10, 10]]}
    b.update(over)
    return b


@pytest.mark.parametrize("extra", [{}, {"layer_key": "backbone.model.layer.3"}])
def test_dinov3_model_is_400_before_any_fetch(extra):
    client = _client({"dino": _dinov3_stub()})
    with patch.object(explain_router, "process_image") as fetch:
        r = client.post("/explain/", json=_body(model_id="dino", **extra))
    assert r.status_code == 400, r.text
    assert "PairX is not supported" in r.text and "dinov3_vitb16" in r.text
    fetch.assert_not_called()


def test_default_model_id_pointing_at_dinov3_is_400():
    explain_router._default_model_id = "dino"
    client = _client({"dino": _dinov3_stub()})
    with patch.object(explain_router, "process_image") as fetch:
        r = client.post("/explain/", json=_body())
    assert r.status_code == 400, r.text
    fetch.assert_not_called()


def test_explain_preprocess_uses_the_instance_transform():
    out = explain_router.preprocess(np.zeros((50, 60, 3), np.uint8), _dinov3_stub())
    assert tuple(out.shape) == (3, 448, 448)


def test_explain_router_no_longer_reads_the_class_constant():
    assert "IMAGE_SIZE" not in inspect.getsource(explain_router)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_explain_router_pairx_capability.py -v`
Expected: FAIL. The DINOv3 stub passes resolution, the preprocess output is 440, and `IMAGE_SIZE` is still in the source.

- [ ] **Step 3: Implement**

In `app/routers/explain_router.py`, replace the body of the `if isinstance(model, MiewidModel):` branch in `preprocess` with:

```python
    if isinstance(model, MiewidModel):
        # The instance's own transform, so PairX sees exactly the tensor
        # /extract embeds (448 for DINOv3, 440 otherwise). Building it here
        # from a class constant once fed 440 inputs to every model.
        augmented = model.preprocess(image=image.astype("uint8"))
        return augmented["image"]
```

In `resolve_pairx_model`, before `return model_entry`:

```python
    if not model_entry.supports_pairx:
        # Also covers an explicit layer_key: a ViT module name passes the
        # named_modules() check and then fails deep inside PairX.
        raise HTTPException(
            status_code=400,
            detail=f"PairX is not supported for model '{model_id}' "
                   f"(backbone {model_entry.backbone_name})",
        )
```

Update the docstring line "Raises 404 ... and 400 when it is loaded but is not a MiewID model" to add "or is a MiewID model PairX cannot explain (DINOv3)". Remove the `albumentations` and `ToTensorV2` imports from `explain_router.py` only if `grep -n "albumentations\|ToTensorV2" app/routers/explain_router.py` shows no other use.

In each of the four stub sites listed under **Files**, add after `inference_lock` is set (use the local variable name at that site, `m` or `entry`):

```python
    # Real instances build this in __init__; spec= only exposes class attrs.
    m.preprocess = build_miewid_transform((440, 440))
```
and extend that file's `from app.models.miewid import MiewidModel` to `from app.models.miewid import MiewidModel, build_miewid_transform`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_explain_router_pairx_capability.py tests/test_explain_router_model_resolution.py tests/test_explain_router_fetch.py tests/test_pairx_offset_and_color.py tests/test_explain_router_model_lock.py tests/test_explain_router_pairx_errors.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/routers/explain_router.py tests/test_explain_router_pairx_capability.py tests/test_explain_router_model_resolution.py tests/test_explain_router_fetch.py tests/test_pairx_offset_and_color.py tests/test_explain_router_model_lock.py
git commit -m "feat(explain): per-instance MiewID transform; PairX fails closed for DINOv3

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Co-Authored-By: Codex <noreply@openai.com>"
```

---

### Task 7: Real-checkpoint integration and prod-image run

**Files:**
- Test: `tests/test_miewid_dinov3_real_checkpoint.py`

**Interfaces:**
- Consumes: the full `MiewidModel` from Tasks 4-5.

- [ ] **Step 1: Write the test**

`tests/test_miewid_dinov3_real_checkpoint.py`:

```python
"""Load the real DEV DINOv3 checkpoint (and the shipped v5 EfficientNet when
present) through the production config path. Skipped where the files are
absent (CI). The DEV checkpoint must never ship or gate a release."""
import io
import os

import numpy as np
import pytest
from PIL import Image

from app.models.miewid import MiewidModel

DEV = os.environ.get(
    "MIEWID_DINOV3_DEV_CKPT",
    "/mnt/c/claude-skills/models/reference/miewid-dinov3-dev/miew_id.msv5_dinov3.DEV-ep10.bin")
DEV_SHA = "1c30dec4cf20e82da5cb9a9e2bf53efe4a7080c75ec3db3416952f592b356735"
EFFNET = os.environ.get(
    "MIEWID_EFFNET_V5_CKPT",
    "/mnt/c/claude-skills/models/reference/miewIDv5/_model/miew_id.msv5_main.bin")

pytestmark = pytest.mark.skipif(not os.path.exists(DEV), reason="DEV checkpoint not present")


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
    cos = float((a1 @ b.T) / (np.linalg.norm(a1) * np.linalg.norm(b)))
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
```

- [ ] **Step 2: Run it**

Run: `python3 -m pytest tests/test_miewid_dinov3_real_checkpoint.py -v`
Expected: 2 passed (not skipped) on this host. If `test_dinov3_and_effnet_coexist` fails on the EfficientNet load with an ArcFace-head or mismatch error, report the message: it means `miew_id.msv5_main.bin` is not the bare format the training session described.

- [ ] **Step 3: Run the full suite on the dev host**

Run: `python3 -m pytest tests/ -q`
Expected: no failures beyond those already failing on `origin/main` (check with `git stash` if any fail).

- [ ] **Step 4: Run the new tests inside the production image (torch 2.1.2, Python 3.10)**

```bash
docker build -t ml-service -f docker/dockerfile .
docker run --rm --gpus all --entrypoint "" -v "$PWD:/src" -w /src \
  -v /mnt/c/claude-skills/models/reference:/mnt/c/claude-skills/models/reference:ro \
  ml-service bash -c "pip install -q pytest==8.3.3 && python3 -m pytest -q \
    tests/test_dinov3_vit.py tests/test_miewid_dinov3_net.py \
    tests/test_miewid_checkpoint_validation.py tests/test_miewid_dispatch.py \
    tests/test_miewid_inference.py tests/test_explain_router_pairx_capability.py \
    tests/test_miewid_dinov3_real_checkpoint.py"
```
Expected: all pass, including the CUDA-only test (no `--gpus all` → it skips; report which). This is the check that the vendored module runs on torch 2.1.2. A failure here that does not reproduce on the dev host is a torch-2.1 incompatibility: fix it in `dinov3_vit.py` and re-run the Task 1 golden test.

- [ ] **Step 4b: Measure GPU memory (spec, "Resource cost")**

Write this to `/tmp/vram_probe.py` (scratch, not committed):

```python
import io

import numpy as np
import torch
from PIL import Image

from app.models.miewid import MiewidModel

R = "/mnt/c/claude-skills/models/reference"
dino, eff = MiewidModel(), MiewidModel()
dino.load(device="cuda", model_id="d", backbone="dinov3_vitb16",
          checkpoint_path=f"{R}/miewid-dinov3-dev/miew_id.msv5_dinov3.DEV-ep10.bin")
eff.load(device="cuda", model_id="e", version=5,
         checkpoint_path=f"{R}/miewIDv5/_model/miew_id.msv5_main.bin")
buf = io.BytesIO()
Image.fromarray(np.zeros((1200, 1600, 3), np.uint8)).save(buf, format="JPEG")
torch.cuda.reset_peak_memory_stats()
dino.extract_embeddings(buf.getvalue())
eff.extract_embeddings(buf.getvalue())
print(f"resident {torch.cuda.memory_allocated() / 2**20:.0f} MiB, "
      f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
```

```bash
docker run --rm --gpus all --entrypoint "" -v "$PWD:/src" -w /src \
  -v /tmp/vram_probe.py:/tmp/vram_probe.py:ro \
  -v /mnt/c/claude-skills/models/reference:/mnt/c/claude-skills/models/reference:ro \
  ml-service python3 /tmp/vram_probe.py
```
Record both numbers in the PR description. The DINOv3 instance's increment is what the prod GPU must have free on top of today's models.

- [ ] **Step 5: List prod MiewID entries that set `imgsz` or `embedding_dim`**

These are now assertions. A prod entry with a wrong or non-int value would fail startup. The prod config is not in this repo (`docs/` and memory: repo `model_config.json` drifts from prod), so print this check for the operator in the PR description:

```bash
python3 - "$MODEL_CONFIG" <<'EOF'
import json, sys
for e in json.load(open(sys.argv[1]))["models"]:
    if e.get("model_type") == "miewid" and ("imgsz" in e or "embedding_dim" in e):
        print(e["model_id"], {k: e.get(k) for k in ("imgsz", "embedding_dim", "version", "backbone")})
EOF
```
Any entry printed with `imgsz` other than the int `440` (EfficientNet/HF) must be fixed in prod config before deploying this branch.

- [ ] **Step 6: Commit**

```bash
git add tests/test_miewid_dinov3_real_checkpoint.py
git commit -m "test(miewid): load the DEV DINOv3 checkpoint beside v5 EfficientNet

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Co-Authored-By: Codex <noreply@openai.com>"
```

---

### Task 8: Parity-gate tooling (release-blocking; runs on the final checkpoint)

**Files:**
- Create: `scripts/preflight/miewid_dinov3/metrics.py`
- Create: `scripts/preflight/miewid_dinov3/reference_runner.py`
- Create: `scripts/preflight/miewid_dinov3/run_gate.py`
- Create: `scripts/preflight/miewid_dinov3/manifest.json`
- Create: `scripts/preflight/miewid_dinov3/README.md`
- Test: `tests/test_miewid_dinov3_gate_metrics.py`

**Interfaces:**
- Consumes: `MiewidModel`, `DinoV3MiewIdNet.forward_stages` (Tasks 2, 4, 5).
- Produces: `metrics.max_abs(a, b) -> float`, `metrics.rel_l2(a, b) -> float` (‖a−b‖/‖b‖), `metrics.cosine_distance(a, b) -> float`, `metrics.topk_agreement(ref: ndarray[N,D], got: ndarray[N,D], k: int) -> float`, `metrics.check_stage(name, got, ref, limits) -> dict` with `{"stage", "max_abs", "rel_l2", "pass"}`; the reference artifact format `reference.npz` + `reference.json` described below.

The gate has two halves because the two implementations need different stacks. `reference_runner.py` runs in the **training venv** (transformers 5.17, torch 2.6) with the training repo's own code. `run_gate.py` runs in the **production image** (torch 2.1.2). They meet through files: same image bytes in, per-stage arrays out.

- [ ] **Step 1: Write the failing metrics tests**

`tests/test_miewid_dinov3_gate_metrics.py`:

```python
"""The parity gate's metrics. The negative control matters most: cosine and
top-k are both blind to a uniformly scaled embedding, so the raw-error check
must catch it or the gate is not live."""
import numpy as np

from scripts.preflight.miewid_dinov3 import metrics

LIMITS = {"max_abs": 5e-2, "rel_l2": 1e-2}


def test_identical_passes():
    a = np.random.default_rng(0).normal(size=(19, 1536)).astype(np.float32)
    assert metrics.check_stage("embedding", a, a, LIMITS)["pass"]
    assert metrics.topk_agreement(a, a, 5) == 1.0
    assert metrics.cosine_distance(a[0], a[0]) < 1e-6


def test_scaled_embedding_fails_raw_check_but_not_cosine():
    a = np.random.default_rng(0).normal(size=(19, 1536)).astype(np.float32)
    scaled = a * 1.05
    assert not metrics.check_stage("embedding", scaled, a, LIMITS)["pass"]
    assert metrics.cosine_distance(scaled[0], a[0]) < 1e-6
    assert metrics.topk_agreement(scaled, a, 5) == 1.0


def test_rel_l2_and_max_abs():
    ref = np.array([3.0, 4.0])
    got = np.array([3.0, 4.5])
    assert metrics.max_abs(got, ref) == 0.5
    assert abs(metrics.rel_l2(got, ref) - 0.1) < 1e-12


def test_topk_agreement_detects_reordering():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(10, 8))
    got = ref.copy()
    got[[0, 1]] = got[[1, 0]]
    assert metrics.topk_agreement(got, ref, 3) < 1.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m pytest tests/test_miewid_dinov3_gate_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.preflight.miewid_dinov3'`.

- [ ] **Step 3: Implement `metrics.py`**

```python
"""Parity-gate metrics for the MiewID DINOv3 port. Pure numpy, no torch, so
the same code scores runs from either environment."""
import numpy as np


def max_abs(got, ref) -> float:
    return float(np.max(np.abs(np.asarray(got, np.float64) - np.asarray(ref, np.float64))))


def rel_l2(got, ref) -> float:
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def cosine_distance(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(1.0 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _topk(emb, k):
    e = np.asarray(emb, np.float64)
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    sim = e @ e.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1, kind="stable")[:, :k]


def topk_agreement(got, ref, k: int) -> float:
    """Fraction of queries whose ordered top-k neighbours match exactly."""
    return float(np.mean(np.all(_topk(got, k) == _topk(ref, k), axis=1)))


def check_stage(name, got, ref, limits) -> dict:
    result = {"stage": name, "max_abs": max_abs(got, ref), "rel_l2": rel_l2(got, ref)}
    result["pass"] = result["max_abs"] <= limits["max_abs"] and result["rel_l2"] <= limits["rel_l2"]
    return result
```

Run: `python3 -m pytest tests/test_miewid_dinov3_gate_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 4: Write `manifest.json`**

```json
{
  "_note": "INPUT manifest for the MiewID DINOv3 parity gate. Inputs and limits only; both implementations are executed live. Thresholds are PROPOSED until agreed with the training owner (spec, Parity gate step 3); the gate prints that status.",
  "thresholds_status": "proposed",
  "thresholds": {
    "arch": {
      "tokens":    {"max_abs": 5e-2, "rel_l2": 1e-2},
      "concat":    {"max_abs": 5e-2, "rel_l2": 1e-2},
      "embedding": {"max_abs": 5e-2, "rel_l2": 1e-2}
    },
    "e2e": {
      "input":     {"max_abs": 5e-2, "rel_l2": 1e-2},
      "embedding": {"max_abs": 5e-2, "rel_l2": 1e-2}
    },
    "cosine_distance_max": 1e-3,
    "top5_agreement_min": 0.99,
    "negative_control_scale": 1.05
  },
  "checkpoint": {
    "path": "/datasets/miew_id.msv5_dinov3.bin",
    "sha256": "<final sha256 from the training session>"
  },
  "reference_source": {
    "repo": "wbia-plugin-miew-id DINOv3 experiment tree (audit_v4_1/dinov3-exp)",
    "hf_backbone": "facebook/dinov3-vitb16-pretrain-lvd1689m@5931719e67bbdb9737e363e781fb0c67687896bc"
  },
  "chips_csv": "/fixtures/reference_chips.csv"
}
```

The `<final sha256 ...>` value is filled in at gate time from the training session's end-of-training answer. `run_gate.py` refuses to run while it still starts with `<`.

- [ ] **Step 5: Write `reference_runner.py` (training venv)**

```python
"""Reference half of the MiewID DINOv3 parity gate. Runs in the TRAINING
venv (transformers 5.17, torch 2.6) with the training repo's own code:
load_image (cv2.imread, EXIF applied) -> get_chip_from_img ->
get_test_transforms((448, 448)) -> MiewIdNet, under CUDA fp16 autocast as
training eval runs. Writes per-chip inputs and stage outputs:

  <out>/reference.npz   input[N,3,448,448] tokens[N,789,768] concat[N,1536] embedding[N,1536]
  <out>/reference.json  chip order, image sha256s, environment, checkpoint sha256

Usage (on the training host):
  python reference_runner.py --repo /path/to/dinov3-exp --checkpoint model_final.bin \
      --chips reference_chips.csv --path-column host_path --out /tmp/ref
"""
import argparse
import ast
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="tree containing wbia_miew_id/")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--chips", required=True)
    ap.add_argument("--path-column", default="host_path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from wbia_miew_id.datasets.helpers import get_chip_from_img, load_image
    from wbia_miew_id.datasets.transforms import get_test_transforms
    from wbia_miew_id.models.model import MiewIdNet
    import transformers

    model = MiewIdNet(n_classes=1, model_name="dinov3:facebook/dinov3-vitb16-pretrain-lvd1689m",
                      use_fc=False, pretrained=True)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval().cuda()
    assert model.backbone.config._attn_implementation == "sdpa", model.backbone.config._attn_implementation

    captured = {}
    model.backbone.register_forward_hook(
        lambda mod, inp, out: captured.__setitem__("tokens", out.last_hidden_state.float()))
    model.bn.register_forward_hook(
        lambda mod, inp, out: captured.__setitem__("concat", inp[0].float()))
    transform = get_test_transforms((448, 448))

    rows = list(csv.DictReader(open(args.chips)))
    inputs, tokens, concats, embs, meta = [], [], [], [], []
    for row in rows:
        path = row[args.path_column]
        raw = pathlib.Path(path).read_bytes()
        img = load_image(path)
        chip = get_chip_from_img(img, ast.literal_eval(row["bbox"]), float(row["theta"]))
        x = transform(image=chip)["image"].unsqueeze(0).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            emb = model(x).float()
        inputs.append(x[0].cpu().numpy())
        tokens.append(captured["tokens"][0].cpu().numpy())
        concats.append(captured["concat"][0].cpu().numpy())
        embs.append(emb[0].cpu().numpy())
        meta.append({"case": row["case"], "species": row["species"], "path": path,
                     "bbox": row["bbox"], "theta": row["theta"], "image_sha256": sha256_bytes(raw)})

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "reference.npz", input=np.stack(inputs), tokens=np.stack(tokens),
             concat=np.stack(concats), embedding=np.stack(embs))
    json.dump({
        "chips": meta,
        "checkpoint_sha256": sha256_bytes(pathlib.Path(args.checkpoint).read_bytes()),
        "env": {"torch": torch.__version__, "transformers": transformers.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "attn_implementation": model.backbone.config._attn_implementation},
    }, open(out / "reference.json", "w"), indent=2)
    print(f"wrote {len(rows)} chips to {out}")


if __name__ == "__main__":
    main()
```

Note for the training session: if `MiewIdNet`'s constructor in the experiment tree needs other keyword arguments, use the values from `configs/EDA-msv5-dinov3.yaml`. If `model(x)` there returns anything other than the BN output, use `model.extract_feat(x)` instead and say so in `reference.json`.

- [ ] **Step 6: Write `run_gate.py` (production image)**

```python
"""Service half of the MiewID DINOv3 parity gate. Runs INSIDE the production
image on the production GPU. Compares the service to reference.npz:

  arch: the reference's own preprocessed tensors -> DinoV3MiewIdNet.forward_stages
        (tokens, concat, embedding), under the service's autocast
  e2e:  the same image FILE BYTES -> MiewidModel.extract_embeddings with the bbox
        converted exactly as /extract converts it (int floor); the model's input
        tensor is captured with a pre-hook, never recomputed here
  retrieval: ordered top-5 over the chip set; cosine distance per chip
  negative control: the e2e embeddings x1.05 must FAIL the raw check

Exit 0 only if every check passes AND the negative control fails.
"""
import argparse
import ast
import csv
import json
import pathlib
import sys

import numpy as np
import torch

from app.models.miewid import MiewidModel
from scripts.preflight.miewid_dinov3 import metrics


def rewrite(path, rules):
    for old, new in rules:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--reference", required=True, help="dir with reference.npz/.json")
    ap.add_argument("--chips", required=True)
    ap.add_argument("--path-column", default="host_path")
    ap.add_argument("--rewrite-prefix", action="append", default=[], help="OLD=NEW")
    ap.add_argument("--artifact", required=True)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    th = manifest["thresholds"]
    if manifest["checkpoint"]["sha256"].startswith("<"):
        sys.exit("manifest checkpoint sha256 is still a placeholder; fill it from the training session")
    rules = [tuple(r.split("=", 1)) for r in args.rewrite_prefix]
    ref = np.load(pathlib.Path(args.reference) / "reference.npz")
    ref_meta = json.load(open(pathlib.Path(args.reference) / "reference.json"))

    m = MiewidModel()
    m.load(device="cuda", model_id="gate", backbone="dinov3_vitb16",
           checkpoint_path=manifest["checkpoint"]["path"],
           checkpoint_sha256=manifest["checkpoint"]["sha256"], version=5)
    if m.checkpoint_sha256 != ref_meta["checkpoint_sha256"]:
        sys.exit(f"service checkpoint {m.checkpoint_sha256} != reference {ref_meta['checkpoint_sha256']}")

    # arch: identical tensors in
    arch = {"tokens": [], "concat": [], "embedding": []}
    for x in ref["input"]:
        with torch.no_grad(), m._autocast():
            stages = m.model.forward_stages(torch.from_numpy(x).unsqueeze(0).cuda())
        for k in arch:
            arch[k].append(stages[k][0].float().cpu().numpy())
    results = [metrics.check_stage(f"arch.{k}", np.stack(v), ref[k], th["arch"][k])
               for k, v in arch.items()]

    # e2e: identical bytes in
    captured = []
    m.model.register_forward_pre_hook(lambda mod, a: captured.append(a[0][0].float().cpu().numpy()))
    rows = list(csv.DictReader(open(args.chips)))
    e2e_inputs, e2e_embs, float_bbox_embs = [], [], []
    for row, meta in zip(rows, ref_meta["chips"]):
        path = rewrite(row[args.path_column], rules)
        raw = pathlib.Path(path).read_bytes()
        bbox = ast.literal_eval(row["bbox"])
        theta = float(row["theta"])
        int_bbox = tuple(int(v) for v in bbox)          # /extract's conversion
        captured.clear()
        e2e_embs.append(m.extract_embeddings(raw, bbox=int_bbox, theta=theta)[0])
        e2e_inputs.append(captured[0])
        # Diagnostic only, not gated: float bbox as training passes it.
        float_bbox_embs.append(m.extract_embeddings(raw, bbox=tuple(bbox), theta=theta)[0])
    e2e_embs = np.stack(e2e_embs)
    results.append(metrics.check_stage("e2e.input", np.stack(e2e_inputs), ref["input"], th["e2e"]["input"]))
    results.append(metrics.check_stage("e2e.embedding", e2e_embs, ref["embedding"], th["e2e"]["embedding"]))

    cos = [metrics.cosine_distance(a, b) for a, b in zip(e2e_embs, ref["embedding"])]
    top5 = metrics.topk_agreement(e2e_embs, ref["embedding"], 5)
    results.append({"stage": "e2e.cosine", "max": max(cos), "pass": max(cos) <= th["cosine_distance_max"]})
    results.append({"stage": "e2e.top5", "agreement": top5, "pass": top5 >= th["top5_agreement_min"]})

    control = metrics.check_stage("negative_control", e2e_embs * th["negative_control_scale"],
                                  ref["embedding"], th["e2e"]["embedding"])
    control_ok = not control["pass"]

    diag = {"float_bbox_cosine_max": max(metrics.cosine_distance(a, b)
                                         for a, b in zip(float_bbox_embs, ref["embedding"]))}
    passed = all(r["pass"] for r in results) and control_ok
    artifact = {
        "pass": passed,
        "thresholds_status": manifest["thresholds_status"],
        "results": results,
        "negative_control": {**control, "failed_as_required": control_ok},
        "per_chip_cosine": [{"case": mt["case"], "species": mt["species"], "cosine_distance": c}
                            for mt, c in zip(ref_meta["chips"], cos)],
        "diagnostics": diag,
        "service_env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0),
                        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                        "precision": m.precision},
        "reference_env": ref_meta["env"],
        "checkpoint_sha256": m.checkpoint_sha256,
    }
    json.dump(artifact, open(args.artifact, "w"), indent=2)
    print(json.dumps({"pass": passed, "thresholds_status": manifest["thresholds_status"]}))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Write the gate `README.md`**

`scripts/preflight/miewid_dinov3/README.md`:

````markdown
# MiewID DINOv3 parity gate

Release-blocking. Run on the **final** checkpoint, never the DEV snapshot.
Spec: `docs/plans/2026-10-15-miewid-dinov3-design.md` ("Parity gate").

## Why

`app/models/dinov3_vit.py` is a vendored port. Its failure mode is
embeddings that are slightly wrong, which retrieval cannot show you until
matches go missing. The only trustworthy oracle is the training code itself,
run live on the same bytes.

## 1. Reference (training host, training venv)

```bash
python scripts/preflight/miewid_dinov3/reference_runner.py \
  --repo /path/to/audit_v4_1/dinov3-exp \
  --checkpoint /path/to/miew_id.msv5_dinov3.bin \
  --chips reference_chips.csv --path-column host_path --out /tmp/dinov3-ref
```

Copy `/tmp/dinov3-ref/` and the 19 chip images to the ml-service host.

## 2. Service (ml-service host, INSIDE the production image)

```bash
docker build -t ml-service -f docker/dockerfile .
docker run --rm --gpus all --entrypoint "" -v "$PWD:/src" -w /src \
  -v "$MODELS_DIR:/datasets:ro" -v /path/to/dinov3-ref:/ref:ro \
  -v /path/to/chip-images:/chips:ro -v "$PWD/preflight-artifacts:/artifacts" \
  ml-service python3 scripts/preflight/miewid_dinov3/run_gate.py \
    --manifest scripts/preflight/miewid_dinov3/manifest.json \
    --reference /ref --chips /ref/reference_chips.csv \
    --rewrite-prefix /nas/=/chips/nas/ --rewrite-prefix /data0/=/chips/data0/ \
    --artifact /artifacts/dinov3-gate.json
```

Fill `manifest.json` `checkpoint.sha256` first; the gate refuses a placeholder.

## Reading the result

- `pass` needs every stage within limits **and** the ×1.05 negative control
  failing. Cosine and top-k are blind to scale, so the control proves the
  raw-error check is live.
- `thresholds_status: "proposed"` means the limits have not yet been agreed
  with the training owner. A pass under proposed limits is not a release
  pass.
- e2e uses the bbox exactly as `/extract` converts it (int floor), because
  that is what production does. `diagnostics.float_bbox_cosine_max` shows the
  same comparison with the training-style float bbox. If e2e fails but the
  float-bbox diagnostic passes, the int floor is the cause: report it, do not
  loosen limits.
- The trout and lion chips carry bboxes whose source frame or resolution
  disagrees with the file. Both sides apply the same bbox to the same decoded
  pixels, so they still compare.
````

- [ ] **Step 8: Smoke-test `run_gate.py` argument handling**

Run: `python3 scripts/preflight/miewid_dinov3/run_gate.py --manifest scripts/preflight/miewid_dinov3/manifest.json --reference /nonexistent --chips /nonexistent --artifact /tmp/x.json; echo "exit=$?"`
Expected: exits non-zero with `manifest checkpoint sha256 is still a placeholder`. That proves the placeholder guard runs before anything is loaded.

- [ ] **Step 9: Commit**

```bash
git add scripts/preflight/miewid_dinov3/ tests/test_miewid_dinov3_gate_metrics.py
git commit -m "feat(preflight): MiewID DINOv3 parity gate (reference runner + service gate)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Co-Authored-By: Codex <noreply@openai.com>"
```

---

## After the plan

Not tasks for this branch. They wait on the training session's end-of-training answers:

1. The final checkpoint's URL and SHA-256 go into `manifest.json` and the prod config entry.
2. The training session runs `reference_runner.py` on the final weights.
3. `run_gate.py` runs in the prod image; thresholds are agreed with the training owner, and `thresholds_status` changes to `"agreed"`.
4. The prod config gets the DINOv3 entry (README example), with `checkpoint_sha256` set.
