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
