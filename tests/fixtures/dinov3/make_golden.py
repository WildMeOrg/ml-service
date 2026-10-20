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
        # Plain str: torch.__version__ is a TorchVersion object, which
        # torch.load(weights_only=True) refuses to unpickle.
        "transformers_version": str(transformers.__version__),
        "torch_version": str(torch.__version__),
    }, out_path)
    print(f"wrote {out_path}: tokens {tuple(tok.shape)}, embedding {tuple(emb.shape)}")


if __name__ == "__main__":
    main(sys.argv[1])
