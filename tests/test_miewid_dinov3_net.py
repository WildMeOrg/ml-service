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
