"""Tests du U-Net from scratch (src/memoire/model/unet.py)."""

import pytest
import torch
from torch import nn

from memoire.model import UNet


def _small_unet(**overrides) -> UNet:
    kwargs = {"in_channels": 3, "num_classes": 2, "base_channels": 16, "depth": 4, "gn_groups": 8}
    kwargs.update(overrides)
    return UNet(**kwargs)


@pytest.mark.parametrize("size", [64, 96])
def test_forward_shape(size):
    model = _small_unet()
    model.eval()
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2, size, size)


def test_forward_rejects_non_multiple_size():
    model = _small_unet()
    with pytest.raises(ValueError, match="multiple"):
        model(torch.randn(1, 3, 60, 60))


def test_no_batchnorm_and_groupnorm_present():
    model = _small_unet()
    batchnorms = [
        m
        for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))
    ]
    assert batchnorms == [], "BatchNorm is forbidden (GroupNorm only)"
    groupnorms = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
    assert len(groupnorms) > 0


def test_gradients_flow_to_all_parameters():
    torch.manual_seed(0)
    model = _small_unet()
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 2, (2, 64, 64))
    loss = nn.functional.cross_entropy(model(x), target)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert param.grad.abs().sum().item() > 0, f"zero gradient for {name}"


def test_param_count_scales_with_base_channels():
    n16 = sum(p.numel() for p in _small_unet(base_channels=16).parameters())
    n32 = sum(p.numel() for p in _small_unet(base_channels=32).parameters())
    ratio = n32 / n16
    # Conv weights dominate and scale as channels**2, so doubling base_channels
    # should roughly quadruple the parameter count.
    assert 3.5 < ratio < 4.5, f"unexpected ratio {ratio:.2f} (n16={n16}, n32={n32})"


def test_init_is_deterministic_with_fixed_seed():
    torch.manual_seed(1234)
    model_a = _small_unet()
    torch.manual_seed(1234)
    model_b = _small_unet()
    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), f"mismatch for {key}"


def test_biases_zero_and_kaiming_scale_at_init():
    torch.manual_seed(0)
    model = _small_unet()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            assert torch.all(module.bias == 0), f"non-zero bias at init for {name}"
            fan_in = module.weight[0].numel()
            expected_std = (2.0 / fan_in) ** 0.5
            actual_std = module.weight.std().item()
            assert actual_std == pytest.approx(expected_std, rel=0.25), (
                f"init std off for {name}: {actual_std} vs {expected_std}"
            )
