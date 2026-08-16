"""Tests du modèle de référence pour l'ablation (src/memoire/model/baseline.py)."""

import pytest
import torch
from torch import nn

from memoire.model import PlainEncoderDecoder, UNet


def _small_baseline(**overrides) -> PlainEncoderDecoder:
    kwargs = {"in_channels": 3, "num_classes": 2, "base_channels": 16, "depth": 4, "gn_groups": 8}
    kwargs.update(overrides)
    return PlainEncoderDecoder(**kwargs)


@pytest.mark.parametrize("size", [64, 96])
def test_forward_shape(size):
    model = _small_baseline()
    model.eval()
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2, size, size)


def test_forward_rejects_non_multiple_size():
    model = _small_baseline()
    with pytest.raises(ValueError, match="multiple"):
        model(torch.randn(1, 3, 60, 60))


def test_no_batchnorm_and_groupnorm_present():
    model = _small_baseline()
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
    model = _small_baseline()
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 2, (2, 64, 64))
    loss = nn.functional.cross_entropy(model(x), target)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert param.grad.abs().sum().item() > 0, f"zero gradient for {name}"


def test_biases_zero_and_kaiming_scale_at_init():
    torch.manual_seed(0)
    model = _small_baseline()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            assert torch.all(module.bias == 0), f"non-zero bias at init for {name}"
            fan_in = module.weight[0].numel()
            expected_std = (2.0 / fan_in) ** 0.5
            actual_std = module.weight.std().item()
            assert actual_std == pytest.approx(expected_std, rel=0.25), (
                f"init std off for {name}: {actual_std} vs {expected_std}"
            )


def test_no_concatenation_decoder_input_matches_up_conv_output():
    # The whole point of this model: decoder blocks consume the up-sampled
    # bottleneck features directly, never a torch.cat with an encoder skip —
    # so each decoder block's first conv has in_channels == up_conv's
    # out_channels, not double that (which is what UNet does).
    model = _small_baseline()
    for up_conv, block in zip(model.up_convs, model.decoder_blocks):
        first_conv = block[0]
        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == up_conv.out_channels


def test_matches_unet_parameter_budget_within_expected_ratio():
    # Same depth/channels/norm as UNet, minus skip-connection concatenation:
    # only the first conv of each decoder block has a narrower input, so the
    # baseline should be close to but strictly smaller than the U-Net.
    unet = UNet(in_channels=3, num_classes=2, base_channels=16, depth=4, gn_groups=8)
    baseline = _small_baseline()
    n_unet = sum(p.numel() for p in unet.parameters())
    n_baseline = sum(p.numel() for p in baseline.parameters())
    assert n_baseline < n_unet
    assert n_baseline / n_unet > 0.6  # comparable capacity, not a toy model


def test_init_is_deterministic_with_fixed_seed():
    torch.manual_seed(1234)
    model_a = _small_baseline()
    torch.manual_seed(1234)
    model_b = _small_baseline()
    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), f"mismatch for {key}"


def test_rejects_invalid_depth_and_gn_groups():
    with pytest.raises(ValueError):
        PlainEncoderDecoder(depth=0)
    with pytest.raises(ValueError):
        PlainEncoderDecoder(base_channels=16, gn_groups=7)
