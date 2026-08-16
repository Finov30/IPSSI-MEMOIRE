"""Plain encoder/decoder baseline (chap. 6.5) — the ablation reference model.

Identical to :class:`~memoire.model.unet.UNet` in every respect (depth,
channel schedule, GroupNorm, bilinear-upsample decoder, He init, from-scratch
only) except one: the decoder never receives the encoder's skip connections.
Holding every other factor fixed isolates what the U-Net's defining feature
(Ronneberger et al., 2015) actually buys over a plain conv encoder/decoder of
matched capacity — the comparison chapter 7.3's ablation plan and chapter
8.3's results table are built on.
"""

import torch
from torch import nn
from torch.nn import functional as F

from memoire.model.unet import _conv_block


class PlainEncoderDecoder(nn.Module):
    """Symmetric encoder/decoder with no skip connections between them.

    Args: identical to :class:`~memoire.model.unet.UNet` (same defaults);
        see there for parameter documentation.

    Shapes:
        forward: ``B x in_channels x H x W -> B x num_classes x H x W``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 32,
        depth: int = 4,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if base_channels % gn_groups != 0:
            raise ValueError(f"gn_groups ({gn_groups}) must divide base_channels ({base_channels})")
        self.depth = depth

        self.encoder_blocks = nn.ModuleList()
        channels = in_channels
        for i in range(depth):
            out = base_channels * (2**i)
            self.encoder_blocks.append(_conv_block(channels, out, gn_groups))
            channels = out
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _conv_block(channels, channels * 2, gn_groups)
        channels *= 2

        # Decoder mirrors the U-Net's channel schedule exactly, but each block
        # consumes only the upsampled features from above — no concatenation
        # with an encoder skip, so no doubled input width either.
        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            out = base_channels * (2**i)
            self.up_convs.append(nn.Conv2d(channels, out, kernel_size=3, padding=1))
            self.decoder_blocks.append(_conv_block(out, out, gn_groups))
            channels = out

        self.head = nn.Conv2d(channels, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """He normal init (fan_in, ReLU gain) on all convs, zero biases."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor = 2**self.depth
        h, w = x.shape[-2], x.shape[-1]
        if h % factor != 0 or w % factor != 0:
            raise ValueError(
                f"input spatial size {h}x{w} must be a multiple of 2**depth = {factor}"
            )

        for block in self.encoder_blocks:
            x = block(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up_conv, block in zip(self.up_convs, self.decoder_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = up_conv(x)
            x = block(x)

        return self.head(x)
