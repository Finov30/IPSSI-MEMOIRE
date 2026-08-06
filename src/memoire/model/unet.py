"""2D U-Net implemented from scratch (Ronneberger et al., 2015).

Design decisions (see docs/CONVENTIONS-TRAINING.md):

- **GroupNorm, never BatchNorm.** Target batch size is 4-8 on a single GPU;
  BatchNorm statistics collapse in that regime, GroupNorm is batch-size
  independent.
- **He (Kaiming) normal initialisation** on every convolution, applied in
  ``__init__``: the whole thesis constraint is random init only, so the init
  scheme is part of the architecture contract, not left to framework defaults.
- **Bilinear upsampling followed by a 3x3 convolution** instead of
  ``ConvTranspose2d``: transposed convolutions with stride 2 are prone to
  checkerboard artifacts (uneven kernel overlap), which matter for dense
  masks, and they add parameters. Fixed bilinear interpolation followed by a
  learned convolution avoids the artifact by construction and keeps the
  parameter budget lower - relevant when training from scratch on a small
  corpus.

Input spatial dimensions must be multiples of ``2 ** depth`` so that the
max-pool / upsample round trip reproduces exactly the input resolution; the
forward pass enforces this with an explicit error.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _conv_block(in_channels: int, out_channels: int, gn_groups: int) -> nn.Sequential:
    """Two 3x3 convolutions, each followed by GroupNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(gn_groups, out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(gn_groups, out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Symmetric encoder/decoder U-Net producing per-pixel class logits.

    Args:
        in_channels: number of input image channels (3 for RGB).
        num_classes: number of output classes (logits, one channel per class).
        base_channels: channels of the first encoder level; doubled at each
            level down (16-32 expected, default 32).
        depth: number of down/up-sampling steps (default 4). Input H and W
            must be multiples of ``2 ** depth``.
        gn_groups: number of groups for every GroupNorm layer; must divide
            every channel count in the network (i.e. divide ``base_channels``).

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

        # Encoder: base_channels * 2**i at level i.
        self.encoder_blocks = nn.ModuleList()
        channels = in_channels
        for i in range(depth):
            out = base_channels * (2**i)
            self.encoder_blocks.append(_conv_block(channels, out, gn_groups))
            channels = out
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck at base_channels * 2**depth.
        self.bottleneck = _conv_block(channels, channels * 2, gn_groups)
        channels *= 2

        # Decoder: bilinear upsample + 3x3 conv halving channels, then a conv
        # block on the concatenation with the skip connection.
        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            skip = base_channels * (2**i)
            self.up_convs.append(nn.Conv2d(channels, skip, kernel_size=3, padding=1))
            self.decoder_blocks.append(_conv_block(skip * 2, skip, gn_groups))
            channels = skip

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

        skips = []
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up_conv, block, skip in zip(self.up_convs, self.decoder_blocks, reversed(skips)):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = up_conv(x)
            x = block(torch.cat([skip, x], dim=1))

        return self.head(x)
