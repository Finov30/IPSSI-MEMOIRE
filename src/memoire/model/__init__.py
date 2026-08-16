"""Model package: from-scratch architectures (no pretrained weights)."""

from memoire.model.baseline import PlainEncoderDecoder
from memoire.model.unet import UNet

__all__ = ["PlainEncoderDecoder", "UNet"]
