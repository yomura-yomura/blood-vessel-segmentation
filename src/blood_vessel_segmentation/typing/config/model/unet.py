from typing import Sequence

from pydantic.dataclasses import dataclass

__all__ = ["MonaiUNetConfig", "UNetConfig"]


@dataclass
class MonaiUNetConfig:
    arch: str = "monai_unet"

    spatial_dims: int = 2
    channels: Sequence[int] = (32, 64, 128, 256, 512)
    strides: Sequence[int] = (2, 2, 2, 2)


@dataclass
class UNetConfig:
    arch: str = "unet"

    # see https://github.com/qubvel/segmentation_models.pytorch?tab=readme-ov-file#encoders-
    encoder_name: str = "resnet34"
    encoder_weights: str | None = "imagenet"
