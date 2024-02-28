from typing import Literal, TypeAlias

import numpy as np
from pydantic import Field, PositiveFloat
from pydantic.dataclasses import dataclass

__all__ = ["TransformElementConfig", "RandCropByPosNegLabelConfig", "RandAffineConfig"]


@dataclass
class RandCropByPosNegLabelConfig:
    type: Literal["RandCropByPosNegLabel"] = "RandCropByPosNegLabel"
    pos: float = 2.0
    neg: float = 1.0


@dataclass
class RandAffineConfig:
    type: Literal["RandAffine"] = "RandAffine"
    prob: float = Field(0.1, ge=0, le=1)
    rotate_range: list[float] | None = Field(ge=0, lt=2 * np.pi)
    shear_range: list[float] | None = Field(ge=0, lt=2 * np.pi)
    translate_range: list[int] | None = Field()
    scale_range: list[PositiveFloat] | None = Field()


TransformElementConfig: TypeAlias = RandCropByPosNegLabelConfig | RandAffineConfig
