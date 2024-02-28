from typing import TypeAlias

from . import unet
from .unet import *

__all__ = ["ModelConfig"] + unet.__all__

ModelConfig: TypeAlias = MonaiUNetConfig | UNetConfig
