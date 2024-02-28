from typing import Literal

from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass

from .transform import TransformElementConfig

__all__ = ["TrainConfig", "ModelCheckpointConfig", "EarlyStoppingConfig"]


@dataclass
class ModelCheckpointConfig:
    monitor: str | None = "valid/metric"
    mode: Literal["min", "max"] = "max"
    save_top_k: NonNegativeInt = 2


@dataclass
class EarlyStoppingConfig:
    monitor: str | None = "valid/metric"
    mode: Literal["min", "max"] = "max"

    min_delta: NonNegativeFloat = 0.0
    patience: PositiveInt = 15


@dataclass
class TrainConfig:
    accelerator: str = "gpu"
    devices: PositiveInt = 1
    precision: Literal[16, 32] = 32
    seed: int = 42
    max_epochs: PositiveInt = 400

    model_checkpoint: "ModelCheckpointConfig" = ModelCheckpointConfig()

    # optimizer
    learning_rate: PositiveFloat = 1e-4
    weight_decay: NonNegativeFloat = 0.0

    early_stopping: "EarlyStoppingConfig" = EarlyStoppingConfig()

    augmentations: list[TransformElementConfig] = ()
