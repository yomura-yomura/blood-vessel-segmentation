from pydantic import Field, constr
from pydantic.dataclasses import dataclass

from .dataset import DatasetConfig
from .model import ModelConfig
from .train import TrainConfig

__all__ = ["Config"]


@dataclass
class Config:
    exp_name: constr(min_length=1)
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig
