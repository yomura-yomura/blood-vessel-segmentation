from typing import Annotated, Literal

from pydantic.dataclasses import dataclass

from ..common import ValueRange

__all__ = ["DatasetConfig"]


@dataclass
class DatasetConfig:
    split_type: Literal["1/3"] = "1/3"

    train_batch_size: int = 1
    valid_batch_size: int = 64
    test_batch_size: int = 64
    patch_size: int = 512

    train_cache_rate: Annotated[float, ValueRange[0.0, 1.0]] = 0.0
    valid_cache_rate: Annotated[float, ValueRange[0.0, 1.0]] = 0.0
    num_workers: int | None = None

    remove_not_labeled_images: bool = True
