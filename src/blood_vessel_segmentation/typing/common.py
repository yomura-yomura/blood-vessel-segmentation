import pathlib
from collections import namedtuple
from typing import Literal, TypeAlias

__all__ = ["FilePath", "DatasetType", "ValueRange"]

FilePath: TypeAlias = pathlib.Path[str] | str

DatasetType: TypeAlias = Literal["train", "test"]

ValueRange = namedtuple("ValueRange", ("min", "max"))
