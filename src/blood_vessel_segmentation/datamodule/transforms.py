import monai
import numpy as np

from ..data import io as _data_io

__all__ = ["LoadImage"]


class LoadImage(monai.transforms.MapTransform):
    def __init__(self, with_label: bool):
        super().__init__(keys=["dataset_type", "kidney_id", "stem"], allow_missing_keys=False)
        self.with_label = with_label

    def __call__(self, row: dict[str, str]) -> dict[str, np.ndarray]:
        slice_id = int(row["stem"])
        image = _data_io.load_image(*row.values())

        if self.with_label:
            label = _data_io.load_label(*row.values())
            ret = {"slice_id": slice_id, "data": image, "label": label}
        else:
            ret = {"slice_id": slice_id, "data": image}

        return ret
