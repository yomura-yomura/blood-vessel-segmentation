from typing import Literal

import pandas as pd
from PIL import Image

from . import fast, official

__all__ = ["calc_metric", "fast", "official"]


def calc_metric(solution: pd.DataFrame, submission: pd.DataFrame, mode: Literal["normal", "fast"] = "fast") -> float:
    solution = solution.copy()
    add_size_columns(solution)
    if mode == "normal":
        return official.score(
            solution,
            submission,
            "id",
            tolerance=0.0,
            image_id_column_name="image_id",
            slice_id_column_name="slice_id",
        )
    elif mode == "fast":
        return fast.compute_surface_dice_score(submission, solution)
    else:
        raise ValueError(f"unexpected {mode=}")


def add_size_columns(df: pd.DataFrame):
    """
    df (DataFrame): including id column, e.g., kidney_1_dense_0000
    """
    from ..data.io import get_image_path

    widths = []
    heights = []
    subdirs = []
    nums = []
    for i, r in df.iterrows():
        file_id = r["id"]
        subdir = file_id[:-5]  # kidney_1_dense
        file_num = file_id[-4:]  # 0000

        img = Image.open(get_image_path("train", subdir.split("_")[1], file_num))
        w, h = img.size
        widths.append(w)
        heights.append(h)
        subdirs.append(subdir)
        nums.append(file_num)

    df["width"] = widths
    df["height"] = heights
    df["image_id"] = subdirs
    df["slice_id"] = nums
