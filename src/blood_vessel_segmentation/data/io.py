import pathlib

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from .. import pj_struct_paths
from ..typing import DatasetType

# See Competition Data Tab: https://www.kaggle.com/competitions/blood-vessel-segmentation/data
AVAILABLE_KIDNEY_ID_DICT = {
    "train": {
        "1": {
            "image": "kidney_1_dense",
            "label": "kidney_1_dense",
        },  # 50um
        "3": {
            "image": "kidney_3_sparse",
            "label": "kidney_3_dense",
        },  # 50.16um
    },
    "test": {
        "5": {
            "image": "kidney_5",
        },  # maybe 50.28um (scanned at 25.14um)
        "6": {
            "image": "kidney_6",
        },  # maybe 63.08um (scanned at 15.77um)
    },
}


def get_train_rle_df(kidney_id: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(pj_struct_paths.get_kaggle_dataset_dir_path() / f"train_rles.csv")

    if kidney_id is not None:
        df["_kidney_dirname"] = [id_.rsplit("_", maxsplit=1)[0] for id_ in df["id"]]
        df = df[df["_kidney_dirname"] == AVAILABLE_KIDNEY_ID_DICT["train"][kidney_id]["label"]]
        df = df.drop(columns="_kidney_dirname").reset_index(drop=True)

    return df


def get_all_image_df(
    dataset_type: DatasetType | None = None,
    kidney_id: str | None = None,
    stem: str | None = None,
    based_on_label_stems: bool = True,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (dataset_type_, kidney_id_, stem_)
            for dataset_type_ in (AVAILABLE_KIDNEY_ID_DICT.keys() if dataset_type is None else [dataset_type])
            for kidney_id_ in (AVAILABLE_KIDNEY_ID_DICT[dataset_type_].keys() if kidney_id is None else [kidney_id])
            for stem_ in (
                get_available_image_stems(dataset_type_, kidney_id_, based_on_label_stems=based_on_label_stems)
                if stem is None
                else [stem]
            )
        ],
        columns=["dataset_type", "kidney_id", "stem"],
    )


def get_available_image_stems(
    dataset_type: DatasetType, kidney_id: str, based_on_label_stems: bool = True
) -> list[str]:
    available_image_stems = sorted(
        (p.stem for p in get_image_dir_path(dataset_type, kidney_id).glob("*.tif")),
        key=lambda stem: int(stem),
    )

    label_dir_path = get_label_dir_path(dataset_type, kidney_id)
    if based_on_label_stems and label_dir_path is not None:
        label_stems = [p.stem for p in label_dir_path.glob("*.tif")]
        available_image_stems = [stem for stem in available_image_stems if stem in label_stems]

    return available_image_stems


# Path


def get_image_dir_path(dataset_type: DatasetType, kidney_id: str) -> pathlib.Path:
    return (
        pj_struct_paths.get_kaggle_dataset_dir_path()
        / dataset_type
        / AVAILABLE_KIDNEY_ID_DICT[dataset_type][kidney_id]["image"]
        / "images"
    )


def get_label_dir_path(dataset_type: DatasetType, kidney_id: str) -> pathlib.Path | None:
    kidney_id_dirname = AVAILABLE_KIDNEY_ID_DICT[dataset_type][kidney_id].get("label", None)
    if kidney_id_dirname is None:
        return None
    return pj_struct_paths.get_kaggle_dataset_dir_path() / dataset_type / kidney_id_dirname / "labels"


def get_image_path(dataset_type: DatasetType, kidney_id: str, stem: str) -> pathlib.Path:
    return get_image_dir_path(dataset_type, kidney_id) / f"{stem}.tif"


def get_label_path(dataset_type: DatasetType, kidney_id: str, stem: str) -> pathlib.Path:
    return get_label_dir_path(dataset_type, kidney_id) / f"{stem}.tif"


# Load


def load_image(dataset_type: DatasetType, kidney_id: str, stem: str) -> NDArray[np.uint16]:
    with Image.open(get_image_path(dataset_type, kidney_id, stem)) as img:
        return np.asarray(img)[..., np.newaxis]


def load_label(dataset_type: DatasetType, kidney_id: str, stem: str) -> NDArray[np.bool_]:
    label_path = get_label_path(dataset_type, kidney_id, stem)
    if label_path is None:
        raise FileNotFoundError(label_path)
    with Image.open(label_path) as img:
        label = np.asarray(img)
    assert np.all(np.isin(label, [0, 255]))
    return (label > 0)[..., np.newaxis]
