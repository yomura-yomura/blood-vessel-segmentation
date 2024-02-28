import pandas as pd
import pytest

import blood_vessel_segmentation.data.io


@pytest.mark.parametrize(
    "dataset_type, kidney_id, image_stem",
    pd.concat(
        [
            blood_vessel_segmentation.data.io.get_all_image_df("train", based_on_label_stems=False),
            blood_vessel_segmentation.data.io.get_all_image_df("test"),
        ]
    ).to_numpy(),
)
def test_load_image(dataset_type, kidney_id, image_stem):
    blood_vessel_segmentation.data.io.load_image(dataset_type, kidney_id, image_stem)


@pytest.mark.parametrize(
    "dataset_type, kidney_id, image_stem", blood_vessel_segmentation.data.io.get_all_image_df("train").to_numpy()
)
def test_load_label(dataset_type, kidney_id, image_stem):
    blood_vessel_segmentation.data.io.load_image(dataset_type, kidney_id, image_stem)
