import logging
import os

import lightning as L
import monai.data
import torch

from ..data import io as _data_io
from ..datamodule import transforms as _transforms
from ..typing.config import Config
from ..typing.config.transform import *

logger = logging.getLogger(__name__)


__all__ = ["TwoDimensionsDataModule"]


class TwoDimensionsDataModule(L.LightningDataModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.dataset_config = config.dataset
        self.train_config = config.train

        self.num_workers = self.dataset_config.num_workers or os.cpu_count()
        self.dataset_dict: dict[str, monai.data.CacheDataset] = {}

    def setup(self, stage: str = None) -> None:
        train_kidney_id, valid_kidney_id = self.dataset_config.split_type.split("/")

        pre_transforms = [
            monai.transforms.EnsureTyped(keys=("data", "label"), dtype=torch.float32, allow_missing_keys=True),
            monai.transforms.EnsureChannelFirstd(keys=["data", "label"], allow_missing_keys=True, channel_dim=-1),
            # monai.transforms.Resized(keys=["data", "label"], spatial_size=(1024, 768), allow_missing_keys=True),
        ]
        suf_transforms = [
            monai.transforms.ToTensord(keys=["data", "label"], allow_missing_keys=True, track_meta=False),
        ]

        if stage == "fit":
            augmentation_transforms = [
                # monai.transforms.RandSpatialCropd(keys=("data", "label"), roi_size=self.dataset_config.patch_size),
            ]
            for transform_config in self.train_config.augmentations:
                logger.info(f"add augmentation: {transform_config}")
                match transform_config:
                    case RandCropByPosNegLabelConfig(type="RandCropByPosNegLabel"):
                        augmentation_transforms.append(
                            monai.transforms.RandCropByPosNegLabeld(
                                keys=("data", "label"),
                                label_key="label",
                                spatial_size=self.dataset_config.patch_size,
                                pos=transform_config.pos,
                                neg=transform_config.neg,
                            ),
                        )
                    case RandAffineConfig(type="RandAffine"):
                        augmentation_transforms.append(
                            monai.transforms.RandAffined(
                                keys=("data", "label"),
                                prob=transform_config.prob,
                                rotate_range=transform_config.rotate_range,
                                shear_range=transform_config.shear_range,
                                translate_range=transform_config.translate_range,
                                scale_range=transform_config.scale_range,
                            ),
                        )
                    case _:
                        raise ValueError(f"unexpected {transform_config.type=}")

            rows = _data_io.get_all_image_df("train", train_kidney_id).to_dict(orient="records")
            if self.dataset_config.remove_not_labeled_images:
                _rows = [row for row in rows if _data_io.load_label(**row).any()]
                logger.info(f"removed all not-labeled images. size changed: {len(rows)} -> {len(_rows)}")
                rows = _rows

            self.dataset_dict["train"] = monai.data.CacheDataset(
                rows,
                transform=monai.transforms.Compose(
                    [
                        _transforms.LoadImage(with_label=True),
                        *pre_transforms,
                        *augmentation_transforms,
                        *suf_transforms,
                    ]
                ).set_random_state(seed=self.train_config.seed),
                cache_rate=self.dataset_config.train_cache_rate,
            )
            logger.info(f"train dataset size: {len(self.dataset_dict['train'])}")
        if stage in ("fit", "validate"):
            self.dataset_dict["valid"] = monai.data.CacheDataset(
                _data_io.get_all_image_df("train", valid_kidney_id).to_dict(orient="records"),
                transform=monai.transforms.Compose(
                    [_transforms.LoadImage(with_label=True), *pre_transforms, *suf_transforms]
                ),
                cache_rate=self.dataset_config.valid_cache_rate,
            )
            logger.info(f"valid dataset size: {len(self.dataset_dict['valid'])}")
        if stage == "predict":
            self.dataset_dict["test"] = monai.data.CacheDataset(
                _data_io.get_all_image_df("test", valid_kidney_id).to_dict(orient="records"),
                transform=monai.transforms.Compose(
                    [_transforms.LoadImage(with_label=False), *pre_transforms, *suf_transforms]
                ),
                cache_rate=0,
            )
            logger.info(f"test dataset size: {len(self.dataset_dict['test'])}")

    def train_dataloader(self) -> monai.data.DataLoader:
        return monai.data.DataLoader(
            self.dataset_dict["train"],
            shuffle=True,
            batch_size=self.dataset_config.train_batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> monai.data.DataLoader:
        return monai.data.DataLoader(
            self.dataset_dict["valid"],
            shuffle=False,
            batch_size=self.dataset_config.valid_batch_size,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def predict_dataloader(self) -> monai.data.DataLoader:
        return monai.data.DataLoader(
            self.dataset_dict["test"],
            shuffle=False,
            batch_size=self.dataset_config.test_batch_size,
            num_workers=self.num_workers,
            drop_last=False,
        )
