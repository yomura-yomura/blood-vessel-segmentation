import dataclasses
import json
import logging
import pathlib
from typing import Any

import lightning as L
import monai.networks.nets
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import AdamW

from .. import metric as _metric
from .. import pj_struct_paths as _pj_struct_paths
from ..data import io as _data_io
from ..data import utils as _data_utils
from ..typing.config import Config

__all__ = ["TwoDimensionsModule"]


logger = logging.getLogger(__name__)


class TwoDimensionsModule(L.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config

        match self.config.model.arch:
            case "monai_unet":
                self.model = monai.networks.nets.UNet(
                    spatial_dims=self.config.model.spatial_dims,
                    in_channels=1,
                    out_channels=1,
                    channels=self.config.model.channels,
                    strides=self.config.model.strides,
                )
            case "unet":
                self.model = smp.Unet(
                    encoder_name=self.config.model.encoder_name,
                    encoder_weights=self.config.model.encoder_weights,
                    in_channels=1,
                    classes=1,
                )
            case _ as arch:
                raise ValueError(f"unexpected {arch=}")

        self.loss_func = monai.losses.DiceLoss(batch=True)

        # fit
        self.validation_step_outputs = []

        # Model Checkpointing
        if self.config.train.model_checkpoint.monitor is None:
            self.save_checkpoints = False
            self.monitor = self.mode = self.save_top_k = None
            self.dir_path_to_save_model = self.config_path = None
            self.best_ckpt_path = self.last_ckpt_path = None
        else:
            self.save_checkpoints = True

            self.monitor = self.config.train.model_checkpoint.monitor
            self.mode = self.config.train.model_checkpoint.mode
            self.save_top_k = self.config.train.model_checkpoint.save_top_k
            logger.info(f"save checkpoints with {self.monitor=}, {self.mode=}, {self.save_top_k=}")

            self.dir_path_to_save_model = (
                _pj_struct_paths.get_project_root_path() / "tools" / "2d" / "models" / config.exp_name
            )
            self.config_path = self.dir_path_to_save_model / "config.json"

            self.best_ckpt_path = self.dir_path_to_save_model / "best.ckpt"
            self.last_ckpt_path = self.dir_path_to_save_model / "last.ckpt"

        self.best_score_paths: list[tuple[float, pathlib.Path]] = []
        self.last_model_path = None

        # Metric
        self.train_kidney_id, self.valid_kidney_id = self.config.dataset.split_type.split("/")
        self.true_df = None
        self.valid_stems = None

    def setup(self, stage: str) -> None:
        if stage == "fit":
            # Model Checkpointing
            if self.save_checkpoints:
                self.true_df = _data_io.get_train_rle_df(self.valid_kidney_id)
                self.valid_stems = _data_io.get_all_image_df("train", self.valid_kidney_id)["stem"]

            if self.config_path.exists():
                raise FileExistsError(self.config_path)
            self.dir_path_to_save_model.mkdir(parents=True, exist_ok=True)

            logger.info(f"export config as {self.config_path}")
            with open(self.config_path, "w") as f:

                def default(item: Any) -> dict[str, Any]:
                    match item:
                        case _ if dataclasses.is_dataclass(item):
                            return dataclasses.asdict(item)
                        case _:
                            raise TypeError(type(item))

                json.dump(self.config, f, indent=2, default=default)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch["data"])

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.float]:
        logits = self.forward(batch)
        probs = logits.sigmoid()
        loss = self.loss_func(probs, batch["label"])
        self.log(
            f"train/loss",
            loss.detach().cpu(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=self.config.dataset.train_batch_size,
        )
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.float]:
        logits = monai.inferers.sliding_window_inference(
            batch["data"],
            roi_size=self.config.dataset.patch_size,
            sw_batch_size=self.config.dataset.valid_batch_size,
            predictor=self.model,
        )

        probs = logits.sigmoid()

        true = torch.where(batch["label"] > 0.5, 1, 0)  # noqa: unexpected type(s)
        preds = torch.where(probs > 0.5, 1, 0)

        metric_dict = {
            "valid/loss": self.loss_func(probs, batch["label"]),
        }
        self.validation_step_outputs.append(
            {
                **{k: v.item() for k, v in metric_dict.items()},
                "preds": preds.detach().cpu().numpy(),
            }
        )
        return metric_dict

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self.validation_step_outputs.clear()
            return
        if len(self.validation_step_outputs) == 0:
            return

        # Competition Metric
        sub_df = pd.DataFrame(
            [
                {
                    "id": "_".join(
                        [
                            _data_io.AVAILABLE_KIDNEY_ID_DICT["train"][self.valid_kidney_id]["label"],
                            stem,
                        ]
                    ),
                    "rle": _data_utils.rle_encode(pred),
                }
                for pred, stem in zip(
                    (pred for batch in self.validation_step_outputs for pred in batch["preds"]),
                    self.valid_stems,
                    strict=True,
                )
            ]
        )
        assert (self.true_df["id"] == sub_df["id"]).all()

        metric_dict = {
            "valid/loss": sum(batch["valid/loss"] * len(batch["preds"]) for batch in self.validation_step_outputs)
            / len(self.true_df),
            "valid/metric": _metric.calc_metric(sub_df, self.true_df, mode="fast"),
        }

        for name, value in metric_dict.items():
            self.log(
                name,
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        if self.dir_path_to_save_model is not None:
            self.save_checkpoint_top_k(metric_dict[self.monitor])
        self.validation_step_outputs.clear()

    def predict_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = monai.inferers.sliding_window_inference(
            batch["data"],
            roi_size=self.config.dataset.patch_size,
            sw_batch_size=self.config.dataset.valid_batch_size,
            predictor=self.model,
        )
        probs = logits.sigmoid()
        return probs

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = AdamW(
            self.parameters(), lr=self.config.train.learning_rate, weight_decay=self.config.train.weight_decay
        )
        return [optimizer]

    def save_checkpoint_top_k(self, score: float):
        epoch = self.trainer.current_epoch
        step = self.trainer.global_step

        current_model_path = self.dir_path_to_save_model / (
            "-".join(
                [
                    f"{epoch=}",
                    f"{step=}",
                    f"{self.monitor}={score:.4f}",
                ]
            ).replace("/", "_")
            + ".ckpt"
        )

        self.last_model_path = current_model_path

        self.best_score_paths.append((score, current_model_path))

        # ascending order
        best_score_paths = sorted(self.best_score_paths, key=lambda pair: pair[0])
        if self.mode == "min":
            pass
        elif self.mode == "max":
            # descending order
            best_score_paths = best_score_paths[::-1]
        else:
            raise ValueError(f"unexpected {self.mode=}")
        best_model_score, best_model_path = best_score_paths[0]

        self.trainer.save_checkpoint(current_model_path)

        if len(best_score_paths) < self.save_top_k or current_model_path in (
            path for _, path in best_score_paths[: self.save_top_k]
        ):
            print(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {score:0.5f}"
                f" (best {best_model_score:0.5f}), saving model to {current_model_path} in top {self.save_top_k}"
            )
            self.best_ckpt_path.unlink(missing_ok=True)
            self.best_ckpt_path.symlink_to(best_model_path.name)
        else:
            print(f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} was not in top {self.save_top_k}")

        if len(best_score_paths) > 0:
            self.last_ckpt_path.unlink(missing_ok=True)
            self.last_ckpt_path.symlink_to(self.last_model_path.name)

        indices_to_remove = []
        for i, (_, model_path) in enumerate(best_score_paths[self.save_top_k :]):
            if model_path == self.last_model_path:
                continue
            model_path.unlink(missing_ok=True)
            indices_to_remove.append(self.save_top_k + i)

        for i in indices_to_remove[::-1]:
            best_score_paths.pop(i)  # noqa: Expected type 'SupportsIndex', got 'float' instead
        self.best_score_paths = best_score_paths
