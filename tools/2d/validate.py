import json
import pathlib

import lightning as L
import torch

from blood_vessel_segmentation.datamodule.two_dimensions import TwoDimensionsDataModule
from blood_vessel_segmentation.module.two_dimensions import TwoDimensionsModule
from blood_vessel_segmentation.typing.config import *

model_checkpoint_dir_path = pathlib.Path("models/test-with-metric9")

with open(model_checkpoint_dir_path / "config.json") as f:
    config = Config(**json.load(f))

config.dataset.train_cache_rate = config.dataset.valid_cache_rate = 0

datamodule = TwoDimensionsDataModule(config)
datamodule.setup("fit")

# model_checkpoint_path = "test/epoch=22-step=52417-valid_loss=0.6882.ckpt"

module = TwoDimensionsModule.load_from_checkpoint(
    model_checkpoint_dir_path / "best.ckpt",
    config=config,
)

trainer = L.Trainer(devices=config.train.devices, accelerator=config.train.accelerator)

preds_list = trainer.predict(module, datamodule.val_dataloader())
preds = torch.concat(preds_list, dim=0).numpy()[:, 0]

import pandas as pd
import tqdm

data = torch.concat([batch["data"][:, 0] for batch in tqdm.tqdm(datamodule.val_dataloader())], dim=0).numpy()
label = torch.concat([batch["label"][:, 0] for batch in tqdm.tqdm(datamodule.val_dataloader())], dim=0).numpy()

import plotly.express as px

fig = px.imshow(data[10], title="data")
fig.show()

fig = px.imshow(preds[10], title="pred")
fig.show()

fig = px.imshow(label[10], title="true")
fig.show()


import blood_vessel_segmentation.data.io
import blood_vessel_segmentation.data.utils

_, valid_kidney_id = config.dataset.split_type.split("/")
true_df = blood_vessel_segmentation.data.io.get_train_rle_df(valid_kidney_id)
sub_df = pd.DataFrame(
    [
        {
            "id": "_".join(
                [
                    blood_vessel_segmentation.data.io.AVAILABLE_KIDNEY_ID_DICT["train"][valid_kidney_id]["label"],
                    stem,
                ]
            ),
            "rle": blood_vessel_segmentation.data.utils.rle_encode(pred > 0.5),
        }
        for pred, stem in zip(
            tqdm.tqdm(preds, desc="rle"),
            blood_vessel_segmentation.data.io.get_all_image_df("train", "3")["stem"],
            strict=True,
        )
    ]
)
assert (true_df["id"] == sub_df["id"]).all()

import blood_vessel_segmentation.metric

print("fast", blood_vessel_segmentation.metric.calc_metric(sub_df, true_df, mode="fast"))
# print("normal", blood_vessel_segmentation.metric.calc_metric(sub_df, true_df, mode="normal"))
