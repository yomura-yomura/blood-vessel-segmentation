import logging

import lightning as L

import blood_vessel_segmentation
from blood_vessel_segmentation.datamodule.two_dimensions import TwoDimensionsDataModule
from blood_vessel_segmentation.module.two_dimensions import TwoDimensionsModule
from blood_vessel_segmentation.typing.config import *
from blood_vessel_segmentation.typing.config.model import *
from blood_vessel_segmentation.typing.config.transform import *

blood_vessel_segmentation.logger.setLevel(logging.INFO)

config = Config(
    exp_name="test-with-metric9",
    dataset=DatasetConfig(
        valid_cache_rate=1,
        # patch_size=256,
    ),
    model=MonaiUNetConfig(),
    train=TrainConfig(
        augmentations=[
            RandAffineConfig(
                rotate_range=None,
                shear_range=None,
                translate_range=None,
                scale_range=[0.5, 1.5],
            ),
            RandCropByPosNegLabelConfig(
                pos=2,
            ),
        ]
    ),
)

datamodule = TwoDimensionsDataModule(config)
# datamodule.setup("fit")
# batch = next(iter(datamodule.train_dataloader()))

module = TwoDimensionsModule(config)
# module(batch)


L.seed_everything(seed=config.train.seed)


from lightning.pytorch.loggers import WandbLogger

trainer = L.Trainer(
    logger=WandbLogger(
        name=f"{config.exp_name}-{config.dataset.split_type}",
        project="blood-vessel-segmentation",
        group=config.exp_name,
    ),
    callbacks=[
        # L.pytorch.callbacks.ModelCheckpoint(
        #     dirpath=model_path / "checkpoints",
        #     filename=filename,
        #     verbose=True,
        #     monitor="valid/loss",
        #     mode="min",
        #     save_weights_only=True,
        #     save_on_train_epoch_end=False
        #     # save_last=not cfg.train.save_best_checkpoint,
        # ),
        L.pytorch.callbacks.LearningRateMonitor("step"),
        L.pytorch.callbacks.EarlyStopping(
            monitor=config.train.early_stopping.monitor,
            mode=config.train.early_stopping.mode,
            min_delta=config.train.early_stopping.min_delta,
            patience=config.train.early_stopping.patience,
            verbose=False,
        ),
    ],
    accelerator=config.train.accelerator,
    devices=config.train.devices,
    precision=config.train.precision,
    max_epochs=config.train.max_epochs,
)
trainer.fit(module, datamodule)
