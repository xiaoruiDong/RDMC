from rdkit import Chem  # need to import this for some reason w pytorch lightning imports
import pytorch_lightning as pl  # causing issues
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers

from ts_ml.dataloaders.ts_screener_loader import TSScreenerDataModule
from ts_ml.trainers.ts_screener_trainer import LitScreenerModule

from argparse import ArgumentParser


def train_ts_screener(config):
    seed_everything(config["seed"], workers=True)
    irc_data = TSScreenerDataModule(config)
    model = LitScreenerModule(config)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["log_dir"],
        filename='best_model',
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_weights_only=True
    )
    trainer = pl.Trainer(
        logger=pl_loggers.TensorBoardLogger(config["log_dir"]),
        gpus=config["gpus"],
        max_epochs=config["n_epochs"],
        callbacks=[LearningRateMonitor(),
                   checkpoint_callback,
                   ],
        gradient_clip_val=10.0,
    )
    trainer.fit(model, irc_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = LitScreenerModule.add_args(parser)
    args = parser.parse_args()
    train_ts_screener(vars(args))
