from rdkit import Chem  # need to import this for some reason w pytorch lightning imports
import pytorch_lightning as pl  # causing issues
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning import seed_everything

from ts_ml.dataloaders.ts_egnn_loader import TSDataModule
from ts_ml.trainers.ts_egnn_trainer import LitTSModule

from argparse import ArgumentParser


def train_ts_egnn(config):
    seed_everything(config["seed"], workers=True)
    ts_data = TSDataModule(config)
    config["node_dim"] = ts_data.num_node_features
    config["edge_dim"] = ts_data.num_edge_features
    model = LitTSModule(config)
    early_stopping_nan_loss = EarlyStopping(
        monitor="train_loss",
        min_delta=0.0,
        check_finite=True
    )
    early_stopping_val_rmsd = EarlyStopping(
        monitor="val_rmsd",
        min_delta=0.0,
        divergence_threshold=1.0
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["log_dir"],
        filename='best_model',
        monitor="val_rmsd",
        mode="min",
        save_top_k=1,
        save_weights_only=True
    )
    neptune_logger = NeptuneLogger(
        project="lagnajit/ts-egnn",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMGM3YmQ0YS0xZDNmLTRmMjAtOWQ2NS1kNTNkZDI1MzcwODgifQ==",
        tags=[],
        mode="offline",
    )
    trainer = pl.Trainer(
        logger=neptune_logger,
        gpus=config["gpus"],
        max_epochs=config["n_epochs"],
        callbacks=[LearningRateMonitor(),
                   checkpoint_callback,
                   ],
        gradient_clip_val=10.0,
        # profiler="pytorch"
    )
    trainer.fit(model, ts_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = LitTSModule.add_args(parser)
    args = parser.parse_args()
    train_ts_egnn(vars(args))
