from rdkit import Chem  # need to import this for some reason w pytorch lightning imports
import pytorch_lightning as pl  # causing issues
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning import seed_everything

from model.data import TSDataModule
from model.ts_trainer import LitTSModule

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
    trainer = pl.Trainer(
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
