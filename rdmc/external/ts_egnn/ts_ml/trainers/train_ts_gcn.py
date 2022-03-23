from rdkit import Chem  # need to import this for some reason w pytorch lightning imports
import pytorch_lightning as pl  # causing issues
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers

from ts_ml.dataloaders.ts_gcn_loader import TSDataModule
from ts_ml.trainers.ts_gcn_trainer import LitTSModule

from argparse import ArgumentParser


def train_ts_gcn(config):
    seed_everything(config["seed"], workers=True)
    ts_data = TSDataModule(config)
    config["node_dim"] = ts_data.num_node_features
    config["edge_dim"] = ts_data.num_edge_features
    model = LitTSModule(config)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["log_dir"],
        filename='best_model',
        monitor="val_rmsd",
        mode="min",
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
    trainer.fit(model, ts_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = LitTSModule.add_args(parser)
    args = parser.parse_args()
    train_ts_gcn(vars(args))
