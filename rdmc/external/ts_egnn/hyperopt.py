from rdkit import Chem  # need to import this for some reason w pytorch lightning imports
import pytorch_lightning as pl  # causing issues
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything

from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

from model.data import TSDataModule
from model.ts_trainer import LitTSModule

import os
from argparse import ArgumentParser

#  change line 177 in ray/tune/integration/pytorch_lightning.py to "if trainer.sanity_checking"
#  add "from rdkit import Chem" to ray/tune/tune.py

config = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "batch_size": tune.choice([32, 64, 128]),
    "coordinate_loss_coeff": tune.quniform(0.0, 1.0, 0.1),
    "distance_loss_coeff": tune.quniform(0.0, 1.0, 0.1),
    "hard_sphere_loss_coeff": tune.quniform(0.0, 1.0, 0.1),
    "hidden_dim": tune.choice([32, 64, 128]),
    "depth": tune.qrandint(4, 10, 2),
    "prod_feat": tune.choice(["dist", "adj"])
}


def train_ts_egnn(config, base_config):
    os.chdir(TUNE_ORIG_WORKING_DIR)  # else ray will switch to ./hyperopt_dir
    base_config.update(config)
    seed_everything(base_config["seed"], workers=True)
    ts_data = TSDataModule(base_config)
    base_config["node_dim"] = ts_data.num_node_features
    base_config["edge_dim"] = ts_data.num_edge_features
    model = LitTSModule(base_config)
    checkpoint_callback = ModelCheckpoint(monitor="val/rmsd")
    ray_metrics = {"val_rmsd": "val/rmsd"}
    trainer = pl.Trainer(
        gpus=base_config["gpus"],
        max_epochs=base_config["n_epochs"],
        callbacks=[LearningRateMonitor(), checkpoint_callback, TuneReportCallback(ray_metrics, on="validation_end")],
        gradient_clip_val=10.0,
    )
    trainer.fit(model, ts_data)


if __name__ == "__main__":

    TUNE_ORIG_WORKING_DIR = os.getcwd()
    parser = ArgumentParser()
    parser = LitTSModule.add_args(parser)
    parser.add_argument("--hyperopt_dir", type=str, default="./hyperopt")
    parser.add_argument("--gpus_per_trial", type=int, default=0)
    parser.add_argument("--cpus_per_trial", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=20)
    args = parser.parse_args()

    trainable = tune.with_parameters(
        train_ts_egnn,
        base_config=vars(args)
    )

    scheduler = ASHAScheduler(
        max_t=args.n_epochs,
        grace_period=3,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size"],
        metric_columns=["val_rmsd", "training_iteration"])

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": args.cpus_per_trial,
            "gpu": args.gpus_per_trial
        },
        metric="val_rmsd",
        mode="min",
        config=config,
        num_samples=args.num_trials,
        name="tune_ts_egnn",
        local_dir=args.hyperopt_dir,
        fail_fast=True
    )

    print(analysis.best_config)
