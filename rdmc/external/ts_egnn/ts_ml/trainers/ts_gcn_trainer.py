import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torch_geometric.utils import to_dense_adj
import numpy as np
from ..models.ts_gcn import TS_GCN
from ..trainers.utils import eval_stats


class LitTSModule(pl.LightningModule):
    def __init__(self, config):
        super(LitTSModule, self).__init__()

        self.save_hyperparameters()
        self.model = TS_GCN(config)
        self.config = config
        self.lr = config["lr"]

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
                                                               patience=5, min_lr=self.lr / 100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_rmsd"}

    def _step(self, data, batch_idx, mode):
        out, mask = self(data)
        loss = F.mse_loss(out, to_dense_adj(data.edge_index, data.batch, data.y), reduction='sum') / mask.sum()

        # logs
        predicted_ts_coords = torch.vstack([c[:m[0].GetNumAtoms()] for c, m in zip(data.coords, data.mols)])
        batch_maes, batch_rmses, batch_rmsds = eval_stats(data, predicted_ts_coords)
        batch_size = len(batch_maes)
        self.log(f'{mode}_loss', loss, batch_size=batch_size)
        self.log(f'{mode}_mae', np.mean(batch_maes), batch_size=batch_size)
        self.log(f'{mode}_rmse', np.mean(batch_rmses), batch_size=batch_size)
        self.log(f'{mode}_rmsd', np.mean(batch_rmsds), batch_size=batch_size)

        return loss

    def training_step(self, data, batch_idx):
        loss = self._step(data, batch_idx, mode="train")
        return loss

    def validation_step(self, data, batch_idx):
        loss = self._step(data, batch_idx, mode="val")
        return loss

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("models")
        parser.add_argument('--hidden_dim', type=int, default=100)
        parser.add_argument('--depth', type=int, default=3)
        parser.add_argument('--n_layers', type=int, default=2)
        parser.add_argument('--chiral_corrections', action='store_true', default=False)
        parser.add_argument("--shuffle_mols", action='store_false', default=True,
                            help="Shuffle reactants and products when choosing starting species")
        parser.add_argument("--prep_mols", action='store_false', default=True,
                            help="Reinitialize reactant/product as if starting from SMILES")
        return parent_parser

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("training")
        parser.add_argument("--gpus", type=int, default=0)
        parser.add_argument("--n_epochs", type=int, default=120)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--seed", type=int, default=0)
        return parent_parser

    @staticmethod
    def add_program_args(parent_parser):
        parser = parent_parser.add_argument_group("program")
        parser.add_argument('--log_dir', type=str)
        parser.add_argument('--data_dir', type=str)
        parser.add_argument('--split_path', type=str)
        return parent_parser

    @classmethod
    def add_args(cls, parent_parser):
        parser = cls.add_program_args(parent_parser)  # program options
        parser = cls.add_argparse_args(parser)  # trainer options
        parser = cls.add_model_specific_args(parser)  # models specific args
        return parser
