import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torch_geometric.utils import to_dense_adj
import numpy as np
from ..models.ts_egnn import TS_EGNN
from ..trainers.utils import hard_sphere_loss_fn, eval_stats


class LitTSModule(pl.LightningModule):
    def __init__(self, config):
        super(LitTSModule, self).__init__()

        self.save_hyperparameters()
        self.model = TS_EGNN(config)
        self.config = config
        self.lr = config["lr"]
        self.coordinate_loss_coeff = config["coordinate_loss_coeff"]
        self.distance_loss_coeff = config["distance_loss_coeff"]
        self.hard_sphere_loss_coeff = config["hard_sphere_loss_coeff"]
        self.soft_hard_sphere_loss = config["soft_hard_sphere_loss"]
        self.dm_loss_coeff = config["dm_loss_coeff"]
        self.annealed_dm_loss = config["annealed_dm_loss"]
        self.coord_reg_coeff = config["coord_reg_coeff"]

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
                                                               patience=5, min_lr=self.lr / 100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_rmsd"}

    def training_step(self, data, batch_idx):
        out = self(data)
        predicted_ts_coords = out[:, :3]

        # distance calc (only for existing bonds on reactants or products)
        b_a1, b_a2 = data.bonded_index
        true_bond_distances = torch.norm(data.pos_ts[b_a1] - data.pos_ts[b_a2], dim=-1)
        predicted_bond_distances = torch.norm(predicted_ts_coords[b_a1] - predicted_ts_coords[b_a2], dim=-1)

        # distance matrix loss
        true_distance_matrix = to_dense_adj(data.edge_index, data.batch, torch.norm(
            data.pos_ts[data.edge_index[0]] - data.pos_ts[data.edge_index[1]], dim=-1))
        predicted_distance_matrix = to_dense_adj(data.edge_index, data.batch, torch.norm(
            predicted_ts_coords[data.edge_index[0]] - predicted_ts_coords[data.edge_index[1]], dim=-1))
        distance_mask = to_dense_adj(data.edge_index, data.batch)

        # distance and coordinate loss
        distance_loss = F.mse_loss(true_bond_distances, predicted_bond_distances)
        coordinate_loss = F.mse_loss(predicted_ts_coords, data.pos_ts)
        hard_sphere_loss = hard_sphere_loss_fn(data, predicted_ts_coords)
        distance_matrix_loss = F.mse_loss(true_distance_matrix, predicted_distance_matrix,
                                          reduction="sum") / distance_mask.sum()

        if self.soft_hard_sphere_loss:
            for x in self.model.coords:
                hard_sphere_loss += hard_sphere_loss_fn(data, x[:, :3])

        annealed_dm_loss = 0.0
        n_steps = len(self.model.coords)
        if self.annealed_dm_loss:
            for n_step, x in enumerate(self.model.coords):
                dm = to_dense_adj(data.edge_index, data.batch,
                                  torch.norm(x[data.edge_index[0]] - x[data.edge_index[1]], dim=-1))
                dm_loss = F.mse_loss(true_distance_matrix, dm, reduction="sum") / distance_mask.sum()
                annealed_dm_loss += (n_step+1)/n_steps * dm_loss

        coordinate_regularization_loss = 0.0
        if self.coord_reg_coeff > 0:
            for i in range(n_steps-1):
                coordinate_regularization_loss += \
                    self.coord_reg_coeff * F.mse_loss(self.model.coords[i], self.model.coords[i+1])

        loss = self.coordinate_loss_coeff * coordinate_loss + \
               self.distance_loss_coeff * distance_loss + \
               self.hard_sphere_loss_coeff * hard_sphere_loss + \
               self.dm_loss_coeff * distance_matrix_loss + \
               annealed_dm_loss + coordinate_regularization_loss

        # logs
        batch_maes, batch_rmses, batch_rmsds = eval_stats(data, predicted_ts_coords)
        batch_size = len(batch_maes)
        self.log('train_loss', loss, batch_size=batch_size)
        self.log('train_mae', np.mean(batch_maes), batch_size=batch_size)
        self.log('train_rmse', np.mean(batch_rmses), batch_size=batch_size)
        self.log('train_rmsd', np.mean(batch_rmsds), batch_size=batch_size)
        self.log('train_distance_loss', distance_loss, batch_size=batch_size)
        self.log('train_coordinate_loss', coordinate_loss, batch_size=batch_size)
        self.log('train_hard_sphere_loss', hard_sphere_loss, batch_size=batch_size)
        self.log('train_distance_matrix_loss', distance_matrix_loss, batch_size=batch_size)
        self.log('train_annealed_distance_matrix_loss', annealed_dm_loss, batch_size=batch_size)
        self.log('train_coordinate_regularization_loss', coordinate_regularization_loss, batch_size=batch_size)

        return loss

    def validation_step(self, data, batch_idx):
        out = self(data)
        predicted_ts_coords = out[:, :3]

        # distance calc (only for existing bonds on reactants or products)
        b_a1, b_a2 = data.bonded_index
        true_bond_distances = torch.norm(data.pos_ts[b_a1] - data.pos_ts[b_a2], dim=-1)
        predicted_bond_distances = torch.norm(predicted_ts_coords[b_a1] - predicted_ts_coords[b_a2], dim=-1)

        # distance matrix loss
        true_distance_matrix = to_dense_adj(data.edge_index, data.batch, torch.norm(
            data.pos_ts[data.edge_index[0]] - data.pos_ts[data.edge_index[1]], dim=-1))
        predicted_distance_matrix = to_dense_adj(data.edge_index, data.batch, torch.norm(
            predicted_ts_coords[data.edge_index[0]] - predicted_ts_coords[data.edge_index[1]], dim=-1))
        distance_mask = to_dense_adj(data.edge_index, data.batch)

        # distance and coordinate loss
        distance_loss = F.mse_loss(true_bond_distances, predicted_bond_distances)
        coordinate_loss = F.mse_loss(predicted_ts_coords, data.pos_ts)
        hard_sphere_loss = hard_sphere_loss_fn(data, predicted_ts_coords)
        distance_matrix_loss = F.mse_loss(true_distance_matrix, predicted_distance_matrix,
                                          reduction="sum") / distance_mask.sum()

        if self.soft_hard_sphere_loss:
            for x in self.model.coords:
                hard_sphere_loss += hard_sphere_loss_fn(data, x[:, :3])

        loss = self.coordinate_loss_coeff * coordinate_loss + \
               self.distance_loss_coeff * distance_loss + \
               self.hard_sphere_loss_coeff * hard_sphere_loss + \
               self.dm_loss_coeff * distance_matrix_loss

        # logs
        batch_maes, batch_rmses, batch_rmsds = eval_stats(data, predicted_ts_coords)
        batch_size = len(batch_maes)
        self.log('val_loss', loss, batch_size=batch_size)
        self.log('val_mae', np.mean(batch_maes), batch_size=batch_size)
        self.log('val_rmse', np.mean(batch_rmses), batch_size=batch_size)
        self.log('val_rmsd', np.mean(batch_rmsds), batch_size=batch_size)
        self.log('val_distance_loss', distance_loss, batch_size=batch_size)
        self.log('val_coordinate_loss', coordinate_loss, batch_size=batch_size)
        self.log('val_hard_sphere_loss', hard_sphere_loss, batch_size=batch_size)
        self.log('val_distance_matrix_loss', distance_matrix_loss, batch_size=batch_size)
        # self.trainer.callback_metrics["loss"] = loss

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("models")
        parser.add_argument("--depth", type=int, default=6)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--cutoff", type=float, default=10.0,
                            help="Cutoff radius for graph construction")
        parser.add_argument("--coordinate_loss_coeff", type=float, default=1.0)
        parser.add_argument("--distance_loss_coeff", type=float, default=0.1)
        parser.add_argument("--hard_sphere_loss_coeff", type=float, default=0.1)
        parser.add_argument("--dm_loss_coeff", type=float, default=0.0)
        parser.add_argument("--coord_reg_coeff", type=float, default=0.0)
        parser.add_argument("--soft_hard_sphere_loss", action='store_true', default=False,
                            help="Enforce hard sphere loss coordinates at every EGNN update")
        parser.add_argument("--annealed_dm_loss", action='store_true', default=False,
                            help="Anneal distance loss over each EGNN update")
        parser.add_argument("--prod_feat", type=str, default="dist", choices=["dist", "adj"],
                            help="Choose to include distances or simply adjacency to featurize product")
        parser.add_argument("--no_shuffle_mols", action='store_true', default=False,
                            help="Don't shuffle reactants and products when choosing starting species")
        parser.add_argument("--set_similar_mols", action='store_true', default=False,
                            help="Choose reactant or product with min RMSD compared to TS as starting species")
        parser.add_argument("--no_mol_prep", action='store_true', default=False,
                            help="Don't reinitialize reactant/product as if starting from SMILES")
        parser.add_argument("--product_loss", action='store_true', default=False,
                            help="Use product instead of TS to compute loss")
        parser.add_argument("--new_config", action='store_true', default=False,
                            help="When fine-tuning, use new config or not")
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
        parser.add_argument('--n_training_points', type=int, default=None)
        return parent_parser

    @staticmethod
    def add_program_args(parent_parser):
        parser = parent_parser.add_argument_group("program")
        parser.add_argument('--log_dir', type=str)
        parser.add_argument('--data_dir', type=str)
        parser.add_argument('--split_path', type=str)
        parser.add_argument('--pretrained_path', type=str, default=None)
        return parent_parser

    @classmethod
    def add_args(cls, parent_parser):
        parser = cls.add_program_args(parent_parser)  # program options
        parser = cls.add_argparse_args(parser)  # trainer options
        parser = cls.add_model_specific_args(parser)  # models specific args
        return parser

