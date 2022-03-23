import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from ..models.ts_screener import SchNetClassifier


class LitScreenerModule(pl.LightningModule):
    def __init__(self, config):
        super(LitScreenerModule, self).__init__()

        self.save_hyperparameters()
        self.model = SchNetClassifier(config)
        self.config = config
        self.lr = config["lr"]

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7,
                                                               patience=5, min_lr=self.lr / 100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_acc"}

    def _step(self, data, batch_idx, mode):
        out = self(data)

        # calculate loss
        loss = F.binary_cross_entropy(out, data.y)

        # calculate batch accuracy
        acc = (out.round() == data.y).detach().cpu().numpy().mean()

        # calculate confusion matrix
        cm = confusion_matrix(data.y.int().detach().cpu().numpy(),
                              out.round().int().detach().cpu().numpy())

        # logs
        batch_size = len(data.ptr)
        self.log(f'{mode}_loss', loss, batch_size=batch_size)
        self.log(f'{mode}_acc', acc, batch_size=batch_size)

        return loss

    def training_step(self, data, batch_idx):
        loss = self._step(data, batch_idx, mode="train")
        return loss

    def validation_step(self, data, batch_idx):
        loss = self._step(data, batch_idx, mode="val")
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("models")
        # general args
        parser.add_argument("--model", type=str, default="schnet",
                            choices=["schnet", "dn_pp"])
        parser.add_argument("--hidden_channels", type=int, default=64)
        parser.add_argument('--num_blocks', type=int, default=3)
        parser.add_argument("--num_filters", type=int, default=64)
        parser.add_argument("--cutoff", type=float, default=10.0,
                            help="Cutoff radius for graph construction")
        parser.add_argument("--hidden_dim", type=int, default=32)
        parser.add_argument("--use_rxn_core", action='store_true', default=False,
                            help="Use only reaction core atoms to generate molecular representation")
        # schnet
        parser.add_argument("--num_gaussians", type=int, default=25)
        # dimenet pp (refer here https://github.com/gasteigerjo/dimenet/blob/master/config_pp.yaml)
        parser.add_argument('--int_emb_size', type=int, default=64)
        parser.add_argument('--basis_emb_size', type=int, default=8)
        parser.add_argument('--num_spherical', type=int, default=7)
        parser.add_argument('--num_radial', type=int, default=6)
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
        parser.add_argument('--data_path', type=str)
        parser.add_argument('--split_path', type=str)
        return parent_parser

    @classmethod
    def add_args(cls, parent_parser):
        parser = cls.add_program_args(parent_parser)  # program options
        parser = cls.add_argparse_args(parser)  # trainer options
        parser = cls.add_model_specific_args(parser)  # models specific args
        return parser
