import torch
from ts_ml.models.schnet import SchNet
from ts_ml.models.dimenet_pp import DimeNetPlusPlus


class SchNetClassifier(torch.nn.Module):
    def __init__(self, config):
        super(SchNetClassifier, self).__init__()

        if config["model"] == "schnet":
            self.featurizer = SchNet(
                node_dim=config["node_dim"],
                hidden_channels=config["hidden_channels"],
                num_filters=config["num_filters"],
                num_interactions=config["num_blocks"],
                num_gaussians=config["num_gaussians"],
                cutoff=config["cutoff"],
                out_dim=config["hidden_dim"],
                use_rxn_core=config["use_rxn_core"],
            )
        elif config["model"] == "dn_pp":
            self.featurizer = DimeNetPlusPlus(
                node_dim=config["node_dim"],
                hidden_channels=config["hidden_channels"],
                out_channels=config["hidden_dim"],
                num_blocks=config["num_blocks"],
                int_emb_size=config["int_emb_size"],
                basis_emb_size=config["basis_emb_size"],
                out_emb_channels=config["hidden_dim"],
                num_spherical=config["num_spherical"],
                num_radial=config["num_radial"],
                num_output_layers=2,
                use_rxn_core=config["use_rxn_core"],
            )

        self.lin = torch.nn.Linear(config["hidden_dim"], 1)

    def forward(self, data):
        latent_v = self.featurizer(data.x, data.z, data.pos, data.edge_index, data.batch, data.rxn_core_mask)
        # latent_r = self.featurizer(data.x, data.z, data.pos, data.edge_index_r, data.batch)
        # latent_p = self.featurizer(data.x, data.z, data.pos, data.edge_index_p, data.batch)
        #
        # latent_v = latent_r * latent_p
        pred = torch.sigmoid(self.lin(latent_v))

        return pred.squeeze(-1)
