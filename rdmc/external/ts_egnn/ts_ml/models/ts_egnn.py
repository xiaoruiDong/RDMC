import torch
from torch import nn
from egnn_pytorch import EGNN_Sparse


class TS_EGNN(nn.Module):
    def __init__(self, config):
        super(TS_EGNN, self).__init__()

        self.egnn_layers = nn.ModuleList()
        self.cutoff = config["cutoff"]
        for _ in range(config["depth"]):
            self.egnn_layers.append(EGNN_Sparse(feats_dim=config["hidden_dim"],
                                                pos_dim=3,
                                                edge_attr_dim=config["hidden_dim"],
                                                m_dim=16,
                                                fourier_features=5,
                                                soft_edge=0,
                                                norm_feats=True,
                                                norm_coors=True,
                                                norm_coors_scale_init=1e-2,
                                                dropout=0.,
                                                coor_weights_clamp_value=None,
                                                aggr="add",
                                                ))

        self.node_init = nn.Linear(config["node_dim"], config["hidden_dim"])
        self.edge_init = nn.Linear(config["edge_dim"], config["hidden_dim"])

    def forward(self, data):

        x = torch.cat([data.pos_r, self.node_init(data.x)], dim=-1)
        edge_index, edge_attr = radius_graph(data.pos_r, self.cutoff, data.edge_index, data.edge_attr)
        edge_attr = self.edge_init(edge_attr)
        batch = data.batch

        self.coords = [data.pos_r]
        for layer in self.egnn_layers:
            x = layer(x, edge_index, edge_attr, batch)
            self.coords.append(x[:, :3])
        return x


def radius_graph(pos, cutoff, edge_index, edge_attr):
    row, col = edge_index
    dist = (pos[row] - pos[col]).norm(dim=-1)
    mask = dist < cutoff
    return edge_index[:, mask], edge_attr[mask]
