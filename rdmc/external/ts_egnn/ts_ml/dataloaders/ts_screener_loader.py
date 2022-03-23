from rdkit import Chem
from rdmc import RDKitMol
from rdmc.ts import get_all_changing_bonds
import torch
import torch_geometric as tg
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.utils import to_undirected
import pytorch_lightning as pl
from ..dataloaders.features import onek_encoding_unk
import numpy as np


ATOMIC_SYMBOLS = ['H', 'C', 'N', 'O']


def atom_features(atom):
    features = onek_encoding_unk(atom.GetSymbol(), ATOMIC_SYMBOLS)
    features += [atom.GetFormalCharge()]
    features += [atom.GetNumRadicalElectrons()]
    return features


class TSScreenerDataset(Dataset):

    def __init__(self, config, mode="train"):
        super(TSScreenerDataset, self).__init__()

        self.split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        self.split = np.load(config["split_path"], allow_pickle=True)[self.split_idx]
        all_irc_data = RDKitMol.FromFile(config["data_path"])
        self.all_irc_data = [all_irc_data[i] for i in self.split]

    def process_key(self, key):
        irc_data = self.all_irc_data[key]
        return mol2data(irc_data)

    def __len__(self):
        return len(self.all_irc_data)

    def __getitem__(self, key):
        return self.process_key(key)

    @staticmethod
    def get_dims():
        mol = RDKitMol.FromSmiles("CCC")
        mol.EmbedConformer()
        mol.SetProp("Name", "CCC>>CCC")
        mol.SetProp("IRC_result", "True")
        data = mol2data(mol)
        node_dim = data.x.size(-1)
        edge_dim = data.edge_attr.size(-1)
        return {"node_dim": node_dim, "edge_dim": edge_dim}


def mol2data(ts_mol):
    rxn_smiles = ts_mol.GetProp("Name")
    y = 1. if ts_mol.GetProp("IRC_result") == "True" else 0.

    r_smi, p_smi = rxn_smiles.split(">>")
    r_mol = RDKitMol.FromSmiles(r_smi)
    p_mol = RDKitMol.FromSmiles(p_smi)
    r_bonds = [tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in r_mol.GetBonds()]
    p_bonds = [tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in p_mol.GetBonds()]
    cgr_bonds = sorted(list(set(r_bonds) | set(p_bonds)))
    rxn_core_mask = get_rxn_core_mask(r_mol, p_mol)
    z = ts_mol.GetAtomicNumbers()
    pos = ts_mol.GetPositions()
    edge_attr = []

    x_r = torch.tensor([atom_features(a) for a in r_mol.GetAtoms()], dtype=torch.float)
    x_p = torch.tensor([atom_features(a) for a in p_mol.GetAtoms()], dtype=torch.float)
    x = torch.cat([x_r, x_p[:, 5:]], dim=-1)

    data = tg.data.Data()
    data.edge_index = to_undirected(torch.tensor(cgr_bonds, dtype=torch.long).t().contiguous())
    data.edge_index_r = to_undirected(torch.tensor(r_bonds, dtype=torch.long).t().contiguous())
    data.edge_index_p = to_undirected(torch.tensor(p_bonds, dtype=torch.long).t().contiguous())
    data.x = x
    data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    data.z = torch.tensor(z, dtype=torch.long)
    data.pos = torch.tensor(pos, dtype=torch.float)
    data.y = torch.tensor(y, dtype=torch.float)
    data.smiles = (r_smi, p_smi)
    data.rxn_core_mask = torch.tensor(rxn_core_mask, dtype=torch.long)

    return data


def get_rxn_core_mask(r_mol, p_mol):
    changing_bonds = get_all_changing_bonds(r_mol, p_mol)
    changing_atoms = set([x for y in [x for y in changing_bonds for x in y] for x in y])
    mask = [1 if i in changing_atoms else 0 for i in np.arange(r_mol.GetNumAtoms())]
    return mask


class TSScreenerDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.config.update(TSScreenerDataset.get_dims())

    def train_dataloader(self):
        train_dataset = TSScreenerDataset(self.config, 'train')
        return DataLoader(dataset=train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataset = TSScreenerDataset(self.config, 'val')
        return DataLoader(dataset=val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        test_dataset = TSScreenerDataset(self.config, 'test')
        return DataLoader(dataset=test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
