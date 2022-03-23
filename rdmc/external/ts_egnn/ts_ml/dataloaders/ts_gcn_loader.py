from rdkit import Chem
import pytorch_lightning as pl
import torch
import torch_geometric as tg
from torch_geometric.data import Dataset, DataLoader

import os
import glob
import numpy as np
from ..dataloaders.ts_gcn_features import MolGraph
from ..dataloaders.features import shuffle_mols
from rdmc import RDKitMol
from rdmc.conformer_generation.align import prepare_mols


class TSGCNDataset(Dataset):

    def __init__(self, config, mode='train'):
        super(TSGCNDataset, self).__init__()

        self.split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        self.split = np.load(config["split_path"], allow_pickle=True)[self.split_idx]

        self.data_dir = config["data_dir"]
        self.mols = self.get_mols()
        self.shuffle_mols = config["shuffle_mols"]  # randomize which is reactant/product
        self.prep_mols = config["prep_mols"]  # prep as if starting from SMILES

    def process_key(self, key):
        mols = self.mols[key]
        return self.process_mols(mols)

    def process_mols(self, mols, no_ts=False):
        if self.shuffle_mols:
            mols = shuffle_mols(mols)
        if self.prep_mols:
            r_mol, ts_mol, p_mol = mols
            new_r_mol, new_p_mol = prepare_mols(RDKitMol.FromMol(r_mol), RDKitMol.FromMol(p_mol))
            mols = (new_r_mol.ToRWMol(), ts_mol, new_p_mol.ToRWMol())
        molgraph = MolGraph(mols)
        mol_data = self.molgraph2data(molgraph)
        mol_data.pos_r = torch.tensor(mols[0].GetConformer().GetPositions(), dtype=torch.float)
        mol_data.pos_ts = None if no_ts else torch.tensor(mols[1].GetConformer().GetPositions(), dtype=torch.float)
        mol_data.pos_p = torch.tensor(mols[2].GetConformer().GetPositions(), dtype=torch.float)
        mol_data.z = torch.tensor([a.GetAtomicNum() for a in mols[0].GetAtoms()], dtype=torch.int)
        mol_data.mols = mols
        return mol_data

    def molgraph2data(self, molgraph):
        data = tg.data.Data()
        data.x = torch.tensor(molgraph.f_atoms, dtype=torch.float)
        data.edge_index = torch.tensor(molgraph.edge_index, dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(molgraph.f_bonds, dtype=torch.float)
        data.y = torch.tensor(molgraph.y, dtype=torch.float)
        # data.bonded_index = torch.tensor(molgraph.bonded_index, dtype=torch.long).t().contiguous()

        return data

    def get_mols(self):

        r_file = glob.glob(os.path.join(self.data_dir, '*_reactants.sdf'))[0]
        ts_file = glob.glob(os.path.join(self.data_dir, '*_ts.sdf'))[0]
        p_file = glob.glob(os.path.join(self.data_dir, '*_products.sdf'))[0]

        data = [Chem.SDMolSupplier(r_file, removeHs=False, sanitize=True),
                Chem.SDMolSupplier(ts_file, removeHs=False, sanitize=False),
                Chem.SDMolSupplier(p_file, removeHs=False, sanitize=True)]

        data = [(x, y, z) for (x, y, z) in zip(data[0], data[1], data[2]) if (x, y, z)]
        return [data[i] for i in self.split]

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, key):
        return self.process_key(key)


class TSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        train_dataset = TSGCNDataset(config, 'train')
        self.num_node_features = train_dataset.num_node_features
        self.num_edge_features = train_dataset.num_edge_features

    def train_dataloader(self):
        train_dataset = TSGCNDataset(self.config, 'train')
        return DataLoader(dataset=train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataset = TSGCNDataset(self.config, 'val')
        return DataLoader(dataset=val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        test_dataset = TSGCNDataset(self.config, 'test')
        return DataLoader(dataset=test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
