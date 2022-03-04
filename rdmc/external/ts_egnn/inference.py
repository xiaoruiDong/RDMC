#!/usr/bin/env python3
#-*- coding: utf-8 -*-


"A standalone scirpt to run TS-EGNN given the reactant and product SDF file. Created based on `use_egnn.ipynb`"

from rdkit import Chem
from rdmc import RDKitMol

from argparse import ArgumentParser
import os
from typing import List

import numpy as np
from model.ts_trainer import LitTSModule
from model.data import TSDataset
from torch_geometric.data import Batch


model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'trained_models',
                         '2022_02_07',
                         'best_model.ckpt')

class EvalTSDataset(TSDataset):
    def __init__(self, config):

        self.set_similar_mols = config["set_similar_mols"]  # use species (r/p) which is more similar to TS as starting mol
        self.shuffle_mols = config["shuffle_mols"]  # randomize which is reactant/product
        self.prep_mols = config["prep_mols"]  # prep as if starting from SMILES
        self.prod_feat = config["prod_feat"]  # whether product features include distance or adjacency


def inference(r_mols: List[Chem.rdchem.Mol],
              p_mols: List[Chem.rdchem.Mol],
              ts_xyz_path: str = 'TS.xyz',
              ):
    """
    Loads in the best weights from hyperparameter optimization to predict a TS guess.
    The TS guess is written to an xyz file.

    Args:
        r_mol: List of RDKit molecule objects for the reactant/s present in the sdf file.
        p_mol: List of RDKit molecule objects for the product/s present in the sdf file.
        ts_xyz_path: String indicating the path to write the TS guess structure to.

    """

    TSModule = LitTSModule.load_from_checkpoint(
        checkpoint_path=model_path,
    )

    config = TSModule.config
    config["shuffle_mols"] = False
    config["prep_mols"] = True
    test_dataset = EvalTSDataset(config)

    mols = (r_mols[0], None, p_mols[0])
    data = test_dataset.process_mols(mols, no_ts=True)
    batch_data = Batch.from_data_list([data])
    predicted_ts_coords = TSModule.model(batch_data)[:, :3].cpu().detach().numpy()
    predicted_ts = RDKitMol(mols[0])
    predicted_ts.EmbedConformer()
    predicted_ts.SetPositions(np.array(predicted_ts_coords, dtype=float))

    with open(ts_xyz_path, 'w') as f:
        f.write(predicted_ts.ToXYZ())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--r_sdf_path', type=str, default='reactant.sdf')
    parser.add_argument('--p_sdf_path', type=str, default='product.sdf')
    parser.add_argument('--ts_xyz_path', type=str, default='TS.xyz')
    args = parser.parse_args()

    # read in sdf files for reactant and product of the atom-mapped reaction
    r_mols = Chem.SDMolSupplier(args.r_sdf_path, removeHs=False, sanitize=True)
    p_mols = Chem.SDMolSupplier(args.p_sdf_path, removeHs=False, sanitize=True)
    inference(r_mols, p_mols, args.ts_xyz_path)
