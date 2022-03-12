#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for providing transition state initial guess geometries
"""

from rdkit import Chem
import os.path as osp
import numpy as np
from time import time
from torch_geometric.data import Batch

try:
    from rdmc.external.ts_egnn.model.ts_trainer import LitTSModule
    from rdmc.external.ts_egnn.model.data import TSDataset

    class EvalTSDataset(TSDataset):
        def __init__(self, config):
            self.set_similar_mols = config[
                "set_similar_mols"]  # use species (r/p) which is more similar to TS as starting mol
            self.shuffle_mols = config["shuffle_mols"]  # randomize which is reactant/product
            self.prep_mols = config["prep_mols"]  # prep as if starting from SMILES
            self.prod_feat = config["prod_feat"]  # whether product features include distance or adjacency

except ImportError:
    print("No TS-EGNN installation detected. Skipping import...")


class TSInitialGuesser:
    def __init__(self, track_stats=False):
        self.track_stats = track_stats
        self.n_success = None
        self.percent_success = None
        self.stats = []

    def generate_ts_guesses(self, mols, save_dir=None):
        raise NotImplementedError

    def __call__(self, mols, save_dir=None):
        time_start = time()
        ts_mol_data = self.generate_ts_guesses(mols, save_dir)

        if not self.track_stats:
            return ts_mol_data

        time_end = time()
        stats = {"time": time_end - time_start}
        self.stats.append(stats)

        return ts_mol_data


class TSEGNNGuesser(TSInitialGuesser):
    def __init__(self, trained_model_dir, track_stats=False):
        super(TSEGNNGuesser, self).__init__(track_stats)

        self.module = LitTSModule.load_from_checkpoint(
            checkpoint_path=osp.join(trained_model_dir, "best_model.ckpt"),
        )

        self.config = self.module.config
        self.module.model.eval()
        self.config["shuffle_mols"] = False
        self.config["prep_mols"] = False  # ts_generator class takes care of prep
        self.test_dataset = EvalTSDataset(self.config)

    def save_guesses(self, save_dir, rp_combos, ts_mol):

        # save stable species used to generate guesses
        r_path = osp.join(save_dir, "reactant_confs.sdf")
        p_path = osp.join(save_dir, "product_confs.sdf")
        r_writer = Chem.rdmolfiles.SDWriter(r_path)
        p_writer = Chem.rdmolfiles.SDWriter(p_path)

        for r, p in rp_combos:
            r, p = r.ToRWMol(), p.ToRWMol()
            r.SetProp("_Name", f"{Chem.MolToSmiles(r)}")
            p.SetProp("_Name", f"{Chem.MolToSmiles(p)}")
            r_writer.write(r)
            p_writer.write(p)

        r_writer.close()
        p_writer.close()

        # save ts initial guesses
        ts_path = osp.join(save_dir, "ts_initial_guess_confs.sdf")
        with Chem.rdmolfiles.SDWriter(ts_path) as ts_writer:
            [ts_writer.write(ts_mol, confId=i) for i in range(ts_mol.GetNumConformers())]

    def generate_ts_guesses(self, mols, save_dir=None):

        rp_inputs = [(x[0].ToRWMol(), None, x[1].ToRWMol()) for x in mols]
        rp_data = [self.test_dataset.process_mols(m, no_ts=True) for m in rp_inputs]
        batch_data = Batch.from_data_list(rp_data)

        # use ts_egnn to make initial guesses
        predicted_ts_coords = self.module.model(batch_data)[:, :3].cpu().detach().numpy()
        predicted_ts_coords = np.array_split(predicted_ts_coords, len(rp_inputs))

        # copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        ts_mol.EmbedMultipleNullConfs(len(rp_inputs))
        [ts_mol.GetConformer(i).SetPositions(np.array(predicted_ts_coords[i], dtype=float)) for i in
         range(len(rp_inputs))];

        if save_dir:
            self.save_guesses(save_dir, mols, ts_mol.ToRWMol())

        return ts_mol
