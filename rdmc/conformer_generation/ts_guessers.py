#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for providing transition state initial guess geometries
"""

from rdkit import Chem
import os.path as osp
import numpy as np
from time import time
import torch
from torch_geometric.data import Batch
from rdmc.external.xtb_tools.opt import run_xtb_calc

import os
from ase import Atoms
from xtb.ase.calculator import XTB
from ase.optimize import QuasiNewton
from ase.autoneb import AutoNEB
from ase.calculators.calculator import CalculationFailed

try:
    from rdmc.external.ts_egnn.ts_ml.trainers.ts_egnn_trainer import LitTSModule
    from rdmc.external.ts_egnn.ts_ml.dataloaders.ts_egnn_loader import TSDataset

    class EvalTSDataset(TSDataset):
        def __init__(self, config):
            self.set_similar_mols = config[
                "set_similar_mols"]  # use species (r/p) which is more similar to TS as starting mol
            self.shuffle_mols = config["shuffle_mols"]  # randomize which is reactant/product
            self.prep_mols = config["prep_mols"]  # prep as if starting from SMILES
            self.prod_feat = config["prod_feat"]  # whether product features include distance or adjacency

except ImportError:
    print("No TS-EGNN installation detected. Skipping import...")

try:
    from rdmc.external.ts_egnn.ts_ml.trainers.ts_gcn_trainer import LitTSModule as LitTSGCNModule
    from rdmc.external.ts_egnn.ts_ml.dataloaders.ts_gcn_loader import TSGCNDataset

    class EvalTSGCNDataset(TSGCNDataset):
        def __init__(self, config):

            self.shuffle_mols = config["shuffle_mols"]  # randomize which is reactant/product
            self.prep_mols = config["prep_mols"]  # prep as if starting from SMILES

except ImportError:
    print("No TS-GCN installation detected. Skipping import...")


class TSInitialGuesser:
    def __init__(self, track_stats=False):
        self.track_stats = track_stats
        self.n_success = None
        self.percent_success = None
        self.stats = []

    def generate_ts_guesses(self, mols, save_dir=None):
        raise NotImplementedError

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
        ts_writer = Chem.rdmolfiles.SDWriter(ts_path)
        for i in range(ts_mol.GetNumConformers()):
            ts_writer.write(ts_mol, confId=i)
        ts_writer.close()

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


class TSGCNGuesser(TSInitialGuesser):
    def __init__(self, trained_model_dir, track_stats=False):
        super(TSGCNGuesser, self).__init__(track_stats)

        self.module = LitTSGCNModule.load_from_checkpoint(
            checkpoint_path=osp.join(trained_model_dir, "best_model.ckpt"),
            strict=False,  # TODO: make sure d_init can be properly loaded
        )

        self.config = self.module.config
        self.module.model.eval()
        self.config["shuffle_mols"] = False
        self.config["prep_mols"] = False  # ts_generator class takes care of prep
        self.test_dataset = EvalTSGCNDataset(self.config)

    def generate_ts_guesses(self, mols, save_dir=None):

        rp_inputs = [(x[0].ToRWMol(), None, x[1].ToRWMol()) for x in mols]
        rp_data = [self.test_dataset.process_mols(m, no_ts=True) for m in rp_inputs]
        batch_data = Batch.from_data_list(rp_data)

        # use ts_gcn to make initial guesses
        _ = self.module.model(batch_data)
        predicted_ts_coords = torch.vstack([c[:m[0].GetNumAtoms()] for c, m in zip(batch_data.coords, batch_data.mols)])
        predicted_ts_coords = np.array_split(predicted_ts_coords.cpu().detach().numpy(), len(rp_inputs))

        # copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        ts_mol.EmbedMultipleNullConfs(len(rp_inputs))
        [ts_mol.GetConformer(i).SetPositions(np.array(predicted_ts_coords[i], dtype=float)) for i in
         range(len(rp_inputs))];

        if save_dir:
            self.save_guesses(save_dir, mols, ts_mol.ToRWMol())

        return ts_mol


class RMSDPPGuesser(TSInitialGuesser):
    def __init__(self, track_stats=False):
        super(RMSDPPGuesser, self).__init__(track_stats)

        pass

    def generate_ts_guesses(self, mols, save_dir=None):

        ts_guesses, used_rp_combos = [], []
        for r_mol, p_mol in mols:
            _, ts_guess = run_xtb_calc((r_mol, p_mol), return_optmol=True, job="--path")
            if ts_guess:
                ts_guesses.append(ts_guess)
                used_rp_combos.append((r_mol, p_mol))

        if len(ts_guesses) == 0:
            return None

        # copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        [ts_mol.AddConformer(t.GetConformer().ToConformer(), assignId=True) for t in ts_guesses]

        if save_dir:
            self.save_guesses(save_dir, used_rp_combos, ts_mol.ToRWMol())

        return ts_mol


def attach_calculators(images):
    for i in range(len(images)):
        images[i].set_calculator(XTB())


class AutoNEBGuesser(TSInitialGuesser):
    def __init__(self, track_stats=False):
        super(AutoNEBGuesser, self).__init__(track_stats)

        pass

    def generate_ts_guesses(self, mols, save_dir=None):

        ts_guesses, used_rp_combos = [], []
        for i, (r_mol, p_mol) in enumerate(mols):

            if save_dir:
                ts_conf_dir = os.path.join(save_dir, f"neb_conf{i}")
                if not os.path.exists(ts_conf_dir):
                    os.makedirs(ts_conf_dir)

            r_traj = os.path.join(ts_conf_dir, "ts000.traj")
            p_traj = os.path.join(ts_conf_dir, "ts001.traj")

            r_coords = r_mol.GetConformer().GetPositions()
            r_numbers = r_mol.GetAtomicNumbers()
            r_atoms = Atoms(positions=r_coords, numbers=r_numbers)
            r_atoms.set_calculator(XTB())
            qn = QuasiNewton(r_atoms, trajectory=r_traj, logfile=None)
            qn.run(fmax=0.05)

            p_coords = p_mol.GetConformer().GetPositions()
            p_numbers = p_mol.GetAtomicNumbers()
            p_atoms = Atoms(positions=p_coords, numbers=p_numbers)
            p_atoms.set_calculator(XTB())
            qn = QuasiNewton(p_atoms, trajectory=p_traj, logfile=None)
            qn.run(fmax=0.05)

            # need to change dirs bc autoneb path settings are messed up
            cwd = os.getcwd()

            try:
                os.chdir(ts_conf_dir)

                autoneb = AutoNEB(attach_calculators,
                                  prefix='ts',
                                  optimizer='BFGS',
                                  n_simul=3,
                                  n_max=7,
                                  fmax=0.05,
                                  k=0.5,
                                  parallel=False,
                                  maxsteps=[50, 1000])

                autoneb.run()
                os.chdir(cwd)

                used_rp_combos.append((r_mol, p_mol))
                ts_guess_idx = np.argmax(autoneb.get_energies())
                ts_guesses.append(autoneb.all_images[ts_guess_idx].positions)

            except (CalculationFailed, AssertionError) as e:
                os.chdir(cwd)

        if len(ts_guesses) == 0:
            return None

        # copy data to mol
        ts_mol = mols[0][0].Copy(quickCopy=True)
        ts_mol.EmbedMultipleNullConfs(len(ts_guesses))
        [ts_mol.GetConformer(i).SetPositions(p) for i, p in enumerate(ts_guesses)]

        if save_dir:
            self.save_guesses(save_dir, used_rp_combos, ts_mol.ToRWMol())

        return ts_mol
