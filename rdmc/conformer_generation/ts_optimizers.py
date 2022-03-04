#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for optimizing transition state geometries
"""

from rdkit import Chem
import os
from time import time
import pickle
from rdmc.external.sella import run_sella_xtb_opt


class TSOptimizer:
    def __init__(self, track_stats=False):
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def optimize_ts_guesses(self, mol, save_dir):
        raise NotImplementedError

    def __call__(self, mol, save_dir):
        time_start = time()
        opt_mol = self.optimize_ts_guesses(mol, save_dir)

        if not self.track_stats:
            return opt_mol

        time_end = time()
        stats = {"time": time_end - time_start}
        self.stats.append(stats)

        return opt_mol


class SellaOptimizer(TSOptimizer):
    def __init__(self, fmax=1e-3, steps=1000, track_stats=False):
        super(SellaOptimizer, self).__init__(track_stats)

        self.fmax = fmax
        self.steps = steps

    def save_opt_mols(self, save_dir, opt_mol):

        # save optimized ts mols
        ts_path = os.path.join(save_dir, "ts_optimized_confs.sdf")
        with Chem.rdmolfiles.SDWriter(ts_path) as ts_writer:
            [ts_writer.write(opt_mol, confId=i) for i in range(opt_mol.GetNumConformers())]

    def optimize_ts_guesses(self, mol, save_dir=None):

        for i in range(mol.GetNumConformers()):
            if save_dir:
                ts_conf_dir = os.path.join(save_dir, f"conf{i}")
                if not os.path.exists(ts_conf_dir):
                    os.makedirs(ts_conf_dir)
            opt_mol = run_sella_xtb_opt(mol, confId=i, fmax=self.fmax, steps=self.steps, save_dir=ts_conf_dir)

        if save_dir:
            self.save_opt_mols(save_dir, opt_mol.ToRWMol())

        return opt_mol
