#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for optimizing transition state geometries
"""

from rdkit import Chem
from rdmc import RDKitMol
import os
from time import time
from rdmc.external.sella import run_sella_opt
from rdmc.external.orca import write_orca_opt
from rdmc.external.gaussian import GaussianLog, write_gaussian_ts_opt
import subprocess


class TSOptimizer:
    def __init__(self, track_stats=False):
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def optimize_ts_guesses(self, mol, save_dir, **kwargs):
        raise NotImplementedError

    def save_opt_mols(self, save_dir, opt_mol):

        # save optimized ts mols
        ts_path = os.path.join(save_dir, "ts_optimized_confs.sdf")
        with Chem.rdmolfiles.SDWriter(ts_path) as ts_writer:
            [ts_writer.write(opt_mol, confId=i) for i in range(opt_mol.GetNumConformers())]

    def __call__(self, mol, save_dir, **kwargs):
        time_start = time()
        opt_mol = self.optimize_ts_guesses(mol, save_dir, **kwargs)

        if not self.track_stats:
            return opt_mol

        time_end = time()
        stats = {"time": time_end - time_start}
        self.stats.append(stats)

        return opt_mol


class SellaOptimizer(TSOptimizer):
    def __init__(self, method="GFN2-xTB", fmax=1e-3, steps=1000, track_stats=False):
        super(SellaOptimizer, self).__init__(track_stats)

        self.method = method
        self.fmax = fmax
        self.steps = steps

    def optimize_ts_guesses(self, mol, save_dir=None, **kwargs):

        opt_mol = mol.Copy()
        for i in range(mol.GetNumConformers()):
            if save_dir:
                ts_conf_dir = os.path.join(save_dir, f"conf{i}")
                if not os.path.exists(ts_conf_dir):
                    os.makedirs(ts_conf_dir)

            opt_mol = run_sella_opt(opt_mol,
                                    method=self.method,
                                    confId=i,
                                    fmax=self.fmax,
                                    steps=self.steps,
                                    save_dir=ts_conf_dir
                                    )
        if save_dir:
            self.save_opt_mols(save_dir, opt_mol.ToRWMol())

        return opt_mol


class OrcaOptimizer(TSOptimizer):
    def __init__(self, method="XTB2", track_stats=False):
        super(OrcaOptimizer, self).__init__(track_stats)

        self.method = method

    def optimize_ts_guesses(self, mol, save_dir=None, **kwargs):

        r_smi, _ = kwargs["rxn_smiles"].split(">>")
        r_mol = RDKitMol.FromSmiles(r_smi)
        multiplicity = r_mol.GetSpinMultiplicity()

        ORCA_BINARY = os.environ.get("ORCA")
        opt_mol = mol.Copy(quickCopy=True)
        for i in range(mol.GetNumConformers()):
            if save_dir:
                ts_conf_dir = os.path.join(save_dir, f"orca_opt{i}")
                if not os.path.exists(ts_conf_dir):
                    os.makedirs(ts_conf_dir)

            orca_str = write_orca_opt(mol, confId=i, method=self.method, mult=multiplicity)
            orca_input_file = os.path.join(ts_conf_dir, "orca_opt.inp")
            with open(orca_input_file, "w") as f:
                f.writelines(orca_str)

            with open(os.path.join(ts_conf_dir, "orca_opt.log"), "w") as f:
                orca_run = subprocess.run(
                    [ORCA_BINARY, orca_input_file],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                )
            if orca_run.returncode == 0:
                new_mol = RDKitMol.FromFile(os.path.join(ts_conf_dir, "orca_opt.xyz"), sanitize=False)
                opt_mol.AddConformer(new_mol.GetConformer().ToConformer(), assignId=True)

        if save_dir:
            self.save_opt_mols(save_dir, opt_mol.ToRWMol())

        return opt_mol


class GaussianOptimizer(TSOptimizer):
    def __init__(self, method="GFN2-xTB", track_stats=False):
        super(GaussianOptimizer, self).__init__(track_stats)

        self.method = method

    def optimize_ts_guesses(self, mol, save_dir=None, **kwargs):

        r_smi, _ = kwargs["rxn_smiles"].split(">>")
        r_mol = RDKitMol.FromSmiles(r_smi)
        multiplicity = r_mol.GetSpinMultiplicity()

        GAUSSIAN_BINARY = os.path.join(os.environ.get("g16root"), "g16", "g16")
        opt_mol = mol.Copy(quickCopy=True)
        for i in range(mol.GetNumConformers()):
            if save_dir:
                ts_conf_dir = os.path.join(save_dir, f"gaussian_opt{i}")
                if not os.path.exists(ts_conf_dir):
                    os.makedirs(ts_conf_dir)

            gaussian_str = write_gaussian_ts_opt(mol, confId=i, method=self.method, mult=multiplicity)
            gaussian_input_file = os.path.join(ts_conf_dir, "gaussian_opt.gjf")
            with open(gaussian_input_file, "w") as f:
                f.writelines(gaussian_str)

            with open(os.path.join(ts_conf_dir, "gaussian_opt.log"), "w") as f:
                gaussian_run = subprocess.run(
                    [GAUSSIAN_BINARY, gaussian_input_file],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                )
            if gaussian_run.returncode == 0:
                g16_log = GaussianLog(os.path.join(ts_conf_dir, "gaussian_opt.log"))
                if g16_log.success:
                    new_mol = g16_log.get_mol(embed_conformers=False)
                    opt_mol.AddConformer(new_mol.GetConformer().ToConformer(), assignId=True)

        if save_dir:
            self.save_opt_mols(save_dir, opt_mol.ToRWMol())

        return opt_mol
