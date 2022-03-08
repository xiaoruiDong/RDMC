#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for optimizing initial guess geometries
"""

from rdmc.forcefield import RDKitFF
from time import time
import numpy as np
from .utils import *
try:
    from rdmc.external.xtb_tools.opt import run_xtb_calc
except ImportError:
    pass


class ConfGenOptimizer:
    def __init__(self, track_stats=False):

        self.iter = 0
        self.track_stats = track_stats
        self.n_failures = None
        self.percent_failures = None
        self.n_opt_cycles = None
        self.stats = []

    def optimize_conformers(self, mol_data):
        raise NotImplementedError

    def __call__(self, mol_data):

        self.iter += 1
        time_start = time()
        mol_data = self.optimize_conformers(mol_data)

        if not self.track_stats:
            return mol_data

        time_end = time()
        stats = {"iter": self.iter,
                 "time": time_end - time_start,
                 "n_failures": self.n_failures,
                 "percent_failures": self.percent_failures,
                 "n_opt_cycles": self.n_opt_cycles}
        self.stats.append(stats)
        return mol_data


class MMFFOptimizer(ConfGenOptimizer):
    def __init__(self, method="rdkit", track_stats=False):
        super(MMFFOptimizer, self).__init__(track_stats)
        if method == "rdkit":
            self.ff = RDKitFF()
        elif method == "openbabel":
            raise NotImplementedError

    def optimize_conformers(self, mol_data):

        if len(mol_data) == 0:
            return mol_data

        # Everytime calling dict_to_mol create a new molecule object
        # No need to Copy the molecule object in this function
        mol = dict_to_mol(mol_data)
        self.ff.setup(mol)
        results = self.ff.optimize_confs()
        _, energies = zip(*results)  # kcal/mol
        opt_mol = self.ff.get_optimized_mol()

        for c_id, energy in zip(range(len(mol_data)), energies):
            conf = opt_mol.GetConformer(c_id)
            positions = conf.GetPositions()
            mol_data[c_id].update({"positions": positions,  # issues if not all opts succeeded?
                                   "conf": conf,  # all confs share the same owning molecule `opt_mol`
                                   "energy": energy})

        if self.track_stats:
            self.n_failures = np.sum([r[0] == 1 for r in results])
            self.percent_failures = self.n_failures / len(mol_data) * 100

        return sorted(mol_data, key=lambda x: x["energy"])


class XTBOptimizer(ConfGenOptimizer):
    def __init__(self, method="gff", level="normal", track_stats=False):
        super(XTBOptimizer, self).__init__(track_stats)
        self.method = method
        self.level = level

    def optimize_conformers(self, mol_data):

        if len(mol_data) == 0:
            return mol_data

        new_mol = dict_to_mol(mol_data)
        correct_atom_mapping = new_mol.GetAtomMapNumbers()

        failed_ids = set()
        all_props = []
        for c_id in range(len(mol_data)):
            try:
                props, opt_mol = run_xtb_calc(new_mol, confId=c_id, job="--opt", return_optmol=True, method=self.method, level=self.level)
                all_props.append(props)
            except ValueError as e:
                failed_ids.add(c_id)
                print(e)
                continue

            opt_mol.SetAtomMapNumbers(correct_atom_mapping)
            # Renumber the molecule based on the atom mapping just set
            opt_mol.RenumberAtoms()
            positions = opt_mol.GetPositions()
            conf = new_mol.GetConformer(id=c_id)
            conf.SetPositions(positions)
            energy = float(opt_mol.GetProp('total energy / Eh'))  # * HARTREE_TO_KCAL_MOL  # kcal/mol (TODO: check)
            mol_data[c_id].update({"positions": positions,  # issues if not all opts succeeded?
                                   "conf": conf,
                                   "energy": energy})

        final_mol_data = [c for i, c in enumerate(mol_data) if i not in failed_ids]

        if self.track_stats:
            self.n_failures = len(failed_ids)
            self.percent_failures = self.n_failures / len(mol_data) * 100
            self.n_opt_cycles = [p["n_opt_cycles"] if "n_opt_cycles" in p else -1 for p in all_props]

        return sorted(final_mol_data, key=lambda x: x["energy"])
