#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for optimizing initial guess geometries
"""

from rdmc.forcefield import RDKitFF
from .utils import *
try:
    from rdmc.external.xtb.opt import run_xtb_calc
except ImportError:
    pass


class MMFFOptimizer:
    def __init__(self, method="rdkit"):
        if method == "rdkit":
            self.ff = RDKitFF()
        elif method == "openbabel":
            raise NotImplementedError

    def __call__(self, mol_data):

        if len(mol_data) == 0:
            return mol_data

        mol = dict_to_mol(mol_data)
        self.ff.setup(mol.Copy())
        results = self.ff.optimize_confs()
        _, energies = zip(*results)  # kcal/mol
        opt_mol = self.ff.get_optimized_mol()

        for c_id, energy in zip(range(len(mol_data)), energies):
            conf = opt_mol.Copy().GetConformer(c_id)
            positions = conf.GetPositions()
            mol_data[c_id].update({"positions": positions,  # issues if not all opts succeeded?
                                   "conf": conf,
                                   "energy": energy})

        return sorted(mol_data, key=lambda x: x["energy"])


class XTBOptimizer:
    def __init__(self, method="gff"):
        self.method = method

    def __call__(self, mol_data):

        if len(mol_data) == 0:
            return mol_data

        new_mol = mol_data[0]["conf"].GetOwningMol().Copy()
        new_mol._mol.RemoveAllConformers()
        new_mol.EmbedNullConformer()

        failed_ids = set()
        for c_id, c_data in enumerate(mol_data):
            pos = c_data["conf"].GetPositions()
            new_mol.SetPositions(pos)
            try:
                _, opt_mol = run_xtb_calc(new_mol, opt=True, return_optmol=True, method=self.method)
            except ValueError as e:
                failed_ids.add(c_id)
                print(e)
                continue

            conf = opt_mol.Copy().GetConformer(0)
            positions = conf.GetPositions()
            energy = float(opt_mol.GetProp('total energy / Eh'))  # * HARTREE_TO_KCAL_MOL  # kcal/mol (TODO: check)
            mol_data[c_id].update({"positions": positions,  # issues if not all opts succeeded?
                                   "conf": conf,
                                   "energy": energy})

        mol_data = [c for i, c in enumerate(mol_data) if i not in failed_ids]
        return sorted(mol_data, key=lambda x: x["energy"])
