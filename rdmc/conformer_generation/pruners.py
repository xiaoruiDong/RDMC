#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Modules for pruning a group of conformers
"""

from rdmc.mol import RDKitMol
import numpy as np
try:
    from rdmc.external.xtb.crest import run_cre_check
except ImportError:
    pass


class TorsionPruner:
    """
    Prune conformers based on torsion angle criteria.
    This method uses a mean and max criteria to prune conformers:
    A conformer is considered unique if it satisfies either of the following criteria:
        mean difference of all torsion angles > mean_chk_threshold
        max difference of all torsion angles > max_chk_threshold
    New conformers are compared to all conformers that have already been deemed unique
    """

    def __init__(self, mean_chk_threshold=10, max_chk_threshold=20):

        self.mean_chk_threshold = mean_chk_threshold
        self.max_chk_threshold = max_chk_threshold
        self.torsions_list = None

    def initialize_torsions_list(self, smiles):

        mol = RDKitMol.FromSmiles(smiles)
        mol.EmbedNullConformer()
        self.torsions_list = mol.GetConformer().GetTorsionalModes()

    def calculate_torsions(self, mol_data):

        for conf_data in mol_data:
            conf = conf_data["conf"]
            torsions = np.array([conf.GetTorsionDeg(t) for t in self.torsions_list]) % 360
            conf_data.update({"torsions": torsions})
        return mol_data

    def rad_angle_compare(self, x, y):

        # compare angles in radians
        return np.abs(np.arctan2(np.sin(x - y), np.cos(x - y))) * 180 / np.pi

    def torsion_list_compare(self, c1_ts, c2_ts):

        # compare two lists of torsions in radians
        return [self.rad_angle_compare(t1, t2) for t1, t2 in zip(c1_ts, c2_ts)]

    def __call__(self, current_mol_data, unique_mol_data=None):

        if unique_mol_data is None:
            unique_mol_data = []

        # calculate torsions for new mols
        current_mol_data = self.calculate_torsions(current_mol_data)

        # prep comparison and compute torsion matrix
        n_unique_mols = max(1, len(unique_mol_data))  # set to 1 if 0
        mols_list = unique_mol_data + current_mol_data
        torsion_matrix = np.stack([c["torsions"] for c in mols_list])
        torsion_matrix_rad = torsion_matrix * np.pi / 180
        n_confs = len(mols_list)
        conf_ids = np.arange(n_confs).tolist()

        # start comparison at new mols
        for i in conf_ids[n_unique_mols:]:

            c_torsions = torsion_matrix_rad[i]  # torsions of this conformer
            c_before_torsions = torsion_matrix_rad[:i]  # torsions of all other conformers already compared

            # mean and max criteria checks
            comp = np.array([self.torsion_list_compare(c_torsions, ct) for ct in c_before_torsions])
            chk1 = (np.mean(comp, axis=1) < self.mean_chk_threshold).any()
            chk2 = (np.max(comp, axis=1) < self.max_chk_threshold).any()

            # remove conformer if either check is satisfied
            if chk1 or chk2:
                conf_ids.remove(i)

        # update mols and sort by energy
        updated_unique_mol_data = sorted([mols_list[i] for i in conf_ids], key=lambda x: x["energy"])
        return updated_unique_mol_data


class CRESTPruner:
    def __init__(self, ethr=0.15, rthr=0.125, bthr=0.01, ewin=10000):

        self.ethr = ethr
        self.rthr = rthr
        self.bthr = bthr
        self.ewin = ewin

    def __call__(self, current_mol_data, unique_mol_data=None):

        if unique_mol_data is None:
            unique_mol_data = []

        all_mol_data = unique_mol_data + current_mol_data
        updated_unique_mol_data = run_cre_check(all_mol_data,
                                                ethr=self.ethr,
                                                rthr=self.rthr,
                                                bthr=self.bthr,
                                                ewin=self.ewin
                                                )
        updated_unique_mol_data = sorted(updated_unique_mol_data, key=lambda x: x["energy"])
        return updated_unique_mol_data
