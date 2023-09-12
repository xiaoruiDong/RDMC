#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for pruning a group of conformers
"""

from time import time
from typing import List, Optional

import numpy as np

from rdmc.mol import RDKitMol
try:
    from rdmc.external.xtb_tools.crest import run_cre_check
except ImportError:
    pass


class ConfGenPruner:
    """
    Base class for conformer pruning.

    Args:
        track_stats (bool, optional): Whether to track statistics. Defaults to ``False``.
    """

    def __init__(self,
                 track_stats: bool = False):

        self.iter = 0
        self.track_stats = track_stats
        self.n_input_confs = None
        self.n_pruned_confs = None
        self.n_output_confs = None
        self.stats = []

    def prune_conformers(self,
                         current_mol_data: List[dict],
                         unique_mol_data: Optional[List[dict]] = None,
                         sort_by_energy: bool = True,
                         return_ids: bool = False):
        """
        Prune conformers.

        Args:
            current_mol_data (list[dict]): conformer data of the current iteration.
            unique_mol_data (list[dict], optional): Unique conformer data of previous iterations. Defaults to ``None``.
            sort_by_energy (bool, optional): Whether to sort conformers by energy. Defaults to ``True``.
            return_ids (bool, optional): Whether to return conformer IDs. Defaults to ``False``.

        Raises:
            NotImplementedError: This method should be implemented in the subclass.
        """
        raise NotImplementedError

    def __call__(self,
                 current_mol_data: List[dict],
                 unique_mol_data: Optional[List[dict]] = None,
                 sort_by_energy: bool = True,
                 return_ids: bool = False):
        """
        Execute the task of pruning conformers.

        Args:
            current_mol_data (list[dict]): conformer data of the current iteration.
            unique_mol_data (list[dict], optional): Unique conformer data of previous iterations. Defaults to ``None``.
            sort_by_energy (bool, optional): Whether to sort conformers by energy. Defaults to ``True``.
            return_ids (bool, optional): Whether to return conformer IDs. Defaults to ``False``.

        Returns:
            list[dict]: Updated conformer data.
        """
        self.iter += 1
        time_start = time()
        mol_data = self.prune_conformers(current_mol_data, unique_mol_data, sort_by_energy, return_ids)

        if not self.track_stats:
            return mol_data

        time_end = time()
        stats = {"iter": self.iter,
                 "time": time_end - time_start,
                 "n_input_confs": self.n_input_confs,
                 "n_pruned_confs": self.n_pruned_confs,
                 "n_output_confs": self.n_output_confs}
        self.stats.append(stats)
        return mol_data


class TorsionPruner(ConfGenPruner):
    """
    Prune conformers based on torsion angle criteria.
    This method uses a mean and max criteria to prune conformers:
    A conformer is considered unique if it satisfies either of the following criteria:

    - mean difference of all torsion angles > mean_chk_threshold
    - max difference of all torsion angles > max_chk_threshold

    New conformers are compared to all conformers that have already been deemed unique.

    Args:
        mean_chk_threshold (float, optional): Mean difference threshold. Defaults to ``10.``.
        max_chk_threshold (float, optional): Max difference threshold. Defaults to ``20.``.
        track_stats (bool, optional): Whether to track statistics. Defaults to ``False``.
    """

    def __init__(self,
                 mean_chk_threshold: float = 10.,
                 max_chk_threshold: float = 20.,
                 track_stats: bool = False):
        super(TorsionPruner, self).__init__(track_stats)

        self.mean_chk_threshold = mean_chk_threshold
        self.max_chk_threshold = max_chk_threshold
        self.torsions_list = None

    def initialize_torsions_list(self,
                                 smiles: Optional[str] = None,
                                 torsions: Optional[list] = None,
                                 excludeMethyl: bool = False):
        """
        Initialize the list of torsions to be used for comparison and pruning.

        Args:
            smiles (str, optional): SMILES of the molecule. Defaults to ``None``. This should be provided if
                                    ``torsions`` is not provided.
            torsions (list, optional): List of torsions. Defaults to ``None``,
                                       in which case the torsions will be extracted from the molecule.
            excludeMethyl (bool, optional): Whether to exclude methyl groups. Defaults to ``False``.
        """
        if torsions:
            self.torsions_list = torsions
        elif smiles:
            mol = RDKitMol.FromSmiles(smiles)
            self.torsions_list = mol.GetTorsionalModes(excludeMethyl=excludeMethyl, includeRings=True)
        else:
            raise ValueError("Either a SMILES or a list of torsional modes should be provided.")

    def initialize_ts_torsions_list(self,
                                    rxn_smiles: Optional[str] = None,
                                    torsions: Optional[list] = None,
                                    excludeMethyl: bool = False):
        """
        Initialize the list of torsions to be used for comparison and pruning for TS molecules.

        Args:
            rxn_smiles (str, optional): SMILES of the reaction. Defaults to ``None``. This should be provided if
                                        ``torsions`` is not provided.
            torsions (list, optional): List of torsions. Defaults to ``None``, in which case the torsions will be
                                       extracted according to the reactants and the products.
            excludeMethyl (bool, optional): Whether to exclude methyl groups. Defaults to ``False``.
        """

        if torsions:
            self.torsions_list = torsions
        elif rxn_smiles:
            r_smi, p_smi = rxn_smiles.split(">>")
            r_mol = RDKitMol.FromSmiles(r_smi)
            p_mol = RDKitMol.FromSmiles(p_smi)
            torsions_list = \
                r_mol.GetTorsionalModes(excludeMethyl=excludeMethyl, includeRings=True) + \
                p_mol.GetTorsionalModes(excludeMethyl=excludeMethyl, includeRings=True)
            self.torsions_list = [list(x) for x in set(tuple(x) for x in torsions_list)]
        else:
            raise ValueError("Either a SMILES or a list of torsional modes should be provided.")

    def calculate_torsions(self,
                           mol_data: List[dict],
                           ) -> List[dict]:
        """
        Calculate torsions for a list of conformers.

        Args:
            mol_data (list[dict]): conformer data.

        Returns:
            list[dict]: Conformer data with values of torsions added.
        """
        for conf_data in mol_data:
            conf = conf_data["conf"]
            torsions = np.array([conf.GetTorsionDeg(t) for t in self.torsions_list]) % 360
            conf_data.update({"torsions": torsions})
        return mol_data

    @staticmethod
    def rad_angle_compare(x: float,
                          y: float,
                          ) -> float:
        """
        Compare two angles in radians.

        Args:
            x (float): angle in degrees.
            y (float): angle in degrees.

        Returns:
            float: Absolute difference between the two angles in radians.
        """
        return np.abs(np.arctan2(np.sin(x - y), np.cos(x - y))) * 180 / np.pi

    @staticmethod
    def torsion_list_compare(c1_ts: List[float],
                             c2_ts: List[float],
                             ) -> list[float]:
        """
        Compare two lists of torsions in radians.

        Args:
            c1_ts (list): list of torsions in degrees.
            c2_ts (list): list of torsions in degress.

        Returns:
            list: Absolute difference between the two lists of torsions in radians.
        """
        # compare two lists of torsions in radians
        return [TorsionPruner.rad_angle_compare(t1, t2) for t1, t2 in zip(c1_ts, c2_ts)]

    def prune_conformers(self,
                         current_mol_data: List[dict],
                         unique_mol_data: Optional[List[dict]] = None,
                         sort_by_energy: bool = True,
                         return_ids: bool = False):
        """
        Prune conformers.

        Args:
            current_mol_data (list[dict]): conformer data of the current iteration.
            unique_mol_data (list[dict], optional): Unique conformer data of previous iterations. Defaults to ``None``.
            sort_by_energy (bool, optional): Whether to sort conformers by energy. Defaults to ``True``.
            return_ids (bool, optional): Whether to return conformer IDs. Defaults to ``False``.

        Returns:
            list[dict]: Updated conformer data.
        """
        if unique_mol_data is None:
            unique_mol_data = []

        # calculate torsions for new mols
        current_mol_data = self.calculate_torsions(current_mol_data)

        # prep comparison and compute torsion matrix
        n_unique_mols = max(1, len(unique_mol_data))  # set to 1 if 0
        all_mol_data = unique_mol_data + current_mol_data
        torsion_matrix = np.stack([c["torsions"] for c in all_mol_data])
        torsion_matrix_rad = torsion_matrix * np.pi / 180
        n_confs = len(all_mol_data)
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
        if sort_by_energy:
            updated_unique_mol_data = sorted([all_mol_data[i] for i in conf_ids], key=lambda x: x["energy"])
        else:
            updated_unique_mol_data = [all_mol_data[i] for i in conf_ids]

        self.n_input_confs = len(all_mol_data)
        self.n_output_confs = len(updated_unique_mol_data)
        self.n_pruned_confs = self.n_input_confs - self.n_output_confs

        if return_ids:
            return updated_unique_mol_data, conf_ids
        else:
            return updated_unique_mol_data


class CRESTPruner(ConfGenPruner):
    """
    Prune conformers using CREST.

    Args:
        ethr (float, optional): Energy threshold. Defaults to ``0.15``.
        rthr (float, optional): RMSD threshold. Defaults to ``0.125``.
        bthr (float, optional): Bond threshold. Defaults to ``0.01``.
        ewin (int, optional): Energy window. Defaults to ``10000``.
        track_stats (bool, optional): Whether to track statistics. Defaults to ``False``.
    """

    def __init__(self,
                 ethr: float = 0.15,
                 rthr: float = 0.125,
                 bthr: float = 0.01,
                 ewin: float = 10000,
                 track_stats: bool = False):
        """
        Initialize the CREST pruner.

        Args:
            ethr (float, optional): Energy threshold. Defaults to ``0.15``.
            rthr (float, optional): RMSD threshold. Defaults to ``0.125``.
            bthr (float, optional): Bond threshold. Defaults to ``0.01``.
            ewin (int, optional): Energy window. Defaults to ``10000``.
            track_stats (bool, optional): Whether to track statistics. Defaults to ``False``.
        """
        super(CRESTPruner, self).__init__(track_stats)

        self.ethr = ethr
        self.rthr = rthr
        self.bthr = bthr
        self.ewin = ewin

    def prune_conformers(self,
                         current_mol_data: List[dict],
                         unique_mol_data: Optional[List[dict]] = None,
                         sort_by_energy: bool = True,
                         return_ids: bool = False):
        """
        Prune conformers.

        Args:
            current_mol_data (list[dict]): conformer data of the current iteration.
            unique_mol_data (list[dict], optional): Unique conformer data of previous iterations. Defaults to ``None``.
            sort_by_energy (bool, optional): Whether to sort conformers by energy. Defaults to ``True``.
            return_ids (bool, optional): Whether to return conformer IDs. Defaults to ``False``.

        Returns:
            list[dict]: Updated conformer data.
        """
        if unique_mol_data is None:
            unique_mol_data = []

        all_mol_data = unique_mol_data + current_mol_data
        updated_unique_mol_data, conf_ids = run_cre_check(all_mol_data,
                                                          ethr=self.ethr,
                                                          rthr=self.rthr,
                                                          bthr=self.bthr,
                                                          ewin=self.ewin
                                                          )

        if sort_by_energy:
            updated_unique_mol_data = sorted(updated_unique_mol_data, key=lambda x: x["energy"])

        self.n_input_confs = len(all_mol_data)
        self.n_output_confs = len(updated_unique_mol_data)
        self.n_pruned_confs = self.n_input_confs - self.n_output_confs

        if return_ids:
            return updated_unique_mol_data, conf_ids
        else:
            return updated_unique_mol_data
