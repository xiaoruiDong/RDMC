#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np

from rdmc.conformer_generation.task import MolTask


class TorsionPruner(MolTask):
    """
    Prune conformers based on torsion angle criteria.
    This method uses a max and root-mean-square (RMS) criteria to prune conformers.
    A conformer is considered different to another conformer if it satisfies both
    of the following criteria:
        - RMS difference of all torsion angles > rms_diff
        - Max difference of all torsion angles > max_diff
    New conformers are compared to all conformers that have already been deemed unique.
    """

    def task_prep(self,
                  max_diff: float = 10.,
                  rms_diff: float = 5.,
                  exclude_methyl: bool = True,
                  include_rings: bool = True,
                  **kwargs):
        """
        Set the threshold for the torsion pruner.

        Args:
            max_diff(float, optional): The maximum difference of torsion angles between two
                                        conformers to be considered different. Defaults to 10 degrees.
            rms_diff(float, optional): The root-mean-square difference of torsion angles between
                                        two conformers to be considered different. Defaults to 5 degrees.
        """
        self.max_diff = max_diff
        self.rms_diff = rms_diff
        self.exclude_methyl = exclude_methyl
        self.include_rings = include_rings
        super().task_prep(**kwargs)

    @MolTask.timer
    def run(self,
            mol: 'RDKitMol',
            torsions: Optional[list] = None,
            **kwargs):
        """
        """
        if torsions is None:
            torsions = mol.GetTorsionalModes(excludeMethyl=self.exclude_methyl,
                                             includeRings=self.include_rings)

        if not self.run_ids:
            return

        torsion_matrix = get_torsion_angles(mol=mol,
                                            cid=self.run_ids[0],
                                            torsions=torsions)
        for subtask_id in self.run_ids[1:]:
            torsion = get_torsion_angles(mol=mol,
                                         cid=subtask_id,
                                         torsions=torsions)
            # Note the maximum difference between two dihedral angles
            # is 180 degrees, considering the periodicity of dihedral.
            diff_matrix = 180 - np.abs((torsion_matrix - torsion) - 180)
            # 1. Check if there is an entry whose max difference is
            #    smaller than the threshold
            check1 = (np.max(diff_matrix, axis=1) < self.max_diff).any()
            # 2. Check if there is an entry whose RMS difference is
            #    smaller than the threshold
            check2 = (np.sqrt(np.mean(diff_matrix ** 2, axis=1)) < self.rms_diff).any()

            if check1 or check2:
                mol.keep_ids[subtask_id] = False
            else:
                torsion_matrix = np.vstack((torsion_matrix, torsion))

        return mol


def get_torsion_angles(mol: 'RDKitMol',
                       cid: int,
                       torsions: list):
    """
    Get the torsion angles of a conformer.
    """
    conf = mol.GetConformer(id=cid)
    return np.array([[conf.GetTorsionDeg(torsion) for torsion in torsions]])
