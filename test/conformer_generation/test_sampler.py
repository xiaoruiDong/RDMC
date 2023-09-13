#!/usr/bin/env python3

"""
Unit tests for the sampler module.
"""

import os
import logging
import unittest
from typing import List
from tempfile import TemporaryDirectory

import pytest

from rdkit import Chem
from rdmc import RDKitMol
from rdmc.conformer_generation.sampler import TorsionalSampler

logging.basicConfig(level=logging.DEBUG)

################################################################################


class TestSampler:
    """
    A class used to test basic operations for the sampler module.
    """

    xyz_string = """20

    C      2.458330   -0.178848   -0.464657
    H      2.748291   -1.029390   -1.117779
    H      2.392455   -0.654430    0.656573
    H      3.251550    0.597307   -0.403447
    C      1.115966    0.368477   -0.787023
    C      0.036708   -0.657818   -0.993102
    H      0.317143   -1.602122   -0.453451
    H     -0.041315   -0.889646   -2.087561
    C     -1.313562   -0.164147   -0.488468
    H     -1.659795    0.715987   -1.095646
    C     -2.345567   -1.268983   -0.485875
    H     -2.026412   -2.101899    0.185643
    H     -3.321363   -0.872831   -0.114973
    H     -2.478035   -1.664461   -1.520591
    O      0.892181    1.584779   -0.834219
    O     -1.120323    0.266868    0.895191
    O     -1.400242    1.510791    1.061614
    H     -0.602542    2.042871    0.823259
    O      1.729903   -1.166551    1.715311
    H      0.896619   -0.700572    1.739545
    """
    mol = RDKitMol.FromXYZ(xyz_string)
    rxn_smiles = "[C:1]([H:2])([H:3])([H:4])[C:5]([C:6]([H:7])([H:8])[C:9]([H:10])([C:11]([H:12])([H:13])[H:14])[O:16][O:17][H:18])=[O:15].[O:19][H:20]>>[C:1]([H:2])([H:4])[C:5]([C:6]([H:7])([H:8])[C:9]([H:10])([C:11]([H:12])([H:13])[H:14])[O:16][O:17][H:18])=[O:15].[H:3][O:19][H:20]"
    sampler = TorsionalSampler(
        n_point_each_torsion=3,
        n_dimension=-1,
    )

    @pytest.mark.parametrize("torsions,no_sample_dangling_bonds,n_conformers",
                             [(None, True, 61,),
                              (None, False, 190,),
                              ([[17, 16, 15, 8],
                                [16, 15, 8, 5],
                                [15, 8, 5, 4],
                                [8, 5, 4, 0],
                                [5, 4, 0, 2],
                                [4, 0, 2, 18],
                                [0, 2, 18, 19],],
                               False,
                               1930,),
                              ([[17, 16, 15, 8],
                                [16, 15, 8, 5],
                                [15, 8, 5, 4],
                                [8, 5, 4, 0],
                                [5, 4, 0, 2],
                                [4, 0, 2, 18],
                                [0, 2, 18, 19],
                               ],
                               True,
                               1930,),
                              ])
    def test_no_greedy_TorsionalSampler(
        self,
        torsions: List,
        no_sample_dangling_bonds: bool,
        n_conformers: int,
    ):
        """
        Test if the TorsionalSampler object with `no_greedy` as `True` works normally.
        """
        with TemporaryDirectory() as save_dir:
            self.sampler(
                mol=self.mol,
                id=0,
                rxn_smiles=self.rxn_smiles,
                torsions=torsions,
                no_sample_dangling_bonds=no_sample_dangling_bonds,
                no_greedy=True,
                save_dir=save_dir,
            )

            # Check results
            sdf_path = os.path.join(save_dir, "torsion_sampling_0/sampling_confs.sdf")
            reader = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
            assert len(reader) == n_conformers


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=3))
