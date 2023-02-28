#!/usr/bin/env python3

"""
Unit tests for the utils module.
"""

import logging
import unittest

import numpy as np
from rdmc.utils import (parse_xyz_by_openbabel,
                        reverse_map,
                        openbabel_mol_to_rdkit_mol)

logging.basicConfig(level=logging.DEBUG)

################################################################################

class TestUtils(unittest.TestCase):
    """
    The general class to test functions in the utils module
    """

    def test_reverse_match(self):
        """
        Test the functionality to reverse a mapping.
        """
        map = [ 1,  2,  3,  4,  5, 17, 18, 19, 20, 21, 22, 23, 24, 25,  6,  7,  8,
                9, 10, 11, 12, 13, 14, 15, 16, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                35, 36, 37, 38, 39]
        r_map = [ 0,  1,  2,  3,  4, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,  5,
                  6,  7,  8,  9, 10, 11, 12, 13, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                  34, 35, 36, 37, 38]

        self.assertSequenceEqual(r_map, reverse_map(map))
        np.testing.assert_equal(np.array(r_map), reverse_map(map, as_list=False))

    def test_openbabel_mol_to_rdkit_mol_single_atom_xyz(self):
        """
        Test if a single-atom openbabel mol with all-zero xyz coordinates can be successfully converted to rdkit mol with a conformer embedded.
        """
        xyz = '1\n[Geometry 1]\nH      0.0000000000    0.0000000000    0.0000000000\n'
        obmol = parse_xyz_by_openbabel(xyz)
        rdmol = openbabel_mol_to_rdkit_mol(obmol)

        self.assertEqual(rdmol.GetNumConformers(), 1)
        self.assertEqual(rdmol.GetNumAtoms(), 1)
        self.assertTrue(np.array_equal(
                                rdmol.GetConformer().GetPositions(),
                                np.array([[0., 0., 0.,]])
                                )
                        )


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=3))
