#!/usr/bin/env python3

"""
Unit tests for the mol module.
"""

import logging
import unittest

import numpy as np

from rdmc import RDKitMol
from rdmc import ts

logging.basicConfig(level=logging.DEBUG)

################################################################################


class TestBondAnalysis(unittest.TestCase):
    """
    A class used to test bond analysis on the RDKitMol instances.
    """

    def test_get_bonds_as_sets(self) -> None:
        """
        Test get bonds for multiple mols.
        """
        mol1 = RDKitMol.FromSmiles("[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]")
        mol2 = RDKitMol.FromSmiles("[C:1]([N:3]=[C:2]=[N:6][N:5]=[O:4])([H:7])([H:8])[H:9]")
        mol3 = RDKitMol.FromSmiles("[C:1]2([C:2]1[N:3]2[O:4][N:5][N:6]1)([H:7])([H:8])[H:9]", sanitize=False)

        self.assertEqual(ts._get_bonds_as_sets(mol1),
                         ({(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (0, 7), (0, 8), (1, 5)},))
        self.assertEqual(ts._get_bonds_as_sets(mol1, mol2),
                         ({(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (0, 7), (0, 8), (1, 5)},
                          {(0, 2), (1, 2), (3, 4), (4, 5), (0, 6), (0, 7), (0, 8), (1, 5)}))
        self.assertEqual(ts._get_bonds_as_sets(mol1, mol2, mol3),
                         ({(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (0, 7), (0, 8), (1, 5)},
                          {(0, 2), (1, 2), (3, 4), (4, 5), (0, 6), (0, 7), (0, 8), (1, 5)},
                          {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (0, 7), (0, 8), (1, 5), (0, 2)}))

    def test_get_formed_bonds(self) -> None:
        """
        Test get formed bonds between the reactant complex and the product complex.
        """
        mol1 = RDKitMol.FromSmiles("[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]")
        mol2 = RDKitMol.FromSmiles("[C:1]([N:3]=[C:2]=[N:6][N:5]=[O:4])([H:7])([H:8])[H:9]")

        # The backend molecule should be the same as the input RWMol object
        self.assertEqual(ts.get_formed_bonds(mol1, mol2), [(0, 2)])
        self.assertEqual(ts.get_formed_bonds(mol2, mol1), [(0, 1), (2, 3)])

    def test_get_broken_bonds(self) -> None:
        """
        Test get broken bonds between the reactant complex and the product complex.
        """
        mol1 = RDKitMol.FromSmiles("[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]")
        mol2 = RDKitMol.FromSmiles("[C:1]([N:3]=[C:2]=[N:6][N:5]=[O:4])([H:7])([H:8])[H:9]")

        # The backend molecule should be the same as the input RWMol object
        self.assertEqual(ts.get_broken_bonds(mol1, mol2), [(0, 1), (2, 3)])
        self.assertEqual(ts.get_broken_bonds(mol2, mol1), [(0, 2)])

    def test_get_broken_bonds(self) -> None:
        """
        Test get broken bonds between the reactant complex and the product complex.
        """
        mol1 = RDKitMol.FromSmiles("[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]")
        mol2 = RDKitMol.FromSmiles("[C:1]([N:3]=[C:2]=[N:6][N:5]=[O:4])([H:7])([H:8])[H:9]")

        # The backend molecule should be the same as the input RWMol object
        self.assertEqual(ts.get_broken_bonds(mol1, mol2), [(0, 1), (2, 3)])
        self.assertEqual(ts.get_broken_bonds(mol2, mol1), [(0, 2)])

    def test_get_formed_and_broken_bonds(self) -> None:
        """
        Test get formed and broken bonds between the reactant complex and the product complex.
        """
        mol1 = RDKitMol.FromSmiles("[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]")
        mol2 = RDKitMol.FromSmiles("[C:1]([N:3]=[C:2]=[N:6][N:5]=[O:4])([H:7])([H:8])[H:9]")

        # The backend molecule should be the same as the input RWMol object
        self.assertEqual(ts.get_formed_and_broken_bonds(mol1, mol2),
                         ([(0, 2)],
                          [(0, 1), (2, 3)]))
        self.assertEqual(ts.get_formed_and_broken_bonds(mol2, mol1),
                         ([(0, 1), (2, 3)],
                          [(0, 2)]))

    def test_get_all_changing_bonds(self) -> None:
        """
        Test get formed and broken bonds between the reactant complex and the product complex.
        """
        mol1 = RDKitMol.FromSmiles("[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]")
        mol2 = RDKitMol.FromSmiles("[C:1]([N:3]=[C:2]=[N:6][N:5]=[O:4])([H:7])([H:8])[H:9]")

        # The backend molecule should be the same as the input RWMol object
        self.assertEqual(ts.get_all_changing_bonds(mol1, mol2),
                         ([(0, 2)],
                          [(0, 1), (2, 3)],
                          [(1, 2), (4, 5), (1, 5), (3, 4)]))
        self.assertEqual(ts.get_all_changing_bonds(mol2, mol1),
                         ([(0, 1), (2, 3)],
                          [(0, 2)],
                          [(1, 2), (4, 5), (1, 5), (3, 4)]))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=3))
