#!/usr/bin/env python3

"""
Unit tests for the mol module.
"""

import logging
import unittest

from rdmc import Reaction, RDKitMol

logging.basicConfig(level=logging.DEBUG)


class TestReaction(unittest.TestCase):
    """
    A class used to test basic operations for the RDKitMol Class.
    """

    # Reaction 1, isomerization of ethanol radicals, with atom mapping
    rsmi1 = "[C:1]([C:2]([O:3][H:8])([H:6])[H:7])([H:4])[H:5]"
    psmi1 = "[C:1]([C:2]([O:3])([H:6])[H:7])([H:4])([H:5])[H:8]"
    psmi1_bad = "[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]"

    rmol1 = RDKitMol.FromSmiles(rsmi1)
    pmol1 = RDKitMol.FromSmiles(psmi1)
    pmol1_bad = RDKitMol.FromSmiles(psmi1_bad)

    rxn1 = Reaction(reactant=rmol1, product=pmol1)
    rxn1_bad = Reaction(reactant=rmol1, product=pmol1_bad)

    def test_initialize_reaction_1_to_1(self):
        """
        Test the initialization of the Reaction class.
        """
        # Test the initialization of the Reaction class with list
        reactant = [self.rmol1]
        product = [self.pmol1]
        rxn = Reaction(reactant=reactant, product=product)
        self.assertEqual(len(rxn.reactant), 1)
        self.assertEqual(len(rxn.product), 1)
        self.assertEqual(rxn.reactant[0].ToSmiles(removeAtomMap=False, removeHs=False),
                         self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False),)
        self.assertEqual(rxn.reactant_complex.ToSmiles(removeAtomMap=False, removeHs=False),
                         self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False),)
        self.assertEqual(rxn.product[0].ToSmiles(removeAtomMap=False, removeHs=False),
                         self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False),)
        self.assertEqual(rxn.product_complex.ToSmiles(removeAtomMap=False, removeHs=False),
                         self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False),)

        # Test the initialization of the Reaction class with RDKitMol
        reactant = self.rmol1
        product = self.pmol1
        rxn = Reaction(reactant=reactant, product=product)
        self.assertEqual(len(rxn.reactant), 1)
        self.assertEqual(len(rxn.product), 1)
        self.assertEqual(rxn.reactant[0].ToSmiles(removeAtomMap=False, removeHs=False),
                         self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False),)
        self.assertEqual(rxn.reactant_complex.ToSmiles(removeAtomMap=False, removeHs=False),
                         self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False),)
        self.assertEqual(rxn.product[0].ToSmiles(removeAtomMap=False, removeHs=False),
                         self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False),)
        self.assertEqual(rxn.product_complex.ToSmiles(removeAtomMap=False, removeHs=False),
                         self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False),)

    def test_from_reaction_smiles_1_to_1(self):
        """
        Test the initialization of the Reaction class from reaction smiles.
        """
        rxn_smiles = self.rsmi1 + ">>" + self.psmi1
        rxn = Reaction.from_reaction_smiles(rxn_smiles)
        self.assertEqual(len(rxn.reactant), 1)
        self.assertEqual(len(rxn.product), 1)
        self.assertEqual(rxn.reactant[0].ToSmiles(removeAtomMap=False, removeHs=False),
                         self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False),)
        self.assertEqual(rxn.reactant_complex.ToSmiles(removeAtomMap=False, removeHs=False),
                         self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False),)
        self.assertEqual(rxn.product[0].ToSmiles(removeAtomMap=False, removeHs=False),
                         self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False),)
        self.assertEqual(rxn.product_complex.ToSmiles(removeAtomMap=False, removeHs=False),
                         self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False),)

    def test_from_reactant_product_smiles_1_to_1(self):
        """
        Test the initialization of the Reaction class from reactant and product smiles.
        """
        rxn = Reaction.from_reactant_and_product_smiles(rsmi=self.rsmi1, psmi=self.psmi1)
        self.assertEqual(len(rxn.reactant), 1)
        self.assertEqual(len(rxn.product), 1)
        self.assertEqual(rxn.reactant[0].ToSmiles(removeAtomMap=False, removeHs=False),
                         self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False),)
        self.assertEqual(rxn.reactant_complex.ToSmiles(removeAtomMap=False, removeHs=False),
                         self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False),)
        self.assertEqual(rxn.product[0].ToSmiles(removeAtomMap=False, removeHs=False),
                         self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False),)
        self.assertEqual(rxn.product_complex.ToSmiles(removeAtomMap=False, removeHs=False),
                         self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False),)

    def test_is_num_atoms_balanced(self):
        """
        Test the is_num_atoms_balanced property.
        """
        self.assertTrue(self.rxn1.is_num_atoms_balanced)
        self.assertFalse(self.rxn1_bad.is_num_atoms_balanced)

    def test_is_element_balanced(self):
        """
        Test the is_element_balanced property.
        """
        self.assertTrue(self.rxn1.is_element_balanced)
        self.assertFalse(self.rxn1_bad.is_element_balanced)

    def test_is_charge_balanced(self):
        """
        Test the is_charge_balanced property.
        """
        self.assertTrue(self.rxn1.is_charge_balanced)

    def test_is_multiplicity_equal(self):
        """
        Test the is_multiplicity_balanced property.
        """
        self.assertTrue(self.rxn1.is_mult_equal)
        self.assertFalse(self.rxn1_bad.is_mult_equal)

    def test_num_reactant(self):
        """
        Test the num_reactant property.
        """
        self.assertEqual(self.rxn1.num_reactants, 1)

    def test_num_product(self):
        """
        Test the num_product property.
        """
        self.assertEqual(self.rxn1.num_products, 1)

    def test_num_atoms(self):
        """
        Test the num_atoms property.
        """
        self.assertEqual(self.rxn1.num_atoms, 8)
        with self.assertRaises(AssertionError):
            self.rxn1_bad.num_atoms

    def test_num_broken_bonds(self):
        """
        Test the num_broken_bonds property.
        """
        # Reaction 1
        self.assertSequenceEqual(self.rxn1.num_broken_bonds, [(2, 7)])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=3))
