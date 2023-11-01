#!/usr/bin/env python3

"""
Unit tests for the mol module.
"""

import logging
import unittest

import pytest

from rdmc import Reaction, RDKitMol

logging.basicConfig(level=logging.DEBUG)


class TestReaction:
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
        assert len(rxn.reactant) == 1
        assert len(rxn.product) == 1
        assert rxn.reactant[0].ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False)
        assert rxn.reactant_complex.ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False)
        assert rxn.product[0].ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False)
        assert rxn.product_complex.ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False)

        # Test the initialization of the Reaction class with RDKitMol
        reactant = self.rmol1
        product = self.pmol1
        rxn = Reaction(reactant=reactant, product=product)
        assert len(rxn.reactant) == 1
        assert len(rxn.product) == 1
        assert rxn.reactant[0].ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False)
        assert rxn.reactant_complex.ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False)
        assert rxn.product[0].ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False)
        assert rxn.product_complex.ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False)

    def test_from_reaction_smiles_1_to_1(self):
        """
        Test the initialization of the Reaction class from reaction smiles.
        """
        rxn_smiles = self.rsmi1 + ">>" + self.psmi1
        rxn = Reaction.from_reaction_smiles(rxn_smiles)
        assert len(rxn.reactant) == 1
        assert len(rxn.product) == 1
        assert rxn.reactant[0].ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False)
        assert rxn.reactant_complex.ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False)
        assert rxn.product[0].ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False)
        assert rxn.product_complex.ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False)

    def test_from_reactant_product_smiles_1_to_1(self):
        """
        Test the initialization of the Reaction class from reactant and product smiles.
        """
        rxn = Reaction.from_reactant_and_product_smiles(rsmi=self.rsmi1, psmi=self.psmi1)
        assert len(rxn.reactant) == 1
        assert len(rxn.product) == 1
        assert rxn.reactant[0].ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False)
        assert rxn.reactant_complex.ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.rmol1.ToSmiles(removeAtomMap=False, removeHs=False)
        assert rxn.product[0].ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False)
        assert rxn.product_complex.ToSmiles(removeAtomMap=False, removeHs=False) \
            == self.pmol1.ToSmiles(removeAtomMap=False, removeHs=False)

    def test_is_num_atoms_balanced(self):
        """
        Test the is_num_atoms_balanced property.
        """
        assert self.rxn1.is_num_atoms_balanced
        assert not self.rxn1_bad.is_num_atoms_balanced

    def test_is_element_balanced(self):
        """
        Test the is_element_balanced property.
        """
        assert self.rxn1.is_element_balanced
        assert not self.rxn1_bad.is_element_balanced

    def test_is_charge_balanced(self):
        """
        Test the is_charge_balanced property.
        """
        assert self.rxn1.is_charge_balanced

    def test_is_multiplicity_equal(self):
        """
        Test the is_multiplicity_balanced property.
        """
        assert self.rxn1.is_mult_equal
        assert not self.rxn1_bad.is_mult_equal

    def test_num_reactant(self):
        """
        Test the num_reactant property.
        """
        assert self.rxn1.num_reactants == 1

    def test_num_product(self):
        """
        Test the num_product property.
        """
        assert self.rxn1.num_products == 1

    def test_num_atoms(self):
        """
        Test the num_atoms property.
        """
        assert self.rxn1.num_atoms == 8
        with pytest.raises(AssertionError):
            self.rxn1_bad.num_atoms

    def test_num_broken_bonds(self):
        """
        Test the num_broken_bonds property.
        """
        # Reaction 1
        assert self.rxn1.num_broken_bonds == 1

    @pytest.mark.parametrize(
        "rxn_smi1, rxn_smi2, expected_forward, expected_both",
        [
            # Case 1: the difference is right after the >> sign
            (
                "[O:1]([C:2]([C:3]([C:4]([C:5]([C:6]([C:7]([H:18])([H:19])[H:20])([H:16])[H:17])([H:14])[H:15])([H:12])"
                "[H:13])[H:11])([H:9])[H:10])[H:8]>>[H:18].[O:1]([C:2]([C:3]1([H:11])[C:4]([H:12])([H:13])[C:5]([H:14])"
                "([H:15])[C:6]([H:16])([H:17])[C:7]1([H:19])[H:20])([H:9])[H:10])[H:8]",
                "[O:1]([C:2]([C:3]([C:4]([C:5]([C:6]([C:7]([H:18])([H:19])[H:20])([H:16])[H:17])([H:14])[H:15])([H:12])"
                "[H:13])[H:11])([H:9])[H:10])[H:8]>>[H:20].[O:1]([C:2]([C:3]1([H:11])[C:4]([H:12])([H:13])[C:5]([H:14])"
                "([H:15])[C:6]([H:16])([H:17])[C:7]1([H:18])[H:19])([H:9])[H:10])[H:8]",
                True,
                True,
            ),
            # Case 2: the first is H abstraction, the second is H transfer; pay attention to the indexes of O-C=O group
            (
                "[H:3][O:8][C:7]([C:5]([C:4]([H:10])([H:11])[H:12])=[C:6]([H:13])[H:14])=[O:9].[O:1]([O:2])[H:15]>>[C:4]([C:5]"
                "(=[C:6]([H:13])[H:14])[C:7](=[O:8])[O:9][H:15])([H:10])([H:11])[H:12].[O:1]([O:2])[H:3]",
                "[C:4]([C:5](=[C:6]([H:13])[H:14])[C:7](=[O:8])[O:9][H:15])([H:10])([H:11])[H:12].[O:1]([O:2])[H:3]>>[H:3]"
                "[O:9][C:7]([C:5]([C:4]([H:10])([H:11])[H:12])=[C:6]([H:13])[H:14])=[O:8].[O:1][O:2][H:15]",
                False,
                False,
            ),
            # Case 3: The reverse of the second reaction is equivalent to the first reaction
            (
                "[C:1]([C:2]([H:5])([H:6])[O:7][O:8][H:9])([H:3])([H:4])[O:10]>>[C:1]([C:2]([H:5])([H:6])[O:7][O:8][H:9])([H:3])=[O:10].[H:4]",
                "[C:1]([C:2]([H:5])([H:6])[O:7][O:8][H:9])([H:4])=[O:10].[H:3]>>[C:1]([C:2]([H:5])([H:6])[O:7][O:8][H:9])([H:3])([H:4])[O:10]",
                False,
                True,
            ),
        ],
    )
    def test_is_equivalent(self, rxn_smi1, rxn_smi2, expected_forward, expected_both):
        """
        Test the is_equivalent method.
        """
        rxn1 = Reaction.from_reaction_smiles(rxn_smi1)
        rxn2 = Reaction.from_reaction_smiles(rxn_smi2)

        assert rxn1.is_equivalent(rxn2, both_directions=False) == expected_forward
        assert rxn1.is_equivalent(rxn2, both_directions=True) == expected_both


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=3))
