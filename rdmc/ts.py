#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
This module provides class and methods for dealing with Transition states.
"""

from typing import List, Union

import numpy as np


def get_formed_bonds(r_mol: Union['RDKitMol', 'Mol'],
                     p_mol:  Union['RDKitMol', 'Mol'],
                     ) -> List:
    """
    Get all bonds formed in the reaction. Both reactant and product complexes
    need to be atom-mapped.

    Args:
        r_mol ('RDKitMol' or 'Mol'): the reactant complex.
        p_mol ('RDKitMol' or 'Mol'): the product complex.

    Returns
        list: A list of length-2 set that contains the atom indexes of the bonded atoms.
    """
    r_bonds = [{bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} for bond in r_mol.GetBonds()]
    p_bonds = [{bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} for bond in p_mol.GetBonds()]
    return [bond for bond in p_bonds if bond not in r_bonds]


def get_broken_bonds(r_mol: Union['RDKitMol', 'Mol'],
                     p_mol:  Union['RDKitMol', 'Mol'],
                     ) -> List:
    """
    Get all bonds broken in the reaction. Both reactant and product complexes
    need to be atom-mapped.

    Args:
        r_mol ('RDKitMol' or 'Mol'): the reactant complex.
        p_mol ('RDKitMol' or 'Mol'): the product complex.

    Returns:
        list: A list of length-2 set that contains the atom indexes of the bonded atoms.
    """
    r_bonds = [{bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} for bond in r_mol.GetBonds()]
    p_bonds = [{bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} for bond in p_mol.GetBonds()]
    return [bond for bond in r_bonds if bond not in p_bonds]


def get_formed_and_broken_bonds(r_mol: Union['RDKitMol', 'Mol'],
                                p_mol:  Union['RDKitMol', 'Mol'],
                                ) -> List:
    """
    Get all bonds broken in the reaction. Both reactant and product complexes
    need to be atom-mapped. This function doesn't count bonds whose bond order
    is lowered but not equal to zero.

    Args:
        r_mol ('RDKitMol' or 'Mol'): the reactant complex.
        p_mol ('RDKitMol' or 'Mol'): the product complex.

    Returns:
        list: - formed bonds: A list of length-2 set that contains the atom indexes of the bonded atoms.
              - broken bonds: A list of length-2 set that contains the atom indexes of the bonded atoms.
    """
    r_bonds = [{bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} for bond in r_mol.GetBonds()]
    p_bonds = [{bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} for bond in p_mol.GetBonds()]
    formed_bonds = [bond for bond in p_bonds if bond not in r_bonds]
    broken_bonds = [bond for bond in r_bonds if bond not in p_bonds]
    return formed_bonds, broken_bonds


def get_all_changing_bonds(r_mol: Union['RDKitMol', 'Mol'],
                           p_mol: Union['RDKitMol', 'Mol'],
                           ) -> List:
    """
    Get all bonds changed in the reaction. Both reactant and product complexes
    need to be atom-mapped.

    Args:
        r_mol ('RDKitMol' or 'Mol'): the reactant complex.
        p_mol ('RDKitMol' or 'Mol'): the product complex.

    Returns:
        list: - formed bonds: A list of length-2 set that contains the atom indexes of the bonded atoms.
              - broken bonds: A list of length-2 set that contains the atom indexes of the bonded atoms.
              - bonds with BO changed: A list of length-2 set that contains the atom indexes of the bonded atoms.
    """
    r_bonds = [{bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} for bond in r_mol.GetBonds()]
    p_bonds = [{bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} for bond in p_mol.GetBonds()]
    formed_bonds = [bond for bond in p_bonds if bond not in r_bonds]
    broken_bonds = [bond for bond in r_bonds if bond not in p_bonds]
    other_bonds = [bond for bond in r_bonds if bond not in broken_bonds]
    changed_bonds = [bond for bond in other_bonds
                     if r_mol.GetBondBetweenAtoms(*bond).GetBondTypeAsDouble() != \
                        p_mol.GetBondBetweenAtoms(*bond).GetBondTypeAsDouble()]
    return formed_bonds, broken_bonds, changed_bonds


