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


def is_DA_rxn_endo(r_mol: 'RDKitMol',
                   p_mol: 'RDKitMol',
                   embed: bool = False):
    """
    Determine the Diels Alder reaction stereo type (endo or exo),
    based on the provided reactants and products.

    Args:
        r_mol (RDKitMol): the reactant complex.
        p_mol (RDKitMol): the product complex.
        embed (bool): bool. If the DA product has no conformer embedded.
                            Whether to embed a conformer. Defaults to ``False``.
    """
    frags = r_mol.GetMolFrags()

    if len(frags) == 1:
        # This reaction is defined in the reverse direction:
        # DA_product <=> diene + dienophile
        r_mol, p_mol = p_mol, r_mol
        frags = r_mol.GetMolFrags()

    assert len(frags) == 2

    if p_mol.GetNumConformers() == 0 and embed:
        p_mol.EmbedConformer()
    elif p_mol.GetNumConformers() == 0:
        raise ValueError('The provided DA product has no geometry available'
                         'Cannot determine the stereotype of the DA reaction')

    # Analyze the reaction center
    formed, _, changing = get_all_changing_bonds(r_mol, p_mol)
    # `fbond_atoms` are atoms in the formed bonds
    fbond_atoms = set([atom for bond in formed for atom in bond])
    for bond in changing:
        if len(bond & fbond_atoms) == 0:
            # Find the single bond in the diene
            dien_sb = list(bond)
        elif len(bond & fbond_atoms) == 2:
            # Find the double bond of the dienophile
            dinp_db = list(bond)
    # Make `fbond_atoms` for convenience in slicing
    fbond_atoms = list(fbond_atoms)

    # Find the atom indexes in diene and dienophile
    _, dienophile = frags if dien_sb[0] in frags[0] else frags[::-1]

    # Get the 3D geometry of the DA product
    # Create a reference plane from atoms in formed bonds
    # The reference point is chosen to be the center of the plane
    xyz = p_mol.GetPositions()
    ref_plane = xyz[fbond_atoms]
    ref_pt = ref_plane.mean(axis=0, keepdims=True)

    # Use the total least square algorithm to find
    # the normal vector of the reference plane
    A = ref_plane - ref_pt
    norm_vec = np.linalg.svd(A.T @ A)[0][:, -1].reshape(1, -1)

    # Use the vector between middle point of the diene single point
    # and the reference point as one direction vector
    dien_vec = xyz[dien_sb, :].mean(axis=0, keepdims=True) - ref_pt

    # Use the vector between mass center of the dienophile
    # and the reference point as one direction vector
    # excluding atom in dienophile's double bond
    atom_scope = [atom for atom in dienophile if atom not in dinp_db]
    mass_vec = [r_mol.GetAtomWithIdx(i).GetMass() for i in atom_scope]
    wt_xyz = (xyz[atom_scope, :] * np.reshape(mass_vec, (-1, 1)))
    dinp_vec = wt_xyz.mean(axis=0, keepdims=True) - ref_pt

    # Endo is determined by if dien_vec has the same direction as the dinp_vec
    # using the normal vector of the reference plane as a reference direction
    endo = ((norm_vec @ dien_vec.T) * (norm_vec @ dinp_vec.T)).item() > 0
    return endo
