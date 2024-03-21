#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the helper functions for fixing parsed molecules.
"""

import copy

from functools import reduce
from typing import List, Optional

import numpy as np

from rdkit import Chem
from rdkit.Chem import BondType

from rdtools.atom import clear_rxn_prop, decrement_radical
from rdtools.atommap import (
    update_product_atom_map_after_reaction,
    renumber_atoms as renumber_atoms_,
)
from rdtools.dist import get_distance_matrix, get_adjacency_matrix
from rdtools.fix.remedy import remedy_manager
from rdtools.fix.mult import saturate_mol
from rdtools.mol import get_heavy_atoms, force_no_implicit


def fix_mol_by_remedy(
    mol: Chem.Mol,
    remedy: Chem.rdChemReactions.ChemicalReaction,
    max_attempts: int = 10,
    sanitize: bool = True,
) -> Chem.Mol:
    """
    Fix the molecule according to the given remedies that are defined as RDKit ChemicalReaction.

    Args:
        mol (Chem.Mol): The molecule to be fixed.
        remedy (ChemicalReaction): The functional group transformation as the remedy to fix the molecule,
                                   defined as an RDKit ChemicalReaction.
        max_attempts (int, optional): The maximum number of attempts to fix the molecule.
                                      Defaults to ``10``.
        sanitize (bool, optional): Whether to sanitize the molecule after the fix. Defaults to ``True``.

    Returns:
        Chem.Mol: The fixed molecule.
    """
    tmp_mol = mol
    fix_flag = False

    for _ in range(max_attempts):
        tmp_mol.UpdatePropertyCache(False)  # Update connectivity
        Chem.GetSymmSSSR(tmp_mol)  # Update ring information
        try:
            # Remedy are designed to be unimolecular (group transformation), so the product will be unimolecular as well
            # If no match, RunReactants will return an empty tuple and thus cause an IndexError.
            # If there is a match, then there is always a single product being generated, and we can
            # query the product by the second index `[0]`
            fix_mol = remedy.RunReactants([tmp_mol], maxProducts=1)[0][0]
            updated_atoms = update_product_atom_map_after_reaction(
                fix_mol,
                ref_mol=tmp_mol,
            )
        except IndexError:
            break

        # TODO: fix each fragment separately, then combine them
        if fix_mol.GetNumAtoms() < tmp_mol.GetNumAtoms():
            # If the input molecule contains multiple fragments (i.e., isolated graphs),
            # RDKit will only keep the fragment matching the reaction pattern.
            # Therefore we need to append the other fragments back to the molecule.
            frag_assign = []
            frags = list(
                Chem.GetMolFrags(
                    tmp_mol, asMols=True, sanitizeFrags=False, frags=frag_assign
                )
            )
            tmp_idx_in_fix_mol = int(updated_atoms[0].GetProp("react_atom_idx"))
            frag_idx = frag_assign[tmp_idx_in_fix_mol]
            frags[frag_idx] = fix_mol
            tmp_mol = reduce(Chem.CombineMols, frags)
        else:
            tmp_mol = fix_mol

        # Clear reaction properties
        [clear_rxn_prop(atom) for atom in updated_atoms]

        fix_flag = True

    else:
        raise RuntimeError(
            "The fix may be incomplete, as the maximum number of attempts has been reached."
        )

    if not fix_flag:
        return mol

    if sanitize:
        Chem.SanitizeMol(tmp_mol)

    return tmp_mol


def fix_mol_by_remedies(
    mol: Chem.Mol,
    remedies: List[Chem.rdChemReactions.ChemicalReaction],
    max_attempts: int = 10,
    sanitize: bool = True,
) -> Chem.Mol:
    """
    Fix the molecule according to the given remedy defined as an RDKit ChemicalReaction.

    Args:
        mol (Chem.Mol): The molecule to be fixed.
        remedies (list): A list of remedies to fix the molecule defined as an RDKit ChemicalReaction.
        max_attempts (int, optional): The maximum number of attempts to fix the molecule.
                                      Defaults to ``10``.
        sanitize (bool, optional): Whether to sanitize the molecule after the fix. Defaults to ``True``.

    Returns:
        Chem.Mol
    """
    for remedy in remedies:
        mol = fix_mol_by_remedy(mol, remedy, max_attempts=max_attempts, sanitize=False)

    if sanitize:
        Chem.SanitizeMol(mol)

    return mol


def fix_mol(
    mol: "Chem.Mol",
    remedies: Optional[List["ChemicalReaction"]] = None,
    max_attempts: int = 10,
    sanitize: bool = True,
    fix_spin_multiplicity: bool = True,
    mult: int = 0,
    renumber_atoms: bool = True,
) -> "Chem.Mol":
    """
    Fix the molecule by applying the given remedies and saturating bi-radical or carbene to fix spin multiplicity.

    Args:
        mol (Chem.Mol): The molecule to be fixed.
        remedies (List[ChemicalReaction], optional): The list of remedies to fix the molecule,
                                                     defined as RDKit ChemicalReaction.
                                                     Defaults to ``rdmc.fix.DEFAULT_REMEDIES``.
        max_attempts (int, optional): The maximum number of attempts to fix the molecule.
                                        Defaults to ``10``.
        sanitize (bool, optional): Whether to sanitize the molecule after the fix. Defaults to ``True``.
                                   Using ``False`` is only recommended for debugging and testing.
        fix_spin_multiplicity (bool, optional): Whether to fix the spin multiplicity of the molecule. The fix can only
                                                reduce the spin multiplicity. Defaults to ``True``.
        mult (int, optional): The desired spin multiplicity. Defaults to ``0``, which means the lowest possible
                              spin multiplicity will be inferred from the number of unpaired electrons.
                              Only used when ``fix_spin_multiplicity`` is ``True``.
        renumber_atoms (bool, optional): Whether to renumber the atoms after the fix. Defaults to ``True``.
                                         Turn this off when the atom map number is not important.

    Returns:
        Chem.Mol: The fixed molecule.
    """
    # Make sure properties are accessible (e.g. numHs)
    # And make sure no implicit to avoid H being added during the fixing
    mol.UpdatePropertyCache(False)
    force_no_implicit(mol)

    if remedies is None:
        remedies = remedy_manager.default_remedies

    fixed_mol = fix_mol_by_remedies(
        mol,
        remedies,
        max_attempts=max_attempts,
        sanitize=sanitize,
    )

    if fix_spin_multiplicity:
        saturate_mol(fixed_mol, multiplicity=mult)

    if renumber_atoms:
        fixed_mol = renumber_atoms_(fixed_mol)

    if isinstance(mol, Chem.RWMol):
        # During fixing, a few operation tends to change the molecule from RWMol to Mol
        fixed_mol = mol.__class__(fixed_mol)

    return fixed_mol


def find_oxonium_bonds(
    mol: "Chem.Mol",
    threshold: float = 1.65,
) -> List[tuple]:
    """
    Find the potential oxonium atom.

    Args:
        mol (Chem.Mol): The molecule to be fixed.
        threshold (float, optional): The threshold to determine if two atoms are connected.

    Returns:
        List[tuple]: a list of (oxygen atom index, the other atom index).
    """
    heavy_idxs = [atom.GetIdx() for atom in get_heavy_atoms(mol)]
    oxygen_idxs = [i for i in heavy_idxs if mol.GetAtomWithIdx(i).GetAtomicNum() == 8]

    if len(oxygen_idxs) == 0:
        return []

    dist_mat = get_distance_matrix(mol, balaban=True)
    adj_mat = get_adjacency_matrix(mol)

    # A detailed check may be done by element type
    # for now we will use the threshold based on the longest C-O bond 1.65 A
    miss_bonds_mat = (dist_mat[np.ix_(oxygen_idxs, heavy_idxs)] <= threshold) & (
        adj_mat[np.ix_(oxygen_idxs, heavy_idxs)] == 0
    )

    miss_bonds = list(set(zip(*np.nonzero(miss_bonds_mat))))

    return [
        (oxygen_idxs[miss_bond[0]], heavy_idxs[miss_bond[1]])
        for miss_bond in miss_bonds
    ]


def fix_oxonium_bonds(
    mol: "Chem.Mol",
    threshold: float = 1.65,
    sanitize: bool = True,
) -> "Chem.Mol":
    """
    Fix the oxonium atom. Openbabel and Jensen perception algorithm do not perceive the oxonium atom correctly.
    This is a fix to detect if the molecule contains oxonium atom and fix it.

    Args:
        mol (Chem.Mol): The molecule to be fixed.
        threshold (float, optional): The threshold to determine if two atoms are connected.
        sanitize (bool, optional): Whether to sanitize the molecule after the fix. Defaults to ``True``.
                                   Using ``False`` is only recommended for debugging and testing.

    Returns:
        Chem.Mol: The fixed molecule.
    """
    oxonium_bonds = find_oxonium_bonds(mol, threshold=threshold)

    if len(oxonium_bonds) == 0:
        return mol

    fixed_mol = copy.copy(mol)
    for aidx1, aidx2 in oxonium_bonds:
        if aidx1 != aidx2 and fixed_mol.GetBondBetweenAtoms(aidx1, aidx2) is None:
            fixed_mol.AddBond(aidx1, aidx2, order=BondType.SINGLE)

            # Usually the connected atom is a radical site
            # So, update the number of radical electrons afterward
            rad_atom = fixed_mol.GetAtomWithIdx(aidx2)
            if rad_atom.GetNumRadicalElectrons() > 0:
                decrement_radical(rad_atom)

    fixed_mol = fix_mol(
        fixed_mol, remedies=remedy_manager.get_remedies("oxonium"), sanitize=sanitize
    )

    if isinstance(mol, Chem.RWMol):
        # During fixing, a few operation tends to change the molecule from RWMol to Mol
        fixed_mol = mol.__class__(fixed_mol)

    return fixed_mol
