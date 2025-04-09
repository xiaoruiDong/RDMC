#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains the function generating resonance structures."""

import logging
from typing import Any

from rdkit import Chem

from rdtools.resonance.base import ResonanceAlgoRegistry
from rdtools.resonance.utils import (
    is_equivalent_structure,
    is_partially_charged,
    unset_aromatic_flags,
)

logger = logging.getLogger(__name__)


def generate_radical_resonance_structures(
    mol: Chem.RWMol,
    keep_isomorphic: bool = False,
    copy: bool = True,
    kekulize: bool = True,
    **kwargs: Any,
) -> list[Chem.Mol]:
    """Generate resonance structures for a radical molecule.

    RDKit by design doesn't
    work for radical resonance. The approach is a temporary workaround by replacing
    radical electrons by positive charges and generating resonance structures by RDKit
    ResonanceMolSupplier. Currently, this function only works for neutral radicals.

    Known issues:

    - Phenyl radical only generate one resonance structure when ``kekulize=True``, expecting 2.

    Args:
        mol (Chem.RWMol): A radical molecule in RDKit RWMol.
        keep_isomorphic (bool, optional): If keep isomorphic resonance structures. Defaults to ``False``.
        copy (bool, optional): If copy the input molecule. Defaults to ``True``.
        kekulize (bool, optional): If kekulize the molecule in generating resonance structures. Defaults to ``True``.
        **kwargs (Any): Additional arguments for the resonance algorithms.

    Returns:
        list[Chem.Mol]: a list of molecules with resonance structures.
    """
    mol_copy = Chem.RWMol(mol, True) if copy else mol

    # Modify the original molecule to make it a positively charged species
    recipe = {}  # Used to record changes. Temporarily not used now.
    for atom in mol_copy.GetAtoms():
        radical_electrons = atom.GetNumRadicalElectrons()
        if radical_electrons > 0:  # Find a radical site
            recipe[atom.GetIdx()] = radical_electrons
            atom.SetFormalCharge(+radical_electrons)
            atom.SetNumRadicalElectrons(0)
    # Make sure conjugation is assigned
    # Only assign the conjugation after changing radical sites to positively charged sites
    Chem.SetConjugation(mol_copy)

    # Avoid generating certain resonance bonds
    for atom in mol_copy.GetAtoms():
        if (atom.GetAtomicNum() == 8 and len(atom.GetNeighbors()) > 1) or (
            atom.GetAtomicNum() == 7 and len(atom.GetNeighbors()) > 2
        ):
            # Avoid X-O-Y be part of the resonance and forms X-O.=Y
            # Avoid X-N(-Y)-Z be part of the resonance and forms X=N.(-Y)(-Z)
            [bond.SetIsConjugated(False) for bond in atom.GetBonds()]
    mol_copy.UpdatePropertyCache()  # Make sure the assignment is broadcast to atoms / bonds

    # Generate Resonance Structures
    flags = Chem.ALLOW_INCOMPLETE_OCTETS | Chem.UNCONSTRAINED_CATIONS
    if kekulize:
        flags |= Chem.KEKULE_ALL
    suppl = Chem.ResonanceMolSupplier(mol_copy, flags=flags)
    res_mols = [Chem.RWMol(mol) for mol in suppl if mol is not None]

    # Post-processing resonance structures
    cleaned_mols = []
    for res_mol in res_mols:
        discard_flag = False
        for atom in res_mol.GetAtoms():
            # Convert positively charged species back to radical species
            charge = atom.GetFormalCharge()
            if charge > 0:  # Find a radical site
                recipe[atom.GetIdx()] = radical_electrons
                atom.SetFormalCharge(0)
                atom.SetNumRadicalElectrons(charge)
            elif charge < 0:
                # Known case: O=CC=C -> [O-]C=C[C+]
                # Discard such resonance structures
                discard_flag = True
        if discard_flag:
            continue

        # If a structure cannot be sanitized, removed it
        try:
            # Sanitization strategy is inspired by
            # https://github.com/rdkit/rdkit/discussions/6358
            flags = Chem.SANITIZE_PROPERTIES | Chem.SANITIZE_SETCONJUGATION
            if not kekulize:
                flags |= (
                    Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                )
            Chem.SanitizeMol(res_mol, sanitizeOps=flags)
        except BaseException as e:
            logger.debug(f"Sanitization failed for a resonance structure. Got {e}")
            continue
        if kekulize:
            unset_aromatic_flags(res_mol)
        cleaned_mols.append(res_mol)

    # To remove duplicate resonance structures
    known_structs: list[Chem.Mol] = []
    for new_struct in cleaned_mols:
        for known_struct in known_structs:
            if is_equivalent_structure(
                ref_mol=known_struct,
                qry_mol=new_struct,
                isomorphic_equivalent=not keep_isomorphic,
            ):
                break
        else:
            new_struct.__setattr__("__sssAtoms", [])
            known_structs.append(new_struct)

    if not known_structs:
        return [
            mol_copy
        ]  # At least return the original molecule if no resonance structure is found

    return known_structs


def generate_charged_resonance_structures(
    mol: Chem.RWMol,
    keep_isomorphic: bool = False,
    copy: bool = True,
    kekulize: bool = True,
    **kwargs: Any,
) -> list[Chem.Mol]:
    """Generate resonance structures for a charged molecule.

    Args:
        mol (Chem.RWMol): A charged molecule in RDKit RWMol.
        keep_isomorphic (bool, optional): If keep isomorphic resonance structures. Defaults to ``False``.
        copy (bool, optional): If copy the input molecule. Defaults to ``True``.
        kekulize (bool, optional): If kekulize the molecule in generating resonance structures. Defaults to ``True``.
        **kwargs (Any): Additional arguments for the resonance algorithms.

    Returns:
        list[Chem.Mol]: a list of molecules with resonance structures.
    """
    mol_copy = Chem.RWMol(mol, True) if copy else mol

    # Generate Resonance Structures
    flags = Chem.ALLOW_INCOMPLETE_OCTETS | Chem.UNCONSTRAINED_CATIONS
    if kekulize:
        flags |= Chem.KEKULE_ALL
    suppl = Chem.ResonanceMolSupplier(mol_copy, flags=flags)
    res_mols = [Chem.RWMol(mol) for mol in suppl if mol is not None]

    # Post-processing resonance structures
    cleaned_mols = []
    for res_mol in res_mols:
        # If a structure cannot be sanitized, removed it
        try:
            # Sanitization strategy is inspired by
            # https://github.com/rdkit/rdkit/discussions/6358
            flags = Chem.SANITIZE_PROPERTIES | Chem.SANITIZE_SETCONJUGATION
            if not kekulize:
                flags |= (
                    Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                )
            Chem.SanitizeMol(res_mol, sanitizeOps=flags)
        except BaseException as e:
            logger.debug(f"Sanitization failed for a resonance structure. Got {e}")
            continue
        if kekulize:
            unset_aromatic_flags(res_mol)
        cleaned_mols.append(res_mol)

    # To remove duplicate resonance structures
    known_structs: list[Chem.Mol] = []
    for new_struct in cleaned_mols:
        for known_struct in known_structs:
            if is_equivalent_structure(
                ref_mol=known_struct,
                qry_mol=new_struct,
                isomorphic_equivalent=not keep_isomorphic,
            ):
                break
        else:
            new_struct.__setattr__("__sssAtoms", [])
            known_structs.append(new_struct)

    if not known_structs:
        return [
            mol_copy
        ]  # At least return the original molecule if no resonance structure is found

    return known_structs


@ResonanceAlgoRegistry.register("rdkit")
def generate_resonance_structures(
    mol: Chem.RWMol,
    keep_isomorphic: bool = False,
    copy: bool = True,
    kekulize: bool = True,
    **kwargs: Any,
) -> list[Chem.Mol]:
    """Generate resonance structures with RDKit algorithm.

    Args:
        mol (Chem.RWMol): A charged molecule in RDKit RWMol.
        keep_isomorphic (bool, optional): If keep isomorphic resonance structures. Defaults to ``False``.
        copy (bool, optional): If copy the input molecule. Defaults to ``True``.
        kekulize (bool, optional): If kekulize the molecule in generating resonance structures. Defaults to ``True``.
        **kwargs (Any): Additional arguments for the resonance algorithms.

    Returns:
        list[Chem.Mol]: a list of molecules with resonance structures.
    """
    if is_partially_charged(mol):
        return generate_charged_resonance_structures(
            mol=mol,
            keep_isomorphic=keep_isomorphic,
            copy=copy,
            kekulize=kekulize,
            **kwargs,
        )
    else:
        return generate_radical_resonance_structures(
            mol=mol,
            keep_isomorphic=keep_isomorphic,
            copy=copy,
            kekulize=kekulize,
            **kwargs,
        )
