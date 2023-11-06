#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the function generating resonance structures.
"""

from rdmc.mol import RDKitMol
from rdmc.mol_compare import get_unique_mols
from rdkit import Chem
from rdkit.Chem import RWMol


def generate_radical_resonance_structures(
    mol: RDKitMol,
    unique: bool = True,
    consider_atommap: bool = False,
    kekulize: bool = False,
):
    """
    Generate resonance structures for a radical molecule.  RDKit by design doesn't work
    for radical resonance. The approach is a temporary workaround by replacing radical electrons by positive
    charges and generating resonance structures by RDKit ResonanceMolSupplier.
    Currently, this function only works for neutral radicals.

    Known issues:

    - Phenyl radical only generate one resonance structure when ``kekulize=True``, expecting 2.

    Args:
        mol (RDKitMol): A radical molecule.
        unique (bool, optional): Filter out duplicate resonance structures from the list. Defaults to ``True``.
        consider_atommap (bool, atommap): If consider atom map numbers in filtration duplicates.
                                          Only effective when ``unique=True``. Defaults to ``False``.
        kekulize (bool, optional): Whether to kekulize the molecule. Defaults to ``False``. As an example,
                                   benzene have one resonance structure if not kekulized (``False``) and
                                   two resonance structures if kekulized (``True``).

    Returns:
        list: a list of molecules with resonance structures.
    """
    assert mol.GetFormalCharge() == 0, "The current function only works for radical species."
    mol_copy = mol.Copy(quickCopy=True)  # Make a copy of the original molecule

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
    Chem.rdmolops.SetConjugation(mol_copy._mol)

    # Avoid generating certain resonance bonds
    for atom in mol_copy.GetAtoms():
        if (atom.GetAtomicNum() == 8 and len(atom.GetNeighbors()) > 1) or \
                (atom.GetAtomicNum() == 7 and len(atom.GetNeighbors()) > 2):
            # Avoid X-O-Y be part of the resonance and forms X-O.=Y
            # Avoid X-N(-Y)-Z be part of the resonance and forms X=N.(-Y)(-Z)
            [bond.SetIsConjugated(False) for bond in atom.GetBonds()]
    mol_copy.UpdatePropertyCache()  # Make sure the assignment is boardcast to atoms / bonds

    # Generate Resonance Structures
    flags = Chem.ALLOW_INCOMPLETE_OCTETS | Chem.UNCONSTRAINED_CATIONS
    if kekulize:
        flags |= Chem.KEKULE_ALL
    suppl = Chem.ResonanceMolSupplier(mol_copy._mol, flags=flags)
    res_mols = [RDKitMol(RWMol(mol)) for mol in suppl if mol is not None]

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
            flags = Chem.SanitizeFlags.SANITIZE_ALL
            if kekulize:
                flags ^= (
                    Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                )
            res_mol.Sanitize(sanitizeOps=flags)
        except BaseException as e:
            print(e)
            # todo: make error type more specific and add a warning message
            continue
        if kekulize:
            _unset_aromatic_flags(res_mol)
        cleaned_mols.append(res_mol)

    # To remove duplicate resonance structures
    if unique:
        cleaned_mols = get_unique_mols(cleaned_mols, consider_atommap=consider_atommap)
        for mol in cleaned_mols:
            # According to
            # https://github.com/rdkit/rdkit/blob/9249ca5cc840fc72ea3bb73c2ff1d71a1fbd3f47/rdkit/Chem/Draw/IPythonConsole.py#L152
            # highlight info is stored in __sssAtoms
            mol._mol.__setattr__("__sssAtoms", [])

    if not cleaned_mols:
        return [
            mol
        ]  # At least return the original molecule if no resonance structure is found

    return cleaned_mols


def _unset_aromatic_flags(mol):
    """
    A helper function to unset aromatic flags in a molecule.
    This is useful when cleaning up the molecules from resonance structure generation.
    In such case, a molecule may have single-double bonds but are marked as aromatic bonds.
    """
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.AROMATIC and bond.GetIsAromatic():
            bond.SetIsAromatic(False)
            bond.GetBeginAtom().SetIsAromatic(False)
            bond.GetEndAtom().SetIsAromatic(False)
    return mol
