#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
This module provides methods that can directly apply to RDKit Mol/RWMol.
"""

from typing import Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType, Mol, RWMol

import openbabel as ob

# Bond order dictionary for RDKit, numbers are the bond order.
ORDERS = {1: BondType.SINGLE, 2: BondType.DOUBLE, 3: BondType.TRIPLE, 1.5: BondType.AROMATIC,
          4: BondType.QUADRUPLE,
          'S': BondType.SINGLE, 'D': BondType.DOUBLE, 'T': BondType.TRIPLE, 'B': BondType.AROMATIC,
          'Q': BondType.QUADRUPLE}

# The rotational bond definition in RDkit
# It is the same as rdkit.Chem.Lipinski import RotatableBondSmarts
ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
ROTATABLE_BOND_SMARTS_WO_METHYL = Chem.MolFromSmarts('[!$(*#*)&!D1!H3]-&!@[!$(*#*)&!D1&!H3]')

# When perceiving molecules, openbabel will always perceive carbon monoxide as [C]=O
# Needs to correct it by [C-]#[O+]
CO_OPENBABEL_PATTERN = ob.OBSmartsPattern()
CO_OPENBABEL_PATTERN.Init('[C;v2]=[O]')

def determine_smallest_atom_index_in_torsion(atom1: 'rdkit.Chem.rdchem.Atom',
                                             atom2: 'rdkit.Chem.rdchem.Atom',
                                             ) -> int:
    """
    Determine the smallest atom index in mol connected to ``atom1`` which is not ``atom2``.
    Returns a heavy atom if available, otherwise a hydrogen atom.
    Useful for deterministically determining the indices of four atom in a torsion.
    This function assumes there ARE additional atoms connected to ``atom1``, and that ``atom2`` is not a hydrogen atom.

    Args:
        atom1 (Atom): The atom who's neighbors will be searched.
        atom2 (Atom): An atom connected to ``atom1`` to exclude (a pivotal atom).

    Returns:
        int: The smallest atom index (1-indexed) connected to ``atom1`` which is not ``atom2``.
    """
    neighbor = [a for a in atom1.GetNeighbors() if a.GetIdx()
                != atom2.GetIdx()]
    atomic_num_list = sorted([nb.GetAtomicNum() for nb in neighbor])
    min_atomic, max_atomic = atomic_num_list[0], atomic_num_list[-1]
    if min_atomic == max_atomic or min_atomic > 1:
        return min([nb.GetIdx() for nb in neighbor])
    else:
        return min([nb.GetIdx() for nb in neighbor if nb.GetAtomicNum() != 1])


def find_internal_torsions(mol: Union['Mol', 'RWMol'],
                           exclude_methyl: bool = False,
                           ) -> list:
    """
    Find the internal torsions from RDkit molecule.

    Args:
        mol (Union[Mol, RWMol]): RDKit molecule.
        exclude_methyl (bool): Whether exclude the torsions with methyl groups.

    Returns:
        list: A list of internal torsions.
    """
    torsions = list()
    smarts = ROTATABLE_BOND_SMARTS if not exclude_methyl \
        else ROTATABLE_BOND_SMARTS_WO_METHYL
    rot_atom_pairs = mol.GetSubstructMatches(smarts)

    for atoms_ind in rot_atom_pairs:
        pivots = [mol.GetAtomWithIdx(i) for i in atoms_ind]
        first_atom_ind = determine_smallest_atom_index_in_torsion(*pivots)
        pivots.reverse()
        last_atom_ind = determine_smallest_atom_index_in_torsion(*pivots)
        torsions.append([first_atom_ind, *atoms_ind, last_atom_ind])
    return torsions


def openbabel_mol_to_rdkit_mol(obmol: 'openbabel.OBMol',
                               remove_hs: bool = False,
                               sanitize: bool = True,
                               embed: bool = True,
                               ) -> 'RWMol':
    """
    Convert a OpenBabel molecular structure to a Chem.rdchem.RWMol object.
    Args:
        obmol (Molecule): An OpenBabel Molecule object for the conversion.
        remove_hs (bool, optional): Whether to remove hydrogen atoms from the molecule, Defaults to False.
        sanitize (bool, optional): Whether to sanitize the RDKit molecule. Defaults to True.
        embed (bool, optional): Whether to embeb 3D conformer from OBMol. Defaults to True.

    Returns:
        RWMol: A writable RDKit RWMol instance.
    """
    rw_mol = Chem.rdchem.RWMol()
    for obatom in ob.OBMolAtomIter(obmol):
        atom = Chem.rdchem.Atom(obatom.GetAtomicNum())
        isotope = obatom.GetIsotope()
        if isotope != 0:
            atom.SetIsotope(isotope)
        spin = obatom.GetSpinMultiplicity()
        if spin == 2:  # radical
            atom.SetNumRadicalElectrons(1)
        elif spin in [1, 3]:  # carbene
            # TODO: Not sure if singlet and triplet are distinguished
            atom.SetNumRadicalElectrons(2)
        atom.SetFormalCharge(obatom.GetFormalCharge())
        if not (remove_hs and obatom.GetAtomicNum == 1):
            rw_mol.AddAtom(atom)

    for bond in ob.OBMolBondIter(obmol):
        # Atom indexes in Openbael is 1-indexed, so we need to convert them to 0-indexed
        atom1_idx = bond.GetBeginAtomIdx() - 1
        atom2_idx = bond.GetEndAtomIdx() - 1
        # Get the bond order. For aromatic molecules, the bond order is not
        # 1.5 but 1 or 2. Manually set them to 1.5
        bond_order = bond.GetBondOrder()
        if bond_order not in [1, 2, 3, 4] or bond.IsAromatic():
            bond_order = 1.5

        rw_mol.AddBond(atom1_idx, atom2_idx, ORDERS[bond_order])

    # Rectify the molecule
    if remove_hs:
        rw_mol = Chem.RemoveHs(rw_mol, sanitize=sanitize)
    elif sanitize:
        Chem.SanitizeMol(rw_mol)

    # If OBMol has 3D information, it can be embed to the RDKit Mol
    if embed and obmol.HasNonZeroCoords():
        coords = []
        for obatom in ob.OBMolAtomIter(obmol):
            coords.append([obatom.GetX(), obatom.GetY(), obatom.GetZ()])
        AllChem.EmbedMolecule(rw_mol)
        set_rdconf_coordinates(rw_mol.GetConformer(), coords)
    return rw_mol

def rdkit_mol_to_openbabel_mol(rdmol: Union['Mol', 'RWMol'],
                               embed: bool = True,
                               ) -> 'openbabel.OBMol':
    """
    Convert a Mol/RWMol to a Openbabel mol. This a temporary replace of
    ``rdkit_mol_to_openbabel_mol_manual``.

    Args:
        rdmol (Mol): The RDKit Mol/RWMol object to be converted.
        embed (bool, optional): Whether to embed conformer into the OBMol. Defaults to True.

    Returns:
        OBMol: An openbabel OBMol instance.
    """
    try:
        # RDKitMol
        sdf_str = rdmol.ToMolBlock()
    except AttributeError:
        # RDKit Mol or RWMol
        sdf_str = Chem.MolToMolBlock(rdmol)
    obconv = ob.OBConversion()
    obconv.SetInFormat('sdf')
    obmol = ob.OBMol()
    obconv.ReadString(obmol, sdf_str)

    if not embed:
        for atom in ob.OBMolAtomIter(obmol):
            atom.SetVector(ob.vector3(0, 0, 0))

    return obmol


def rdkit_mol_to_openbabel_mol_manual(rdmol: Union['Mol', 'RWMol'],
                                      embed: bool = True,
                                      ) -> 'openbabel.OBMol':
    """
    Convert a Mol/RWMol to a Openbabel mol. This function has a problem converting
    aromatic molecules. Example: 'c1nc[nH]n1'. Currently use a workaround, converting an
    RDKit Mol to sdf string and read by openbabel.

    Args:
        rdmol (Mol): The RDKit Mol/RWMol object to be converted.
        embed (bool, optional): Whether to embed conformer into the OBMol. Defaults to True.

    Returns:
        OBMol: An openbabel OBMol instance.
    """
    obmol = ob.OBMol()
    for rdatom in rdmol.GetAtoms():
        obatom = obmol.NewAtom()
        obatom.SetAtomicNum(rdatom.GetAtomicNum())
        isotope = rdatom.GetIsotope()
        if isotope != 0:
            obatom.SetIsotope(isotope)
        obatom.SetFormalCharge(rdatom.GetFormalCharge())
    bond_type_dict = {BondType.SINGLE: 1,
                      BondType.DOUBLE: 2,
                      BondType.TRIPLE: 3,
                      BondType.QUADRUPLE: 4,
                      BondType.AROMATIC: 5}
    for bond in rdmol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx() + 1
        atom2_idx = bond.GetEndAtomIdx() + 1
        order = bond_type_dict[bond.GetBondType()]
        obmol.AddBond(atom1_idx, atom2_idx, order)

    # Note: aromatic is not correctly handeled for
    # heteroatom involved rings in the current molecule buildup.
    # May need to update in the future

    obmol.AssignSpinMultiplicity(True)

    if embed:
        try:
            conf = rdmol.GetConformer()
        except ValueError:
            # No conformer
            pass
        else:
            coords = conf.GetPositions()
            for atom_idx, atom in enumerate(ob.OBMolAtomIter(obmol)):
                atom.SetVector(ob.vector3(*coords[atom_idx].tolist()))

    return obmol


def rmg_mol_to_rdkit_mol(rmgmol: 'rmgpy.molecule.Molecule',
                         remove_hs: bool = False,
                         sanitize: bool = True,
                         ) -> 'RWMol':
    """
    Convert a RMG molecular structure to an RDKit Mol object. Uses
    `RDKit <http://rdkit.org/>`_ to perform the conversion.
    Perceives aromaticity.
    Adopted from rmgpy/molecule/converter.py

    Args:
        rmgmol (Molecule): An RMG Molecule object for the conversion.
        remove_hs (bool, optional): Whether to remove hydrogen atoms from the molecule, ``True`` to remove.
        sanitize (bool, optional): Whether to sanitize the RDKit molecule, ``True`` to sanitize.

    Returns:
        RWMol: An RWMol molecule object corresponding to the input RMG Molecule object.
    """
    atom_id_map = dict()

    # only manipulate a copy of ``mol``
    mol_copy = rmgmol.copy(deep=True)
    if not mol_copy.atom_ids_valid():
        mol_copy.assign_atom_ids()
    for i, atom in enumerate(mol_copy.atoms):
        # keeps the original atom order before sorting
        atom_id_map[atom.id] = i
    atoms_copy = mol_copy.vertices

    rwmol = Chem.rdchem.RWMol()
    for rmg_atom in atoms_copy:
        rd_atom = Chem.rdchem.Atom(rmg_atom.element.symbol)
        if rmg_atom.element.isotope != -1:
            rd_atom.SetIsotope(rmg_atom.element.isotope)
        rd_atom.SetNumRadicalElectrons(rmg_atom.radical_electrons)
        rd_atom.SetFormalCharge(rmg_atom.charge)
        if rmg_atom.element.symbol == 'C' and rmg_atom.lone_pairs == 1 \
                and not rmg_atom.charge and mol_copy.multiplicity == 1:
            # hard coding for carbenes
            rd_atom.SetNumRadicalElectrons(2)
        if not (remove_hs and rmg_atom.symbol == 'H'):
            rwmol.AddAtom(rd_atom)

    # Add the bonds
    for atom1 in atoms_copy:
        for atom2, bond12 in atom1.edges.items():
            if bond12.is_hydrogen_bond():
                continue
            if atoms_copy.index(atom1) < atoms_copy.index(atom2):
                rwmol.AddBond(
                    atom_id_map[atom1.id],
                    atom_id_map[atom2.id],
                    ORDERS[bond12.get_order_str()])

    # Rectify the molecule
    if remove_hs:
        rwmol = Chem.RemoveHs(rwmol, sanitize=sanitize)
    elif sanitize:
        Chem.SanitizeMol(rwmol)
    return rwmol


def set_rdconf_coordinates(conf: Union['Conformer', 'RDKitConf'],
                           coords: Union[tuple, list, np.ndarray]):
    """
    Set the Positions of atoms of the conformer.

    Args:
        conf (Union[Conformer, 'RDKitConf']): The conformer to be set.
        coords (Union[tuple, list, np.ndarray]): The coordinates to be set.

    Raises:
        ValueError: Not a valid ``coords`` input, when giving something else.
    """
    try:
        num_atoms = coords.shape[0]
    except AttributeError:
        coords = np.array(coords)
        num_atoms = coords.shape[0]
    finally:
        for i in range(num_atoms):
            conf.SetAtomPosition(i, coords[i, :])


def get_obmol_coords(obmol: ob.OBMol):
    """
    Get the atom coordinates from an openbabel molecule. If all coordinates are zero,
    None will be returned.

    Args:
        obmol (OBMol): The openbabel OBMol to get coordinates from.

    Returns:
        np.array: The coordinates.
    """
    coords = []
    for obatom in ob.OBMolAtomIter(obmol):
        coords.append([obatom.GetX(), obatom.GetY(), obatom.GetZ()])
    return np.array(coords)


def set_obmol_coords(obmol: ob.OBMol,
                     coords: np.array):
    """
    Get the atom coordinates from an openbabel molecule. If all coordinates are zero,
    It will return None

    Args:
        obmol (OBMol): The openbabel OBMol to get coordinates from.
        coords (np.array): The coordinates to set.
    """
    for atom_idx, atom in enumerate(ob.OBMolAtomIter(obmol)):
        atom.SetVector(ob.vector3(*coords[atom_idx].tolist()))


def parse_xyz_by_openbabel(xyz: str,
                           correct_CO: bool = True):
    """
    Perceive a xyz str using openbabel and generate the corresponding OBMol.

    Args:
        xyz (str): A str in xyz format containing atom positions.
        correctCO (bool, optional): It is known that openbabel will parse carbon monoxide
                                    as [C]=O instead of [C-]#[O+]. This function contains
                                    a patch to correct that. Defaults to ``True``.

    Returns:
        ob.OBMol: An openbabel molecule from the xyz
    """
    obconversion = ob.OBConversion()
    obconversion.SetInFormat('xyz')
    obmol = ob.OBMol()
    success = obconversion.ReadString(obmol, xyz)
    assert success, ValueError('Unable to parse the provided xyz.')

    if correct_CO and CO_OPENBABEL_PATTERN.Match(obmol):
        index_pairs = [x for x in CO_OPENBABEL_PATTERN.GetUMapList()]
        for pair in index_pairs:
            obmol.GetBond(*pair).SetBondOrder(3)
            for idx in pair:
                atom = obmol.GetAtom(idx)
                if atom.GetAtomicNum() == 6:
                    atom.SetSpinMultiplicity(0)
                    atom.SetFormalCharge(-1)
                elif atom.GetAtomicNum() == 8:
                    atom.SetSpinMultiplicity(0)
                    atom.SetFormalCharge(+1)

    return obmol
