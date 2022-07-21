#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
This module provides methods that can directly apply to RDKit Mol/RWMol.
"""

from typing import Iterable, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType, Mol, RWMol
from rdkit import RDLogger
from rdkit.Chem.TorsionFingerprints import CalculateTorsionLists

# Mute RDKit's error logs
# They can be confusing at places where try ... except ... are implemented.
RDLogger.DisableLog("rdApp.*")

try:
    # Openbabel 3
    from openbabel import openbabel as ob
except ImportError:
    # Openbabel 2
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
CO_OPENBABEL_PATTERN.Init('[Cv2X1]=[OX1]')

# Carbene, nitrene, and atomic oxygen templates. RDKit and Openbabel have difficulty
# distinguish their multiplicity when input as SMILES or XYZ
CARBENE_PATTERN = Chem.MolFromSmarts('[Cv0,Cv1,Cv2,Nv0,Nv1,Ov0]')

PERIODIC_TABLE = Chem.GetPeriodicTable()
VDW_RADII = {i: PERIODIC_TABLE.GetRvdw(i) for i in range(1, 36)}


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


def find_ring_torsions(mol: Union['Mol', 'RWMol']) -> list:
    """
    Find the ring from RDkit molecule.

    Args:
        mol (Union[Mol, RWMol]): RDKit molecule.

    Returns:
        list: A list of ring torsions.
    """
    try:
        _, ring_torsions = CalculateTorsionLists(mol.ToRWMol())
    except AttributeError:
        _, ring_torsions = CalculateTorsionLists(mol)
    if ring_torsions:
        ring_torsions = [list(t) for t in ring_torsions[0][0]]
    return ring_torsions


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
        if not remove_hs:
            atom.SetNoImplicit(True)
        if spin == 2:  # radical
            atom.SetNumRadicalElectrons(1)
        elif spin in [1, 3]:  # carbene
            # TODO: Not sure if singlet and triplet are distinguished
            atom.SetNumRadicalElectrons(2)
        atom.SetFormalCharge(obatom.GetFormalCharge())
        if not (remove_hs and obatom.GetAtomicNum == 1):
            rw_mol.AddAtom(atom)

    for bond in ob.OBMolBondIter(obmol):
        # Atom indexes in Openbabel is 1-indexed, so we need to convert them to 0-indexed
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
        coords = get_obmol_coords(obmol)
        conf = Chem.rdchem.Conformer(coords.shape[0])  # Create a conformer that has number of atoms specified
        set_rdconf_coordinates(conf, coords)
        rw_mol.AddConformer(conf, assignId=True)
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

    # Temporary Fix for Issue # 1
    # This function works okay with openbabel 2.4.1 but not 3.1.1
    # The atom spin multiplicity looks not right in the obmol
    # A naive fix for carbons and oxygens
    # This fix cannot deal with any charged species!!!
    for obatom in ob.OBMolAtomIter(obmol):
        # Find the unsaturated carbons
        if obatom.GetAtomicNum() == 6 and obatom.GetTotalValence() < 4:
            obatom.SetSpinMultiplicity(5 - obatom.GetTotalValence())
        elif obatom.GetAtomicNum() == 7 and obatom.GetTotalValence() < 3:
            obatom.SetSpinMultiplicity(4 - obatom.GetTotalValence())
        elif obatom.GetAtomicNum() == 8 and obatom.GetTotalValence() < 2:
            obatom.SetSpinMultiplicity(3 - obatom.GetTotalValence())
        elif obatom.GetAtomicNum() == 1 and obatom.GetTotalValence() == 0:
            obatom.SetSpinMultiplicity(2)

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
            set_obmol_coords(obmol, coords)

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
    reset_num_electron = {}
    for i, rmg_atom in enumerate(atoms_copy):
        rd_atom = Chem.rdchem.Atom(rmg_atom.element.symbol)
        if rmg_atom.element.isotope != -1:
            rd_atom.SetIsotope(rmg_atom.element.isotope)
        if not remove_hs:
            # Avoid `SanitizeMol` adding undesired hydrogens
            rd_atom.SetNoImplicit(True)
        else:
            explicit_Hs = [True for a, b in rmg_atom.edges.items()
                           if a.is_hydrogen() and b.is_single()]
            rd_atom.SetNumExplicitHs(sum(explicit_Hs))
            rd_atom.SetNoImplicit(True)
        rd_atom.SetNumRadicalElectrons(rmg_atom.radical_electrons)
        rd_atom.SetFormalCharge(rmg_atom.charge)

        # There are cases requiring to reset electrons after sanitization
        # for carbene, nitrene and atomic oxygen
        # For other atoms, to be added once encountered
        if rmg_atom.is_carbon() and rmg_atom.lone_pairs >= 1 and not rmg_atom.charge:
            reset_num_electron[i] = rmg_atom.radical_electrons
        elif rmg_atom.is_nitrogen() and rmg_atom.lone_pairs >= 2 and not rmg_atom.charge:
            reset_num_electron[i] = rmg_atom.radical_electrons
        elif rmg_atom.is_oxygen and rmg_atom.lone_pairs >= 3 and not rmg_atom.charge:
            reset_num_electron[i] = rmg_atom.radical_electrons
        if not (remove_hs and rmg_atom.symbol == 'H'):
            rwmol.AddAtom(rd_atom)

    # Add the bonds
    for atom1 in atoms_copy:
        if remove_hs and atom1.is_hydrogen():
            continue
        for atom2, bond12 in atom1.edges.items():
            if remove_hs and atom2.is_hydrogen():
                continue
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

    for key, val in reset_num_electron.items():
        rwmol.GetAtomWithIdx(key).SetNumRadicalElectrons(val)

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


def fix_CO_openbabel(obmol: 'Openbabel.OBMol',
                     correct_CO: bool = True):
    """
    Fix the CO perception issue for openbabel molecule.

    Args:
        obmol (Openbabel.OBMol): The Openbabel molecule instance.
        correct_CO (bool, optional): Whether to fix this issue. Defaults to True.
    """
    if not correct_CO:
        return
    CO_OPENBABEL_PATTERN.Match(obmol)
    for pair in CO_OPENBABEL_PATTERN.GetUMapList():
        obmol.GetBond(*pair).SetBondOrder(3)
        for idx in pair:
            atom = obmol.GetAtom(idx)
            if atom.GetAtomicNum() == 6:
                atom.SetSpinMultiplicity(0)
                atom.SetFormalCharge(-1)
            elif atom.GetAtomicNum() == 8:
                atom.SetSpinMultiplicity(0)
                atom.SetFormalCharge(+1)


def find_problematic_nitrogens(obmol: 'openbabel.OBMol'):
    """
    A function to detect problematic nitrogen (neutral with four valences).

    Args:
        obmol (openbabel.OBMol): The openbabel molecule instance.

    Returns:
        list: a list of indexes.
    """
    return [atom for atom in ob.OBMolAtomIter(obmol)
            if atom.GetAtomicNum() == 7 and atom.GetTotalValence() > 3 and atom.GetFormalCharge() < 1]


def find_problematic_carbons(obmol: 'openbabel.OBMol'):
    """
    A function to detect problematic nitrogen (neutral with four valences).

    Args:
        obmol (openbabel.OBMol): The openbabel molecule instance.

    Returns:
        list: a list of indexes.
    """
    return [atom for atom in ob.OBMolAtomIter(obmol)
            if atom.GetAtomicNum() == 6 and atom.GetTotalValence() > 4]


def fix_N_terminalXrad(atom,
                       nb_atom,):
    """
    Fix the case where R=N=[X] or R-N#[X]. E.g., R-[N]#[C], R=N=[N]
    Increase the charge of N and decrease the charge and spin multiplicity of X.
    E.g., R-[N+]#[C-], R=[N+]=[N-].

    Args:
        atom (OBAtom): The nitrogen atom
        nb_atom (OBAtom): The neighbor atom
    """
    atom.SetFormalCharge(+1)
    atom.SetSpinMultiplicity(0)
    nb_atom.SetFormalCharge(nb_atom.GetFormalCharge() - 1)
    if nb_atom.GetSpinMultiplicity() > 2:
        nb_atom.SetSpinMultiplicity(nb_atom.GetSpinMultiplicity() - 1)
    else:
        nb_atom.SetSpinMultiplicity(0)


def fix_N_nonterminalX(atom,
                       nb_atom,):
    atom.SetFormalCharge(0)
    atom.SetSpinMultiplicity(0)
    nb_atom.SetSpinMultiplicity(2)
    bond = atom.GetBond(nb_atom)
    bond.SetBondOrder(bond.GetBondOrder() - 1)


def fix_carbon_five_valences(atom, nb_atom):
    atom.SetFormalCharge(0)
    atom.SetSpinMultiplicity(0)
    if nb_atom.GetSpinMultiplicity() > 1:
        nb_atom.SetSpinMultiplicity(nb_atom.GetSpinMultiplicity() + 1)
    else:
        nb_atom.SetSpinMultiplicity(2)
    bond = atom.GetBond(nb_atom)
    bond.SetBondOrder(bond.GetBondOrder() - 1)


def fix_problematic_nitrogen(atom: 'openbabel.OBAtom'):
    if atom.GetTotalValence() == 4:
        terminal_nbs = sorted([nbatom for nbatom in ob.OBAtomAtomIter(atom)
                               if nbatom.GetTotalDegree() == 1 \
                                and nbatom.GetSpinMultiplicity() > 1 \
                                and nbatom.GetFormalCharge() == 0],  # terminal and has unpaired electrons and no charge
                              key=lambda x: x.GetBond(atom).GetBondOrder())
        unsaturated_bonds = sorted([bond for bond in ob.OBAtomBondIter(atom)
                                    if bond.GetBondOrder() > 1 \
                                    and bond.GetBondOrder() != 1.5 ],  # avoid aromatic bonds
                                   key=lambda x: x.GetBondOrder())
        if terminal_nbs:
            fix_N_terminalXrad(atom, terminal_nbs[-1])  # Use the one with the highest bond order
        elif unsaturated_bonds:
            fix_N_nonterminalX(atom, unsaturated_bonds[-1].GetNbrAtom(atom))   # Use the one with the highest bond order
        else:
            atom_dists = [(bond.GetNbrAtom(atom).GetAtomicNum(), bond.GetLength()) for bond in ob.OBAtomBondIter(atom)]
            raise NotImplementedError(f'Encountered a nitrogen ({atom.GetIdx()}) with 4 valences that cannot be fixed.'
                                      f'The neighboring atoms and their corresponding bond distances are {atom_dists}.')
    else:
        raise NotImplementedError(f'Encountered a nitrogen ({atom.GetIdx()}) with 5 valences that cannot be fixed')


def fix_problematic_carbon(atom: 'openbabel.OBAtom'):
    unsaturated_bonds = sorted([bond for bond in ob.OBAtomBondIter(atom)
                                if bond.GetBondOrder() > 1 \
                                and bond.GetBondOrder() != 1.5 ],  # avoid aromatic bonds
                                key=lambda x: x.GetBondOrder())
    if unsaturated_bonds:
        fix_carbon_five_valences(atom, unsaturated_bonds[-1].GetNbrAtom(atom))
    else:
        raise NotImplementedError(f'Encountered a nitrogen ({atom.GetIdx()}) with 5 valences that cannot be fixed')


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
    if not success:
        raise ValueError('Unable to parse the provided xyz.')

    # Temporary Fix for Issue # 1
    # This function works okay with openbabel 2.4.1 but not 3.1.1
    # The atom spin multiplicity looks not right in the obmol
    # A naive fix for carbons and oxygens
    # This fix cannot deal with any charged species!!!
    for obatom in ob.OBMolAtomIter(obmol):
        # Find the unsaturated carbons
        if obatom.GetAtomicNum() == 6 and obatom.GetTotalValence() < 4:
            obatom.SetSpinMultiplicity(5 - obatom.GetTotalValence())
        # Find the unsaturated nitrogen
        elif obatom.GetAtomicNum() == 7 and obatom.GetTotalValence() < 3:
            obatom.SetSpinMultiplicity(4 - obatom.GetTotalValence())
        # Find the unsaturated oxygen
        elif obatom.GetAtomicNum() == 8 and obatom.GetTotalValence() < 2:
            obatom.SetSpinMultiplicity(3 - obatom.GetTotalValence())
        # Find the unsaturated nitrogen and halogen
        elif obatom.GetAtomicNum() in [1, 9, 17, 35, 53] and obatom.GetTotalValence() == 0:
            obatom.SetSpinMultiplicity(2)

    # Correct [C]=O to [C-]#[O+]
    fix_CO_openbabel(obmol, correct_CO=correct_CO)

    # Currently using a pre-check strategy to solve problematic cases
    # If a significant bottleneck of speed is observed
    # they can also be moved out and only be fixed whenever encountered an issue in the latter step
    for _ in range(3):
        problematic_carbons = find_problematic_carbons(obmol)
        if not problematic_carbons:
            break
        for atom in problematic_carbons:
            fix_problematic_carbon(atom)
    else:
        raise NotImplementedError(f'Cannot fix this molecule due to problematic carbons {problematic_carbons}')

    for _ in range(3):
        problematic_nitrogens = find_problematic_nitrogens(obmol)
        if not problematic_nitrogens:
            break
        for atom in problematic_nitrogens:
            fix_problematic_nitrogen(atom)
    else:
        raise NotImplementedError(f'Cannot fix this molecule due to problematic nitrogens {problematic_nitrogens}')
    return obmol


def get_element_symbols(atom_nums: Iterable):
    """
    Get the element symbols for a given atom index list.

    Args:
        atom_nums (Iterable): A list of elemental numbers.
    Returns:
        list: A list of element symbols.
    """
    return [PERIODIC_TABLE.GetElementSymbol(int(atom_num)) for atom_num in atom_nums]


def get_atom_masses(atom_nums: Iterable):
    """
    Get the atom masses for a given atom index list.

    Args:
        atom_nums (Iterable): A list of elemental numbers.
    Returns:
        list: A list of element symbols.
    """
    return [PERIODIC_TABLE.GetAtomicWeight(int(atom_num)) for atom_num in atom_nums]


def get_internal_coords(obmol,
                        nonredundant: bool = True,
                        ) -> list:
    """
    Generate a non_redundant_internal coordinate.

    Args:
        obmol (OBMol): Openbabel mol.
        nonredundant (bool): whether non-redundant. Defaults to ``True``.
    """
    obconv = ob.OBConversion()
    obconv.SetOutFormat('gzmat')
    gzmat_str = obconv.WriteString(obmol)
    lines = gzmat_str.split('Variables:')[0].splitlines()[6:]
    bonds = []
    angles = []
    torsions = []
    for idx, line in enumerate(lines):
        items = line.split()
        try:
            bonds.append((idx + 1, int(items[1])))
            angles.append([idx + 1, int(items[1]), int(items[3])])
            torsions.append([idx + 1, int(items[1]), int(items[3]), int(items[5])])
        except IndexError:
            # First, second, and third lines are special
            pass
    if nonredundant:
        non_red_torsions = []
        pivots = []
        for tor in torsions:
            if tor[1:3] not in pivots and tor[-2:-4:-1] not in pivots:
                pivots.append(tor[1:3])
                non_red_torsions.append(tor)
        pass
        torsions = non_red_torsions
    return bonds, angles, torsions

def reverse_map(map: Iterable,
                as_list: bool = True):
    """
    Inverse-transform the index and value relationship in a mapping.
    E.g., when doing a subgraph match, RDKit will returns a list
    that the indexes correspond to the reference molecule and the values
    correspond to the probing molecule. One by renumber the atoms in the
    probing molecule according to the reverse_map, the atom numbering between
    the two molecules should be consistent

    Args:
        map (Iterable): An atom mapping.
        as_list (bool, optional): Output result as a `list` object. Otherwise,
                                  the output is a np.ndarray.

    Returns:
        An inverted atom map from the given ``match`` atom map
    """
    if as_list:
        return np.argsort(map).tolist()
    else:
        return np.argsort(map)
