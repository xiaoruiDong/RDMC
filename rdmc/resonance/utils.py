#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import math
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import Lipinski, rdqueries


PERIODIC_TABLE = Chem.GetPeriodicTable()

AROMATIC = Chem.BondType.AROMATIC


electronegativity = {
    1: 2.20,
    3: 0.98,
    4: 1.57,
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    11: 0.93,
    12: 1.31,
    13: 1.61,
    14: 1.90,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    19: 0.82,
    20: 1.00,
    21: 1.36,
    22: 1.54,
    23: 1.63,
    24: 1.66,
    25: 1.55,
    26: 1.83,
    27: 1.91,
    29: 1.90,
    30: 1.65,
    31: 1.81,
    32: 2.01,
    33: 2.18,
    34: 2.55,
    35: 2.96,
    53: 2.66,
}


bond_order_dicts = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.QUADRUPLE,
    5: Chem.BondType.QUINTUPLE,
    1.5: Chem.BondType.ONEANDAHALF,
    2.5: Chem.BondType.TWOANDAHALF,
    3.5: Chem.BondType.THREEANDAHALF,
    4.5: Chem.BondType.FOURANDAHALF,
    5.5: Chem.BondType.FIVEANDAHALF,
}

aryl_radical_query = rdqueries.NumRadicalElectronsGreaterQueryAtom(0)
aryl_radical_query.ExpandQuery(rdqueries.IsAromaticQueryAtom())

aryne_template1_aro = Chem.MolFromSmarts("[*]1#[*]:[*]:[*]:[*]:[*]1")
aryne_template1_kek = Chem.MolFromSmarts("[*]1#[*]-[*]=[*]-[*]=[*]1")
aryne_template2 = Chem.MolFromSmarts("[*]1=[*]=[*]=[*]-[*]=[*]1")


def force_no_implicit(mol: "Mol"):
    """
    Force RDKit to not use implicit hydrogens.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1 and not atom.GetTotalNumHs():
            atom.SetNoImplicit(True)


# RDKit / RDMC compatible
def get_electronegativity(atom: "Atom") -> float:
    """
    Get the electronegativity of an atom. Currently only supports atom 1-35 and 53. Others will
    return 1.0.

    Args:
        atom (Atom): The atom whose electronegativity is to be returned.

    Returns:
        float: The electronegativity of the atom.
    """
    return electronegativity.get(atom.GetAtomicNum(), 1.0)


# RDKit / RDMC compatible
def get_total_bond_order(atom: "Atom") -> float:
    """
    Get the total bond order of an atom.

    Args:
        atom (Atom): The atom whose total bond order is to be returned.

    Returns:
        float: The total bond order of the atom.
    """
    # b.GetValenceContrib(atom) is more robust than b.GetBondTypeAsDouble()
    # as it considers cases like dative bonds
    return (
        sum([b.GetValenceContrib(atom) for b in atom.GetBonds()])
        + atom.GetNumImplicitHs()
    )


# RDKit / RDMC compatible
def get_lone_pair(atom: "Atom") -> int:
    """
    Get the number of lone pairs on an atom.

    Args:
        atom (Atom): The atom whose lone pair is to be returned.

    Returns:
        int: The number of lone pairs on the atom.
    """
    order = get_total_bond_order(atom)
    return (
        PERIODIC_TABLE.GetNOuterElecs(atom.GetAtomicNum())
        - atom.GetNumRadicalElectrons()
        - atom.GetFormalCharge()
        - int(order)
    ) // 2


# RDKit / RDMC compatible
def get_charge_span(mol: "RDKitMol") -> float:
    """
    Get the charge span of a molecule.

    Args:
        mol (RDKitMol): The molecule whose charge span is to be returned.

    Returns:
        float: The charge span of the molecule.
    """
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    abs_net_charge = abs(sum(charges))
    sum_of_abs_charges = sum([abs(charge) for charge in charges])
    return (sum_of_abs_charges - abs_net_charge) / 2


# RDKit / RDMC compatible
def get_radical_count(mol) -> int:
    """
    Return the total number of radical electrons on all atoms in the
    molecule. In this function, monoradical atoms count as one, biradicals
    count as two, etc.

    Args:
        mol (RDKitMol, RWMol): The molecule to be checked.

    Returns:
        int: The total number of radical electrons on all atoms in the molecule.
    """
    return sum([atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()])


# RDKit / RDMC compatible
def get_num_occupied_orbitals(atom: "Atom") -> int:
    """
    Get the number of occupied orbitals on an atom.

    Args:
        atom (Atom): The atom whose number of occupied orbitals is to be returned.

    Returns:
        int: The number of occupied orbitals on the atom.
    """
    order = get_total_bond_order(atom)
    return math.ceil(
        (
            PERIODIC_TABLE.GetNOuterElecs(atom.GetAtomicNum())
            - atom.GetFormalCharge()
            + atom.GetNumRadicalElectrons()
            + int(order)
        )
        / 2
    )


# Pure RDKit
def get_num_aromatic_rings(mol: "Mol") -> int:
    """
    Get the number of aromatic rings in a molecule.
    """
    return Lipinski.NumAromaticRings(mol)


# RDKit / RDMC compatible
def get_aryne_rings(mol: "Mol") -> List[List[int]]:
    """
    Get the indices of all aryne rings in a molecule.
    """
    return (
        (
            mol.GetSubstructMatches(aryne_template1_aro)
            + mol.GetSubstructMatches(aryne_template1_kek)
        ),
        mol.GetSubstructMatches(aryne_template2),
    )


# RDKit / RDMC compatible
def has_empty_orbitals(atom: "Atom") -> bool:
    """
    Determine whether an atom has empty orbitals.

    Args:
        atom (Atom): The atom to be checked.

    Returns:
        bool: ``True`` if the atom has empty orbitals, ``False`` otherwise.
    """
    atomic_num = atom.GetAtomicNum()
    num_occupied_orbitals = get_num_occupied_orbitals(atom)
    if atomic_num == 1:
        # s
        return num_occupied_orbitals < 1
    elif atomic_num <= 10:
        # sp3
        return num_occupied_orbitals < 4
    elif atomic_num < 36:
        # sp3d2
        return num_occupied_orbitals < 6
    else:
        # sp3d3. IF7. But let's not worry about it for now
        return num_occupied_orbitals < 7


# RDKit / RDMC compatible
def is_radical(mol) -> bool:
    """
    Determine whether a molecule is a radical.

    Args:
        mol (RDKitMol, RWMol): The molecule to be checked.

    Returns:
        bool: ``True`` if the molecule is a radical, ``False`` otherwise.
    """
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return True
    return False


# RDKit / RDMC compatible
def is_aromatic(mol) -> bool:
    """
    Determine whether a molecule is aromatic.

    Args:
        mol (RDKitMol, RWMol): The molecule to be checked.

    Returns:
        bool: ``True`` if the molecule is aromatic, ``False`` otherwise.
    """
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            return True
    return False


# RDKit / RDMC compatible
def is_cyclic(mol) -> bool:
    """
    Determine whether a molecule is cyclic.

    Args:
        mol (RDKitMol, RWMol): The molecule to be checked.

    Returns:
        bool: ``True`` if the molecule is cyclic, ``False`` otherwise.
    """
    for atom in mol.GetAtoms():
        if atom.IsInRing():
            return True
    return False


# RDKit / RDMC compatible
def get_order_str(bond: "Bond") -> str:
    """
    Get the string representation of the bond order.

    Args:
        bond (Bond): The bond whose order is to be returned.

    Returns:
        str: The string representation of the bond order.
    """
    bond_type = bond.GetBondType()
    if bond_type == 1:
        return "S"
    elif bond_type == 2:
        return "D"
    elif bond_type == 1.5:
        return "B"
    elif bond_type == 3:
        return "T"
    elif bond_type == 4:
        return "Q"
    else:
        return "?"


# RDKit / RDMC compatible
def get_relevant_cycles(mol) -> list:
    """
    Returns all relevant cycles as a list of bond indices.
    Note, as of now, this function only returns the smallest set of smallest rings.

    Args:
        mol (RDKitMol, RWMol): The molecule to be checked.

    Returns:
        list: A list of rings, each represented as a list of bond indices.
    """
    # TODO: improve this function with the actual relevant cycle algorithm
    # This is a temporary solution works for simple cases
    # TODO: determine if it is better to use bond indices or atom indices
    return mol.GetRingInfo().BondRings()


def get_aromatic_rings(mol):
    """
    Returns all aromatic rings as a list of atoms and a list of bonds.

    Identifies rings using `Graph.get_smallest_set_of_smallest_rings()`, then uses RDKit to perceive aromaticity.
    RDKit uses an atom-based pi-electron counting algorithm to check aromaticity based on Huckel's Rule.
    Therefore, this method identifies "true" aromaticity, rather than simply the RMG bond type.

    The method currently restricts aromaticity to six-membered carbon-only rings. This is a limitation imposed
    by RMG, and not by RDKit.

    By default, the atom order will be sorted to get consistent results from different runs. The atom order can
    be saved when dealing with problems that are sensitive to the atom map.
    """
    rings = get_relevant_cycles(mol)

    def filter_fused_rings(_rings):
        """
        Given a list of rings, remove ones which share more than 2 atoms.
        """
        if len(_rings) < 2:
            return _rings

        to_remove = set()
        for i, j in itertools.combinations(range(len(_rings)), 2):
            if len(set(_rings[i]) & set(_rings[j])) > 2:
                to_remove.add(i)
                to_remove.add(j)

        to_remove_sorted = sorted(to_remove, reverse=True)

        for i in to_remove_sorted:
            del _rings[i]

        return _rings

    # Remove rings that share more than 3 atoms, since they cannot be planar
    rings = filter_fused_rings(rings)

    # Only keep rings with exactly 6 atoms, since RMG can only handle aromatic benzene
    rings = [ring for ring in rings if len(ring) == 6]

    if not rings:
        return [], []

    aromatic_rings = []
    aromatic_bonds = []
    for ring0 in rings:
        aromatic_bonds_in_ring = []
        # Figure out which atoms and bonds are aromatic and reassign appropriately:
        for idx in ring0:
            bond = mol.GetBondWithIdx(idx)
            if (
                bond.GetBeginAtom().GetAtomicNum()
                == bond.GetEndAtom().GetAtomicNum()
                == 6
            ):
                if bond.GetBondType() is AROMATIC:
                    aromatic_bonds_in_ring.append(bond)
                else:
                    break
        else:  # didn't break so all atoms are carbon
            if len(aromatic_bonds_in_ring) == 6:
                aromatic_rings.append(ring0)
                aromatic_bonds.append(aromatic_bonds_in_ring)

    return aromatic_rings, aromatic_bonds


def is_aryl_radical(
    mol,
) -> bool:
    """
    Determine if the molecule only contains aryl radicals, i.e., radical on an aromatic ring.

    Args:
        mol (RDKitMol, RWMol): The molecule to be checked.
    Returns:
        bool: ``True`` if the molecule only contains aryl radicals, ``False`` otherwise.
    """
    total = get_radical_count(mol)
    if not total:
        return False  # not a radical molecule

    num_aryl_rad = len(mol.GetAtomsMatchingQuery(aryl_radical_query))

    return total == num_aryl_rad


# RDKit / RDMC compatible
def is_identical(mol1: "RDKitMol", mol2: "RDKitMol") -> bool:
    """
    Determine whether two molecules are identical. This method assumes two molecules have
    the same composition.

    Args:
        mol1 (RDKitMol, RWMol): The first molecule to be compared.
        mol2 (RDKitMol, RWMol): The second molecule to be compared.
    Returns:
        bool: ``True`` if the two molecules are identical, ``False`` otherwise.
    """
    return mol1.GetSubstructMatch(mol2) == tuple(range(mol1.GetNumAtoms()))


# Pure RDKit
def increment_radical(atom: "Atom"):
    """
    Increment the number of radical electrons on an atom by one.

    Args:
        atom (Atom): The atom whose radical count is to be incremented.
    """
    atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() + 1)


# RDKit / RDMC compatible
def decrement_radical(atom: "Atom"):
    """
    Decrement the number of radical electrons on an atom by one.

    Args:
        atom (Atom): The atom whose radical count is to be decremented.
    """
    new_radical_count = atom.GetNumRadicalElectrons() - 1
    if new_radical_count < 0:
        raise ValueError("Radical count cannot be negative")
    atom.SetNumRadicalElectrons(new_radical_count)


# RDKit / RDMC compatible
def increment_order(bond: "Bond"):
    """
    Increment the bond order of a bond by one.

    Args:
        bond (Bond): The bond whose order is to be incremented.
    """
    bond.SetBondType(bond_order_dicts[bond.GetBondTypeAsDouble() + 1])


# RDKit / RDMC compatible
def decrement_order(bond: "Bond"):
    """
    Decrement the bond order of a bond by one.

    Args:
        bond (Bond): The bond whose order is to be decremented.
    """
    new_order = bond.GetBondTypeAsDouble() - 1
    if new_order <= 0.5:  # Also avoid decrementing aromatic bonds with bond order 1.5
        raise ValueError("Bond order cannot be negative")
    bond.SetBondType(bond_order_dicts[new_order])


# RDKit / RDMC compatible
def update_charge(atom: "Atom", lone_pair: int = 0):
    """
    Update the formal charge of an atom based on its number of lone pairs and bond orders.
    Since there is no way to set the number of lone pairs directly, the formal charge is
    set to reflect the number of lone pairs. The formal charge is set to the number of
    valence electrons minus the number of lone pair electrons, the number of radical electrons,
    and the total bond orders.

    Args:
        atom (Atom): The atom whose formal charge is to be updated.
        lone_pair (int): The number of lone pairs on the atom.
    """
    order = get_total_bond_order(atom)
    charge = (
        PERIODIC_TABLE.GetNOuterElecs(atom.GetAtomicNum())
        - atom.GetNumRadicalElectrons()
        - int(order)
        - 2 * lone_pair
    )
    atom.SetFormalCharge(int(charge))


# RDKit / RDMC compatible
def unset_aromatic_flags(mol):
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


resonance_sanitize_flag = (
    Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
)


# Pure RDKit
def sanitize_resonance_mol(
    mol: "RWMol",
    sanitize_flag=resonance_sanitize_flag,
):
    """
    A helper function to clean up a molecule from resonance structure generation.

    Args:
        mol: The molecule to be sanitized.
        sanitize_flag: The sanitize flag used to sanitize the molecule.
    """
    Chem.SanitizeMol(mol, sanitize_flag)


def _find_shortest_path(start, end, path=None, path_idxs=None):
    """
    Get the shortest path between two atoms in a molecule.

    """
    path = path if path else []
    path_idxs = path_idxs if path else []
    path = path + [start]
    path_idx = path_idxs + [start.GetIdx()]
    if path_idx[-1] == end.GetIdx():
        return path

    shortest = None
    for node in start.GetNeighbors():
        if node.GetIdx() not in path_idx:
            newpath = _find_shortest_path(node, end, path, path_idx)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest


# Pure RDKit
def get_shortest_path(mol, idx1, idx2):
    """
    Get the shortest path between two atoms in a molecule. The RDKit GetShortestPath
    has a very long setup time ~ 0.5ms (on test machine) regardless of the size of the molecule.
    As a comparison, on the same machine, a naive python implementation of DFS (`_find_shortest_path`)
    takes ~0.5 ms for a 100-C normal alkane end to end. Therefore, it make more sense to use a method
    with a shorter setup time though scaling worse for smaller molecules while using GetShortestPath
    for larger molecules.

    Args:
        mol (RDKitMol, RWMol): The molecule to be checked.
        idx1 (int): The index of the first atom.
        idx2 (int): The index of the second atom.

    Returns:
        list: A list of atoms in the shortest path between the two atoms.
    """
    if mol.GetNumHeavyAtoms() > 100:
        return Chem.GetShortestPath(mol, idx1, idx2)

    return _find_shortest_path(
        mol.GetAtomWithIdx(idx1),
        mol.GetAtomWithIdx(idx2)
    )
