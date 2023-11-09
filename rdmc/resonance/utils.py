#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

from typing import List, Optional

from rdkit import Chem


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


# Pure RDKit
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


# Pure RDKit
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


# Pure RDKit
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


def get_charge_span(mol: "RDKitMol") -> float:
    """
    Get the charge span of a molecule.

    Args:
        mol (RDKitMol): The molecule whose charge span is to be returned.

    Returns:
        float: The charge span of the molecule.
    """
    abs_net_charge = abs(mol.GetFormalCharge())
    sum_of_abs_charges = sum([abs(atom.GetFormalCharge()) for atom in mol.GetAtoms()])
    return (sum_of_abs_charges - abs_net_charge) / 2


# Pure RDKit
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


def get_sorting_label(atom) -> int:
    return -1


def set_reactive(mol, reactive: bool):
    mol.reactive = reactive


def is_reactive(mol) -> bool:
    try:
        return mol.reactive
    except AttributeError:
        set_reactive(mol, True)  # by default, assume it is reactive
    return mol.reactive


# Pure RDKit
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


# Pure RDKit
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


# Pure RDKit
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


# Pure RDKit
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
    elif bond_type == 1.5:
        return "B"
    elif bond_type == 2:
        return "D"
    elif bond_type == 3:
        return "T"
    elif bond_type == 4:
        return "Q"
    else:
        return "?"


# Pure RDKit
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
    aromatic_rings: Optional[list] = None,
) -> bool:
    """
    Determine if the molecule only contains aryl radicals, i.e., radical on an aromatic ring.
    If no ``aromatic_rings`` provided, aromatic rings will be searched in-place,
    and this process may involve atom order change by default.

    Args:
        mol (RDKitMol, RWMol): The molecule to be checked.
        aromatic_rings (list, optional): A list of aromatic rings, each represented as a list of bond indices.

    Returns:
        bool: ``True`` if the molecule only contains aryl radicals, ``False`` otherwise.
    """
    total = get_radical_count(mol)
    if not total:
        return False  # not a radical molecule

    if aromatic_rings is None:
        aromatic_rings = get_aromatic_rings(mol)[0]

    # TODO: This is currently wrong as aromatic_rings are bond indices, not atom indices
    aromatic_atoms = set(
        [atom for atom in itertools.chain.from_iterable(aromatic_rings)]
    )
    aryl = sum(
        [
            mol.GetAtomWithIdx(atom_idx).GetNumRadicalElectrons()
            for atom_idx in aromatic_atoms
        ]
    )

    return total == aryl


# Pure RDKit
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


# Pure RDKit
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


# Pure RDKit
def increment_order(bond: "Bond"):
    """
    Increment the bond order of a bond by one.

    Args:
        bond (Bond): The bond whose order is to be incremented.
    """
    bond.SetBondType(bond_order_dicts[bond.GetBondTypeAsDouble() + 1])


# Pure RDKit
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


# Pure RDKit
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
