#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

from rdmc.utils import PERIODIC_TABLE

from rdkit.Chem.rdchem import BondType


AROMATIC = BondType.AROMATIC


electronegativity = {1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
                     11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
                     19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55,
                     26: 1.83, 27: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18,
                     34: 2.55, 35: 2.96, 53: 2.66}


bond_order_dicts = {
    1: BondType.SINGLE,
    2: BondType.DOUBLE,
    3: BondType.TRIPLE,
    4: BondType.QUADRUPLE,
    5: BondType.QUINTUPLE,
    1.5: BondType.ONEANDAHALF,
    2.5: BondType.TWOANDAHALF,
    3.5: BondType.THREEANDAHALF,
    4.5: BondType.FOURANDAHALF,
    5.5: BondType.FIVEANDAHALF,
}

def get_electronegativity(atom):
    return electronegativity.get(atom.GetAtomicNum(), 1.0)


def get_total_bond_order(atom):
    return sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()])


def get_lone_pair(atom):
    """
    Helper function
    Returns the lone pair of an atom
    """
    atomic_num = atom.GetAtomicNum()
    if atomic_num == 1:
        return 0
    order = get_total_bond_order(atom)
    return (PERIODIC_TABLE.GetNOuterElecs(atomic_num) - atom.GetNumRadicalElectrons() - atom.GetFormalCharge() - int(order)) / 2


def in_path(atom, path):
    return atom.GetIdx() in [a.GetIdx() for a in path]


def get_charge_span(mol):
    abs_net_charge = abs(mol.GetFormalCharge())
    sum_of_abs_charges = sum([abs(atom.GetFormalCharge()) for atom in mol.GetAtoms()])
    return (sum_of_abs_charges - abs_net_charge) / 2


def is_reactive(mol):
    try:
        return mol.reactive
    except AttributeError:
        set_reactive(mol, True)  # by default, assume it is reactive
    return mol.reactive


def is_radical(mol):
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return True
    return False


def is_aromatic(mol):
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            return True
    return False


def is_cyclic(mol):
    for atom in mol.GetAtoms():
        if atom.IsInRing():
            return True
    return False


def set_reactive(mol, reactive):
    mol.reactive = reactive


def get_order_str(bond):
    bond_type = bond.GetBondType()
    if bond_type == 1:
        return 'S'
    elif bond_type == 1.5:
        return 'B'
    elif bond_type == 2:
        return 'D'
    elif bond_type == 3:
        return 'T'
    elif bond_type == 4:
        return 'Q'
    else:
        return '?'


def get_sorting_label(atom):
    return -1


def get_relevant_cycles(mol):
    # TODO: improve this function with the actual relevant cycle algorithm
    # This is a temporary solution works for simple cases
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
            if bond.GetBeginAtom().GetAtomicNum() == bond.GetEndAtom().GetAtomicNum() == 6:
                if bond.GetBondType() is AROMATIC:
                    aromatic_bonds_in_ring.append(bond)
                else:
                    break
        else:  # didn't break so all atoms are carbon
            if len(aromatic_bonds_in_ring) == 6:
                aromatic_rings.append(ring0)
                aromatic_bonds.append(aromatic_bonds_in_ring)

    return aromatic_rings, aromatic_bonds


def get_radical_count(mol):
    """
    Return the total number of radical electrons on all atoms in the
    molecule. In this function, monoradical atoms count as one, biradicals
    count as two, etc.
    """
    radicals = 0
    for atom in mol.GetAtoms():
        radicals += atom.GetNumRadicalElectrons()
    return radicals


def is_aryl_radical(mol, aromatic_rings=None):
    """
    Return ``True`` if the molecule only contains aryl radicals,
    i.e., radical on an aromatic ring, or ``False`` otherwise.
    If no ``aromatic_rings`` provided, aromatic rings will be searched in-place,
    and this process may involve atom order change by default. Set ``save_order`` to
    ``True`` to force the atom order unchanged.
    """
    if aromatic_rings is None:
        aromatic_rings = get_aromatic_rings(mol)[0]

    total = get_radical_count(mol)
    if not total:
        return False  # not a radical molecule
    aromatic_atoms = set([atom for atom in itertools.chain.from_iterable(aromatic_rings)])
    aryl = sum([mol.GetAtomWithIdx(atom_idx).GetNumRadicalElectrons() for atom_idx in aromatic_atoms])

    return total == aryl


def is_identical(mol1, mol2):
    return mol1.GetSubstructMatch(mol2) == tuple(range(mol1.GetNumAtoms()))


def increment_radical(atom):
    atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() + 1)


def decrement_radical(atom):
    new_radical_count = atom.GetNumRadicalElectrons() - 1
    if new_radical_count < 0:
        raise ValueError('Radical count cannot be negative')
    atom.SetNumRadicalElectrons(new_radical_count)


def increment_order(bond):
    bond.SetBondType(bond_order_dicts[bond.GetBondTypeAsDouble() + 1])

def decrement_order(bond):
    new_order = bond.GetBondTypeAsDouble() - 1
    if new_order <= 0.5:
        raise ValueError('Bond order cannot be negative')
    bond.SetBondType(bond_order_dicts[new_order])


def update_charge(atom, lone_pair):
    order = get_total_bond_order(atom)
    charge = PERIODIC_TABLE.GetNOuterElecs(atom.GetAtomicNum()) - atom.GetNumRadicalElectrons() - int(order) - 2 * lone_pair
    atom.SetFormalCharge(int(charge))
