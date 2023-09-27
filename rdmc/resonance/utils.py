#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.utils import PERIODIC_TABLE


electronegativity = {1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
                     11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
                     19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55,
                     26: 1.83, 27: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18,
                     34: 2.55, 35: 2.96, 53: 2.66}


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


def is_aromatic(mol):
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            return True


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
