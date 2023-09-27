#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.utils import PERIODIC_TABLE


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


