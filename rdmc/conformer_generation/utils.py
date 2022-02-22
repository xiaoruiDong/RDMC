#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Utilities for conformer generation modules
"""


def mol_to_dict(mol, iter=None):
    mol_data = []
    for c_id in range(mol.GetNumConformers()):
        conf = mol.Copy().GetConformer(c_id)
        positions = conf.GetPositions()
        mol_data.append({"positions": positions,
                         "conf": conf})
        if iter:
            mol_data[c_id].update({"iter": iter})
    return mol_data


def dict_to_mol(mol_data):
    mol = mol_data[0]["conf"].GetOwningMol().Copy()
    [mol._mol.AddConformer(c["conf"].ToConformer(), assignId=True) for c in mol_data]
    return mol
