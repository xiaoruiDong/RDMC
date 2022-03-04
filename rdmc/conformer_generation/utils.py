#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Utilities for conformer generation modules
"""

from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


def mol_to_dict(mol, iter=None):
    mol_data = []
    for c_id in range(mol.GetNumConformers()):

        conf = mol.Copy().GetConformer(c_id)
        new_mol = mol.Copy()
        new_mol.RemoveAllConformers()
        new_mol._mol.AddConformer(conf.ToConformer(), assignId=True)
        conf.SetOwningMol(new_mol)
        positions = conf.GetPositions()
        mol_data.append({"positions": positions,
                         "conf": conf})
        if iter:
            mol_data[c_id].update({"iter": iter})
    return mol_data


def dict_to_mol(mol_data):
    mol = mol_data[0]["conf"].GetOwningMol().Copy()
    mol.RemoveAllConformers()
    [mol._mol.AddConformer(c["conf"].ToConformer(), assignId=True) for c in mol_data]
    return mol


def cluster_confs(mol, cutoff=1.0):
    rmsmat = AllChem.GetConformerRMSMatrix(mol.ToRWMol(), prealigned=False)
    num = mol.GetNumConformers()
    clusters = Butina.ClusterData(rmsmat, num, cutoff, isDistData=True, reordering=True)
    confs_to_keep = [c[0] for c in clusters]

    updated_mol = mol.Copy()
    updated_mol.RemoveAllConformers()
    [updated_mol.AddConformer(c.ToConformer(), assignId=True) for c in mol.GetConformers(confs_to_keep)]

    return updated_mol
