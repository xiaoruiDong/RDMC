#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Utilities for conformer generation modules
"""

from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


def mol_to_dict(mol,
                copy: bool = True,
                iter: int = None):
    """
    Convert a molecule to a dictionary that stores its conformers object, atom coordinates,
    and iteration numbers for a certain calculation (optional).

    Args:
        mol ('RDKitMol'): An RDKitMol object.
        copy (bool, optional): Use a copy of the molecule to process data. Defaults to True.
        iter (int, optional): Number of iterations. Defaults to None.

    Returns:
        list: mol data as a list of dict; each dict corresponds to a conformer.
    """
    mol_data = []
    if copy:
        mol = mol.Copy()
    for c_id in range(mol.GetNumConformers()):
        conf = mol.GetConformer(id=c_id)
        positions = conf.GetPositions()
        mol_data.append({"positions": positions,
                         "conf": conf})
        if iter is not None:
            mol_data[c_id].update({"iter": iter})
    return mol_data


def dict_to_mol(mol_data):
    mol = mol_data[0]["conf"].GetOwningMol().Copy(quickCopy=True)
    [mol.AddConformer(c["conf"].ToConformer(), assignId=True) for c in mol_data]
    return mol


def cluster_confs(mol, cutoff=1.0):
    rmsmat = AllChem.GetConformerRMSMatrix(mol.ToRWMol(), prealigned=False)
    num = mol.GetNumConformers()
    clusters = Butina.ClusterData(rmsmat, num, cutoff, isDistData=True, reordering=True)
    confs_to_keep = [c[0] for c in clusters]

    updated_mol = mol.Copy(quickCopy=True)
    [updated_mol.AddConformer(c.ToConformer(), assignId=True) for c in mol.GetConformers(confs_to_keep)]

    return updated_mol
