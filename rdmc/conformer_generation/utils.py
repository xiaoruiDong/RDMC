#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Utilities for conformer generation modules
"""

from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


def mol_to_dict(mol,
                copy: bool = True,
                iter: int = None,
                conf_copy_attrs: list = None):
    """
    Convert a molecule to a dictionary that stores its conformers object, atom coordinates,
    and iteration numbers for a certain calculation (optional).

    Args:
        mol ('RDKitMol'): An RDKitMol object.
        copy (bool, optional): Use a copy of the molecule to process data. Defaults to True.
        iter (int, optional): Number of iterations. Defaults to None.
        conf_copy_attrs (list, optional): Conformer-level attributes to copy to the dictionary.

    Returns:
        list: mol data as a list of dict; each dict corresponds to a conformer.
    """
    mol_data = []
    if copy:
        mol = mol.Copy(copy_attrs=conf_copy_attrs)
    if conf_copy_attrs is None:
        conf_copy_attrs = []
    for c_id in range(mol.GetNumConformers()):
        conf = mol.GetConformer(id=c_id)
        positions = conf.GetPositions()
        mol_data.append({"positions": positions,
                         "conf": conf})
        if iter is not None:
            mol_data[c_id].update({"iter": iter})
        for attr in conf_copy_attrs:
            mol_data[c_id].update({attr: getattr(mol, attr)[c_id]})
    return mol_data


def dict_to_mol(mol_data,
                conf_copy_attrs: list = None):
    """
    Convert a dictionary that stores its conformers object, atom coordinates,
    and conformer-level attributes to an RDKitMol. The method assumes that the
    first conformer's owning mol contains the conformer-level attributes, which
    are extracted through the Copy function (this should be the case if the
    dictionary was generated with the mol_to_dict function).

    Args:
        mol_data (list) List containing dictionaries of data entries for each conformer.
        conf_copy_attrs (list, optional): Conformer-level attributes to copy to the mol.

    Returns:
        mol ('RDKitMol'): An RDKitMol object.
    """
    if conf_copy_attrs is None:
        conf_copy_attrs = []
    mol = mol_data[0]["conf"].GetOwningMol().Copy(quickCopy=True, copy_attrs=conf_copy_attrs)
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
