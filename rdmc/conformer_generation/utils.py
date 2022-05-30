#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Utilities for conformer generation modules
"""

from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

import os
import pickle
import numpy as np
from collections import defaultdict

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


def get_conf_failure_mode(rxn_dir):
    """
    Parse a reaction directory for a TS generation run and extract failure modes (which conformer failed the
    full workflow and for what reason)

    Args:
        rxn_dir (str) Path to the reaction directory.

    Returns:
        failure_dict ('dict'): Dictionary of conformer ids mapped to the corresponding failure mode.
    """

    failure_modes = {
        0: "opt",
        1: "prune",
        2: "freq",
        3: "irc",
        4: "workflow",
        5: "none",
    }

    opt_check_file = os.path.join(rxn_dir, "opt_check_ids.pkl")
    freq_check_file = os.path.join(rxn_dir, "freq_check_ids.pkl")
    prune_check_file = os.path.join(rxn_dir, "prune_check_ids.pkl")
    irc_check_file = os.path.join(rxn_dir, "irc_check_ids.pkl")
    workflow_check_file = os.path.join(rxn_dir, "workflow_check_ids.pkl")

    opt_check_ids = pickle.load(open(opt_check_file, "rb"))
    prune_check_ids = pickle.load(open(prune_check_file, "rb"))
    freq_check_ids = pickle.load(open(freq_check_file, "rb"))
    irc_check_ids = pickle.load(open(irc_check_file, "rb"))
    workflow_check_ids = pickle.load(open(workflow_check_file, "rb"))

    all_checks = defaultdict(list)
    for d in [opt_check_ids, prune_check_ids, freq_check_ids, irc_check_ids, workflow_check_ids]:
        for k, v in d.items():
            all_checks[k].append(v)

    all_checks = np.concatenate([np.array([*all_checks.values()]), np.array([[False]] * len(all_checks))], axis=1)
    modes = np.argmax(~all_checks, axis=1)
    failure_dict = {i: failure_modes[m] for i, m in enumerate(modes)}

    return failure_dict
