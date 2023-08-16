#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Utilities for conformer generation modules
"""

from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from collections import defaultdict
import os
import pickle
import time
from typing import Union

import numpy as np

from rdmc.utils import PERIODIC_TABLE as PT
from rdmc.external.logparser import GaussianLog


SOFTWARE_AVAIL = {}


def timer(func):
    """
    Timer decorator for recording the time of a function.
    """
    def wrapper(self, *args, **kwargs):
        time_start = time.time()
        result = func(self, *args, **kwargs)
        time_end = time.time()
        self._last_run_time = time_end - time_start
        return result
    return wrapper


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


def get_conf_failure_mode(rxn_dir, pruner=True):
    """
    Parse a reaction directory for a TS generation run and extract failure modes (which conformer failed the
    full workflow and for what reason)

    Args:
        rxn_dir (str) Path to the reaction directory.
        pruner (bool: Optional) Whether or not pruner was used during workflow

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
    prune_check_ids = pickle.load(open(prune_check_file, "rb")) if pruner else {i: True for i in range(len(opt_check_ids))}
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


def get_frames_from_freq(log,
                         amplitude: float = 1.0,
                         num_frames: int = 10,
                         weights: Union[bool, np.array] = False):
    """
    Args:
        log (GaussianLog): A gaussian log object with vibrational freq calculated.
        amplitude (float): The amplitude of the motion. If a single value is provided then the guess
                           will be unique (if available). 0.25 will be the default. Otherwise, a list
                           can be provided, and all possible results will be returned.
        num_frames (int): The number of frames in each direction (forward and reverse). Defaults to 10.
        weights (bool or np.array): If ``True``, use the sqrt(atom mass) as a scaling factor to the displacement.
                              If ``False``, use the identity weights. If a N x 1 ``np.array` is provided, then
                              The concern is that light atoms (e.g., H) tend to have larger motions
                              than heavier atoms.

    Returns:
        np.array: The atomic numbers as an 1D array
        np.array: The 3D geometries at each frame as a 3D array (number of frames x 2 + 1, number of atoms, 3)
    """
    assert log.num_neg_freqs == 1

    equ_xyz = log.converged_geometries[-1]
    disp = log.cclib_results.vibdisps[0]
    amp_factors = np.linspace(-amplitude, amplitude, 2 * num_frames + 1)

    # Generate weights
    if isinstance(weights, bool) and weights:
        atom_masses = np.array([PT.GetAtomicWeight(int(num)) for num in log.cclib_results.atomnos]).reshape(-1, 1)
        weights = np.sqrt(atom_masses)
    elif isinstance(weights, bool) and not weights:
        weights = np.ones((equ_xyz.shape[0], 1))

    xyzs = equ_xyz - np.einsum('i,jk->ijk', amp_factors, weights * disp)

    return log.cclib_results.atomnos, xyzs


def convert_log_to_mol(log_path: str,
                       amplitude: float = 1.0,
                       num_frames: int = 10,
                       weights: Union[bool, np.array] = False):
    """
    Args:
        log_path (str): The path to the log file.
        amplitude (float): The amplitude of the motion. If a single value is provided then the guess
                           will be unique (if available). 0.25 will be the default. Otherwise, a list
                           can be provided, and all possible results will be returned.
        num_frames (int): The number of frames in each direction (forward and reverse). Defaults to 10.
        weights (bool or np.array): If ``True``, use the sqrt(atom mass) as a scaling factor to the displacement.
                              If ``False``, use the identity weights. If a N x 1 ``np.array` is provided, then
                              The concern is that light atoms (e.g., H) tend to have larger motions
                              than heavier atoms.
    """
    glog = GaussianLog(log_path)

    try:
        assert glog.success
        assert glog.is_ts
        assert glog.num_neg_freqs == 1
    except AssertionError:
        return None

    # Get TS mol object and construct geometries as numpy arrays for all frames
    mol = glog.get_mol(converged=True, embed_conformers=False, sanitize=False)
    _, xyzs = get_frames_from_freq(glog, amplitude=amplitude, num_frames=num_frames, weights=weights)

    # Embed geometries to the mol object for output
    mol.EmbedMultipleNullConfs(xyzs.shape[0])
    [mol.SetPositions(xyzs[i, :, :], id=i) for i in range(xyzs.shape[0])]

    return mol
