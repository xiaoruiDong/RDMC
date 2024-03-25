#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xTB optimization and single point routines.
Taken from https://github.com/josejimenezluna/delfta/blob/f33dbe4fc2b860cef287880e5678829cebc8b94d/delfta/utils.py
"""

import os
import logging
import numpy as np

# Path handling shortcuts

ROOT_PATH = os.path.dirname(__file__)
TS_PATH_INP = os.path.join(ROOT_PATH, "ts_path.inp")
UTILS_PATH = os.path.join(ROOT_PATH, "utils")
XTB_BINARY = os.path.join(os.environ.get("CONDA_PREFIX"), "bin", "xtb")
CREST_BINARY = os.path.join(os.environ.get("CONDA_PREFIX"), "bin", "crest")
XTB_GAUSSIAN_PL = os.path.join(ROOT_PATH, "xtb_gaussian.pl")

XTB_ENV = {
    "OMP_STACKSIZE": "1G",
    "OMP_NUM_THREADS": "1",
    "OMP_MAX_ACTIVE_LEVELS": "1",
    "MKL_NUM_THREADS": "1",
}

# Constants

EV_TO_HARTREE = (
    1 / 27.211386245988
)  # https://physics.nist.gov/cgi-bin/cuu/Value?hrev (04.06.21)

AU_TO_DEBYE = 1 / 0.3934303  # https://en.wikipedia.org/wiki/Debye (04.06.21)

WBO_CUTOFF = 0.05

ELEM_TO_ATOMNUM = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}

ATOMNUM_TO_ELEM = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}

ATOM_ENERGIES_XTB = {
    "H": -0.393482763936,
    "C": -1.793296371365,
    "O": -3.767606950376,
    "N": -2.605824161279,
    "F": -4.619339964238,
    "S": -3.146456870402,
    "P": -2.374178794732,
    "Cl": -4.482525134961,
    "Br": -4.048339371234,
    "I": -3.779630263390,
}


COLUMN_ORDER = {
    "E_form": 0,
    "E_homo": 1,
    "E_lumo": 2,
    "E_gap": 3,
    "dipole": 4,
    "charges": 5,
}


def preds_to_lists(preds):
    preds_list = {}
    for key, val in preds.items():
        if isinstance(val, list):
            preds_list[key] = [elem.tolist() for elem in val]
        elif isinstance(val, np.ndarray):
            preds_list[key] = val.tolist()
    return preds_list


def get_bond_aidxs(mol):
    atom_idxs = []
    for i in range(mol.OBMol.NumBonds()):
        begin_idx = mol.OBMol.GetBondById(i).GetBeginAtomIdx() - 1
        end_idx = mol.OBMol.GetBondById(i).GetEndAtomIdx() - 1
        atom_idxs.append((min(begin_idx, end_idx), max(begin_idx, end_idx)))
    return atom_idxs
