#!/usr/bin/env python3
#-*- coding: utf-8 -*-

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

EV_TO_HARTREE = 1 / 27.211386245988  # https://physics.nist.gov/cgi-bin/cuu/Value?hrev (04.06.21)
AU_TO_DEBYE = 1 / 0.3934303  # https://en.wikipedia.org/wiki/Debye (04.06.21)
WBO_CUTOFF = 0.05

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

# Job creation and parsing

COLUMN_ORDER = {
    "E_form": 0,
    "E_homo": 1,
    "E_lumo": 2,
    "E_gap": 3,
    "dipole": 4,
    "charges": 5,
}

METHOD_DICT = {
    'gfn0': '--gfn 0',
    'gfn1': '--gfn 1',
    'gfn2': '--gfn 2',
    'gfnff': '--gfnff',
}

FILE_NAME_DICT = {
    'log': 'xtb.log',
    'out': 'xtbout.json',
    'wbo': 'wbo',
    'g98': 'g98.out',
    'ts': 'xtbpath_ts.xyz',
    'coord': 'mol.sdf',
    'coord_xyz': 'mol.xyz',
    'end_coord': 'end.sdf',
    'opt': 'xtbopt.sdf',
    'restart': 'xtbrestart',
}
