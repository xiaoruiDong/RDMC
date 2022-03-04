#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Transition state initial guess optimization with ASE optimizers (requires xtb-python).
"""

import os
import io
from contextlib import redirect_stdout
from shutil import rmtree
import tempfile
from ase import Atoms
from xtb.ase.calculator import XTB
from sella import Sella
import pandas as pd


def run_sella_xtb_opt(rdmc_mol, confId=0):
    temp_dir = tempfile.mkdtemp()
    trajfile = os.path.join(temp_dir, "ts.traj")
    logfile = os.path.join(temp_dir, "ts.log")

    coords = rdmc_mol.GetConformer(confId).GetPositions()
    numbers = rdmc_mol.GetAtomicNumbers()

    atoms = Atoms(positions=coords, numbers=numbers)
    atoms.calc = XTB()

    with io.StringIO() as buf, redirect_stdout(buf):
        opt = Sella(
            atoms,
            logfile=logfile,
            trajectory=trajfile,
        )
        opt.run(1e-3, 1000)

    opt_rdmc_mol = rdmc_mol.Copy()
    opt_rdmc_mol.GetConformer(confId).SetPositions(opt.atoms.positions)

    prop_headers = ["step", "time", "energy", "fmax", "cmax", "trust", "rho"]
    with open(logfile) as f:
        data = f.readlines()
    props = pd.DataFrame([l.split()[1:] for l in data[1:]], columns=prop_headers, dtype=float).set_index("step").to_dict()
    rmtree(temp_dir)

    return props, opt_rdmc_mol
