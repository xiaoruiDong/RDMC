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
from ase.io.trajectory import Trajectory


def run_sella_xtb_opt(rdmc_mol, confId=0, fmax=1e-3, steps=1000, save_dir=None):
    temp_dir = tempfile.mkdtemp() if not save_dir else save_dir
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
        opt.run(fmax, steps)

    opt_rdmc_mol = rdmc_mol.Copy()
    opt_rdmc_mol.GetConformer(confId).SetPositions(opt.atoms.positions)

    if not save_dir:
        rmtree(temp_dir)

    return opt_rdmc_mol
