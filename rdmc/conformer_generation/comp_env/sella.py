#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transition state initial guess optimization with ASE optimizers (requires xtb-python).
"""

import os
import io
from contextlib import redirect_stdout
from shutil import rmtree
import tempfile

from rdmc.conformer_generation.comp_env.software import register_module

with register_module('sella'):
    from sella import Sella

with register_module('ase'):
    from ase.calculators.orca import ORCA

with register_module('xtb-python'):
    from xtb.ase.calculator import XTB

import pandas as pd


def run_sella_opt(
    rdmc_mol,
    confId=0,
    fmax=1e-3,
    steps=1000,
    save_dir=None,
    method="GFN2-xTB",
    copy_attrs=None,
):
    temp_dir = tempfile.mkdtemp() if not save_dir else save_dir
    trajfile = os.path.join(temp_dir, "ts.traj")
    logfile = os.path.join(temp_dir, "ts.log")
    orca_name = os.path.join(temp_dir, "ts")

    coords = rdmc_mol.GetConformer(confId).GetPositions()
    numbers = rdmc_mol.GetAtomicNumbers()
    atoms = rdmc_mol.ToAtoms()

    # set calculator; use xtb-python for xtb and orca for everything else
    if method == "GFN2-xTB":
        atoms.calc = XTB(method="GFN2-xTB")
    elif method == "AM1":
        atoms.calc = ORCA(label=orca_name, orcasimpleinput="AM1")
    elif method == "PM3":
        atoms.calc = ORCA(label=orca_name, orcasimpleinput="PM3")
    else:
        raise NotImplementedError(
            f"Method ({method}) is not supported with Sella. Only `GFN2-xTB`, `AM1`, and `PM3` "
            f"are supported."
        )

    with io.StringIO() as buf, redirect_stdout(buf):
        opt = Sella(
            atoms,
            logfile=logfile,
            trajectory=trajfile,
        )
        opt.run(fmax, steps)

    opt_rdmc_mol = rdmc_mol.Copy(copy_attrs=copy_attrs)
    opt_rdmc_mol.SetPositions(opt.atoms.positions, confId=confId)
    energy = float(pd.read_csv(logfile).iloc[-1].values[0].split()[3])
    opt_rdmc_mol.energy.update({confId: energy})

    if not save_dir:
        rmtree(temp_dir)

    return opt_rdmc_mol
