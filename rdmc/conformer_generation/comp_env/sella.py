#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transition state initial guess optimization with ASE optimizers (requires xtb-python).
"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile

from rdmc.conformer_generation.comp_env.software import try_import
from rdmc.conformer_generation.comp_env.ase import ORCA
from rdmc.conformer_generation.comp_env.xtb import xtb_calculator
from rdmc.conformer_generation.comp_env.software import has_binary, package_available
import pandas as pd


try_import("sella.Sella", namespace=globals())  # from sella import Sella

sella_available = package_available["sella"] and (
    (has_binary("xtb") and package_available["xtb-python"])
    or (has_binary("orca") and package_available["ase"])
)


def run_sella_opt(
    mol: "RDKitMol",
    conf_id: int = 0,
    method: str = "GFN2-xTB",
    fmax: float = 1e-3,
    steps: int = 1000,
    save_dir: Optional[str] = None,
) -> tuple:
    work_dir = Path(save_dir or tempfile.mkdtemp())
    trajfile = work_dir / "ts.traj"
    logfile = work_dir / "ts.log"
    orca_name = work_dir / "ts"

    atoms = mol.ToAtoms(confId=conf_id)

    # set calculator; use xtb-python for xtb and orca for everything else
    if method.lower() == "GFN2-xTB":
        atoms.calc = xtb_calculator(method="GFN2-xTB")
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

    pos = opt.atoms.positions
    energy = float(pd.read_csv(logfile).iloc[-1].values[0].split()[3])

    return pos, True, energy, None  # positions, success, energy, freq
