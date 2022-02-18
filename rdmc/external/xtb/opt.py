#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
xTB optimization and single point routines.
Taken from https://github.com/josejimenezluna/delfta/blob/f33dbe4fc2b860cef287880e5678829cebc8b94d/delfta/xtb.py
"""

import json
import os
from shutil import rmtree
import subprocess
import tempfile
import numpy as np

from rdmc import RDKitMol
from rdmc.external.xtb.utils import (
    ATOM_ENERGIES_XTB,
    ATOMNUM_TO_ELEM,
    AU_TO_DEBYE,
    EV_TO_HARTREE,
    UTILS_PATH,
    XTB_BINARY,
    XTB_ENV,
)

XTB_INPUT_FILE = os.path.join(UTILS_PATH, "xtb.inp")


def read_xtb_json(json_file, mol):
    """Reads JSON output file from xTB.
    Parameters
    ----------
    json_file : str
        path to output file
    mol : pybel molecule object
        molecule object, needed to compute atomic energy
    Returns
    -------
    dict
        dictionary of xTB properties
    """

    with open(json_file, "r") as f:
        data = json.load(f)
    E_homo, E_lumo = get_homo_and_lumo_energies(data)
    atoms = [ATOMNUM_TO_ELEM[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    atomic_energy = sum([ATOM_ENERGIES_XTB[atom] for atom in atoms])
    props = {
        "E_form": data["total energy"] - atomic_energy,  # already in Hartree
        "E_homo": E_homo * EV_TO_HARTREE,
        "E_lumo": E_lumo * EV_TO_HARTREE,
        "E_gap": data["HOMO-LUMO gap/eV"] * EV_TO_HARTREE,
        "dipole": np.linalg.norm(data["dipole"]) * AU_TO_DEBYE,
        "charges": data["partial charges"],
    }
    return props


def get_homo_and_lumo_energies(data):
    """Extracts HOMO and LUMO energies.
    Parameters
    ----------
    data : dict
        dictionary from xTB JSON output
    Returns
    -------
    tuple(float)
        HOMO/LUMO energies in eV
    Raises
    ------
    ValueError
        in case of unpaired electrons (not supported)
    """
    if data["number of unpaired electrons"] != 0:
        raise ValueError("Unpaired electrons are not supported.")
    num_occupied = (
            np.array(data["fractional occupation"]) > 1e-6
    ).sum()  # number of occupied orbitals; accounting for occassional very small values
    E_homo = data["orbital energies/eV"][num_occupied - 1]  # zero-indexing
    E_lumo = data["orbital energies/eV"][num_occupied]
    return E_homo, E_lumo


def get_wbo(wbo_file):
    """Reads WBO output file from xTB and generates a dictionary with the results.
    Parameters
    ----------
    wbo_file : str
        path to xTB wbo output file
    Returns
    -------
    list
        list with Wiberg bond orders (only covalent bonds)
    """
    with open(wbo_file, "r") as f:
        lines = [elem.rstrip("\n") for elem in f.readlines()]
    tmp = [
        [int(line[:12]) - 1, int(line[12:24]) - 1, float(line[24:])] for line in lines
    ]
    wbo_dict = {f"{min((a1, a2))}-{max((a1, a2))}": wbo for a1, a2, wbo in tmp}
    return wbo_dict


def run_xtb_calc(mol, opt=False, return_optmol=False, method="gfn2", level="normal"):
    """Runs xTB single-point calculation with optional geometry optimization.
    Parameters
    ----------
    mol : pybel molecule object
        assumes hydrogens are present
    opt : bool, optional
        Whether to optimize the geometry, by default False
    return_optmol : bool, optional
        Whether to return the optimized molecule, in case optimization was requested, by default False
    Returns
    -------
    dict
        Molecular properties as computed by GFN2-xTB (formation energy, HOMO/LUMO/gap energies, dipole, atomic charges)
    Raises
    ------
    ValueError
        If xTB calculation yield a non-zero return code.
    """

    if return_optmol and not opt:
        LOGGER.info(
            "Can't have `return_optmol` set to True with `opt` set to False. Setting the latter to True now."
        )
        opt = True

    xtb_command = "--opt" if opt else ""
    method = "--" + method
    temp_dir = tempfile.mkdtemp()
    logfile = os.path.join(temp_dir, "xtb.log")
    xtb_out = os.path.join(temp_dir, "xtbout.json")
    xtb_wbo = os.path.join(temp_dir, "wbo")

    sdf_path = os.path.join(temp_dir, "mol.sdf")
    mol.ToSDFFile(sdf_path)

    with open(logfile, "w") as f:
        xtb_run = subprocess.run(
            [
                XTB_BINARY,
                sdf_path,
                xtb_command,
                method,
                level,
                "--chrg",
                str(mol.GetFormalCharge()),
                "--wbo",
                "--json",
                "true",
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=temp_dir,
            env=XTB_ENV,
        )
    if xtb_run.returncode != 0:
        # error_out = os.path.join(temp_dir, "xtb.log")
        rmtree(temp_dir)
        raise ValueError(f"xTB calculation failed.")

    else:
        if method == "--gff":
            opt_mol = RDKitMol.FromFile(os.path.join(temp_dir, "xtbopt.sdf"))[0]
            rmtree(temp_dir)
            return (None, opt_mol) if return_optmol else None
        props = read_xtb_json(xtb_out, mol)
        if return_optmol:
            opt_mol = RDKitMol.FromFile(os.path.join(temp_dir, "xtbopt.sdf"))[0]
        props.update({"wbo": get_wbo(xtb_wbo)})
        rmtree(temp_dir)
        return (props, opt_mol) if return_optmol else props
