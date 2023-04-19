#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
xTB optimization and single point routines.
Modified based on https://github.com/josejimenezluna/delfta/blob/f33dbe4fc2b860cef287880e5678829cebc8b94d/delfta/xtb.py
by Lagnajit Pattanaik and Xiaorui Dong
"""

import json
import os
import os.path as osp
from shutil import rmtree
import subprocess
import tempfile
from typing import Optional

import numpy as np

from rdmc import RDKitMol
from rdmc.external.xtb_tools.utils import (
    ATOM_ENERGIES_XTB,
    AU_TO_DEBYE,
    EV_TO_HARTREE,
    METHOD_DICT,
    FILE_NAME_DICT,
    TS_PATH_INP,
    XTB_BINARY,
    XTB_ENV,
)


def read_xtb_json(json_file: str,
                  mol: 'RDkitMol',
                  ) -> dict:
    """
    Reads JSON output file from xTB.

    Args:
        json_file (str): path to output file
        mol (RDKitMol): molecule object, needed to compute atomic energy

    Returns:
        dict: dictionary of xTB properties
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    E_homo, E_lumo = get_homo_and_lumo_energies(data)
    atomic_energy = sum([ATOM_ENERGIES_XTB[atom]
                         for atom in mol.GetElementSymbols()])
    props = {
        "E_form": data["total energy"] - atomic_energy,  # already in Hartree
        "E_homo": E_homo * EV_TO_HARTREE,
        "E_lumo": E_lumo * EV_TO_HARTREE,
        "E_gap": data["HOMO-LUMO gap/eV"] * EV_TO_HARTREE,
        "dipole": np.linalg.norm(data["dipole"]) * AU_TO_DEBYE,
        "charges": data["partial charges"],
        }
    return props


def get_homo_and_lumo_energies(data: dict):
    """
    Extracts HOMO and LUMO energies.

    Args:
        data (dict): dictionary from xTB JSON output

    Returns:
        tuple(float): HOMO/LUMO energies in eV

    Raises:
        ValueError: in case of unpaired electrons (not supported)
    """
    if data["number of unpaired electrons"] != 0:
        print("Unpaired electrons are not supported for HOMO/LUMO data extraction.")
        return np.nan, np.nan
    num_occupied = (np.array(data["fractional occupation"]) > 1e-6).sum()  # number of occupied orbitals; accounting for occassional very small values
    E_homo = data["orbital energies/eV"][num_occupied - 1]  # zero-indexing
    E_lumo = data["orbital energies/eV"][num_occupied]
    return E_homo, E_lumo


def get_wbo(wbo_file: str):
    """
    Reads WBO output file from xTB and generates a dictionary with the results.

    Args:
        wbo_file (str): path to xTB wbo output file.

    Returns:
        list: a list of Wiberg bond orders (only covalent bonds)
    """
    with open(wbo_file, "r") as f:
        lines = [elem.rstrip("\n") for elem in f.readlines()]
    tmp = [
        [int(line[:12]) - 1,
         int(line[12:24]) - 1,
         float(line[24:])
         ] for line in lines
        ]
    wbo_dict = {f"{min((a1, a2))}-{max((a1, a2))}": wbo
                for a1, a2, wbo in tmp}
    return wbo_dict


def run_xtb_calc(mol: 'RDKitMol',
                 conf_id: int = 0,
                 uhf: int = 0,
                 charge: int = 0,
                 job: str = "",
                 method: str = "gfn2",
                 level: str = "normal",
                 end_mol: Optional['RDKitMol'] = None,
                 end_conf_id: int = 0,
                 save_dir: Optional[str] = None,
                 env_var: dict = XTB_ENV,
                 ):
    """
    Runs xTB single-point calculation with optional geometry optimization.

    Args:
        mol (RDKitMol): molecule object, assumes hydrogens are present
        conf_id (int, optional): conformer ID to use. Defaults to 0.
        uhf (int, optional): UHF spin state. Defaults to 0.
        charge (int, optional): charge of the molecule. Defaults to 0.
        job (str, optional): xTB job type. Defaults to "" for single-point calculation.
        method (str, optional): xTB method. Defaults to "gfn2".
        level (str, optional): xTB level of theory. Defaults to "normal".
        end_mol (RDKitMol, optional): end molecule for path calculation. Defaults to None.
        end_conf_id (int, optional): end conformer ID for path calculation. Defaults to 0.
        save_dir (str, optional): path to save directory. Defaults to None. If None, a temporary directory is created.
        env_var (dict, optional): environment variables to pass to xTB. Defaults to XTB_ENV.

    Returns:
        mol (RDKitMol): optimized molecule for opt/path jobs; original mol for hess and single-point jobs.
        dict: Molecular properties as computed by GFN2-xTB (formation energy, HOMO/LUMO/gap energies, dipole, atomic charges)

    Raises:
        RuntimeError: If xTB calculation fails.
    """
    assert (end_mol is None) ^ (job == "--path"), "A path job is required if end_mol is provided"

    if mol.GetNumAtoms() == 1:
        return run_xtb_calc_uniatom(mol=mol,
                                    uhf=uhf,
                                    charge=charge,
                                    job=job,
                                    method=method)

    # xTB allows key word like --gfn 2 and --gfn2
    # but the latter seems to be deprecated soon.
    # And a way to correct it.
    method = method if not method else "--" + METHOD_DICT[method]

    # Paths
    work_dir = osp.abspath(save_dir) if save_dir else tempfile.mkdtemp()
    paths = {cat: osp.join(work_dir, fname) for cat, fname in FILE_NAME_DICT.items()}
    if job.startswith('--path'):
        job_type = f'--path {paths["end_coord"]} --input {TS_PATH_INP}'
        end_mol.ToSDFFile(paths["end_coord"], confId=end_conf_id)
    else:
        job_type = job

    # Create coordinate file
    # Note for xtb version >=6.5.0,
    # the sdf file from rdkit is no longer readable by xtb
    # Todo: add a workaround for version >= 6.5.0
    mol.ToSDFFile(paths['coord'], confId=conf_id)

    command = [
        XTB_BINARY,
        paths['coord'],
        job_type,
        method,
        level,
        f"--uhf {uhf}",
        f"--chrg {charge}",
        "--json true",
        "--parallel",
    ]

    try:
        os.remove(paths['restart'])  # to avoid xTB reading old restart file
    except FileNotFoundError:
        pass

    with open(paths['log'], "w", encoding='utf-8') as log_f:
        xtb_run = subprocess.run(command,
                                 stdout=log_f,
                                 stderr=subprocess.STDOUT,
                                 cwd=work_dir,
                                 env=env_var,
                                 check=False,
                                 )
    if xtb_run.returncode != 0:
        if not save_dir:
            rmtree(work_dir)
        raise RuntimeError("xTB calculation failed.")

    # Parse results
    res_parser = get_xtb_results_parser(job)
    try:
        res_mol, props = res_parser(mol, paths)
    except Exception as exc:
        if not save_dir:
            rmtree(work_dir)
        raise exc

    # Read energies, homo, lumo, etc...
    props.update(read_xtb_json(paths['out'], mol))
    props.update({'wbo': get_wbo(paths['wbo'])})

    return res_mol, props


def parse_xtb_opt(mol: 'RDKitMol',
                  paths: dict):
    """
    parse xTB optimization job results.

    Args:
        mol (RDKitMol): Not used in this function
        paths (dict): paths to files
    """
    with open(paths['log'], "r") as f:
        log_data = f.readlines()
    try:
        n_opt_cycles = int([line for line in log_data
                            if "GEOMETRY OPTIMIZATION CONVERGED AFTER" in line][-1].split()[-3])
        # Example:
        #    *** GEOMETRY OPTIMIZATION CONVERGED AFTER 4 ITERATIONS ***
    except IndexError as exc:
        # logfile doesn't exist for uni-atom molecules
        if not os.path.exists(paths['opt']):
            raise RuntimeError("xTB optimization failed.") from exc
        else:
            n_opt_cycles = 1
    res_mol = RDKitMol.FromFile(paths['opt'])[0]

    return res_mol, {'n_opt_cycles': n_opt_cycles}


def parse_xtb_hess(mol: 'RDKitMol',
                   paths: dict):
    """
    Parse xTB hessian job results.

    Args:
        mol (RDKitMol): RDKit molecule object
        paths (dict): paths to files
    """
    with open(paths['g98'], 'r') as f:
        data = f.readlines()
    frequencies = np.array(
                        [line.split()[-3:]
                         for line in data
                         if 'Frequencies' in line],
                        dtype=float
                        ).ravel()
    if len(frequencies) == 0:  # Assumes a poly-atomic molecule
        raise RuntimeError('xTB frequency calculation failed.')

    return mol, {'frequencies': frequencies}


def parse_xtb_path(mol: 'RDKitMol',
                   paths: dict):
    """
    Parse xTB path job results

    Args:
        mol (RDKitMol): RDKit molecule object
        paths (dict): paths to files
    """
    try:
        res_mol = RDKitMol.FromFile(paths['ts'],
                                    sanitize=False)
    except FileNotFoundError as exc:
            raise RuntimeError("xTB path calculation failed.") from exc
    return res_mol, {}


def parse_xtb_sp(mol: 'RDKitMol',
                 paths: dict):
    """
    Parse xTB single-point job results

    Args:
        mol (RDKitMol): RDKit molecule object
        paths (dict): paths to files
    """
    return mol, {}


def get_xtb_results_parser(job: 'str'):
    """
    Get the parser function for xTB results.
    """
    if job.startswith('--opt'):
        return parse_xtb_opt
    elif job.startswith('--hess'):
        return parse_xtb_hess
    elif job.startswith('--path'):
        return parse_xtb_path
    else:
        return parse_xtb_sp


def run_xtb_calc_uniatom(mol: 'RDKitMol',
                         uhf: int = 0,
                         charge: int = 0,
                         job: str = "",
                         method: str = "gfn2",
                         ):
    """
    Run xTB calculation for uni-atom molecules. This is a workaround for
    xTB not being able to generate valid output files for uni-atom molecules.

    Args:
        mol (RDKitMol): RDKit molecule object
        uhf (int): spin multiplicity
        charge (int): charge
        job (str): xTB job type
        method (str): xTB method
    """
    props = {}
    if job.startswith('--opt'):
        props.update({'n_opt_cycles': 1})
    elif job.startswith('--hess'):
        props.update({'frequencies': np.array([])})

    try:
        from xtb.interface import Calculator
        from xtb.utils import get_method
    except ImportError:
        # xtb-python is not installed
        # neglect energy information
        props['charges'] = np.array([0.0])
        props['dipole'] = np.array([0.0, 0.0, 0.0])
        props['E_form'] = np.array([0.0])
        return mol, props

    method_name_dict = {
        'gfn0': 'GFN0-xTB',
        'gfn1': 'GFN1-xTB',
        'gfn2': 'GFN2-xTB',
        'gfnff': 'GFN-FF',
    }

    numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    positions = np.array([[0.0, 0.0, 0.0]])  # doesn't really matter

    calc = Calculator(param=get_method(method_name_dict.get(method)),
                      numbers=numbers,
                      positions=positions,
                      charge=charge,
                      uhf=uhf)
    res = calc.singlepoint()  # other calculations are meaningless

    num_occupied = (res.get_orbital_occupations() > 1e-6).sum()  # number of occupied orbitals; accounting for occassional very small values
    E_homo = res.get_orbital_eigenvalues()[num_occupied - 1]  # zero-indexing
    props.update({
            "E_form": res.get_energy(),  # already in Hartree
            "E_homo": E_homo,
            "dipole": res.get_dipole() * AU_TO_DEBYE,
            "charges": res.get_charges(),
            "wbo": {},
            })
    return mol, props
