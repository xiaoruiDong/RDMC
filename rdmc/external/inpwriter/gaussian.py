#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A module contains functions to write Gaussian input files.
"""

from typing import Optional
from rdmc.external.inpwriter.utils import _get_mult_and_chrg
from rdmc.external.xtb_tools.utils import XTB_GAUSSIAN_PL


def _write_gaussian_inp(memory: int,
                        nprocs: int,
                        scheme: str,
                        charge: int,
                        mult: int,
                        coord: str,) -> str:
    return f"""%mem={memory}gb
%nprocshared={nprocs}
{scheme}

Title Card Required

{charge} {mult}
{coord}


"""


def write_gaussian_opt(mol,
                       conf_id: int = 0,
                       ts: bool = False,
                       charge: Optional[int] = None,
                       mult: Optional[int] = None,
                       memory: int = 1,
                       nprocs: int = 1,
                       method: str = "gfn2-xtb",
                       max_iter: int = 100,
                       coord_type: str = "",
                       modify_internal: Optional[dict] = None,
                       scf_level: str = "tight",
                       opt_level: str = "tight",
                       hess: Optional[str] = None,
                       follow_freq: bool = False,
                       nosymm: bool = False,
                       **kwargs,
                       ) -> str:
    """
    Write the input file for Gaussian optimization calculation.

    Args:
        mol (RDKitMol): The molecule to be run.
        conf_id (int, optional): The conformer ID to be run. Defaults to 0.
        ts (bool, optional): Whether the molecule is a TS. Defaults to False.
        charge (int, optional): The charge of the molecule. Defaults to None, to use the charge of mol.
        mult (int, optional): The multiplicity of the molecule. Defaults to None, to use the multiplicity of mol.
        memory (int, optional): The memory to be used in GB. Defaults to 1.
        nprocs (int, optional): The number of processors to be used. Defaults to 1.
        method (str, optional): The method to be used. Defaults to "gfn2-xtb".
        max_iter (int, optional): The maximum number of iterations. Defaults to 100.
        coord_type (str, optional): The coordinate type. Defaults to "cartesian".
        modify_internal (dict, optional): The internal coordinates to be modified. Defaults to None. # todo: implement this
        scf_level (str, optional): The SCF level. Defaults to "tight".
        opt_level (str, optional): The optimization level. Defaults to "tight".
        hess (str, optional): The initial Hessian. Defaults to None.
        follow_freq (bool, optional): Whether to follow a frequency calculation. Defaults to False.
        nosymm (bool, optional): Whether to use nosymm. Defaults to False.

    Returns:
        str: The input file for ORCA optimization calculation.
    """
    mult, charge = _get_mult_and_chrg(mol, mult, charge)

    opt_scheme = []
    if ts:
        opt_scheme.append('ts')
    if hess:
        opt_scheme.append(hess)
    else:
        opt_scheme.append('calcall')
    if method in ['gfn2-xtb', 'gfn1-xtb']:
        opt_scheme.append('nomicro')
    opt_scheme.append('noeig')
    opt_scheme.append(f'maxcycle={max_iter}')
    if coord_type:
        opt_scheme.append(coord_type)
    opt_scheme.append(opt_level)

    scheme_str = '#P'
    scheme_str += f' opt=({",".join(opt_scheme)}) {method}'
    scheme_str += f' scf=({scf_level})'
    scheme_str += ' nosymm' if nosymm else ''
    scheme_str += ' freq' if follow_freq else ''

    if method.lower() in ['gfn2-xtb', 'gfn1-xtb']:
        scheme_str += f'\nexternal="{XTB_GAUSSIAN_PL} --gfn {method[3]} -P"'

    # todo: modify internal coordinates
    return _write_gaussian_inp(memory,
                               nprocs,
                               scheme_str,
                               charge,
                               mult,
                               mol.ToXYZ(header=False,
                                         confId=conf_id))


def write_gaussian_freq(mol,
                        conf_id: int = 0,
                        charge: Optional[int] = None,
                        mult: Optional[int] = None,
                        memory: int = 1,
                        nprocs: int = 1,
                        method: str = "gfn2-xtb",
                        scf_level: str = "tight",
                        nosymm: bool = False,
                        **kwargs,
                        ):
    """
    Write the input file for Gaussian frequency calculation.

    Args:
        mol (RDKitMol): The molecule to be run.
        conf_id (int, optional): The conformer ID to be run. Defaults to 0.
        charge (int, optional): The charge of the molecule. Defaults to None, to use the charge of mol.
        mult (int, optional): The multiplicity of the molecule. Defaults to None, to use the multiplicity of mol.
        memory (int, optional): The memory to be used in GB. Defaults to 1.
        nprocs (int, optional): The number of processors to be used. Defaults to 1.
        method (str, optional): The method to be used. Defaults to "gfn2-xtb".
        scf_level (str, optional): The SCF level. Defaults to "tight".
        nosymm (bool, optional): Whether to use nosymm. Defaults to False.

    Returns:
        str: The input file for ORCA frequency calculation.
    """
    mult, charge = _get_mult_and_chrg(mol, mult, charge)

    scheme_str = '#P'
    scheme_str += ' freq'
    scheme_str += f' scf=({scf_level})'
    scheme_str += ' nosymm' if nosymm else ''

    if method.lower() in ['gfn2-xtb', 'gfn1-xtb']:
        scheme_str += f'\nexternal="{XTB_GAUSSIAN_PL} --gfn {method[3]} -P"'

    # todo: modify internal coordinates
    return _write_gaussian_inp(memory,
                               nprocs,
                               scheme_str,
                               charge,
                               mult,
                               mol.ToXYZ(header=False,
                                         confId=conf_id))


def write_gaussian_irc(mol,
                       conf_id: int = 0,
                       charge: Optional[int] = None,
                       mult: Optional[int] = None,
                       memory: int = 1,
                       nprocs: int = 1,
                       method: str = "gfn2-xtb",
                       direction: str = 'forward',
                       max_iter: int = 20,
                       max_points: int = 100,
                       step_size: float = 7,
                       algorithm: str = 'hpc',
                       coord_type: str = "massweighted",
                       hess: str = 'calcall',
                       scf_level: str = "tight",
                       nosymm: bool = False,
                       **kwargs,
                       ):
    """
    Write the input file for Gaussian IRC calculation

    Args:
        mol (RDKitMol): The molecule to be run
        conf_id (int, optional): The conformer ID to be run. Defaults to 0.
        charge (int, optional): The charge of the molecule. Defaults to None, to use the charge of mol.
        mult (int, optional): The multiplicity of the molecule. Defaults to None, to use the multiplicity of mol.
        memory (int, optional): The memory to be used in GB. Defaults to 1.
        nprocs (int, optional): The number of processors to be used. Defaults to 1.
        method (str, optional): The method to be used. Defaults to "gfn2-xtb".
        direction (str, optional): The direction of the IRC. Defaults to "both". other options: "forward", "backward".
        max_iter (int, optional): The maximum number of IRC steps. Defaults to 20, same as Gaussian's default.
        max_points (int, optional): The maximum number of IRC points. Defaults to 100.
        step_size (float, optional): The step size of IRC. Defaults to 7.
        algorithm (str, optional): The IRC algorithm. Defaults to "hpc".
        coord_type (str, optional): The coordinate type. Defaults to "massweighted".
        hess (str, optional): The Hessian calculation method. Defaults to "calcall".
        scf_level (str, optional): The SCF level. Defaults to "tight".
        nosymm (bool, optional): Whether to use nosymm. Defaults to False.

    Returns:
        str: The input file for ORCA IRC calculation
    """
    mult, charge = _get_mult_and_chrg(mol, mult, charge)

    irc_scheme = [algorithm, direction]
    if hess:
        irc_scheme.append(hess)
    else:
        irc_scheme.append('calcall')
    if method in ['gfn2-xtb', 'gfn1-xtb']:
        irc_scheme.append('nomicro')
    irc_scheme.append(f'maxcycle={max_iter}')
    irc_scheme.append(f'maxpoints={max_points}')
    irc_scheme.append(f'stepsize={step_size}')
    irc_scheme.append(coord_type)

    scheme_str = '#P'
    scheme_str += f' irc=({",".join(irc_scheme)}) {method}'
    scheme_str += f' scf=({scf_level})'
    scheme_str += ' nosymm' if nosymm else ''

    if method in ['gfn2-xtb', 'gfn1-xtb']:
        scheme_str += f'\nexternal="{XTB_GAUSSIAN_PL} --gfn {method[3]} -P"'

    # todo: modify internal coordinates
    return _write_gaussian_inp(memory,
                               nprocs,
                               scheme_str,
                               charge,
                               mult,
                               mol.ToXYZ(header=False,
                                         confId=conf_id))


def write_gaussian_gsm(method="GFN2-xTB", memory=1, nprocs=1):

    if method == "GFN2-xTB":
        title_section = (
            f'#N NoSymmetry scf(xqc) force\n'
            f'external="{XTB_GAUSSIAN_PL} --gfn 2 -P"'
        )
    else:
        title_section = f"#N NoSymmetry scf(xqc) force {method}"

    gaussian_gsm_input = (f'%mem={memory}gb\n'
                          f'%nprocshared={nprocs}\n'
                          f'{title_section}\n'
                          f'\n'
                          f'Title Card Required'
                          )
    return gaussian_gsm_input
