#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A module contains functions to write Gaussian input files.
"""

from typing import Optional
from rdmc.external.inpwriter.utils import _get_mult_and_chrg
from rdmc.external.xtb_tools.utils import XTB_GAUSSIAN_PL


def write_gaussian_inp(memory: int,
                       nprocs: int,
                       scheme: str,
                       charge: int,
                       mult: int,
                       coord: str = '',
                       extra_sys_settings: str = '',
                       title: str = 'title',
                       extra: str = '',
                       **kwargs,
                       ) -> str:
    """
    Write the base structure of Gaussian input file.

    Args:
        memory (int): The memory to be used in GB.
        nprocs (int): The number of processors to be used.
        scheme (str): The scheme to be used.
        charge (int): The charge of the molecule.
        mult (int): The multiplicity of the molecule.
        coord (str, optional): The coordinates of the molecule. Defaults to ``''``.
        extra_sys_settings (str, optional): Extra system settings. Defaults to ``''``.
        title (str, optional): The title of the calculation. Defaults to ``'title'``.
        extra (str, optional): Extra settings. Defaults to ``''``.

    Returns:
        str: The Gaussian input file.
    """
    output = f"%mem={memory}gb\n%nprocshared={nprocs}\n"
    if extra_sys_settings:
        output += f"{extra_sys_settings.strip()}\n"
    output += f"{scheme}\n\n{title}\n\n{charge} {mult}\n"
    if coord:
        output += coord.strip() + "\n"
    if extra:
        output += f"\n{extra.strip()}\n\n\n"
    else:
        output += "\n\n"
    return output


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
        conf_id (int, optional): The conformer ID to be run. Defaults to ``0``.
        ts (bool, optional): Whether the molecule is a TS. Defaults to ``False``.
        charge (int, optional): The charge of the molecule. Defaults to ``None``, to use the charge of mol.
        mult (int, optional): The multiplicity of the molecule. Defaults to ``None``, to use the multiplicity of mol.
        memory (int, optional): The memory to be used in GB. Defaults to ``1``.
        nprocs (int, optional): The number of processors to be used. Defaults to ``1``.
        method (str, optional): The method to be used. Defaults to ``"gfn2-xtb"``.
        max_iter (int, optional): The maximum number of iterations. Defaults to ``100``.
        coord_type (str, optional): The coordinate type. Defaults to ``"cartesian"``.
        scf_level (str, optional): The SCF level. Defaults to ``"tight"``.
        opt_level (str, optional): The optimization level. Defaults to ``"tight"``.
        hess (str, optional): The initial Hessian. Defaults to ``None``.
        follow_freq (bool, optional): Whether to follow a frequency calculation. Defaults to ``False``.
        nosymm (bool, optional): Whether to use nosymm. Defaults to ``False``.

    Returns:
        str: The input file for Gaussian optimization calculation.
    """
    mult, charge = _get_mult_and_chrg(mol, mult, charge)

    opt_scheme = []
    if ts:
        opt_scheme.append('ts')
    if hess:
        opt_scheme.append(hess)
    elif hess is None:
        opt_scheme.append('calcall')
    if method.lower() in ['gfn2-xtb', 'gfn1-xtb']:
        opt_scheme.append('nomicro')
    opt_scheme.append('noeig')
    opt_scheme.append(f'maxcycle={max_iter}')
    if coord_type:
        opt_scheme.append(coord_type)
    if opt_level:
        opt_scheme.append(opt_level)

    scheme_str = '#P'
    scheme_str += f' opt=({",".join(opt_scheme)})'
    if scf_level:
        scheme_str += f' scf=({scf_level})'
    scheme_str += ' nosymm' if nosymm else ''
    scheme_str += ' freq' if follow_freq else ''

    if method.lower() in ['gfn2-xtb', 'gfn1-xtb']:
        scheme_str += f'\nexternal="{XTB_GAUSSIAN_PL} --gfn {method[3]} -P"'
    else:
        scheme_str += f' {method}'

    # todo: modify internal coordinates
    return write_gaussian_inp(memory=memory,
                              nprocs=nprocs,
                              scheme=scheme_str,
                              charge=charge,
                              mult=mult,
                              coord=mol.ToXYZ(header=False,
                                              confId=conf_id),
                              **kwargs,
                              )


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
        str: The input file for Gaussian frequency calculation.
    """
    mult, charge = _get_mult_and_chrg(mol, mult, charge)

    scheme_str = '#P'
    scheme_str += ' freq'
    scheme_str += f' scf=({scf_level})' if scf_level else ''
    scheme_str += ' nosymm' if nosymm else ''

    if method.lower() in ['gfn2-xtb', 'gfn1-xtb']:
        scheme_str += f'\nexternal="{XTB_GAUSSIAN_PL} --gfn {method[3]} -P"'
    else:
        scheme_str += f' {method}'

    # todo: modify internal coordinates
    return write_gaussian_inp(memory=memory,
                              nprocs=nprocs,
                              scheme=scheme_str,
                              charge=charge,
                              mult=mult,
                              coord=mol.ToXYZ(header=False,
                                              confId=conf_id),
                              **kwargs,
                              )


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
                       hess: Optional[str] = None,
                       irc_level: str = "tight",
                       scf_level: str = "tight",
                       nosymm: bool = False,
                       **kwargs,
                       ):
    """
    Write the input file for Gaussian IRC calculation.

    Args:
        mol (RDKitMol): The molecule to be run.
        conf_id (int, optional): The conformer ID to be run. Defaults to 0.
        charge (int, optional): The charge of the molecule. Defaults to None, to use the charge of mol.
        mult (int, optional): The multiplicity of the molecule. Defaults to None, to use the multiplicity of mol.
        memory (int, optional): The memory to be used in GB. Defaults to 1.
        nprocs (int, optional): The number of processors to be used. Defaults to 1.
        method (str, optional): The method to be used. Defaults to "gfn2-xtb".
        direction (str, optional): The direction of the IRC. Defaults to "forward". other options: "backward".
        max_iter (int, optional): The maximum number of IRC steps. Defaults to 20, same as Gaussian's default.
        max_points (int, optional): The maximum number of IRC points. Defaults to 100.
        step_size (float, optional): The step size of IRC. Defaults to 7. Unit 0.01 bohr.
        algorithm (str, optional): The IRC algorithm. Defaults to "hpc". Other options: "eulerpc", "lqc", "gs2".
        coord_type (str, optional): The coordinate type. Defaults to "massweighted".
        hess (str, optional): The Hessian calculation method. Defaults to "calcall".
        scf_level (str, optional): The SCF level. Defaults to "tight".
        nosymm (bool, optional): Whether to use nosymm. Defaults to False.

    Returns:
        str: The input file for Gaussian IRC calculation
    """
    mult, charge = _get_mult_and_chrg(mol, mult, charge)

    irc_scheme = []
    if direction:
        irc_scheme.append(direction)
    if algorithm:
        irc_scheme.append(algorithm)
    if hess:
        irc_scheme.append(hess)
    elif hess is None:
        irc_scheme.append('calcall')
    if method.lower() in ['gfn2-xtb', 'gfn1-xtb']:
        irc_scheme.append('nomicro')
    irc_scheme.append(f'maxcycle={max_iter}')
    irc_scheme.append(f'maxpoints={max_points}')
    irc_scheme.append(f'stepsize={step_size}')
    if irc_level:
        irc_scheme.append(irc_level)
    irc_scheme.append(coord_type)

    scheme_str = '#P'
    scheme_str += f' irc=({",".join(irc_scheme)})'
    scheme_str += f' scf=({scf_level})' if scf_level else ''
    scheme_str += ' nosymm' if nosymm else ''

    if method in ['gfn2-xtb', 'gfn1-xtb']:
        scheme_str += f'\nexternal="{XTB_GAUSSIAN_PL} --gfn {method[3]} -P"'
    else:
        scheme_str += f' {method}'

    # todo: modify internal coordinates
    return write_gaussian_inp(memory=memory,
                              nprocs=nprocs,
                              scheme=scheme_str,
                              charge=charge,
                              mult=mult,
                              coord=mol.ToXYZ(header=False,
                                              confId=conf_id),
                              **kwargs,
                              )


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
