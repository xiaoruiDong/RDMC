#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A module contains functions to write Gaussian input files.
"""

from typing import Optional, Union
from rdmc.external.inpwriter._register import register_qm_writer
from rdmc.external.inpwriter.utils import (
    _avoid_empty_line,
    _get_mult_and_chrg,
    XTB_GAUSSIAN_PERL_PATH,
)


def write_gaussian_config(
    memory: Union[float, int],
    nprocs: int,
    scheme: str,
    extra_sys_settings: str = "",
    title: str = "title",
    **kwargs,
) -> str:
    """
    Write the configuration section of Gaussian input file.
    This is useful when a calculation program only needs the configuration section.

    Args:
        memory (float or int): The memory to be used in GB.
        nprocs (int): The number of processors to be used.
        scheme (str): The scheme to be used.
        extra_sys_settings (str, optional): Extra system settings. Defaults to ``''``.
        title (str, optional): The title of the calculation. Defaults to ``'title'``.

    Returns:
        str: The configuration section of Gaussian input file.
    """
    return (
        f"%mem={memory}gb\n"
        f"%nprocshared={nprocs}\n"
        f"{_avoid_empty_line(extra_sys_settings)}"
        f"{scheme}\n\n"
        f"{title}"
    )


def write_gaussian_inp(
    memory: Union[float, int],
    nprocs: int,
    scheme: str,
    charge: int,
    mult: int,
    coord: str = "",
    extra_sys_settings: str = "",
    title: str = "title",
    extra: str = "",
    **kwargs,
) -> str:
    """
    Write the base structure of Gaussian input file.

    Args:
        memory (float or int): The memory to be used in GB.
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
    output = write_gaussian_config(
        memory=memory,
        nprocs=nprocs,
        scheme=scheme,
        extra_sys_settings=extra_sys_settings,
        title=title,
    )
    if extra:
        extra = "\n" + _avoid_empty_line(extra)
    output += (
        f"\n\n" f"{charge} {mult}\n" f"{_avoid_empty_line(coord)}" f"{extra}" f"\n"
    )
    return output


def write_gaussian_opt(
    mol,
    conf_id: int = 0,
    ts: bool = False,
    charge: Optional[int] = None,
    mult: Optional[int] = None,
    memory: Union[float, int] = 1,
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
        memory (float or int, optional): The memory to be used in GB. Defaults to ``1``.
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
        opt_scheme.append("ts")
    if hess:
        opt_scheme.append(hess)
    elif hess is None:
        opt_scheme.append("calcall")
    if method.lower() in ["gfn2-xtb", "gfn1-xtb"]:
        opt_scheme.append("nomicro")
    opt_scheme.append("noeig")
    opt_scheme.append(f"maxcycle={max_iter}")
    if coord_type:
        opt_scheme.append(coord_type)
    if opt_level:
        opt_scheme.append(opt_level)

    scheme_str = "#P"
    scheme_str += f' opt=({",".join(opt_scheme)})'
    if scf_level:
        scheme_str += f" scf=({scf_level})"
    scheme_str += " nosymm" if nosymm else ""
    scheme_str += " freq" if follow_freq else ""

    if method.lower() in ["gfn2-xtb", "gfn1-xtb"]:
        if nprocs == [1, 2]:
            xtb_nprocs = nprocs
        elif nprocs in [3, 4]:
            xtb_nprocs, nprocs = nprocs - 1, 1
        else:
            xtb_nprocs, nprocs = nprocs - 2, 2
        scheme_str += (
            f'\nexternal="{XTB_GAUSSIAN_PERL_PATH} --gfn {method[3]} -P {xtb_nprocs}"'
        )
    else:
        scheme_str += f" {method}"

    # todo: modify internal coordinates
    return write_gaussian_inp(
        memory=memory,
        nprocs=nprocs,
        scheme=scheme_str,
        charge=charge,
        mult=mult,
        coord=mol.ToXYZ(header=False, confId=conf_id),
        **kwargs,
    )


def write_gaussian_freq(
    mol,
    conf_id: int = 0,
    charge: Optional[int] = None,
    mult: Optional[int] = None,
    memory: Union[float, int] = 1,
    nprocs: int = 1,
    method: str = "gfn2-xtb",
    scf_level: str = "tight",
    nosymm: bool = False,
    **kwargs,
) -> str:
    """
    Write the input file for Gaussian frequency calculation.

    Args:
        mol (RDKitMol): The molecule to be run.
        conf_id (int, optional): The conformer ID to be run. Defaults to ``0``.
        charge (int, optional): The charge of the molecule. Defaults to ``None``, to use the charge of mol.
        mult (int, optional): The multiplicity of the molecule. Defaults to ``None``, to use the multiplicity of mol.
        memory (float or int, optional): The memory to be used in GB. Defaults to ``1``.
        nprocs (int, optional): The number of processors to be used. Defaults to ``1``.
        method (str, optional): The method to be used. Defaults to ``"gfn2-xtb"``.
        scf_level (str, optional): The SCF level. Defaults to ``"tight"``.
        nosymm (bool, optional): Whether to use nosymm. Defaults to ``False``.

    Returns:
        str: The input file for Gaussian frequency calculation.
    """
    mult, charge = _get_mult_and_chrg(mol, mult, charge)

    scheme_str = "#P"
    scheme_str += " freq"
    scheme_str += f" scf=({scf_level})" if scf_level else ""
    scheme_str += " nosymm" if nosymm else ""

    if method.lower() in ["gfn2-xtb", "gfn1-xtb"]:
        scheme_str += f'\nexternal="{XTB_GAUSSIAN_PERL_PATH} --gfn {method[3]} -P"'
    else:
        scheme_str += f" {method}"

    # todo: modify internal coordinates
    return write_gaussian_inp(
        memory=memory,
        nprocs=nprocs,
        scheme=scheme_str,
        charge=charge,
        mult=mult,
        coord=mol.ToXYZ(header=False, confId=conf_id),
        **kwargs,
    )


def write_gaussian_irc(
    mol,
    conf_id: int = 0,
    charge: Optional[int] = None,
    mult: Optional[int] = None,
    memory: Union[float, int] = 1,
    nprocs: int = 1,
    method: str = "gfn2-xtb",
    direction: str = "forward",
    max_iter: int = 20,
    max_points: int = 100,
    step_size: float = 7,
    algorithm: str = "hpc",
    coord_type: str = "massweighted",
    hess: Optional[str] = None,
    irc_level: str = "tight",
    scf_level: str = "tight",
    nosymm: bool = False,
    **kwargs,
) -> str:
    """
    Write the input file for Gaussian IRC calculation.

    Args:
        mol (RDKitMol): The molecule to be run.
        conf_id (int, optional): The conformer ID to be run. Defaults to ``0``.
        charge (int, optional): The charge of the molecule. Defaults to ``None``, to use the charge of mol.
        mult (int, optional): The multiplicity of the molecule. Defaults to ``None``, to use the multiplicity of mol.
        memory (float or int, optional): The memory to be used in GB. Defaults to ``1``.
        nprocs (int, optional): The number of processors to be used. Defaults to ``1``.
        method (str, optional): The method to be used. Defaults to ``"gfn2-xtb"``.
        direction (str, optional): The direction of the IRC. Defaults to ``"forward"``. other options: ``"backward"``.
        max_iter (int, optional): The maximum number of IRC steps. Defaults to ``20``, same as Gaussian's default.
        max_points (int, optional): The maximum number of IRC points. Defaults to ``100``.
        step_size (float, optional): The step size of IRC. Defaults to ``7``. Unit 0.01 bohr.
        algorithm (str, optional): The IRC algorithm. Defaults to ``"hpc"`` (g16 and g09 default). Other options: ``"eulerpc"``, ``"lqc"``, ``"gs2"``.
        coord_type (str, optional): The coordinate type. Defaults to ``"massweighted"``.
        hess (str, optional): The Hessian calculation method. Defaults to ``"calcall"``.
        scf_level (str, optional): The SCF level. Defaults to ``"tight"``.
        nosymm (bool, optional): Whether to use nosymm. Defaults to ``False``.

    Returns:
        str: The input file for Gaussian IRC calculation.
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
        irc_scheme.append("calcall")
    if method.lower() in ["gfn2-xtb", "gfn1-xtb"]:
        irc_scheme.append("nomicro")
    irc_scheme.append(f"maxcycle={max_iter}")
    irc_scheme.append(f"maxpoints={max_points}")
    irc_scheme.append(f"stepsize={step_size}")
    if irc_level:
        irc_scheme.append(irc_level)
    irc_scheme.append(coord_type)

    scheme_str = "#P"
    scheme_str += f' irc=({",".join(irc_scheme)})'
    scheme_str += f" scf=({scf_level})" if scf_level else ""
    scheme_str += " nosymm" if nosymm else ""

    if method.lower() in ["gfn2-xtb", "gfn1-xtb"]:
        scheme_str += f'\nexternal="{XTB_GAUSSIAN_PERL_PATH} --gfn {method[3]} -P"'
    else:
        scheme_str += f" {method}"

    # todo: modify internal coordinates
    return write_gaussian_inp(
        memory=memory,
        nprocs=nprocs,
        scheme=scheme_str,
        charge=charge,
        mult=mult,
        coord=mol.ToXYZ(header=False, confId=conf_id),
        **kwargs,
    )


def write_gaussian_gsm(
    method: str = "gfn2-xtb",
    memory: Union[float, int] = 1,
    nprocs: int = 1,
    extra_sys_settings: str = "",
    title: str = "title",
) -> str:
    """
    Write the computation setup section of the input file for GSM calculation using Gaussian.

    Args:
        method (str, optional): The method to be used. Defaults to ``"GFN2-xTB"``.
        memory (int, optional): The memory to be used in GB. Defaults to ``1``.
        nprocs (int, optional): The number of processors to be used. Defaults to ``1``.
        extra_sys_settings (str, optional): Extra system settings. Defaults to ``''``.

    Returns:
        str: The computation setup section of the input file for GSM calculation using Gaussian.
    """

    scheme_str = "#N force scf=(xqc) nosymm"
    if method.lower() in ["gfn2-xtb", "gfn1-xtb"]:
        scheme_str += f'\nexternal="{XTB_GAUSSIAN_PERL_PATH} --gfn {method[3]} -P"'
    else:
        scheme_str += f" {method}"

    return write_gaussian_config(
        memory=memory,
        nprocs=nprocs,
        extra_sys_settings=extra_sys_settings,
        scheme=scheme_str,
        title=title,
    )


register_qm_writer("gaussian", "opt", write_gaussian_opt)
register_qm_writer("gaussian", "freq", write_gaussian_freq)
register_qm_writer("gaussian", "irc", write_gaussian_irc)
register_qm_writer("gaussian", "gsm", write_gaussian_gsm)
