#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
A module contains functions to interface with Orca.
"""

from typing import Optional, Tuple, Union

anhess_dict = {
    "am1": False,
    "pm3": False,
    "xtb2": False,
    "xtb1": False,
}


def _get_mult_and_chrg(mol: 'RDKitMol',
                       multiplicity: Optional[int] = None,
                       charge: Optional[int] = None,
                       )-> Tuple[int, int]:
    """
    Get the multiplicity and charge of a molecule.

    Args:
        mol: The molecule.
        multiplicity: The multiplicity.
        charge: The charge.

    Returns:
        The multiplicity and charge of the molecule.
    """
    if multiplicity is None:
        multiplicity = mol.GetSpinMultiplicity()
    if charge is None:
        charge = mol.GetFormalCharge()
    return multiplicity, charge


def write_orca_opt(mol,
                   conf_id: int = 0,
                   ts: bool = False,
                   charge: Optional[int] = None,
                   mult: Optional[int] = None,
                   memory: int = 1,
                   nprocs: int = 1,
                   method: str = "xtb2",
                   max_iter: int = 100,
                   coord_type: str = "redundant",
                   modify_internal: Optional[dict] = None,
                   scf_level: str = "normal",
                   opt_level: str = "normal",
                   hess: Optional[str] = None,
                   follow_freq = False,
                   ) -> str:
    """
    Write the input file for ORCA optimization calculation.

    Args:
        mol (RDKitMol): The molecule to be run.
        conf_id (int, optional): The conformer ID to be run. Defaults to 0.
        ts (bool, optional): Whether the molecule is a TS. Defaults to False.
        charge (int, optional): The charge of the molecule. Defaults to None, to use the charge of mol.
        mult (int, optional): The multiplicity of the molecule. Defaults to None, to use the multiplicity of mol.
        memory (int, optional): The memory to be used in GB. Defaults to 1.
        nprocs (int, optional): The number of processors to be used. Defaults to 1.
        method (str, optional): The method to be used. Defaults to "xtb2".
        max_iter (int, optional): The maximum number of iterations. Defaults to 100.
        coord_type (str, optional): The coordinate type. Defaults to "cartesian".
        modify_internal (dict, optional): The internal coordinates to be modified. Defaults to None. # todo: implement this
        convergence_level (str, optional): The convergence level. Defaults to "normal".
        hess (dict, optional): The initial Hessian. Defaults to None.

    Returns:
        str: The input file for ORCA optimization calculation.
    """
    mult, charge = _get_mult_and_chrg(mol, mult, charge)

    opt = "opt" if not ts else "optts"

    # Determine if analytical hessian is available
    hess_str = ""
    if not anhess_dict.get(method.lower(), True):
        hess_str += "    numhess true\n"
    if hess is None:  # use default
        hess_str += "    calc_hess true\n    recalc_hess 5"
    else:
        hess_str += hess

    if modify_internal is not None:
        raise NotImplementedError("modify_internal is not implemented yet.")

    if follow_freq and anhess_dict.get(method.lower(), True):
        opt += ' anfreq'
    elif follow_freq:
        opt += ' numfreq'

    # GeomString
    orca_opt_input = f"""! {method} {opt}
%maxcore {memory * 1024}
%pal
nprocs {nprocs}
end
%scf
    convergence {scf_level}
end
%geom
    maxiter {max_iter}
    coordsys {coord_type}
    convergence {opt_level}
{hess_str}
end
*xyz {charge} {mult}
{mol.ToXYZ(header=False, confId=conf_id)}
*
"""
    return orca_opt_input


def write_orca_freq(mol,
                    conf_id: int = 0,
                    charge: Optional[int] = None,
                    mult: Optional[int] = None,
                    memory: int = 1,
                    nprocs: int = 1,
                    method: str = "xtb2",
                    ):
    """
    Write the input file for ORCA frequency calculation.

    Args:
        mol (RDKitMol): The molecule to be run.
        conf_id (int, optional): The conformer ID to be run. Defaults to 0.
        charge (int, optional): The charge of the molecule. Defaults to None, to use the charge of mol.
        mult (int, optional): The multiplicity of the molecule. Defaults to None, to use the multiplicity of mol.
        memory (int, optional): The memory to be used in GB. Defaults to 1.
        nprocs (int, optional): The number of processors to be used. Defaults to 1.
        method (str, optional): The method to be used. Defaults to "xtb2".

    Returns:
        str: The input file for ORCA frequency calculation.
    """
    mult, charge = _get_mult_and_chrg(mol, mult, charge)
    # Determine if analytical hessian is available
    if anhess_dict.get(method.lower(), True):
        freq = "anfreq"
    else:
        freq = "numfreq"

    orca_freq_input = \
        f"""
! {method} {freq}
%maxcore {memory * 1024}
%pal
nprocs {nprocs}
end
%scf
    convergence tight
end
*xyz {charge } {mult}
{mol.ToXYZ(header=False, confId=conf_id)}
*
"""
    return orca_freq_input


def write_orca_irc(mol,
                   conf_id: int = 0,
                   charge: Optional[int] = None,
                   mult: Optional[int] = None,
                   memory: int = 1,
                   nprocs: int = 1,
                   method: str = "xtb2",
                   direction: str = "both",
                   max_iter: int = 100,
                   ):
    """
    Write the input file for ORCA IRC calculation

    Args:
        mol (RDKitMol): The molecule to be run
        conf_id (int, optional): The conformer ID to be run. Defaults to 0.
        charge (int, optional): The charge of the molecule. Defaults to None, to use the charge of mol.
        mult (int, optional): The multiplicity of the molecule. Defaults to None, to use the multiplicity of mol.
        memory (int, optional): The memory to be used in GB. Defaults to 1.
        nprocs (int, optional): The number of processors to be used. Defaults to 1.
        method (str, optional): The method to be used. Defaults to "xtb2".
        direction (str, optional): The direction of the IRC. Defaults to "both". other options: "forward", "backward".
        max_iter (int, optional): The maximum number of IRC steps. Defaults to 100.

    Returns:
        str: The input file for ORCA IRC calculation
    """
    mult, charge = _get_mult_and_chrg(mol, mult, charge)

    # Determine if analytical hessian is available
    if anhess_dict.get(method.lower(), True):
        hess = "calc_anfreq"
    else:
        hess = "calc_numfreq"

    orca_irc_input = f"""! {method}
%maxcore {memory * 1024}
%pal
nprocs {nprocs}
end
%scf
    convergence tight
end
%irc
    direction {direction}
    {hess}
    maxiter {max_iter}
*xyz {charge } {mult}
{mol.ToXYZ(header=False, confId=conf_id)}
*
"""
    return orca_irc_input


def write_orca_gsm(method="XTB2", memory=1, nprocs=1):

    if method.upper() in ['AM1', 'PM3']:  # NDO methods cannot be used in parallel runs yet
        nprocs = 1

    orca_gsm_input= f"""! {method} Engrad TightSCF
%maxcore {memory*1024}
%pal
nprocs {nprocs}
end
%scf
maxiter 350
end
"""
    return orca_gsm_input
