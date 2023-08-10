#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A module contains functions to write QChem input files.
"""

# Developer notes:
# From QChem6.0, libopt3 is supported to optimize non-TS geometries.

from typing import Optional
from packaging import version

from rdmc.external.inpwriter.utils import _get_mult_and_chrg


def _write_base_block(job_type: str,
                      uhf: bool,
                      method: str,
                      basis: str,
                      scf_level: int,
                      nosymm: bool,
                      other_args: Optional[list] = None,
                      ) -> str:
    """
    Write the prejob for hessian calculation.

    Args:
        method (str): The method to be used.
        basis (str): The basis to be used.
        mult (int): The multiplicity of the molecule.
        max_iter (int): The maximum number of iterations.
        scf_level (int): The scf convergence level.
        nosymm (bool): Whether to ignore symmetry.

    Returns:
        str: The prejob for hessian calculation.
    """
    if other_args:
        extra = "\n".join(other_args + ['$end'])
    else:
        extra = '\n$end'
    return f"""$rem
jobtype {job_type}
method {method}
basis {basis}
unrestricted {str(uhf).lower()}
scf_algorithm diis
max_scf_cycles 100
scf_convergence {scf_level}
sym_ignore {str(nosymm).lower()}
symmetry {str(not nosymm).lower()}
wavefunction_analysis false
""" + extra


def _write_read_molecule_block():
    """
    Write the read molecule block.
    """
    return """$molecule
read
end"""


def _write_molecule_block(mol: 'RDKitMol',
                          conf_id: int,
                          charge: int,
                          mult: int,
                          ) -> str:
    """
    Write the molecule block.
    """
    return f"""$molecule
{charge} {mult}
{mol.ToXYZ(header=False, confId=conf_id)}
$end"""


def _write_opt_block(ts: bool,
                     uhf: bool,
                     method: str,
                     basis: str,
                     scf_level: int,
                     nosymm: bool,
                     max_iter: int,
                     read: bool = False,
                     ) -> str:
    """
    Write the optimization block.

    Args:
        ts (bool): Whether the molecule is a TS.
        mult (int): The multiplicity of the molecule.
        method (str): The method to be used.
        basis (str): The basis to be used.
        read (bool, optional): Whether to read the hessian and scf from the previous job. Defaults to False.
    """
    job_type = "ts" if ts else "opt"
    args = [f'geom_opt_max_cycles {max_iter}',
            'geom_opt_tol_gradient 100',
            'geom_opt_tol_displacement 400',
            'geom_opt_tol_energy 33']
    if read:
        args += ['scf_guess read', 'geom_opt_hessian read']
    return _write_base_block(job_type=job_type,
                             uhf=uhf,
                             method=method,
                             basis=basis,
                             scf_level=scf_level,
                             nosymm=nosymm,
                             other_args=args,
                             )


def _write_libopt3_block(hess: str = '',
                         coord_type: str = 'delocalized',
                         max_iter: int = 100,
                         follow_freq: bool = False,
                         ) -> str:
    """
    Write the libopt3 block.

    Args:
        hess (str): The hessian file.
        coord_type (str): The coordinate system. Defaults to 'delocalized',
                          Delocalized Natural Internal Coordinates (QChem Default).
                          Other options are 'cartesian' and 'redundant'.
        max_iter (int): The maximum number of iterations. Defaults to 100.
    """
    items = ['$geom_opt',
             f'maxiter {max_iter}',
             f'coordinates {coord_type}']
    if not hess:
        items += ['initial_hessian exact  ! equivalent to calcfc',
                  'recompute_hessian recompute',
                  'recompute_hessian_cycles 1  ! equivalent to calcall']
    else:
        items += hess.splitlines()
    if follow_freq:
        items += ['hessian_verify recomputed',
                  'final_vibrational_analysis true']
    return '\n'.join(items + ['$end'])


def _write_irc_block(uhf: bool,
                     method: str,
                     basis: str,
                     direction: str,
                     max_iter: int,
                     step_size: int,
                     coord_type: str,
                     scf_level: int,
                     nosymm: bool,
                     ) -> str:
    """
    Write the irc block.
    """
    rpath_dir = {'forward': 1,
                 'reverse': -1,
                 }[direction.lower()]
    rpath_coords = {'mass-weighted': 0,
                    'cartesian': 1,
                    'z-matrix': 2,
                    'zmatrix': 2,
                    }[coord_type.lower()]
    args = [f'rpath_direction {rpath_dir}',
            f'rpath_coords {rpath_coords}',
            f'rpath_max_cycles {max_iter}',
            f'rpath_max_stepsize {step_size}',
            'scf_guess read',
            'geom_opt_hessian read']
    return _write_base_block(job_type='irc',
                             uhf=uhf,
                             method=method,
                             basis=basis,
                             scf_level=scf_level,
                             nosymm=nosymm,
                             other_args=args,)


def write_qchem_opt(mol,
                    conf_id: int = 0,
                    ts: bool = False,
                    charge: Optional[int] = None,
                    mult: Optional[int] = None,
                    method: str = "wb97x-d3",
                    basis: str = "def2-svp",
                    max_iter: int = 100,
                    coord_type: str = "redundant",
                    modify_internal: Optional[dict] = None,
                    scf_level: int = 8,
                    hess: str = '',
                    follow_freq: bool = False,
                    nosymm: bool = False,
                    qchemversion: str = "6.0",
                    **kwargs,
                    ) -> str:
    """
    Write the input file for QChem optimization calculation.
    Note, for version >= 6.0, Libopt3 is utilized by qchem, and hessian can be computed
    analytically during the optimization, while for version < 6.0, hessian is numerically updated
    by BFGS method.

    Args:
        mol (RDKitMol): The molecule to be run.
        conf_id (int, optional): The conformer ID to be run. Defaults to 0.
        ts (bool, optional): Whether the molecule is a TS. Defaults to False.
        charge (int, optional): The charge of the molecule. Defaults to None, to use the charge of mol.
        mult (int, optional): The multiplicity of the molecule. Defaults to None, to use the multiplicity of mol.
        method (str, optional): The method to be used. Defaults to "wb97x-d3".
        basis (str, optional): The basis set to be used. Defaults to "def2-svp".
        max_iter (int, optional): The maximum number of iterations. Defaults to 100.
        coord_type (str, optional): The coordinate type. Defaults to "cartesian".
        modify_internal (dict, optional): The internal coordinates to be modified. Defaults to None. # todo: implement this
        scf_level (int, optional): The scf convergence level. Defaults to 8 (recommanded by QChem).
        hess (dict, optional): The initial Hessian. Defaults to None. Only valid for version > 6.0

    Returns:
        str: The input file for QChem optimization calculation.
    """
    mult, charge = _get_mult_and_chrg(mol, mult, charge)

    mol_block = _write_molecule_block(mol, conf_id, charge, mult)

    if version.parse(qchemversion) >= version.parse("6.0") and not ts:
        opt_block = _write_opt_block(ts=ts,
                                     uhf=(mult > 1),
                                     method=method,
                                     basis=basis,
                                     scf_level=scf_level,
                                     nosymm=nosymm,
                                     max_iter=max_iter,
                                     read=False,)
        libopt3_block = _write_libopt3_block(hess=hess,
                                             coord_type=coord_type,
                                             max_iter=max_iter,
                                             follow_freq=follow_freq,)
        return '\n\n'.join([mol_block, opt_block, libopt3_block])

    freq_block = _write_base_block(job_type='freq',
                                   uhf=(mult > 1),
                                   method=method,
                                   basis=basis,
                                   scf_level=scf_level,
                                   nosymm=nosymm)
    read_mol_block = _write_read_molecule_block()
    opt_block = _write_opt_block(ts=ts,
                                 uhf=(mult > 1),
                                 method=method,
                                 basis=basis,
                                 scf_level=scf_level,
                                 nosymm=nosymm,
                                 max_iter=max_iter,
                                 read=True,)

    if not follow_freq:
        return '\n\n'.join([mol_block,
                            freq_block,
                            '@@@',
                            read_mol_block,
                            opt_block])
    else:
        return '\n\n'.join([mol_block,
                            freq_block,
                            '@@@',
                            read_mol_block,
                            opt_block,
                            '@@@',
                            read_mol_block,
                            freq_block])


def write_qchem_freq(mol,
                     conf_id: int = 0,
                     charge: Optional[int] = None,
                     mult: Optional[int] = None,
                     method: str = "wb97x-d3",
                     basis: str = "def2-svp",
                     scf_level: int = 8,
                     nosymm: bool = False,
                     **kwargs,
                     ):
    """
    Write the input file for QChem frequency calculation.
    """
    mult, charge = _get_mult_and_chrg(mol, mult, charge)
    mol_block = _write_molecule_block(mol=mol,
                                      conf_id=conf_id,
                                      charge=charge,
                                      mult=mult,)

    freq_block = _write_base_block(job_type='freq',
                                   uhf=(mult > 1),
                                   method=method,
                                   basis=basis,
                                   scf_level=scf_level,
                                   nosymm=nosymm,)
    return '\n\n'.join([mol_block, freq_block])


def write_qchem_irc(mol,
                    conf_id: int = 0,
                    charge: Optional[int] = None,
                    mult: Optional[int] = None,
                    method: str = "wb97x-d3",
                    basis: str = "def2-svp",
                    direction: str = "forward",
                    max_iter: int = 20,
                    step_size: int = 70,
                    coord_type: str = "mass-weighted",
                    scf_level: int = 8,
                    nosymm: bool = False,
                    **kwargs,
                    ) -> str:
    """
    Write the input file for QChem optimization calculation.
    Note, for version >= 6.0, Libopt3 is utilized by qchem, and hessian can be computed
    analytically during the optimization, while for version < 6.0, hessian is numerically updated
    by BFGS method.

    Args:
        mol (RDKitMol): The molecule to be run.
        conf_id (int, optional): The conformer ID to be run. Defaults to 0.
        ts (bool, optional): Whether the molecule is a TS. Defaults to False.
        charge (int, optional): The charge of the molecule. Defaults to None, to use the charge of mol.
        mult (int, optional): The multiplicity of the molecule. Defaults to None, to use the multiplicity of mol.
        method (str, optional): The method to be used. Defaults to "wb97x-d3".
        basis (str, optional): The basis set to be used. Defaults to "def2-svp".
        max_iter (int, optional): The maximum number of iterations. Defaults to 100.
        coord_type (str, optional): The coordinate type. Defaults to "mass-weighted". Note, the default in gaussian
                                    program is "mass-weighted", and the default in QChem program is cartesian.
        modify_internal (dict, optional): The internal coordinates to be modified. Defaults to None. # todo: implement this
        scf_level (int, optional): The scf convergence level. Defaults to 8 (recommanded by QChem).
        hess (dict, optional): The initial Hessian. Defaults to None. Only valid for version > 6.0

    Returns:
        str: The input file for QChem IRC optimization calculation.
    """
    mult, charge = _get_mult_and_chrg(mol, mult, charge)

    mol_block = _write_molecule_block(mol=mol,
                                      conf_id=conf_id,
                                      charge=charge,
                                      mult=mult,)

    freq_block = _write_base_block(job_type='freq',
                                   uhf=(mult > 1),
                                   method=method,
                                   basis=basis,
                                   scf_level=scf_level,
                                   nosymm=nosymm,)
    read_mol_block = _write_read_molecule_block()
    irc_block = _write_irc_block(uhf=(mult > 1),
                                 method=method,
                                 basis=basis,
                                 direction=direction,
                                 max_iter=max_iter,
                                 step_size=step_size,
                                 coord_type=coord_type,
                                 scf_level=scf_level,
                                 nosymm=nosymm,)
    return '\n\n'.join([mol_block,
                        freq_block,
                        '@@@',
                        read_mol_block,
                        irc_block])
