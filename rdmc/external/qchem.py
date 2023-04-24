#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A module contains functions to read QChem output file.
"""


# TODO: Let the input format be more flexible
def write_qchem_ts_opt(mol, confId=0, method="wB97x-d3", basis="def2-tzvp", mult=1):

    qchem_ts_opt_input = (f'$rem\n'
                          f'JOBTYPE FREQ\n'
                          f'METHOD {method}\n'
                          f'BASIS {basis}\n'
                          f'UNRESTRICTED TRUE\n'
                          f'SCF_ALGORITHM DIIS\n'
                          f'MAX_SCF_CYCLES 100\n'
                          f'SCF_CONVERGENCE 8\n'
                          f'SYM_IGNORE TRUE\n'
                          f'SYMMETRY FALSE\n'
                          f'WAVEFUNCTION_ANALYSIS FALSE\n'
                          f'$end\n\n'
                          f'$molecule\n'
                          f'{mol.GetFormalCharge()} {mult}\n'
                          f'{mol.ToXYZ(header=False, confId=confId)}\n'
                          f'$end\n\n'
                          f'@@@\n\n'
                          f'$molecule\n'
                          f'read\n'
                          f'$end\n\n'
                          f'$rem\n'
                          f'JOBTYPE TS\n'
                          f'METHOD {method}\n'
                          f'BASIS {basis}\n'
                          f'UNRESTRICTED TRUE\n'
                          f'SCF_GUESS READ\n'
                          f'GEOM_OPT_HESSIAN READ\n'
                          f'SCF_ALGORITHM DIIS\n'
                          f'MAX_SCF_CYCLES 100\n'
                          f'SCF_CONVERGENCE 8\n'
                          f'SYM_IGNORE TRUE\n'
                          f'SYMMETRY FALSE\n'
                          f'GEOM_OPT_MAX_CYCLES 100\n'
                          f'GEOM_OPT_TOL_GRADIENT 100\n'
                          f'GEOM_OPT_TOL_DISPLACEMENT 400\n'
                          f'GEOM_OPT_TOL_ENERGY 33\n'
                          f'WAVEFUNCTION_ANALYSIS FALSE\n'
                          f'$end\n\n'
                          f'@@@\n\n'
                          f'$molecule\n'
                          f'read\n'
                          f'$end\n\n'
                          f'$rem\n'
                          f'JOBTYPE FREQ\n'
                          f'METHOD {method}\n'
                          f'BASIS {basis}\n'
                          f'UNRESTRICTED TRUE\n'
                          f'SCF_ALGORITHM DIIS\n'
                          f'MAX_SCF_CYCLES 100\n'
                          f'SCF_CONVERGENCE 8\n'
                          f'SYM_IGNORE TRUE\n'
                          f'SYMMETRY FALSE\n'
                          f'WAVEFUNCTION_ANALYSIS FALSE\n'
                          f'$end\n\n'
    )
    return qchem_ts_opt_input


def write_qchem_opt(mol, confId=0, method="wB97x-d3", basis="def2-tzvp", mult=1):

    qchem_opt_input = (f'$rem\n'
                      f'JOBTYPE OPT\n'
                      f'METHOD {method}\n'
                      f'BASIS {basis}\n'
                      f'UNRESTRICTED TRUE\n'
                      f'SCF_ALGORITHM DIIS\n'
                      f'MAX_SCF_CYCLES 100\n'
                      f'SCF_CONVERGENCE 8\n'
                      f'SYM_IGNORE TRUE\n'
                      f'SYMMETRY FALSE\n'
                      f'GEOM_OPT_MAX_CYCLES 100\n'
                      f'GEOM_OPT_TOL_GRADIENT 100\n'
                      f'GEOM_OPT_TOL_DISPLACEMENT 400\n'
                      f'GEOM_OPT_TOL_ENERGY 33\n'
                      f'WAVEFUNCTION_ANALYSIS FALSE\n'
                      f'$end\n\n'
                      f'$molecule\n'
                      f'{mol.GetFormalCharge()} {mult}\n'
                      f'{mol.ToXYZ(header=False, confId=confId)}\n'
                      f'$end\n\n'
                      f'@@@\n\n'
                      f'$molecule\n'
                      f'read\n'
                      f'$end\n\n'
                      f'$rem\n'
                      f'JOBTYPE FREQ\n'
                      f'METHOD {method}\n'
                      f'BASIS {basis}\n'
                      f'UNRESTRICTED TRUE\n'
                      f'SCF_GUESS READ\n'
                      f'SCF_ALGORITHM DIIS\n'
                      f'MAX_SCF_CYCLES 100\n'
                      f'SCF_CONVERGENCE 8\n'
                      f'SYM_IGNORE TRUE\n'
                      f'SYMMETRY FALSE\n'
                      f'WAVEFUNCTION_ANALYSIS FALSE\n'
                      f'$end\n\n'
    )
    return qchem_opt_input


def write_qchem_irc(mol, confId=0, method="wB97x-d3", basis="def2-tzvp", mult=1, direction='forward'):

    qchem_irc_input = (f'$rem\n'
                      f'JOBTYPE FREQ\n'
                      f'METHOD {method}\n'
                      f'BASIS {basis}\n'
                      f'UNRESTRICTED TRUE\n'
                      f'SCF_ALGORITHM DIIS\n'
                      f'MAX_SCF_CYCLES 100\n'
                      f'SCF_CONVERGENCE 8\n'
                      f'SYM_IGNORE TRUE\n'
                      f'SYMMETRY FALSE\n'
                      f'WAVEFUNCTION_ANALYSIS FALSE\n'
                      f'$end\n\n'
                      f'$molecule\n'
                      f'{mol.GetFormalCharge()} {mult}\n'
                      f'{mol.ToXYZ(header=False, confId=confId)}\n'
                      f'$end\n\n'
                      f'@@@\n\n'
                      f'$molecule\n'
                      f'read\n'
                      f'$end\n\n'
                      f'$rem\n'
                      f'JOBTYPE RPATH\n'
                      f'RPATH_DIRECTION {int(direction == 'forward')}\n'
                      f'METHOD {method}\n'
                      f'BASIS {basis}\n'
                      f'UNRESTRICTED TRUE\n'
                      f'SCF_GUESS READ\n'
                      f'GEOM_OPT_HESSIAN READ\n'
                      f'SCF_ALGORITHM DIIS\n'
                      f'MAX_SCF_CYCLES 100\n'
                      f'SCF_CONVERGENCE 8\n'
                      f'SYM_IGNORE TRUE\n'
                      f'SYMMETRY FALSE\n'
                      f'WAVEFUNCTION_ANALYSIS FALSE\n'
                      f'$end\n\n'
    )
    return qchem_irc_input
