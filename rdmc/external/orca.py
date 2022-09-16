#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
A module contains functions to interface with Orca.
"""


def write_orca_irc(mol, confId=0, maxcores=1000, nprocs=1, method="XTB2", mult=1):

    orca_irc_input = f"""! {method} TightSCF IRC
    %maxcore {maxcores}
    %pal
    nprocs {nprocs}
    end
    *xyz {mol.GetFormalCharge()} {mult}
    {mol.ToXYZ(header=False, confId=confId)}
    *
    """
    return orca_irc_input


def write_orca_opt(mol, confId=0, maxcores=1000, nprocs=1, method="XTB2", mult=1):

    orca_opt_input = f"""! {method} OptTS NumFreq
    %maxcore {maxcores}
    %pal
    nprocs {nprocs}
    end
    %geom
    Calc_Hess true # Calculate Hessian in the beginning
    NumHess true  # Numerical hessian for semiempirical methods
    Recalc_Hess 5  # Recalculate the Hessian every 5 steps
    end
    *xyz {mol.GetFormalCharge()} {mult}
    {mol.ToXYZ(header=False, confId=confId)}
    *
    """
    return orca_opt_input

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
