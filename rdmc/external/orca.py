#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
A module contains functions to interface with Orca.
"""


def write_orca_irc(mol, confId=0, maxcores=1000, nprocs=1, method="XTB2"):

    orca_irc_input = f"""! {method} TightSCF IRC
    %maxcore {maxcores}
    %pal
    nprocs {nprocs}
    end
    *xyz {mol.GetFormalCharge()} 1
    {mol.ToXYZ(header=False, confId=confId)}
    *
    """
    return orca_irc_input


def write_orca_opt(mol, confId=0, maxcores=1000, nprocs=1, method="XTB2"):

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
    *xyz {mol.GetFormalCharge()} 1
    {mol.ToXYZ(header=False, confId=confId)}
    *
    """
    return orca_opt_input
