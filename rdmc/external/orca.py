#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
A module contains functions to interface with Orca.
"""


def write_orca_irc(mol, confId=0, maxcores=1000, nprocs=1):

    orca_irc_input = f"""! XTB1 TightSCF IRC
    %maxcore {maxcores}
    %pal
    nprocs {nprocs}
    end
    *xyz {mol.GetFormalCharge()} {mol.GetSpinMultiplicity()}
    {mol.ToXYZ(header=False, confId=confId)}
    *
    """
    return orca_irc_input
