#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the gaussian input writer.
"""

import pytest

from rdmc import RDKitMol
from rdmc.external.inpwriter.gaussian import (write_gaussian_opt,
                                              write_gaussian_freq,
                                              write_gaussian_irc)
from rdmc.external.xtb_tools.utils import XTB_GAUSSIAN_PL


@pytest.fixture
def mol():
    return RDKitMol.FromXYZ('H      0.349009    0.000000    0.000000\nH     -0.349009    0.000000    0.000000',
                            header=False)

def test_write_gaussian_opt(mol):
    """
    Test the gaussian input writer for geometry optimizations.
    """
    # test_1
    test_input_1 = """%mem=1gb
%nprocshared=1
#P opt=(calcall,noeig,maxcycle=100,tight) scf=(tight) b3lyp/def2-tzvp

title

0 1
H      0.349009    0.000000    0.000000
H     -0.349009    0.000000    0.000000


"""
    gaussian_input = write_gaussian_opt(mol,
                                        method='b3lyp/def2-tzvp')
    assert gaussian_input == test_input_1

    test_input_2 = f"""%mem=16gb
%nprocshared=4
#P opt=(ts,calcfc,nomicro,noeig,maxcycle=200,cartesian) scf=(verytight) nosymm freq
external="{XTB_GAUSSIAN_PL} --gfn 2 -P"

title

0 1
H      0.349009    0.000000    0.000000
H     -0.349009    0.000000    0.000000

B 1 2


"""
    gaussian_input = write_gaussian_opt(mol,
                                        ts=True,
                                        method='gfn2-xtb',
                                        memory=16,
                                        nprocs=4,
                                        max_iter=200,
                                        coord_type='cartesian',
                                        scf_level='verytight',
                                        opt_level='',
                                        hess='calcfc',
                                        follow_freq=True,
                                        nosymm=True,
                                        extra='B 1 2')
    assert gaussian_input == test_input_2


def test_write_gaussian_freq(mol):
    """
    Test the gaussian input writer for frequency calculations.
    """
    # test_1
    test_input_1 = """%mem=1gb
%nprocshared=1
#P freq scf=(tight) b3lyp/def2-tzvp

title

0 1
H      0.349009    0.000000    0.000000
H     -0.349009    0.000000    0.000000


"""
    gaussian_input = write_gaussian_freq(mol,
                                         method='b3lyp/def2-tzvp')
    assert gaussian_input == test_input_1

    test_input_2 = f"""%mem=16gb
%nprocshared=4
#P freq scf=(verytight) nosymm
external="{XTB_GAUSSIAN_PL} --gfn 2 -P"

title

0 1
H      0.349009    0.000000    0.000000
H     -0.349009    0.000000    0.000000


"""
    gaussian_input = write_gaussian_freq(mol,
                                         method='gfn2-xtb',
                                         memory=16,
                                         nprocs=4,
                                         scf_level='verytight',
                                         nosymm=True)
    assert gaussian_input == test_input_2


def test_write_gaussian_irc(mol):
    """
    Test the gaussian input writer for IRC calculations.
    """
    # test_1
    test_input_1 = """%mem=1gb
%nprocshared=1
#P irc=(forward,hpc,calcall,maxcycle=20,maxpoints=100,stepsize=7,tight,massweighted) scf=(tight) b3lyp/def2-tzvp

title

0 1
H      0.349009    0.000000    0.000000
H     -0.349009    0.000000    0.000000


"""
    gaussian_input = write_gaussian_irc(mol,
                                        method='b3lyp/def2-tzvp')
    assert gaussian_input == test_input_1

    test_input_2 = f"""%mem=16gb
%nprocshared=4
#P irc=(reverse,lqc,calcfc,nomicro,maxcycle=100,maxpoints=50,stepsize=5,cartesian) scf=(verytight) nosymm
external="{XTB_GAUSSIAN_PL} --gfn 2 -P"

title

0 1
H      0.349009    0.000000    0.000000
H     -0.349009    0.000000    0.000000


"""
    gaussian_input = write_gaussian_irc(mol,
                                        ts=True,
                                        method='gfn2-xtb',
                                        memory=16,
                                        nprocs=4,
                                        direction='reverse',
                                        algorithm='lqc',
                                        max_iter=100,
                                        max_points=50,
                                        step_size=5,
                                        coord_type='cartesian',
                                        irc_level='',
                                        scf_level='verytight',
                                        hess='calcfc',
                                        nosymm=True)
    assert gaussian_input == test_input_2

