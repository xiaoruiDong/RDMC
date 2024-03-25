#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the gaussian input writer.
"""

import os

import pytest

from rdmc import RDKitMol
from rdmc.external.inpwriter.gaussian import (
    write_gaussian_opt,
    write_gaussian_freq,
    write_gaussian_irc,
    write_gaussian_gsm,
)
from rdmc.external.inpwriter.utils import XTB_GAUSSIAN_PERL_PATH


@pytest.fixture
def mol1():
    return RDKitMol.FromXYZ(
        "H      0.349009    0.000000    0.000000\nH     -0.349009    0.000000    0.000000",
        header=False,
    )


@pytest.fixture
def mol2():
    return RDKitMol.FromXYZ(
        f"""C         -1.81332        2.97999        0.00000
H         -0.74332        2.97999        0.00000
H         -2.16998        1.97232       -0.04779
H         -2.16998        3.52521       -0.84878
H         -2.16999        3.44244        0.89656""",
        header=False,
    )


def test_write_gaussian_opt(mol1, mol2):
    """
    Test the gaussian input writer for geometry optimizations.
    """
    # test_1
    test_input_1 = """%mem=1gb
%nprocshared=1
#P opt=(calcall,noeig,maxcycle=100,tight) scf=(tight) freq b3lyp/def2-tzvp

title

0 1
H      0.349009    0.000000    0.000000
H     -0.349009    0.000000    0.000000

B 1 2

"""
    gaussian_input = write_gaussian_opt(
        mol1, method="b3lyp/def2-tzvp", follow_freq=True, extra="B 1 2"
    )
    assert gaussian_input == test_input_1

    test_input_2 = f"""%mem=16gb
%nprocshared=4
#P opt=(ts,calcfc,nomicro,noeig,maxcycle=200,cartesian) scf=(verytight) nosymm
external="{XTB_GAUSSIAN_PERL_PATH} --gfn 2 -P"

title

0 1
C     -1.813320    2.979990    0.000000
H     -0.743320    2.979990    0.000000
H     -2.169980    1.972320   -0.047790
H     -2.169980    3.525210   -0.848780
H     -2.169990    3.442440    0.896560

"""
    gaussian_input = write_gaussian_opt(
        mol2,
        ts=True,
        method="gfn2-xtb",
        memory=16,
        nprocs=4,
        max_iter=200,
        coord_type="cartesian",
        scf_level="verytight",
        opt_level="",
        hess="calcfc",
        nosymm=True,
    )
    assert gaussian_input == test_input_2


def test_write_gaussian_freq(mol1, mol2):
    """
    Test the gaussian input writer for frequency calculations.
    """
    # test_1
    test_input_1 = """%mem=1gb
%nprocshared=1
#P freq scf=(tight) b3lyp/def2tzvp

title

0 1
H      0.349009    0.000000    0.000000
H     -0.349009    0.000000    0.000000

"""
    gaussian_input = write_gaussian_freq(mol1, method="b3lyp/def2tzvp")
    assert gaussian_input == test_input_1

    test_input_2 = f"""%mem=16gb
%nprocshared=4
#P freq scf=(verytight) nosymm
external="{XTB_GAUSSIAN_PERL_PATH} --gfn 2 -P"

title

0 1
C     -1.813320    2.979990    0.000000
H     -0.743320    2.979990    0.000000
H     -2.169980    1.972320   -0.047790
H     -2.169980    3.525210   -0.848780
H     -2.169990    3.442440    0.896560

"""
    gaussian_input = write_gaussian_freq(
        mol2, method="gfn2-xtb", memory=16, nprocs=4, scf_level="verytight", nosymm=True
    )
    assert gaussian_input == test_input_2


def test_write_gaussian_irc(mol1, mol2):
    """
    Test the gaussian input writer for IRC calculations.
    """
    # test_1
    test_input_1 = """%mem=1gb
%nprocshared=1
#P irc=(forward,hpc,calcall,maxcycle=20,maxpoints=100,stepsize=7,tight,massweighted) scf=(tight) b3lyp/def2tzvp

title

0 1
H      0.349009    0.000000    0.000000
H     -0.349009    0.000000    0.000000

"""
    gaussian_input = write_gaussian_irc(
        mol1,
        method="b3lyp/def2tzvp",
    )
    assert gaussian_input == test_input_1

    test_input_2 = f"""%mem=16gb
%nprocshared=4
#P irc=(reverse,lqc,calcfc,nomicro,maxcycle=100,maxpoints=50,stepsize=5,cartesian) scf=(verytight) nosymm
external="{XTB_GAUSSIAN_PERL_PATH} --gfn 2 -P"

title

0 1
C     -1.813320    2.979990    0.000000
H     -0.743320    2.979990    0.000000
H     -2.169980    1.972320   -0.047790
H     -2.169980    3.525210   -0.848780
H     -2.169990    3.442440    0.896560

"""
    gaussian_input = write_gaussian_irc(
        mol2,
        ts=True,
        method="gfn2-xtb",
        memory=16,
        nprocs=4,
        direction="reverse",
        algorithm="lqc",
        max_iter=100,
        max_points=50,
        step_size=5,
        coord_type="cartesian",
        irc_level="",
        scf_level="verytight",
        hess="calcfc",
        nosymm=True,
    )
    assert gaussian_input == test_input_2


def test_write_gaussian_gsm():
    """
    Test the gaussian input writer for GSM calculations.
    """
    test_input_1 = """%mem=1gb
%nprocshared=1
#N force scf=(xqc) nosymm b3lyp/def2tzvp

title"""
    gaussian_input = write_gaussian_gsm(method="b3lyp/def2tzvp")
    assert gaussian_input == test_input_1

    test_input_2 = f"""%mem=16gb
%nprocshared=4
%chk=check.chk
#N force scf=(xqc) nosymm
external="{XTB_GAUSSIAN_PERL_PATH} --gfn 2 -P"

title"""
    gaussian_input = write_gaussian_gsm(
        method="gfn2-xtb",
        memory=16,
        nprocs=4,
        extra_sys_settings="%chk=check.chk",
    )
    assert gaussian_input == test_input_2
