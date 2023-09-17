#!/usr/bin/env python3

"""
Unit tests for the utils module.
"""

import logging
import unittest

import numpy as np
from rdkit import Chem
from rdmc.utils import (parse_xyz_by_openbabel,
                        parse_xyz_by_jensen,
                        reverse_map,
                        openbabel_mol_to_rdkit_mol)

import pytest

logging.basicConfig(level=logging.DEBUG)

################################################################################

@pytest.fixture(params=[
    ('[C-]#[O+]',"""2

C      0.559061    0.000000    0.000000
O     -0.559061    0.000000    0.000000"""),
    ('C', """5

C      0.005119   -0.010620    0.006014
H      0.549668    0.755438   -0.596981
H      0.749764   -0.587944    0.585285
H     -0.586753   -0.652136   -0.676092
H     -0.717798    0.495262    0.681774"""),
    ('CO', """6

C     -0.350753   -0.005073   -0.018028
O      0.964370   -0.362402   -0.260120
H     -0.598457    0.061649    1.052373
H     -0.986255   -0.815675   -0.462603
H     -0.657857    0.926898   -0.501077
H      1.628951    0.194603    0.189455"""),
    ('CC', """8

C     -0.745523    0.041444    0.011706
C      0.747340    0.002879    0.001223
H     -1.129707   -0.637432    0.814421
H     -1.184900    1.025570    0.199636
H     -1.199871   -0.334603   -0.938879
H      1.084153   -0.736520   -0.773193
H      1.226615    0.961738   -0.268073
H      1.201893   -0.323076    0.953158"""),
    ('[CH3+]', """4

C     -0.006776    0.000178    0.000029
H      1.023025   -0.296754   -0.005142
H     -0.777660   -0.758724   -0.000060
H     -0.238590    1.055301    0.005172"""),
    ('[OH-]', """2

O      0.490127    0.000000    0.000000
H     -0.490127    0.000000    0.000000"""),
])
def smi_xyz_pair(request):
    """
    The SMILES and XYZ pair for testing.
    """
    return request.param


@pytest.fixture(params=[
    ('[H]', """1

H     0.000000    0.000000    0.000000"""),
    ('[H].[H]', """2

H     0.000000    0.000000    0.000000
H     0.000000    0.000000    10.000000"""),
])
def smi_xyz_pair_hs(request):
    """
    The SMILES and XYZ pair of molecules only with Hs for testing.
    """
    return request.param


@pytest.fixture
def smi_xyz_charge(smi_xyz_pair):
    """
    The SMILES, XYZ and charge pair for testing.
    """
    smi, xyz = smi_xyz_pair
    mol_smi = Chem.MolFromSmiles(smi)
    charge = Chem.GetFormalCharge(mol_smi)
    return smi, xyz, charge


@pytest.fixture
def smi_xyz_charge_hs(smi_xyz_pair_hs):
    """
    The SMILES, XYZ and charge of molecules only with Hs for testing.
    """
    smi, xyz = smi_xyz_pair_hs
    mol_smi = Chem.MolFromSmiles(smi)
    charge = Chem.GetFormalCharge(mol_smi)
    return smi, xyz, charge


class TestUtils:
    """
    The general class to test functions in the utils module
    """

    def test_reverse_match(self):
        """
        Test the functionality to reverse a mapping.
        """
        map = [1, 2, 3, 4, 5, 17, 18, 19, 20, 21, 22, 23, 24, 25, 6, 7, 8,
               9, 10, 11, 12, 13, 14, 15, 16, 26, 27, 28, 29, 30, 31, 32, 33, 34,
               35, 36, 37, 38, 39]
        r_map = [0, 1, 2, 3, 4, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 5,
                 6, 7, 8, 9, 10, 11, 12, 13, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                 34, 35, 36, 37, 38]

        assert r_map == reverse_map(map)
        np.testing.assert_equal(np.array(r_map), reverse_map(map, as_list=False))

    def test_openbabel_mol_to_rdkit_mol_single_atom_xyz(self):
        """
        Test if a single-atom openbabel mol with all-zero xyz coordinates can be successfully
        converted to rdkit mol with a conformer embedded.
        """
        xyz = '1\n[Geometry 1]\nH      0.0000000000    0.0000000000    0.0000000000\n'
        obmol = parse_xyz_by_openbabel(xyz)
        rdmol = openbabel_mol_to_rdkit_mol(obmol)

        assert rdmol.GetNumConformers() == 1
        assert rdmol.GetNumAtoms() == 1
        assert np.array_equal(rdmol.GetConformer().GetPositions(),
                              np.array([[0., 0., 0.,]])
                              )

    def test_parse_xyz_by_jensen_builtin(self, smi_xyz_charge):
        """
        Test if XYZ can be parsed correctly using RDKit's built-in module
        """
        # Since RDKit has its own unit / functional test, here we just test whether
        # the function works properly with some simple molecules
        smi, xyz, charge = smi_xyz_charge
        mol_xyz = parse_xyz_by_jensen(xyz=xyz,
                                      charge=charge,
                                      allow_charged_fragments=(charge != 0),
                                      force_rdmc=False)
        assert mol_xyz.GetNumAtoms() == len(xyz.splitlines()) - 2
        assert smi == Chem.MolToSmiles(Chem.RemoveAllHs(mol_xyz),
                                       canonical=True)

    def test_parse_xyz_by_jensen_builtin_of_hs(self, smi_xyz_charge_hs):
        """
        Test if XYZ of molecules with only H atoms can be parsed correctly using RDKit's built-in module
        """
        smi, xyz, charge = smi_xyz_charge_hs
        mol_xyz = parse_xyz_by_jensen(xyz=xyz,
                                      charge=charge,
                                      allow_charged_fragments=(charge != 0),
                                      force_rdmc=False)
        assert mol_xyz.GetNumAtoms() == len(xyz.splitlines()) - 2
        assert smi == Chem.MolToSmiles(mol_xyz)

    def test_parse_xyz_by_jensen_rdmc(self, smi_xyz_charge):
        """
        Test if XYZ can be parsed correctly using xyz2mol implemented in RDMC
        """
        smi, xyz, charge = smi_xyz_charge
        mol_xyz = parse_xyz_by_jensen(xyz=xyz,
                                charge=charge,
                                allow_charged_fragments=(charge != 0),
                                force_rdmc=True)
        assert mol_xyz.GetNumAtoms() == len(xyz.splitlines()) - 2
        assert smi == Chem.MolToSmiles(Chem.RemoveAllHs(mol_xyz),
                                       canonical=True)

    def test_parse_xyz_by_jensen_rdmc_of_hs(self, smi_xyz_charge_hs):
        """
        Test if XYZ of molecules with only H atoms can be parsed correctly using xyz2mol implemented in RDMC
        """
        smi, xyz, charge = smi_xyz_charge_hs
        mol_xyz = parse_xyz_by_jensen(xyz=xyz,
                                      charge=charge,
                                      allow_charged_fragments=(charge != 0),
                                      force_rdmc=True)
        assert mol_xyz.GetNumAtoms() == len(xyz.splitlines()) - 2
        assert smi == Chem.MolToSmiles(mol_xyz)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=3))
