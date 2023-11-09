#!/usr/bin/env python3

###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2023 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_dev@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################


from rdmc import RDKitMol
from rdmc.resonance.pathfinder_rewrite import (
    find_adj_lone_pair_multiple_bond_delocalization_paths,
    find_adj_lone_pair_radical_delocalization_paths,
    find_adj_lone_pair_radical_multiple_bond_delocalization_paths,
    find_allyl_delocalization_paths,
    find_lone_pair_multiple_bond_paths,
)

import pytest


class TestFindAllylDelocalizationPaths:
    """
    test the find_allyl_delocalization_paths method
    """

    def test_allyl_radical(self):
        smiles = "[CH2]C=C"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_allyl_delocalization_paths(mol.ToRWMol())
        assert paths

    def test_nitrogenated_birad(self):
        smiles = "[N]C=[CH]"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_allyl_delocalization_paths(mol.ToRWMol())
        assert paths


class TestFindLonePairMultipleBondPaths:
    """
    test the find_lone_pair_multiple_bond_paths method
    """

    def test_azide(self):
        smiles = "[N-]=[N+]=N"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.ToRWMol())
        assert paths

    def test_nh2cho(self):
        smiles = "NC=O"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.ToRWMol())
        assert paths

    def test_n2oa(self):
        smiles = "[N-]=[N+]=O"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.ToRWMol())
        assert paths

    def test_n2ob(self):
        smiles = "N#[N+][O-]"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.ToRWMol())
        assert paths

    def test_hn3(self):
        smiles = "[NH-][N+]#N"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.ToRWMol())
        assert paths

    def test_sn2(self):
        smiles = "OS(O)=[N+]=[N-]"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.ToRWMol())
        assert paths

    def test_h2nnoo(self):
        smiles = "N[N+]([O-])=O"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_lone_pair_multiple_bond_paths(mol.ToRWMol())
        assert paths


class TestFindAdjLonePairRadicalDelocalizationPaths:
    """
    test the find_lone_pair_radical_delocalization_paths method
    """

    def test_no2a(self):
        smiles = "[O]N=O"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_adj_lone_pair_radical_delocalization_paths(mol.ToRWMol())
        assert paths

    def test_no2b(self):
        smiles = "[O-][N+]=O"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_adj_lone_pair_radical_delocalization_paths(mol.ToRWMol())
        assert paths

    def test_hoso(self):
        smiles = "[O]SO"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_adj_lone_pair_radical_delocalization_paths(mol.ToRWMol())
        assert paths

    def test_double_bond(self):
        mol = RDKitMol.FromSmiles("[O+:1]=[N-:2]")
        paths = find_adj_lone_pair_radical_delocalization_paths(mol.ToRWMol())
        assert paths


class TestFindAdjLonePairMultipleBondDelocalizationPaths:
    """
    test the find_lone_pair_multiple_bond_delocalization_paths method
    """

    def test_sho3(self):
        smiles = "O=[SH](=O)[O]"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_adj_lone_pair_multiple_bond_delocalization_paths(mol.ToRWMol())
        assert paths


class TestFindAdjLonePairRadicalMultipleBondDelocalizationPaths:
    """
    test the find_lone_pair_radical_multiple_bond_delocalization_paths method
    """

    def test_ns(self):
        smiles = "N#[S]"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_adj_lone_pair_radical_multiple_bond_delocalization_paths(
            mol.ToRWMol()
        )
        assert paths

    def test_hso3(self):
        smiles = "O[S](=O)=O"
        mol = RDKitMol.FromSmiles(smiles)
        paths = find_adj_lone_pair_radical_multiple_bond_delocalization_paths(
            mol.ToRWMol()
        )
        assert paths
