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


from rdkit import Chem

from rdmc.resonance.pathfinder_rewrite import (
    find_adj_lone_pair_multiple_bond_delocalization_paths,
    find_adj_lone_pair_radical_delocalization_paths,
    find_adj_lone_pair_radical_multiple_bond_delocalization_paths,
    find_allyl_delocalization_paths,
    find_lone_pair_multiple_bond_paths,
)

import pytest


@pytest.mark.parametrize(
    "smiles",
    [
        "[CH2]C=C",  # allyl radical
        "[N]C=[CH]",  # nitrogenated birad
    ],
)
def test_find_allyl_delocalization_paths(smiles):
    mol = Chem.MolFromSmiles(smiles)
    paths = find_allyl_delocalization_paths(mol)
    assert paths


@pytest.mark.parametrize(
    "smiles",
    [
        "[N-]=[N+]=N",  # azide
        "NC=O",
        "[N-]=[N+]=O",
        "N#[N+][O-]",
        "[NH-][N+]#N",
        "OS(O)=[N+]=[N-]",
        "N[N+]([O-])=O",
    ],
)
def test_find_lone_pair_multiple_bond_paths(smiles):
    mol = Chem.MolFromSmiles(smiles)
    paths = find_lone_pair_multiple_bond_paths(mol)
    assert paths


@pytest.mark.parametrize(
    "smiles",
    [
        "[O]N=O",
        "[O-][N+]=O",
        "[O]SO",
        "[O+:1]=[N-:2]",
    ],
)
def test_find_adj_lone_pair_radical_delocalization_paths(smiles):
    mol = Chem.MolFromSmiles(smiles)
    paths = find_adj_lone_pair_radical_delocalization_paths(mol)
    assert paths


@pytest.mark.parametrize(
    "smiles",
    [
        "O=[SH](=O)[O]",
    ],
)
def test_find_adj_lone_pair_multiple_bond_delocalization_paths(smiles):
    mol = Chem.MolFromSmiles(smiles)
    paths = find_adj_lone_pair_multiple_bond_delocalization_paths(mol)
    assert paths


@pytest.mark.parametrize(
    "smiles",
    [
        "N#[S]",
        "O[S](=O)=O"
    ],
)
def test_find_adj_lone_pair_radical_multiple_bond_delocalization_paths(smiles):
    mol = Chem.MolFromSmiles(smiles)
    paths = find_adj_lone_pair_radical_multiple_bond_delocalization_paths(mol)
    assert paths
