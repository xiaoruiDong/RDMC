#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the fingerprint module. This module doesn't test the actual values (which are tested
in RDKit's CI pipeline), but rather the functionality of the ``get_fingerprint`` function.
"""

import logging

import numpy as np
import pytest

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdmc import RDKitMol
from rdmc.featurizer import get_fingerprint

logging.basicConfig(level=logging.DEBUG)


smis = ['Fc1cccc(C2(c3nnc(Cc4cccc5ccccc45)o3)CCOCC2)c1',
        'O=C(NCc1ccnc(Oc2ccc(F)cc2)c1)c1[nH]nc2c1CCCC2',]


@pytest.fixture(params=smis)
def mol(request):
    return Chem.MolFromSmiles(request.param)


@pytest.fixture(params=smis)
def rdkitmol(request):
    return RDKitMol.FromSmiles(request.param, addHs=False)


@pytest.fixture(
    params=[1024, 2048, 4096]
)
def num_bits(request):
    return request.param


@pytest.fixture(
    params=[True, False]
)
def count(request):
    return request.param


@pytest.fixture(
    params=[2, 3]
)
def radius(request):
    return request.param


generator_name = {
    ('Morgan', True): 'GetHashedMorganFingerprint',
    ('Morgan', False): 'GetMorganFingerprintAsBitVect',
    ('AtomPair', True): 'GetHashedAtomPairFingerprint',
    ('AtomPair', False): 'GetHashedAtomPairFingerprintAsBitVect',
    ('TopologicalTorsion', True): 'GetHashedTopologicalTorsionFingerprint',
    ('TopologicalTorsion', False): 'GetHashedTopologicalTorsionFingerprintAsBitVect',
    ('RDKitFP', False): 'RDKFingerprint',
}  # The names


def get_allchem_fingerprint(mol, count, fp_type, num_bits, **kwargs):
    # This is another way with different APIs people used to calculate fingerprints
    generator = getattr(AllChem, generator_name[(fp_type, count)])
    features_vec = generator(mol, nBits=num_bits, **kwargs)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


class TestFingerprint:
    @pytest.mark.parametrize('fp_type, count',
                             [('Morgan', True),
                              ('Morgan', False),
                              ])
    def test_morgan_fingerprint(self, fp_type, count, mol, num_bits, radius):
        """
        Test the ``get_fingerprint`` function get a reproducible count fingerprint for morgan fingerprints.
        """
        assert np.isclose(get_fingerprint(mol, count=count, num_bits=num_bits, fp_type=fp_type, radius=radius),
                          get_allchem_fingerprint(mol, count=count, num_bits=num_bits, fp_type=fp_type, radius=radius)).all()

    @pytest.mark.parametrize('fp_type, count',
                             [('Morgan', True),
                              ('Morgan', False),
                              ])
    def test_morgan_fingerprint_rdkitmol(self, fp_type, count, rdkitmol, num_bits, radius):
        """
        Test the ``get_fingerprint`` function get a reproducible count fingerprint for morgan fingerprints and RDKitMol.
        """
        assert np.isclose(get_fingerprint(rdkitmol, count=count, num_bits=num_bits, fp_type=fp_type, radius=radius),
                          get_allchem_fingerprint(rdkitmol.ToRWMol(), count=count, num_bits=num_bits, fp_type=fp_type, radius=radius)).all()

    @pytest.mark.parametrize('fp_type, count',
                             [('AtomPair', True),
                              ('AtomPair', False),
                              ('TopologicalTorsion', True),
                              ('TopologicalTorsion', False),
                              ])
    def test_atompair_and_topological_torsion_fingerprint_rdkitmol(self, fp_type, count, rdkitmol, num_bits):
        """
        Test the ``get_fingerprint`` function get a reproducible count fingerprint for AtomPair and TopologicalTorsion Fingerprints.
        """
        assert np.isclose(get_fingerprint(rdkitmol, count=count, num_bits=num_bits, fp_type=fp_type),
                          get_allchem_fingerprint(rdkitmol.ToRWMol(), count=count, num_bits=num_bits, fp_type=fp_type)).all()

    def test_rdkitfp_bit(self, rdkitmol, num_bits):
        """
        Test the ``get_fingerprint`` function get a reproducible bit-based fingerprint for RDKitFP.
        I don't find the the count-based fingerprint for RDKitFP implementation in AllChem.
        """
        features_vec = AllChem.RDKFingerprint(rdkitmol.ToRWMol(), fpSize=num_bits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        assert np.isclose(get_fingerprint(rdkitmol, count=False, num_bits=num_bits, fp_type='rdkitfp'),
                          features).all()
