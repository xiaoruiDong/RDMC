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
from rdtools.featurizer import get_fingerprint
from rdmc.mol import RDKitMol

logging.basicConfig(level=logging.DEBUG)


smis = [
    "Fc1cccc(C2(c3nnc(Cc4cccc5ccccc45)o3)CCOCC2)c1",
    "O=C(NCc1ccnc(Oc2ccc(F)cc2)c1)c1[nH]nc2c1CCCC2",
    "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
]


@pytest.fixture(params=smis)
def mol(request):
    return RDKitMol.FromSmiles(request.param, addHs=False, removeHs=True)


@pytest.fixture(params=smis)
def mol_with_hs(request):
    return RDKitMol.FromSmiles(request.param, addHs=True, removeHs=False)


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


@pytest.fixture(params=['int16', 'int32', 'float16', 'float32'])
def dtype(request):
    return request.param


generator_name = {
    ('morgan', True): 'GetHashedMorganFingerprint',
    ('morgan', False): 'GetMorganFingerprintAsBitVect',
    ('atom_pair', True): 'GetHashedAtomPairFingerprint',
    ('atom_pair', False): 'GetHashedAtomPairFingerprintAsBitVect',
    ('topological_torsion', True): 'GetHashedTopologicalTorsionFingerprint',
    ('topological_torsion', False): 'GetHashedTopologicalTorsionFingerprintAsBitVect',
}  # The names


def get_allchem_fingerprint(mol, count, fp_type, num_bits, **kwargs):
    # This is another way with different APIs people used to calculate fingerprints
    generator = getattr(AllChem, generator_name[(fp_type, count)])
    features_vec = generator(mol, nBits=num_bits, **kwargs)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


class TestFingerprint:

    def test_morgan_fingerprint(self, count, mol, num_bits, radius, dtype):
        """
        Test the ``get_fingerprint`` function get a reproducible count fingerprint for morgan fingerprints.
        """
        fp = get_fingerprint(
            mol, count=count, num_bits=num_bits, fp_type='morgan', radius=radius, dtype=dtype
        )
        fp_exp = get_allchem_fingerprint(mol, count=count, num_bits=num_bits, fp_type='morgan', radius=radius)

        assert np.allclose(fp, fp_exp)
        assert fp.dtype == np.dtype(dtype)

    def test_morgan_fingerprint_for_mol_with_hs(self, count, mol_with_hs, num_bits, radius, dtype):
        """
        Test the ``get_fingerprint`` function get a reproducible count fingerprint for morgan fingerprints.
        """
        fp = get_fingerprint(
            mol_with_hs, count=count, num_bits=num_bits, fp_type='morgan', radius=radius, dtype=dtype
        )
        fp_exp = get_allchem_fingerprint(mol_with_hs, count=count, num_bits=num_bits, fp_type='morgan', radius=radius)

        assert np.allclose(fp, fp_exp)
        assert fp.dtype == np.dtype(dtype)

    def test_atom_pair_fingerprint(self, mol, count, num_bits, dtype):
        """
        Test the ``get_fingerprint`` function get a reproducible count fingerprint for AtomPair and TopologicalTorsion Fingerprints.
        """
        fp = get_fingerprint(
            mol, count=count, num_bits=num_bits, fp_type='atom_pair', dtype=dtype
        )
        fp_exp = get_allchem_fingerprint(mol, count=count, num_bits=num_bits, fp_type='atom_pair')

        assert np.allclose(fp, fp_exp)
        assert fp.dtype == np.dtype(dtype)

    def test_atom_pair_fingerprint_for_mol_with_hs(self, mol_with_hs, count, num_bits, dtype):
        """
        Test the ``get_fingerprint`` function get a reproducible count fingerprint for AtomPair and TopologicalTorsion Fingerprints.
        """
        fp = get_fingerprint(
            mol_with_hs, count=count, num_bits=num_bits, fp_type='atom_pair', dtype=dtype
        )
        fp_exp = get_allchem_fingerprint(mol_with_hs, count=count, num_bits=num_bits, fp_type='atom_pair')

        assert np.allclose(fp, fp_exp)
        assert fp.dtype == np.dtype(dtype)

    def test_topological_torsion_fingerprint(self, mol, count, num_bits, dtype):
        """
        Test the ``get_fingerprint`` function get a reproducible count fingerprint for AtomPair and TopologicalTorsion Fingerprints.
        """
        fp = get_fingerprint(
            mol, count=count, num_bits=num_bits, fp_type="topological_torsion", dtype=dtype
        )
        fp_exp = get_allchem_fingerprint(
            mol, count=count, num_bits=num_bits, fp_type="topological_torsion",
        )

        assert np.allclose(fp, fp_exp)
        assert fp.dtype == np.dtype(dtype)

    def test_topological_torsion_fingerprint_for_mol_with_hs(self, mol_with_hs, count, num_bits, dtype):
        """
        Test the ``get_fingerprint`` function get a reproducible count fingerprint for AtomPair and TopologicalTorsion Fingerprints.
        """
        fp = get_fingerprint(
            mol_with_hs,
            count=count,
            num_bits=num_bits,
            fp_type="topological_torsion",
            dtype=dtype,
        )
        fp_exp = get_allchem_fingerprint(
            mol_with_hs,
            count=count,
            num_bits=num_bits,
            fp_type="topological_torsion",
        )

        assert np.allclose(fp, fp_exp)
        assert fp.dtype == np.dtype(dtype)

    def test_rdkitfp_bit(self, mol, num_bits, dtype):
        """
        Test the ``get_fingerprint`` function get a reproducible bit-based fingerprint for RDKitFP.
        I don't find the the count-based fingerprint for RDKitFP implementation in AllChem.
        """
        fp = get_fingerprint(mol, count=False, num_bits=num_bits, fp_type='rdkit', dtype=dtype)

        features_vec = AllChem.RDKFingerprint(mol, fpSize=num_bits)
        fp_exp = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, fp_exp)

        assert np.allclose(fp, fp_exp)
        assert fp.dtype == np.dtype(dtype)

    def test_rdkitfp_bit_for_mol_with_hs(self, mol_with_hs, num_bits, dtype):
        """
        Test the ``get_fingerprint`` function get a reproducible bit-based fingerprint for RDKitFP.
        I don't find the the count-based fingerprint for RDKitFP implementation in AllChem.
        """
        fp = get_fingerprint(mol_with_hs, count=False, num_bits=num_bits, fp_type='rdkit', dtype=dtype)

        features_vec = AllChem.RDKFingerprint(mol_with_hs, fpSize=num_bits)
        fp_exp = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, fp_exp)

        assert np.allclose(fp, fp_exp)
        assert fp.dtype == np.dtype(dtype)
