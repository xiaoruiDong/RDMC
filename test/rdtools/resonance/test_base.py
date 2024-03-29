#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the base resonance generation function.
"""

import pytest

from rdkit import Chem

from rdtools.resonance import generate_resonance_structures


smi_params = Chem.SmilesParserParams()
smi_params.removeHs = False
smi_params.sanitize = True


@pytest.mark.parametrize(
    "smi, keep_isomorphic, expected_num",
    [
        ("[CH3+]", True, 1),
        ("[CH3+]", False, 1),
        ("[OH-]", True, 1),
        ("[OH-]", False, 1),
        ("[CH2+]C=C", True, 2),
        ("[CH2+]C=C", False, 1),
        ("[CH2+]C=CC", True, 2),
        ("[CH2+]C=CC", False, 2),
        ("[CH2-]C=C", True, 2),
        ("[CH2-]C=C", False, 1),
        ("[CH2-]C=CC", True, 2),
        ("[CH2-]C=CC", False, 2),
        ("C(=O)[O-]", True, 2),
        ("C(=O)[O-]", False, 1),
    ],
)
def test_rdkit_backend(
    smi,
    keep_isomorphic,
    expected_num,
):
    """
    Test the function for generating charged resonance structures.
    """
    mol = Chem.MolFromSmiles(smi, smi_params)
    res_mols = generate_resonance_structures(
        mol,
        keep_isomorphic=keep_isomorphic,
        backend="rdkit",
    )

    assert len(res_mols) == expected_num


@pytest.mark.parametrize(
    "smi, expected_num",
    [
        ("[CH3+]", 1),
        ("[OH-]", 1),
        ("[CH2+]C=C", 1),  # RMG's template doesn't work for allyl cation
        ("[CH2+]C=CC", 1),  # RMG's template doesn't work for allyl cation
        ("[CH2-]C=C", 1),  # RMG's template doesn't work for allyl anion
        ("[CH2-]C=CC", 1),  # RMG's template doesn't work for allyl anion
        ("C(=O)[O-]", 2),
    ],
)
def test_rmg_backend(
    smi,
    expected_num,
):
    """
    Test the function for generating charged resonance structures.
    """
    mol = Chem.MolFromSmiles(smi, smi_params)
    res_mols = generate_resonance_structures(
        mol,
        keep_isomorphic=True,
        backend="rmg",
    )

    assert len(res_mols) == expected_num


@pytest.mark.parametrize(
    "smi, expected_num",
    [
        ("[CH3+]", 1),
        ("[OH-]", 1),
        ("c1ccccc1", 3),  # 1 aromatic form, 2 kekulized form
        ("c1ccccc1[CH2]", 6),  # 1 aromatic, 2 kekulized, 2 ortho, 1 para
        # TODO: add a test case where both algorithm can only find partial results
    ],
)
def test_use_all_backends(
    smi,
    expected_num,
):
    """
    Test the function for generating charged resonance structures.
    """
    mol = Chem.MolFromSmiles(smi, smi_params)
    res_mols = generate_resonance_structures(
        mol,
        keep_isomorphic=True,
        backend="all",
    )

    assert len(res_mols) == expected_num


def test_bad_backend():
    mol = Chem.MolFromSmiles("C", smi_params)  # some randome molecule
    with pytest.raises(ValueError):
        generate_resonance_structures(
            mol,
            backend="bad_backend",
        )
