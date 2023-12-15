#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the resonance module.
"""

import pytest

from rdkit import Chem

from rdmc.rdtools.resonance.rdkit_backend import generate_resonance_structures


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
def test_generate_charged_resonance_structures(smi, keep_isomorphic, expected_num):
    """
    Test the function for generating charged resonance structures.
    """
    mol = Chem.MolFromSmiles(smi, smi_params)
    res_mols = generate_resonance_structures(mol, keep_isomorphic=keep_isomorphic)

    assert len(res_mols) == expected_num


@pytest.mark.parametrize(
    "keep_isomorphic, kekulize, expected_num",
    [
        (
            # Without kekulization, RDKit returns 1 resonance structures anyway
            False,
            False,
            1,
        ),
        (
            # Without kekulization, RDKit returns 1 resonance structures anyway
            True,
            False,
            1,
        ),
        (
            # With kekulization, RDKit returns 2 resonance structures
            # we generate kekulized structures, so the number is 2
            True,
            True,
            2,
        ),
        (
            # With kekulization, RDKit returns 1 resonance structures
            # if not keep isomorphic
            False,
            True,
            1,
        ),
    ],
)
def test_benzene(keep_isomorphic, kekulize, expected_num):
    """
    Test the function for generating radical resonance structures for benzene.
    """

    # Test case 1: benzene
    smi = "c1ccccc1"
    mol = Chem.MolFromSmiles(smi, smi_params)
    res_mols = generate_resonance_structures(
        mol, keep_isomorphic=keep_isomorphic, kekulize=kekulize
    )

    assert len(res_mols) == expected_num


@pytest.mark.parametrize(
    "keep_isomorphic, kekulize, expected_num",
    [
        (
            # Without kekulization, RDKit returns 3 resonance structures
            # 2 with radical site on the ring and 1 with radial site on the alkyl chain
            False,
            False,
            3,
        ),
        (
            # Without kekulization, RDKit returns 4 resonance structures
            # 2 ortho, 1 para, and 1 on the alkyl chain
            True,
            False,
            4,
        ),
        (
            # With kekulization and keeping isomorphic structures,
            # RDKit returns 5 resonance structures
            # 2 ortho, 1 para, and 2 on the alkyl chain
            True,
            True,
            5,
        ),
        (
            # With kekulization and not keeping isomorphic structures,
            # RDKit returns 3 resonance structures
            # 1 ortho, 1 para, and 1 on the alkyl chain
            False,
            True,
            3,
        ),
    ],
)
def test_phenylethyl_radical(
    keep_isomorphic, kekulize, expected_num
):
    """
    Test the function for generating radical resonance structures for phenylethyl.
    """
    smi = "c1ccccc1[CH]C"
    mol = Chem.MolFromSmiles(smi, smi_params)
    res_mols = generate_resonance_structures(
        mol, keep_isomorphic=keep_isomorphic, kekulize=kekulize
    )

    assert len(res_mols) == expected_num


@pytest.mark.parametrize(
    "keep_isomorphic, kekulize, expected_num",
    [
        (
            # Without kekulization, RDKit returns 3 resonance structures
            # 2 with radical site on the ring and 1 with radial site on the alkyl chain
            False,
            False,
            3,
        ),
        (
            # Without kekulization, RDKit returns 4 resonance structures
            # 2 ortho, 1 para, and 1 on the alkyl chain
            True,
            False,
            4,
        ),
        (
            # With kekulization and keeping isomorphic structures,
            # RDKit returns 5 resonance structures
            # 2 ortho, 1 para, and 2 on the alkyl chain
            True,
            True,
            5,
        ),
        (
            # With kekulization and not keeping isomorphic structures,
            # RDKit returns 3 resonance structures
            # 1 ortho, 1 para, and 1 on the alkyl chain
            False,
            True,
            3,
        ),
    ],
)
def test_phenylethyl_cation(
    keep_isomorphic, kekulize, expected_num
):
    """
    Test the function for generating radical resonance structures for phenylethyl.
    """

    # Test case 1: benzene
    smi = "c1ccccc1[CH+]C"
    mol = Chem.MolFromSmiles(smi, smi_params)
    res_mols = generate_resonance_structures(
        mol, keep_isomorphic=keep_isomorphic, kekulize=kekulize
    )

    assert len(res_mols) == expected_num


@pytest.mark.parametrize(
    "keep_isomorphic, kekulize, expected_num",
    [
        (
            # Without kekulization and not keeping isomorphic structures,
            # RDKit returns 2 resonance structures, 1 at end, 1 at middle
            False,
            False,
            2,
        ),
        (
            # Without kekulization and not keeping isomorphic structures,
            # RDKit returns 2 resonance structures, 2 at end, 1 at middle
            True,
            False,
            3,
        ),
        (
            # With kekulization and not keeping isomorphic structures,
            # RDKit returns 2 resonance structures, 2 at end, 1 at middle
            True,
            True,
            3,
        ),
        (
            # With kekulization and not keeping isomorphic structures,
            # RDKit returns 2 resonance structures, 1 at end, 1 at middle
            False,
            True,
            2,
        ),
    ],
)
def test_bisallyl_radical(
    keep_isomorphic, kekulize, expected_num
):
    """
    Test the function for generating radical resonance structures for phenylethyl.
    """
    smi = "C=C[CH]C=C"
    mol = Chem.MolFromSmiles(smi, smi_params)
    res_mols = generate_resonance_structures(
        mol, keep_isomorphic=keep_isomorphic, kekulize=kekulize
    )

    assert len(res_mols) == expected_num


@pytest.mark.parametrize(
    "keep_isomorphic, kekulize, expected_num",
    # Kekulize / keep_isomorphic expect to have no effect
    [
        (
            False,
            False,
            2,
        ),
        (
            True,
            False,
            2,
        ),
        (
            True,
            True,
            2,
        ),
        (
            False,
            True,
            2,
        ),
    ],
)
def test_alpha_ketone_radical(keep_isomorphic, kekulize, expected_num):
    smi = "[CH2]C=O"
    mol = Chem.MolFromSmiles(smi, smi_params)
    res_mols = generate_resonance_structures(
        mol, keep_isomorphic=keep_isomorphic, kekulize=kekulize
    )

    assert len(res_mols) == expected_num


@pytest.mark.parametrize(
    "keep_isomorphic, kekulize, expected_num",
    # Kekulize / keep_isomorphic expect to have no effect
    [
        (
            False,
            False,
            1,
        ),
        (
            True,
            False,
            1,
        ),
        (
            True,
            True,
            1,
        ),
        (
            False,
            True,
            1,
        ),
    ],
)
def test_beta_unsaturated_ketone(keep_isomorphic, kekulize, expected_num):
    smi = "C=CC=O"
    mol = Chem.MolFromSmiles(smi, smi_params)
    res_mols = generate_resonance_structures(
        mol, keep_isomorphic=keep_isomorphic, kekulize=kekulize
    )

    assert len(res_mols) == expected_num
