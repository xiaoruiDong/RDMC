#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the resonance module.
"""

import pytest

from rdmc import RDKitMol
from rdmc.resonance import generate_radical_resonance_structures


def test_generate_radical_resonance_structures():
    """
    Test the function for generating radical resonance structures.
    """
    # Currently couldn't handle charged molecules
    charged_smis = ['[CH3+]', '[OH-]', '[CH2+]C=C', '[CH2-]C=C']
    for smi in charged_smis:
        with pytest.raises(AssertionError):
            generate_radical_resonance_structures(
                RDKitMol.FromSmiles(smi)
            )

    # Test case 1: benzene
    smi = 'c1ccccc1'
    mol = RDKitMol.FromSmiles(smi)
    # Without kekulization, RDKit returns 1 resonance structures
    assert len(generate_radical_resonance_structures(
        mol,
        unique=False,
        kekulize=False,
    )) == 1
    # With kekulization, RDKit returns 2 resonance structures
    assert len(generate_radical_resonance_structures(
        mol,
        unique=False,
        kekulize=True,
    )) == 2
    # With kekulization, RDKit returns 1 resonance structures
    # during uniquifyication without considering atom map
    assert len(generate_radical_resonance_structures(
        mol,
        unique=True,
        consider_atommap=False,
        kekulize=True,
    )) == 1
    # With kekulization, RDKit returns 2 resonance structures
    # during uniquifyication with considering atom map
    assert len(generate_radical_resonance_structures(
        mol,
        unique=True,
        consider_atommap=True,
        kekulize=True,
    )) == 2

    # Test case 2: 1-Phenylethyl radical
    smi = 'c1ccccc1[CH]C'
    mol = RDKitMol.FromSmiles(smi)
    # Without kekulization, RDKit returns 4 resonance structures
    # 3 with radical site on the ring and 1 with differently kekulized benzene
    assert len(generate_radical_resonance_structures(
        mol,
        unique=False,
        kekulize=False,
    )) == 4
    # With kekulization, RDKit returns 5 resonance structures
    # 3 with radical site on the ring and 2 with differently kekulized benzene
    assert len(generate_radical_resonance_structures(
        mol,
        unique=False,
        kekulize=True,
    )) == 5
    # With filtration, kekulization, and not considering atom map,
    # There will be 3 resonance structures, 2 with radical site on the ring
    # and 1 with radial site on the alkyl chain
    assert len(generate_radical_resonance_structures(
        RDKitMol.FromSmiles(smi),
        unique=True,
        consider_atommap=False,
        kekulize=True,
    )) == 3
    # With filtration and considering atom map, and without kekulization,
    # RDKit returns 4 structures, 3 with radical site on the ring
    # and 1 with radial site on the alkyl chain
    assert len(generate_radical_resonance_structures(
        RDKitMol.FromSmiles(smi),
        unique=True,
        consider_atommap=True
    )) == 4

    # Test case 3: Phenyl radical
    smi = 'C=C[CH]C=C'
    # No dependence on kekulization
    assert len(generate_radical_resonance_structures(
        RDKitMol.FromSmiles(smi),
        unique=False,
        kekulize=True,
    )) == len(generate_radical_resonance_structures(
        RDKitMol.FromSmiles(smi),
        unique=False,
        kekulize=False,
    )) == 3
    # With filtering and considering atom map, RDKit returns 3 resonance structures
    # radical site at two ends and the middle
    assert len(generate_radical_resonance_structures(
        RDKitMol.FromSmiles(smi),
        unique=True,
        consider_atommap=True,
        kekulize=True,
    )) == 3
    # With filtration and not considering atom map, RDKit returns 2 structures
    # radical site at the end and the middle
    assert len(generate_radical_resonance_structures(
        RDKitMol.FromSmiles(smi),
        unique=True,
        consider_atommap=False,
        kekulize=True,
    )) == 2
