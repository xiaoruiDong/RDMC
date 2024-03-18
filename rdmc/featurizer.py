#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains functions for generating molecular fingerprints.
"""

import numpy as np

from rdkit.Chem import Mol
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganGenerator,
    GetRDKitFPGenerator,
    GetTopologicalTorsionGenerator
)


fingerprint_dicts = {
    'atom_pair': GetAtomPairGenerator,
    'morgan': GetMorganGenerator,
    'rdkit': GetRDKitFPGenerator,
    'topological_torsion': GetTopologicalTorsionGenerator,
}


def get_fingerprint(
    mol: Mol,
    count: bool = False,
    fp_type: str = 'morgan',
    num_bits: int = 2048,
    dtype: str = 'int32',
    **kwargs,
) -> np.ndarray:
    """
    A helper function for generating molecular fingerprints. Please visit
    `RDKit <https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html>`_ for
    more information. This function also supports fingerprint-specific arguments,
    please visit the above website and find ``GetXXXGenerator`` for the corresponding
    argument names and allowed value types.

    Args:
        mol: The molecule to generate a fingerprint for.
        count: Whether to generate a count fingerprint. Default is ``False``.
        fp_type: The type of fingerprint to generate. Options are:
                 ``'atom_pair'``, ``'morgan'`` (default), ``'rdkit'``,
                 and ``'topological_torsion'``.
        num_bits: The length of the fingerprint. Default is ``2048``.

    Returns:
        np.ndarray: A numpy array of the molecular fingerprint.
    """
    generator = fingerprint_dicts[fp_type.lower()](
        fpSize=num_bits,
        **kwargs,
    )
    return getattr(
        generator,
        f'Get{"Count" if count else ""}FingerprintAsNumPy'
    )(mol).astype(dtype)
