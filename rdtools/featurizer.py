#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains functions for generating molecular fingerprints.
"""

from functools import lru_cache
from typing import Optional

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem import Mol
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganGenerator,
    GetRDKitFPGenerator,
    GetTopologicalTorsionGenerator,
)


def rdkit_vector_to_array(
    vector,
    num_bits: Optional[int] = None,
    dtype: str = "int32",
) -> np.array:
    """
    A helper function to convert a sparse rdkit.DataStructs.cDataStructs.ExplicitBitVect
    or rdkit.DataStructs.cDataStructs.UIntSparseIntVect vector to a numpy array.

    Args:
        vector: RDkit Vector generated from fingerprint algorithms
        num_bits (int, optional): The length of the vector, defaults to `None`
        dtype (str, optional)
    """
    num_bits = num_bits or len(vector)
    arr = np.zeros((num_bits,), dtype=dtype)
    DataStructs.ConvertToNumpyArray(vector, arr)  # overwrites arr
    return arr


def GetAvalonGenerator(fpSize=512, *args, **kwargs):

    class AvalonGenerator:

        @staticmethod
        def GetCountFingerprintAsNumPy(mol):
            return rdkit_vector_to_array(
                pyAvalonTools.GetAvalonCountFP(mol, nBits=fpSize),
                fpSize,
            )

        @staticmethod
        def GetFingerprintAsNumPy(mol):
            return rdkit_vector_to_array(
                pyAvalonTools.GetAvalonFP(mol, nBits=fpSize),
                fpSize,
            )

    return AvalonGenerator()


def GetMACCSGenerator(*args, **kwargs):

    class MACCSGenerator:

        @staticmethod
        def GetFingerprintAsNumPy(mol):
            return rdkit_vector_to_array(GetMACCSKeysFingerprint(mol))

    return MACCSGenerator()


@lru_cache
def get_fingerprint_generator(
    fp_type: str = "morgan",
    num_bits: int = 1024,
    count: bool = True,
    **kwargs,
):

    fingerprint_dicts = {
        "avalon": GetAvalonGenerator,
        "atom_pair": GetAtomPairGenerator,
        "morgan": GetMorganGenerator,
        "maccs": GetMACCSGenerator,
        "rdkit": GetRDKitFPGenerator,
        "topological_torsion": GetTopologicalTorsionGenerator,
    }

    generator = fingerprint_dicts[fp_type.lower()](
        fpSize=num_bits,
        **kwargs,
    )

    return getattr(generator, f'Get{"Count" if count else ""}FingerprintAsNumPy')


def get_fingerprint(
    mol: Mol,
    count: bool = False,
    fp_type: str = "morgan",
    num_bits: int = 2048,
    dtype: str = "int32",
    **kwargs,
) -> np.ndarray:
    """
    A helper function for generating molecular fingerprints. Please visit
    `RDKit <https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html>`_ for
    more information. This function also supports fingerprint-specific arguments,
    please visit the above website and find ``GetXXXGenerator`` for the corresponding
    argument names and allowed value types.

    Args:
        mol (Mol): The molecule to generate a fingerprint for.
        count (bool, optional): Whether to generate a count fingerprint. Default is ``False``.
        fp_type (str,  optional): The type of fingerprint to generate. Options are:
            ``'atom_pair'``, ``'morgan'`` (default), ``'rdkit'``,
            ``'topological_torsion'``, ``'avalon'``, and ``'maccs'``.
        num_bits (int, optional): The length of the fingerprint. Default is ``2048``. It has no effect on
            ``'maccs'`` generator.
        dtype (str, optional): The data type of the output numpy array. Defaults to ``'int32'``.

    Returns:
        np.ndarray: A numpy array of the molecular fingerprint.
    """
    fp_generator = get_fingerprint_generator(
        fp_type,
        num_bits,
        count,
        **kwargs,
    )

    return fp_generator(mol).astype(dtype)


def _get_rxn_fingerprint(rfp: np.array, pfp: np.array, mode: str):
    if mode == "REAC":
        return rfp
    elif mode == "PROD":
        return pfp
    elif mode == "DIFF":
        return rfp - pfp
    elif mode == "SUM":
        return rfp + pfp
    elif mode == "REVD":
        return pfp - rfp
    else:
        raise NotImplementedError(f"The reaction mode ({mode}) is not implemented.")


def get_rxn_fingerprint(
    rmol: Mol,
    pmol: Mol,
    mode: str = "REAC_DIFF",
    fp_type: str = "morgan",
    count: bool = False,
    num_bits: int = 2048,
    **kwargs,
):
    """
    A helper function for generating molecular fingerprints based on the reactant molecule
    complex and the product molecule complex.

    Args:
        rmol (Mol): the reactant complex molecule object
        pmol (Mol): the product complex molecule object
        mode (str): The fingerprint combination of ``'REAC'`` (reactant), ``'PROD'`` (product),
            ``'DIFF'`` (reactant - product), ``'REVD'`` (product - reactant), ``'SUM'`` (reactant + product),
            separated by ``'_'``. Defaults to ``REAC_DIFF``, with the fingerprint to be a concatenation of
            reactant fingerprint and the difference between the reactant complex and the product complex.
        fp_type (str,  optional): The type of fingerprint to generate. Options are:
            ``'atom_pair'``, ``'morgan'`` (default), ``'rdkit'``,
            ``'topological_torsion'``, ``'avalon'``, and ``'maccs'``.
        num_bits (int, optional): The length of the molecular fingerprint. For a mode with N blocks, the eventual length
            is ``num_bits * N``. Default is ``2048``. It has no effect on ``'maccs'`` generator.
        dtype (str, optional): The data type of the output numpy array. Defaults to ``'int32'``.
    """
    rfp, pfp = (
        get_fingerprint(
            mol,
            count=count,
            fp_type=fp_type,
            num_bits=num_bits,
            **kwargs,
        )
        for mol in (rmol, pmol)
    )

    seg_fps = [_get_rxn_fingerprint(rfp, pfp, rxn_mode) for rxn_mode in mode.split("_")]

    return np.concatenate(seg_fps)
