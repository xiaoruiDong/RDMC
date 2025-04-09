#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains functions for generating molecular fingerprints."""

from functools import lru_cache
from typing import Any, Callable, Optional, Protocol, Union

import numpy as np
import numpy.typing as npt
from rdkit import DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Mol
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganGenerator,
    GetRDKitFPGenerator,
    GetTopologicalTorsionGenerator,
)
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint


def rdkit_vector_to_array(
    vector: Union[DataStructs.ExplicitBitVect, DataStructs.UIntSparseIntVect],
    num_bits: Optional[int] = None,
    dtype: str = "int32",
) -> npt.NDArray[np.int_]:
    """Convert a RDKit vector to a numpy array.

    This function converts a RDKit
    :class:`rdkit.DataStructs.cDataStructs.ExplicitBitVect` or
    :class:`rdkit.DataStructs.cDataStructs.UIntSparseIntVect` vector to a numpy array.

    Args:
        vector (Union[DataStructs.ExplicitBitVect, DataStructs.UIntSparseIntVect]): RDkit Vector generated from fingerprint algorithms.
        num_bits (Optional[int], optional): The length of the vector, defaults to ``None``.
        dtype (str, optional):
            The data type of the output numpy array. Defaults to ``'int32'``.

    Returns:
        npt.NDArray[np.int_]: A numpy array of the vector.
    """
    num_bits = num_bits or len(vector)
    arr = np.zeros((num_bits,), dtype=dtype)
    DataStructs.ConvertToNumpyArray(vector, arr)  # overwrites arr
    return arr


class AvalonGenerator(Protocol):
    """A protocol for the Avalon fingerprint generator."""

    @staticmethod
    def GetCountFingerprintAsNumPy(mol: Mol) -> npt.NDArray[np.int_]:
        """Get the count fingerprint as a numpy array."""
        pass

    @staticmethod
    def GetFingerprintAsNumPy(mol: Mol) -> npt.NDArray[np.int_]:
        """Get the fingerprint as a numpy array."""
        pass


def GetAvalonGenerator(fpSize: int = 512, *args: Any, **kwargs: Any) -> AvalonGenerator:
    """Get the Avalon fingerprint generator.

    Args:
        fpSize (int, optional): The length of the fingerprint. Defaults to ``512``.
        *args (Any): Additional arguments for the generator.
        **kwargs (Any): Additional keyword arguments for the generator.

    Returns:
        AvalonGenerator: The Avalon fingerprint generator.
    """

    class AvalonGenerator:
        @staticmethod
        def GetCountFingerprintAsNumPy(mol: Mol) -> npt.NDArray[np.int_]:
            """Get the count fingerprint as a numpy array.

            Args:
                mol (Mol): The molecule to generate a fingerprint for.

            Returns:
                npt.NDArray[np.int_]: A numpy array of the count fingerprint.
            """
            return rdkit_vector_to_array(
                pyAvalonTools.GetAvalonCountFP(mol, nBits=fpSize),
                fpSize,
            )

        @staticmethod
        def GetFingerprintAsNumPy(mol: Mol) -> npt.NDArray[np.int_]:
            """Get the fingerprint as a numpy array.

            Args:
                mol (Mol): The molecule to generate a fingerprint for.

            Returns:
                npt.NDArray[np.int_]: A numpy array of the fingerprint.
            """
            return rdkit_vector_to_array(
                pyAvalonTools.GetAvalonFP(mol, nBits=fpSize),
                fpSize,
            )

    return AvalonGenerator()


class MACCSGenerator(Protocol):
    """A protocol for the MACCS fingerprint generator."""

    @staticmethod
    def GetFingerprintAsNumPy(mol: Mol) -> npt.NDArray[np.int_]:
        """Get the fingerprint as a numpy array."""
        pass


def GetMACCSGenerator(*args: Any, **kwargs: Any) -> MACCSGenerator:
    """Get the MACCS fingerprint generator.

    Args:
        *args (Any): Additional arguments for the generator.
        **kwargs (Any): Additional keyword arguments for the generator.

    Returns:
        MACCSGenerator: The MACCS fingerprint generator.
    """

    class MACCSGenerator:
        @staticmethod
        def GetFingerprintAsNumPy(mol: Mol) -> npt.NDArray[np.int_]:
            """Get the fingerprint as a numpy array.

            Args:
                mol (Mol): The molecule to generate a fingerprint for.

            Returns:
                npt.NDArray[np.int_]: A numpy array of the fingerprint.
            """
            return rdkit_vector_to_array(GetMACCSKeysFingerprint(mol))

    return MACCSGenerator()


@lru_cache
def get_fingerprint_generator(
    fp_type: str = "morgan",
    num_bits: int = 1024,
    count: bool = True,
    **kwargs: Any,
) -> Any:
    """Get the fingerprint generator for the specified type.

    Args:
        fp_type (str, optional): The type of fingerprint to generate. Options are:
            ``'atom_pair'``, ``'morgan'`` (default), ``'rdkit'``,
            ``'topological_torsion'``, ``'avalon'``, and ``'maccs'``.
        num_bits (int, optional): The length of the fingerprint. Default is ``1024``.
        count (bool, optional): Whether to generate a count fingerprint. Default is ``True``.
        **kwargs (Any): Additional arguments for the generator.a

    Returns:
        Any: The fingerprint generator.
    """
    fingerprint_dicts: dict[str, Callable[..., Any]] = {
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

    return getattr(generator, f"Get{'Count' if count else ''}FingerprintAsNumPy")


def get_fingerprint(
    mol: Mol,
    count: bool = False,
    fp_type: str = "morgan",
    num_bits: int = 2048,
    dtype: str = "int32",
    **kwargs: Any,
) -> npt.NDArray[np.int_]:
    """A helper function for generating molecular fingerprints.

    Please visit
    `RDKit <https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html>`_ for
    more information. This function also supports fingerprint-specific arguments,
    please visit the above website and find ``GetXXXGenerator`` for the corresponding
    argument names and allowed value types.

    Args:
        mol (Mol): The molecule to generate a fingerprint for.
        count (bool, optional): Whether to generate a count fingerprint. Default is ``False``.
        fp_type (str, optional): The type of fingerprint to generate. Options are:
            ``'atom_pair'``, ``'morgan'`` (default), ``'rdkit'``,
            ``'topological_torsion'``, ``'avalon'``, and ``'maccs'``.
        num_bits (int, optional): The length of the fingerprint. Default is ``2048``. It has no effect on
            ``'maccs'`` generator.
        dtype (str, optional): The data type of the output numpy array. Defaults to ``'int32'``.
        **kwargs (Any): Additional arguments for the generator.

    Returns:
        npt.NDArray[np.int_]: A numpy array of the molecular fingerprint.
    """
    fp_generator = get_fingerprint_generator(
        fp_type,
        num_bits,
        count,
        **kwargs,
    )

    return fp_generator(mol).astype(dtype)


def _get_rxn_fingerprint(
    rfp: npt.NDArray[np.int_],
    pfp: npt.NDArray[np.int_],
    mode: str,
) -> npt.NDArray[np.int_]:
    """Get the reaction fingerprint based on the mode.

    The mode can be one of the following:
        - ``'REAC'``: Reactant fingerprint
        - ``'PROD'``: Product fingerprint
        - ``'DIFF'``: Reactant - Product fingerprint
        - ``'SUM'``: Reactant + Product fingerprint
        - ``'REVD'``: Product - Reactant fingerprint

    Args:
        rfp (npt.NDArray[np.int_]): The reactant fingerprint.
        pfp (npt.NDArray[np.int_]): The product fingerprint.
        mode (str): The mode for the reaction fingerprint.

    Returns:
        npt.NDArray[np.int_]: The reaction fingerprint based on the mode.

    Raises:
        NotImplementedError: If the mode is not implemented.
    """
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
    **kwargs: Any,
) -> npt.NDArray[np.int_]:
    """Generate reaction fingerprints.

    based on the reactant molecule complex and the product molecule complex.

    Args:
        rmol (Mol): the reactant complex molecule object
        pmol (Mol): the product complex molecule object
        mode (str, optional): The fingerprint combination of ``'REAC'`` (reactant), ``'PROD'`` (product),
            ``'DIFF'`` (reactant - product), ``'REVD'`` (product - reactant), ``'SUM'`` (reactant + product),
            separated by ``'_'``. Defaults to ``REAC_DIFF``, with the fingerprint to be a concatenation of
            reactant fingerprint and the difference between the reactant complex and the product complex.
        fp_type (str, optional): The type of fingerprint to generate. Options are:
            ``'atom_pair'``, ``'morgan'`` (default), ``'rdkit'``,
            ``'topological_torsion'``, ``'avalon'``, and ``'maccs'``.
        count (bool, optional): Whether to generate a count fingerprint. Default is ``False``.
        num_bits (int, optional): The length of the molecular fingerprint. For a mode with N blocks, the eventual length
            is ``num_bits * N``. Default is ``2048``. It has no effect on ``'maccs'`` generator.
        **kwargs (Any): Additional arguments for the generator.

    Returns:
        npt.NDArray[np.int_]: A numpy array of the molecular fingerprint.
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
