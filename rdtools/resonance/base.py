#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modules for base operations of resonance structure generation and analysis."""

from typing import Any, Callable, Literal

from rdkit import Chem

from rdtools.resonance.utils import is_equivalent_structure


class ResonanceAlgoRegistry:
    """Registry for resonance algorithms."""

    _registry: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[..., Any]:
        """Register a resonance algorithm.

        Args:
            name (str): Name of the resonance algorithm.

        Returns:
            Callable[..., Any]: A decorator that registers the algorithm.
        """

        def decorator(some_function: Callable[..., Any]) -> Callable[..., Any]:
            """Decorator that registers the algorithm.

            Args:
                some_function (Callable[..., Any]): The function to register.

            Returns:
                Callable[..., Any]: The decorated function.
            """
            cls._registry[name] = some_function
            return some_function

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        """Get a registered resonance algorithm by name.

        Args:
            name (str): Name of the resonance algorithm.

        Returns:
            Callable[..., Any]: The registered algorithm function.
        """
        return cls._registry[name]


def _merge_resonance_structures(
    known_structs: list[Chem.Mol],
    new_structs: list[Chem.Mol],
    keep_isomorphic: bool = False,
) -> None:
    """Merge resonance structures by removing duplicates.

    This is only used in combining resonance structures from different backends.
    It modifies the ``known_structs`` list in place.

    Args:
        known_structs (list[Chem.Mol]): A list of known resonance structures. This list will be modified in place.
        new_structs (list[Chem.Mol]): A list of new resonance structures.
        keep_isomorphic (bool, optional): If keep isomorphic resonance structures. Defaults to ``False``.
    """
    if len(new_structs) <= 1:
        # The new algorithm failed or only return the original molecule
        return

    structs_to_add = []
    # Each method has its own de-duplicate method
    # so don't append to known structs until all checked
    for new_struct in new_structs:
        for known_struct in known_structs:
            if is_equivalent_structure(
                ref_mol=known_struct,
                qry_mol=new_struct,
                isomorphic_equivalent=not keep_isomorphic,
            ):
                break
        else:
            structs_to_add.append(new_struct)
    known_structs.extend(structs_to_add)


def generate_resonance_structures(
    mol: Chem.RWMol,
    keep_isomorphic: bool = False,
    copy: bool = True,
    backend: Literal["all", "rdkit", "rmg"] = "all",
    **kwargs: Any,
) -> list[Chem.Mol]:
    """Generate resonance structures for a molecule.

    Args:
        mol (Chem.RWMol): A charged molecule in RDKit RWMol.
        keep_isomorphic (bool, optional): If keep isomorphic resonance structures. Defaults to ``False``.
        copy (bool, optional): If copy the input molecule. Defaults to ``True``.
        backend (Literal["all", "rdkit", "rmg"], optional): The backend to use for generating resonance structures. Defaults to ``"all"``.
        **kwargs (Any): Additional arguments for the resonance algorithms.


    Returns:
        list[Chem.Mol]: A list of resonance structures.

    Raises:
        ValueError: If the backend is invalid.
    """
    if backend == "all":
        algos = list(ResonanceAlgoRegistry._registry.values())
        known_structs = algos[0](
            mol=mol,
            keep_isomorphic=keep_isomorphic,
            copy=copy,
            **kwargs,
        )
        for algo in algos[1:]:
            new_structs = algo(
                mol=mol,
                keep_isomorphic=keep_isomorphic,
                copy=copy,
                **kwargs,
            )
            _merge_resonance_structures(known_structs, new_structs)

        return known_structs

    try:
        algo = ResonanceAlgoRegistry.get(backend)
    except KeyError:
        raise ValueError(f"Invalid backend {backend}")

    return algo(
        mol=mol,
        keep_isomorphic=keep_isomorphic,
        copy=copy,
        **kwargs,
    )
