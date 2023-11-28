#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdmc.resonance.utils import is_equivalent_structure


class ResonanceAlgoRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(some_class):
            cls._registry[name] = some_class
            return some_class

        return decorator

    @classmethod
    def get(cls, name: str):
        return cls._registry.get(name)


def _merge_resonance_structures(
    known_structs: list,
    new_structs: list,
    keep_isomorphic: bool = False,
):
    """
    Merge resonance structures by removing duplicates.
    This is only used in combining resonance structures from different backends.

    Args:
        known_structs (list): A list of known resonance structures. This list will be modified in place.
        new_structs (list): A list of new resonance structures.
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
    mol: "Chem.RWMol",
    keep_isomorphic: bool = False,
    copy: bool = True,
    backend: str = "all",
    **kwargs,
):
    """
    Generate resonance structures for a molecule.
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

    algo = ResonanceAlgoRegistry.get(backend)
    if algo:
        return algo(
            mol=mol,
            keep_isomorphic=keep_isomorphic,
            copy=copy,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend {backend}")
