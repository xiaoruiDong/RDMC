#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains functions for generating resonance structures.
This is a rewrite of the original resonance_rmg.py module.
"""

from collections import defaultdict
from typing import List, Optional, Tuple
import logging

from rdkit import Chem

from rdmc.resonance import filtration
from rdmc.resonance.pathfinder_rewrite import PathFinderRegistry
from rdmc.resonance.utils import (
    decrement_radical,
    decrement_order,
    get_aromatic_rings,
    has_empty_orbitals,
    increment_radical,
    increment_order,
    is_aryl_radical,
    is_cyclic,
    is_identical,
    is_radical,
    get_charge_span,
    get_lone_pair,
    get_num_aromatic_rings,
)


logger = logging.getLogger(__name__)


def analyze_molecule(mol):
    """
    Identify key features of molecule important for resonance structure generation.

    Returns:
        dict: a dictionary of features. The keys are:
            - is_radical: whether the molecule is a radical
            - is_cyclic: whether the molecule is cyclic
            - is_aromatic: whether the molecule is aromatic
            - isPolycyclicAromatic: whether the molecule is polycyclic aromatic
            - is_aryl_radical: whether the molecule is an aryl radical
            - hasLonePairs: whether the molecule has lone pairs
    """
    features = {
        "is_radical": is_radical(mol),
        "is_cyclic": is_cyclic(mol),
        "is_aromatic": False,
        "isPolycyclicAromatic": False,
        "is_aryl_radical": False,
        "hasLonePairs": False,
    }

    if features["is_cyclic"]:
        num_aromatic_rings = get_num_aromatic_rings(mol)
        if num_aromatic_rings > 0:
            features["is_aromatic"] = True
        if num_aromatic_rings > 1:
            features["isPolycyclicAromatic"] = True
        if features["is_radical"] and features["is_aromatic"]:
            features["is_aryl_radical"] = is_aryl_radical(mol)
    for atom in mol.GetAtoms():
        if get_lone_pair(atom) > 0:
            features["hasLonePairs"] = True
            break

    return features


def populate_resonance_algorithms(
    features: Optional[dict] = None,
) -> Tuple[list]:
    """
    Generate two list of resonance structure algorithms
        - The first are aromatic-agnostic approaches
        - The second are aromatic-specific approaches

    Args:
        features (dict, optional): a dictionary of features from ``analyze_molecule``.

    Returns:
        Tuple[list]: a tuple of two lists of resonance structure algorithms

    Returns a list of resonance algorithms.
    """
    if features is None:
        return (
            list(PathFinderRegistry._registry.keys()),
            [
                generate_optimal_aromatic_resonance_structures,
                generate_aryne_resonance_structures,
                generate_kekule_structure,
                # generate_clar_structures,
            ],
        )

    aromatic_agnostic, aromatic_specific = [], []
    # If the molecule is aromatic, then radical resonance has already been considered
    # If the molecule was falsely identified as aromatic, then is_aryl_radical will still accurately capture
    # cases where the radical is in an orbital that is orthogonal to the pi orbitals.
    if (
        features["is_radical"]
        and not features["is_aromatic"]
        and not features["is_aryl_radical"]
    ):
        aromatic_agnostic.append("allyl_radical")
    if features["is_cyclic"]:
        aromatic_specific.append(generate_aryne_resonance_structures)
    if features["hasLonePairs"]:
        aromatic_agnostic.extend(
            [
                "adj_lone_pair_radical",
                "forward_adj_lone_pair_multiple_bond",
                "reverse_adj_lone_pair_multiple_bond",
                "forward_adj_lone_pair_radical_multiple_bond",
                "reverse_adj_lone_pair_radical_multiple_bond",
            ]
        )
        if not features["is_aromatic"]:
            # The generate_lone_pair_multiple_bond_resonance_structures method may perturb the electronic
            # configuration of a conjugated aromatic system, causing a major slow-down (two orders of magnitude
            # slower in one observed case), and it doesn't necessarily result in new representative localized
            # structures. Here we forbid it for all structures bearing at least one aromatic ring as a "good enough"
            # solution. A more holistic approach would be to identify these cases in generate_resonance_structures,
            # and pass a list of forbidden atom ID's to find_lone_pair_multiple_bond_paths.
            aromatic_agnostic.append(
                "lone_pair_multiple_bond",
            )

    return (aromatic_agnostic, aromatic_specific)


def _generate_resonance_structures(
    mol_list: list,
    aromatic_agnostic: list,
    aromatic_specific: list,
    keep_isomorphic: bool = False,
):
    """
    Iteratively generate all resonance structures for a list of starting molecules using the specified methods.

    Args:
        mol_list             starting list of molecules
        method_list          list of resonance structure algorithms
        keep_isomorphic      if ``False``, removes any structures that are isomorphic (default)
                             if ``True``, keep all unique molecules.
    """
    octate_deviations = filtration.get_octet_deviation_list(mol_list)
    charge_spans = filtration.get_charge_span_list(mol_list)
    min_octet_deviation = min(octate_deviations)
    min_charge_span = min(charge_spans)
    ref_charge = Chem.GetFormalCharge(mol_list[0])

    prune_book = {i: defaultdict(set) for i in range(len(mol_list))}

    # Iterate over resonance structures
    index = 0

    while index < len(mol_list):
        # On-the-fly filtration: Extend methods only for molecule that don't deviate too much from the octet rule
        # (a +2 distance from the minimal deviation is used, octet deviations per species are in increments of 2)
        # Sometimes rearranging the structure requires an additional higher charge span structure, so allow
        # structures with a +1 higher charge span compared to the minimum, e.g., [O-]S#S[N+]#N
        # Filtration is always called.
        octet_deviation = octate_deviations[index]
        charge_span = charge_spans[index]
        if (
            octet_deviation > min_octet_deviation + 2
            or charge_span > min_charge_span + 1
        ):
            # Skip this resonance structure
            index += 1
            continue

        if octet_deviation < min_octet_deviation:
            # update min_octet_deviation to make this criterion tighter
            min_octet_deviation = octet_deviation
        if charge_span < min_charge_span:
            # update min_charge_span to make this criterion tighter
            min_charge_span = charge_span

        for method in aromatic_agnostic:
            pathfinder = PathFinderRegistry.get(method)
            for new_structure, path in generate_resonance_structures_with_pathfinder(
                mol_list[index], pathfinder, prune_paths=prune_book[index][method]
            ):
                charge = Chem.GetFormalCharge(new_structure)
                if charge != ref_charge:  # Not expected to happen
                    logger.debug(
                        f"Resonance generation created a molecule {Chem.MolToSmiles(new_structure)} "
                        f"with a net charge of {charge} which does not match the input mol charge of {ref_charge}.\n"
                        f"Removing it from resonance structures"
                    )
                    continue
                for j, known_structure in enumerate(mol_list):
                    if filtration.is_equivalent_structure(
                        known_structure,
                        new_structure,
                        isomorphic_equivalent=not keep_isomorphic,
                    ):
                        if j > index:
                            prune_book[j][pathfinder.reverse_abbr].add(path[::-1])
                        break
                else:
                    mol_list.append(new_structure)
                    octate_deviations.append(
                        filtration.get_octet_deviation(new_structure)
                    )
                    charge_spans.append(get_charge_span(new_structure))
                    prune_book[len(mol_list) - 1] = defaultdict(set)
                    prune_book[len(mol_list) - 1][pathfinder.reverse_abbr].add(
                        path[::-1]
                    )

        for method in aromatic_specific:
            new_structures = method(mol_list[index])
            for new_structure in new_structures:
                charge = Chem.GetFormalCharge(new_structure)
                if charge != ref_charge:  # Not expected to happen
                    logger.debug(
                        f"Resonance generation created a molecule {Chem.MolToSmiles(new_structure)} "
                        f"with a net charge of {charge} which does not match the input mol charge of {ref_charge}.\n"
                        f"Removing it from resonance structures"
                    )
                    continue
                for j, known_structure in enumerate(mol_list):
                    if filtration.is_equivalent_structure(
                        known_structure,
                        new_structure,
                        isomorphic_equivalent=not keep_isomorphic,
                    ):
                        break
                else:
                    mol_list.append(new_structure)
                    octate_deviations.append(
                        filtration.get_octet_deviation(new_structure)
                    )
                    charge_spans.append(get_charge_span(new_structure))
                    prune_book[len(mol_list) - 1] = defaultdict(set)

        # Move to the next resonance structure
        index += 1

    return mol_list


def generate_resonance_structures(
    mol,
    clar_structures: bool = False,
    keep_isomorphic: bool = False,
    filter_structures: bool = True,
) -> list:
    """
    Generate and return all of the resonance structures for the input molecule.

    Most of the complexity of this method goes into handling aromatic species, particularly to generate an accurate
    set of resonance structures that is consistent regardless of the input structure. The following considerations
    are made:

    1. False positives from RDKit aromaticity detection can occur if a molecule has exocyclic double bonds
    2. False negatives from RDKit aromaticity detection can occur if a radical is delocalized into an aromatic ring
    3. sp2 hybridized radicals in the plane of an aromatic ring do not participate in hyperconjugation
    4. Non-aromatic resonance structures of PAHs are not important resonance contributors (assumption)

    Aromatic species are broken into the following categories for resonance treatment:

    - Radical polycyclic aromatic species: Kekule structures are generated in order to generate adjacent resonance
      structures. The resulting structures are then used for Clar structure generation. After all three steps, any
      non-aromatic structures are removed, under the assumption that they are not important resonance contributors.
    - Radical monocyclic aromatic species: Kekule structures are generated along with adjacent resonance structures.
      All are kept regardless of aromaticity because the radical is more likely to delocalize into the ring.
    - Stable polycyclic aromatic species: Clar structures are generated
    - Stable monocyclic aromatic species: Kekule structures are generated
    """
    # TODO: Clar_structure is not the first priority. will be implemented later.

    # RMG avoid generating resonance structures for charged species
    # The concerns are due to invalid atom types. This may not be an issue for RDKit.
    # Comment out the check for formal charge until significant issues are found.
    # if mol.GetFormalCharge() != 0:
    #     return [mol]

    mol_list = [mol]
    # Analyze molecule
    features = analyze_molecule(mol)

    # Use generate_optimal_aromatic_resonance_structures to check for false positives and negatives
    if features["is_aromatic"] or (
        features["is_cyclic"]
        and features["is_radical"]
        and not features["is_aryl_radical"]
    ):
        new_mol_list = generate_optimal_aromatic_resonance_structures(mol, features)
        if not new_mol_list:
            # Encountered false positive, i.e., the molecule is not actually aromatic
            features["is_aromatic"] = False
            features["isPolycyclicAromatic"] = False
        else:
            features["is_aromatic"] = True
            features["isPolycyclicAromatic"] = len(get_aromatic_rings(new_mol_list[0])[0]) > 1
            for new_mol in new_mol_list:
                if not filtration.is_equivalent_structure(
                    mol, new_mol, not keep_isomorphic
                ):
                    mol_list.append(new_mol)

    # Special handling for aromatic species
    if features["is_aromatic"]:
        method_list = []
        if features["is_radical"] and not features["is_aryl_radical"]:
            method_list.append(generate_kekule_structure)
            method_list.append(generate_allyl_delocalization_resonance_structures)
        if features["isPolycyclicAromatic"] and clar_structures:
            method.append(generate_clar_structures)
        else:
            method.append(generate_aromatic_resonance_structure)

        for method in method_list:
            # For some reason unknown, RMG choose to apply each method separately.
            # Here we follow the same approach.
            _generate_resonance_structures(
                mol_list, [method], keep_isomorphic=keep_isomorphic
            )

    # Generate remaining resonance structures
    method_list = populate_resonance_algorithms(features)
    _generate_resonance_structures(
        mol_list, method_list, keep_isomorphic=keep_isomorphic
    )

    if filter_structures:
        return filtration.filter_structures(mol_list, features=features)

    return mol_list


def require_radical(fun):
    """
    A decorator for resonance structure generation functions that require a radical.

    Returns an empty list if the input molecule is not a radical.
    """

    def wrapper(mol):
        if not is_radical(mol):
            return []
        return fun(mol)

    return wrapper


@require_radical
def generate_allyl_delocalization_resonance_structures(mol):
    """
    Generate all of the resonance structures formed by one allyl radical shift.

    Biradicals on a single atom are not supported.
    """
    structures = []
    paths = pathfinder.find_allyl_delocalization_paths(mol)
    logger.debug(f"Found paths: {paths}")
    for a1_idx, a2_idx, a3_idx in paths:
        if mol.GetAtomWithIdx(a1_idx).GetNumRadicalElectrons() < 1:
            continue
        structure = Chem.RWMol(mol, True)
        a1, a3 = structure.GetAtomWithIdx(a1_idx), structure.GetAtomWithIdx(a3_idx)
        b12 = structure.GetBondBetweenAtoms(a1_idx, a2_idx)
        b23 = structure.GetBondBetweenAtoms(a2_idx, a3_idx)
        try:
            decrement_radical(a1)
            increment_radical(a3)
            increment_order(b12)
            decrement_order(b23)
            sanitize_resonance_mol(structure)
        except BaseException as e:  # cannot make the change
            logger.debug(
                f"Cannot transform path {(a1_idx, a2_idx, a3_idx)} "
                f"in `generate_allyl_delocalization_resonance_structures`."
                f"\nGot: {e}"
            )
        else:
            structures.append(structure)
    return structures


def generate_lone_pair_multiple_bond_resonance_structures(mol):
    """
    Generate all of the resonance structures formed by lone electron pair - multiple bond shifts in 3-atom systems.
    Examples: aniline (Nc1ccccc1), azide, [:NH2]C=[::O] <=> [NH2+]=C[:::O-]
    (where ':' denotes a lone pair, '.' denotes a radical, '-' not in [] denotes a single bond, '-'/'+' denote charge)
    """
    structures = []
    paths = pathfinder.find_lone_pair_multiple_bond_paths(mol)
    logger.debug(f"Found paths: {paths}")
    for a1_idx, a2_idx, a3_idx in paths:
        # preprocessing before attempting to make the change
        a1 = mol.GetAtomWithIdx(a1_idx)
        a3 = mol.GetAtomWithIdx(a3_idx)
        charge1, charge3 = a1.GetFormalCharge(), a3.GetFormalCharge()
        if charge1 >= 1 or charge3 <= -1 or get_lone_pair(a1) <= 0:
            # cannot decrease lone pair on atom1 if no lone pair
            # avoid creating +2 charged atoms
            # avoid creating -2 charged atoms
            continue
        # Copy the molecule and make the change
        structure = Chem.RWMol(mol, True)
        a1 = structure.GetAtomWithIdx(a1_idx)
        a3 = structure.GetAtomWithIdx(a3_idx)
        b12 = structure.GetBondBetweenAtoms(a1_idx, a2_idx)
        b23 = structure.GetBondBetweenAtoms(a2_idx, a3_idx)
        try:
            increment_order(b12)
            decrement_order(b23)
            a1.SetFormalCharge(charge1 + 1)
            a3.SetFormalCharge(charge3 - 1)
            sanitize_resonance_mol(structure)
        except BaseException as e:
            logger.debug(
                f"Cannot transform path {(a1_idx, a2_idx, a3_idx)} "
                f"in `generate_lone_pair_multiple_bond_resonance_structures`."
                f"\nGot: {e}"
            )
        else:
            structures.append(structure)
    return structures


@require_radical
def generate_adj_lone_pair_radical_resonance_structures(mol):
    """
    Generate all of the resonance structures formed by lone electron pair - radical shifts between adjacent atoms.
    These resonance transformations do not involve changing bond orders.
    NO2 example: O=[:N]-[::O.] <=> O=[N.+]-[:::O-]
    (where ':' denotes a lone pair, '.' denotes a radical, '-' not in [] denotes a single bond, '-'/'+' denote charge)
    """
    structures = []
    paths = pathfinder.find_adj_lone_pair_radical_delocalization_paths(mol)
    logger.debug(f"Found paths: {paths}")
    for a1_idx, a2_idx in paths:
        # preprocessing before attempting to make the change
        a1, a2 = mol.GetAtomWithIdx(a1_idx), mol.GetAtomWithIdx(a2_idx)
        charge1, charge2 = a1.GetFormalCharge(), a2.GetFormalCharge()
        if (
            charge1 <= -1
            or charge2 >= 1
            or a1.GetNumRadicalElectrons() < 1
            or get_lone_pair(a2) <= 0
        ):
            continue
        # Copy the molecule and make the change
        structure = Chem.RWMol(mol, True)
        a1 = structure.GetAtomWithIdx(a1_idx)
        a2 = structure.GetAtomWithIdx(a2_idx)
        try:
            decrement_radical(a1)
            increment_radical(a2)
            a1.SetFormalCharge(charge1 - 1)
            a2.SetFormalCharge(charge2 + 1)
            sanitize_resonance_mol(structure)
        except BaseException as e:
            logger.debug(
                f"Cannot transform path {(a1_idx, a2_idx)}) "
                f"in `generate_adj_lone_pair_radical_resonance_structures`."
                f"\nGot: {e}"
            )
        else:
            structures.append(structure)
    return structures


def generate_adj_lone_pair_multiple_bond_resonance_structures(mol):
    """
    Generate all of the resonance structures formed by lone electron pair - multiple bond shifts between adjacent atoms.
    Example: [:NH]=[CH2] <=> [::NH-]-[CH2+]
    (where ':' denotes a lone pair, '.' denotes a radical, '-' not in [] denotes a single bond, '-'/'+' denote charge)
    Here atom1 refers to the N/S/O atom, atom 2 refers to the any R!H (atom2's lone_pairs aren't affected)
    (In direction 1 atom1 <losses> a lone pair, in direction 2 atom1 <gains> a lone pair)
    """
    structures = []
    paths = pathfinder.find_adj_lone_pair_multiple_bond_delocalization_paths(mol)
    logger.debug(f"Found paths: {paths}")
    for a1_idx, a2_idx, direction in paths:
        # preprocessing before attempting to make the change
        a1, a2 = mol.GetAtomWithIdx(a1_idx), mol.GetAtomWithIdx(a2_idx)
        charge1, charge2 = a1.GetFormalCharge(), a2.GetFormalCharge()
        if direction == 1 and (
            charge1 >= 1
            or charge2 <= -1
            or get_lone_pair(a1) <= 0
            or not has_empty_orbitals(a2)
        ):
            # atom1 needs to have at least 1 lone pair to lose
            # atom2 needs to have at least 1 empty orbital to form a bond
            logger.debug(
                f"Remove paths: {(a1_idx, a2_idx, direction)} in preprocessing."
            )
            continue
        elif direction == 2 and (charge1 <= -1 or charge2 >= 1):
            logger.debug(
                f"Remove paths: {(a1_idx, a2_idx, direction)} in preprocessing."
            )
            continue

        # Copy the molecule and make the change
        structure = Chem.RWMol(mol, True)
        a1 = structure.GetAtomWithIdx(a1_idx)
        a2 = structure.GetAtomWithIdx(a2_idx)
        b12 = structure.GetBondBetweenAtoms(a1_idx, a2_idx)
        try:
            if direction == 1:  # The direction <increasing> the bond order
                increment_order(b12)
                a1.SetFormalCharge(charge1 + 1)
                a2.SetFormalCharge(charge2 - 1)
            elif direction == 2:  # The direction <decreasing> the bond order
                decrement_order(b12)
                a1.SetFormalCharge(charge1 - 1)
                a2.SetFormalCharge(charge2 + 1)
            sanitize_resonance_mol(structure)
        except BaseException as e:
            logger.debug(
                f"Cannot transform path {(a1_idx, a2_idx, direction)} "
                f"in `generate_adj_lone_pair_multiple_bond_resonance_structures`."
                f"\nGot: {e}"
            )
        else:
            # TODO: This is a constraint used in RMG, check if it is still needed here.
            if not Chem.rdmolops.GetFormalCharge(structure):
                structures.append(structure)
    return structures


@require_radical
def generate_adj_lone_pair_radical_multiple_bond_resonance_structures(mol):
    """
    Generate all of the resonance structures formed by lone electron pair - radical - multiple bond shifts between adjacent atoms.
    Example: [:N.]=[CH2] <=> [::N]-[.CH2]
    (where ':' denotes a lone pair, '.' denotes a radical, '-' not in [] denotes a single bond, '-'/'+' denote charge)
    Here atom1 refers to the N/S/O atom, atom 2 refers to the any R!H (atom2's lone_pairs aren't affected)
    This function is similar to generate_adj_lone_pair_multiple_bond_resonance_structures() except for dealing with the
    radical transformations.
    (In direction 1 atom1 <losses> a lone pair, gains a radical, and atom2 looses a radical.
    In direction 2 atom1 <gains> a lone pair, looses a radical, and atom2 gains a radical)
    """
    structures = []
    paths = pathfinder.find_adj_lone_pair_radical_multiple_bond_delocalization_paths(
        mol
    )
    for a1_idx, a2_idx, direction in paths:
        # preprocessing before attempting to make the change
        a1, a2 = mol.GetAtomWithIdx(a1_idx), mol.GetAtomWithIdx(a2_idx)
        if direction == 1 and (
            a2.GetNumRadicalElectrons() < 1
            or get_lone_pair(a1) <= 0
            or not has_empty_orbitals(a1)
        ):
            continue
        elif direction == 2 and (a1.GetNumRadicalElectrons() < 1):
            continue
        # Copy the molecule and make the change
        structure = Chem.RWMol(mol, True)
        a1, a2 = structure.GetAtomWithIdx(a1_idx), structure.GetAtomWithIdx(a2_idx)
        b12 = structure.GetBondBetweenAtoms(a1_idx, a2_idx)
        try:
            if direction == 1:  # The direction <increasing> the bond order
                increment_order(b12)
                increment_radical(a1)
                decrement_radical(a2)
            elif direction == 2:  # The direction <decreasing> the bond order
                decrement_order(b12)
                decrement_radical(a1)
                increment_radical(a2)
            sanitize_resonance_mol(structure)
        except BaseException as e:
            logger.debug(
                f"Cannot transform path {(a1_idx, a2_idx, direction)} "
                f"in `generate_adj_lone_pair_multiple_bond_resonance_structures`."
                f"\nGot: {e}"
            )
        else:
            structures.append(structure)
    return structures
