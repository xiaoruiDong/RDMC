#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains functions for generating resonance structures.
This is a rewrite of the original resonance_rmg.py module.
"""

from collections import defaultdict
from itertools import chain
import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from rdkit import Chem

from rdmc.resonance.base import ResonanceAlgoRegistry
from rdmc.resonance.filtration import (
    get_octet_deviation,
    get_octet_deviation_list,
    get_charge_span_list,
)
from rdmc.resonance.filtration import filter_structures as filter_resonance_structures
from rdmc.resonance.pathfinder import PathFinderRegistry
from rdmc.resonance.utils import (
    force_no_implicit,
    get_aryne_rings,
    get_charge_span,
    get_lone_pair,
    get_num_aromatic_rings,
    is_aryl_radical,
    is_cyclic,
    is_equivalent_structure,
    is_radical,
)


logger = logging.getLogger(__name__)

sanitize_flag_kekule = Chem.SANITIZE_PROPERTIES | Chem.SANITIZE_SETCONJUGATION
sanitize_flag_aromatic = sanitizeOps = (
    Chem.SANITIZE_PROPERTIES
    | Chem.SANITIZE_SETAROMATICITY
    | Chem.SANITIZE_SETCONJUGATION
)


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
                generate_clar_structures,
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
    update_aromatic_flags: bool = False,
):
    """
    Iteratively generate all resonance structures for a list of starting molecules using the specified methods.

    Args:
        mol_list             starting list of molecules
        method_list          list of resonance structure algorithms
        keep_isomorphic      if ``False``, removes any structures that are isomorphic (default)
                             if ``True``, keep all unique molecules.
    """
    octate_deviations = get_octet_deviation_list(mol_list)
    charge_spans = get_charge_span_list(mol_list)
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
                if update_aromatic_flags:
                    Chem.SanitizeMol(
                        new_structure,
                        sanitizeOps=sanitize_flag_aromatic,
                    )
                for j, known_structure in enumerate(mol_list):
                    if is_equivalent_structure(
                        known_structure,
                        new_structure,
                        isomorphic_equivalent=not keep_isomorphic,
                    ):
                        if j > index:
                            prune_book[j][pathfinder.reverse_abbr].add(path[::-1])
                        break
                else:
                    mol_list.append(new_structure)
                    octate_deviations.append(get_octet_deviation(new_structure))
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
                    if is_equivalent_structure(
                        known_structure,
                        new_structure,
                        isomorphic_equivalent=not keep_isomorphic,
                    ):
                        break
                else:
                    mol_list.append(new_structure)
                    octate_deviations.append(get_octet_deviation(new_structure))
                    charge_spans.append(get_charge_span(new_structure))
                    prune_book[len(mol_list) - 1] = defaultdict(set)

        # Move to the next resonance structure
        index += 1

    return mol_list


@ResonanceAlgoRegistry.register("rmg")
def generate_resonance_structures(
    mol,
    clar_structures: bool = True,
    keep_isomorphic: bool = False,
    filter_structures: bool = True,
    copy: bool = True,
    **kwargs,
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

    Note, RDKit atom obeys the octet rule, that is when making an atom electron deficient, it may automatically add
    implicit hydrogens if possible. This is not desired in resonance structure generation, so the molecule will be processed
    by setting atoms with no hydrogen (either implicit or explicit) to be "no implicit".
    """

    # RMG avoid generating resonance structures for charged species
    # The concerns are due to invalid atom types. This may not be an issue for RDKit.
    # Comment out the check for formal charge until significant issues are found.
    # if mol.GetFormalCharge() != 0:
    #     return [mol]
    if copy:
        mol = Chem.RWMol(mol, True)

    force_no_implicit(mol)

    # Analyze molecule
    features = analyze_molecule(mol)

    if features["is_aromatic"]:
        # This may be an intended behavior in RMG!
        # `mol` (`mol_list[0]`) will be changed inplace to kekule form during `analyze_molecule`.
        # So the optimal aromatic structure, generated in the next block will always be different
        # from the initial one, if it is indeed an aromatic molecule. We will then always have an
        # optimal aromatic structure and fully kekulized structure carried over to the later steps
        # for aromatic molecules. And in the filtration step, all aromatic structures will be kept.
        # The kekulized structure will be removed by aromaticity_filtration. But, it will be added
        # back in the end, since it is the first structure in the list.
        # However, in RDMC workflow, `mol` is not modified in `analyze_molecule`
        # Therefore, to mimic RMG's behavior, we need to generate kekulized form of `mol` here
        mol_list = generate_kekule_structure(mol)
    else:
        mol_list = [mol]

    # Use generate_optimal_aromatic_resonance_structures to check for false positives and negatives
    if features["is_aromatic"] or (
        features["is_cyclic"]
        and features["is_radical"]
        and not features["is_aryl_radical"]
    ):
        new_mol_list = generate_optimal_aromatic_resonance_structures(
            mol_list[0], features
        )
        if not new_mol_list:
            # Encountered false positive, i.e., the molecule is not actually aromatic
            features["is_aromatic"] = False
            features["isPolycyclicAromatic"] = False
        else:
            features["is_aromatic"] = True
            features["isPolycyclicAromatic"] = (
                get_num_aromatic_rings(new_mol_list[0]) > 1
            )
            for new_mol in new_mol_list:
                if not is_equivalent_structure(
                    mol_list[0], new_mol, not keep_isomorphic
                ):
                    mol_list.append(new_mol)

    # Special handling for aromatic species
    if features["is_aromatic"]:
        if features["is_radical"] and not features["is_aryl_radical"]:
            _generate_resonance_structures(
                mol_list,
                ["allyl_radical"],
                [],
                keep_isomorphic=keep_isomorphic,
                update_aromatic_flags=True,
            )
        if features["isPolycyclicAromatic"] and clar_structures:
            aromatic_specific = [generate_clar_structures]
        else:
            aromatic_specific = [generate_aromatic_resonance_structure]
        _generate_resonance_structures(
            mol_list, [], aromatic_specific, keep_isomorphic=keep_isomorphic
        )

    # Generate remaining resonance structures
    aromatic_agnostic, aromatic_specific = populate_resonance_algorithms(features)
    _generate_resonance_structures(
        mol_list, aromatic_agnostic, aromatic_specific, keep_isomorphic=keep_isomorphic
    )

    if filter_structures:
        return filter_resonance_structures(mol_list, features=features)

    return mol_list


def generate_resonance_structures_with_pathfinder(
    mol: "Mol",
    pathfinder: "PathFinder",
    prune_paths=set(),
) -> List[tuple]:
    """
    Generate resonance structures using a path finder.

    Args:
        mol (Mol): the input molecule.
        pathfinder (PathFinder): the path finder registered in the registry.
        prune_paths (set, optional): A set of paths to prune. Defaults to an empty set.

    Returns:
        List[tuple]: a list of tuples, each tuple contains a resonance structure and a path.
    """
    structures = []
    paths = pathfinder.find(mol)
    logger.debug(f"Found paths: {paths} with method {pathfinder.__name__}")
    paths.difference_update(prune_paths)
    logger.debug(f"{paths} After pruning")
    for path in paths:
        if pathfinder.verify(mol, path):
            structure = pathfinder.transform(mol, path)
            if structure is not None:
                structures.append((structure, path))
        else:
            logger.debug(f"{path} does not pass the check")

    return structures


def generate_optimal_aromatic_resonance_structures(
    mol: "Mol",
    features: dict = None,
):
    """
    Generate the aromatic form of the molecule. For radicals, generates the form with the most aromatic rings.

    Args:
        mol (Mol): the input molecule.
        features (dict, optional): a dictionary of features from ``analyze_molecule``.

    Returns:
        list: a list of resonance structures. In most cases, only one structure will be returned.
              In certain cases where multiple forms have the same number of aromatic rings,
              multiple structures will be returned. It just returns an empty list if an error is raised.
    """
    features = features if features is not None else analyze_molecule(mol)

    if not features["is_cyclic"]:
        return []

    # Copy the molecule so we don't affect the original
    molecule = Chem.RWMol(mol, True)

    # Attempt to rearrange electrons to obtain a structure with the most aromatic rings
    # Possible rearrangements include aryne resonance and allyl resonance
    aromatic_specific_methods = [generate_aryne_resonance_structures]
    aromatic_agnostic_methods = []
    if features["is_radical"] and not features["is_aryl_radical"]:
        aromatic_agnostic_methods.append("allyl_radical")

    kekule_list = generate_kekule_structure(molecule)

    _generate_resonance_structures(
        kekule_list,
        aromatic_agnostic=aromatic_agnostic_methods,
        aromatic_specific=aromatic_specific_methods,
    )

    new_structures = []
    max_num_aromatic_rings = 0
    for mol in kekule_list:
        # The mol is aromatized in place
        result = generate_aromatic_resonance_structure(mol, copy=False)
        if (
            not result
        ):  # we only use result to check if the molecule is aromatized successfully
            continue

        num_aromatic_rings = get_num_aromatic_rings(mol)
        if num_aromatic_rings > max_num_aromatic_rings:
            # Find a molecule with more aromatic rings
            max_num_aromatic_rings = num_aromatic_rings
            new_structures = [mol]
        elif num_aromatic_rings == max_num_aromatic_rings:
            # Find a molecule with the same number of aromatic rings
            for struct in new_structures:
                if is_equivalent_structure(struct, mol):
                    break
            else:
                new_structures.append(mol)

    return new_structures


def generate_aromatic_resonance_structure(
    mol: "Mol",
    copy: bool = True,
) -> list:
    """
    Generate the aromatic form of the molecule in place without considering other resonance.

    This method completely get rid of the implementation in the original rmg algorithm, to
    perceive aromaticity and use purely RDKit aromaticity perception. So the performance may be different.

    Args:
        mol: molecule to generate aromatic resonance structure for
        aromatic_bonds (optional): list of previously identified aromatic bonds
        copy (optional): copy the molecule if ``True``, otherwise modify in place

    Returns:
        List of one molecule if successful, empty list otherwise
    """
    mol = Chem.RWMol(mol, True) if copy else mol
    try:
        Chem.SanitizeMol(mol, sanitizeOps=sanitize_flag_aromatic)
        return [mol]
    except BaseException:
        return []


def generate_kekule_structure(mol, copy: bool = True):
    """
    Generate a kekulized (single-double bond) form of the molecule.
    The specific arrangement of double bonds is non-deterministic, and depends on RDKit.

    Returns a single Kekule structure as an element of a list of length 1.
    If there's an error (eg. in RDKit) then it just returns an empty list.
    """
    mol = Chem.RWMol(mol, True) if copy else mol
    try:
        Chem.KekulizeIfPossible(mol, clearAromaticFlags=True)
        Chem.SanitizeMol(mol, sanitizeOps=sanitize_flag_kekule)
        return [mol]
    except BaseException:
        return []


def generate_aryne_resonance_structures(mol):
    """
    Generate aryne resonance structures, including the cumulene and alkyne forms.

    For all 6-membered rings, check for the following bond patterns:

      - TSDSDS (pattern 1)
      - DDDSDS (pattern 2)

    This does NOT cover all possible aryne resonance forms, only the simplest ones.
    Especially for polycyclic arynes, enumeration of all resonance forms is
    related to enumeration of all Kekule structures, which is very difficult.
    """
    pattern1_rings, pattern2_rings = get_aryne_rings(mol)

    operations = [[ring, "DDDSDS"] for ring in pattern1_rings] + [
        [ring, "DSTSDS"] for ring in pattern2_rings
    ]

    new_mol_list = []
    for ring, new_orders in operations:
        new_struct = Chem.RWMol(mol, True)

        for i in range(6):
            bond = new_struct.GetBondBetweenAtoms(ring[i - 1], ring[i])
            if new_orders[i] == "S":
                bond.SetBondType(Chem.BondType.SINGLE)
            elif new_orders[i] == "D":
                bond.SetBondType(Chem.BondType.DOUBLE)
            elif new_orders[i] == "T":
                bond.SetBondType(Chem.BondType.TRIPLE)

            try:
                Chem.SanitizeMol(
                    new_struct,
                    sanitizeOps=sanitize_flag_aromatic,
                )
            except BaseException:
                pass  # Don't append resonance structure if it creates an undefined atomtype
            else:
                new_mol_list.append(new_struct)

    return new_mol_list


def generate_clar_structures(mol):
    """
    Generate Clar structures for a given molecule.

    Returns a list of :class:`Molecule` objects corresponding to the Clar structures.
    """
    if not is_cyclic(mol):
        return []

    # Atom IDs are necessary in order to maintain consistent matrices between iterations

    try:
        solutions, aromatic_rings, bonds = _clar_optimization(mol)
    except BaseException:
        # The optimization algorithm did not work on the first iteration
        return []

    structures = []

    for solution in solutions:
        new_struct = Chem.RWMol(mol, True)
        ring_assign = solution[0: len(aromatic_rings)]
        bond_assign = solution[len(aromatic_rings):]

        for index, bidx in enumerate(bonds):
            bond = new_struct.GetBondWithIdx(bidx)
            if bond_assign[index] == 0:
                bond.SetBondType(Chem.BondType.SINGLE)
            else:  # the value is either 0 or 1 and we don't need to check
                bond.SetBondType(Chem.BondType.DOUBLE)
            bond.SetIsAromatic(False)
            bond.GetBeginAtom().SetIsAromatic(False)
            bond.GetEndAtom().SetIsAromatic(False)

        _clar_transformation(new_struct, aromatic_rings, ring_assign)

        try:
            new_struct.UpdatePropertyCache()
        except BaseException:
            pass
        else:
            structures.append(new_struct)

    return structures


def _clar_optimization(mol):
    """
    Implements linear programming algorithm for finding Clar structures. This algorithm maximizes the number
    of Clar sextets within the constraints of molecular geometry and atom valency.

    Returns a list of valid Clar solutions in the form of a tuple, with the following entries:
        [0] Molecule object
        [1] List of aromatic rings
        [2] List of bonds
        [3] Optimization solution

    The optimization solution is a list of boolean values with sextet assignments followed by double bond assignments,
    with indices corresponding to the list of aromatic rings and list of bonds, respectively.

    Method adapted from:
        Hansen, P.; Zheng, M. The Clar Number of a Benzenoid Hydrocarbon and Linear Programming.
            J. Math. Chem. 1994, 15 (1), 93-107.
    """
    ring_info = mol.GetRingInfo()
    bond_rings = [
        ring for ring in ring_info.BondRings() if len(ring) == 6
    ]  # RMG only consider 6 member rings
    aromatic_rings = [
        ring
        for ring in bond_rings
        if all(
            [
                mol.GetBondWithIdx(bidx).GetBondType() == Chem.BondType.AROMATIC
                for bidx in ring
            ]
        )
    ]
    if not aromatic_rings:
        return []
    aromatic_rings.sort(key=lambda x: sum(x))

    bonds = set(chain(*aromatic_rings))
    bonds = sorted(bonds)

    n_ring = len(aromatic_rings)
    n_ring_bond = len(bonds)

    atom_bond_map = defaultdict(set)
    atom_ring_map = defaultdict(set)
    for i, ring in enumerate(aromatic_rings):
        for bidx in ring:
            bond = mol.GetBondWithIdx(bidx)
            for atom in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
                atom_ring_map[atom].add(i)
                atom_bond_map[atom].add(n_ring + bonds.index(bidx))

    atoms = sorted(atom_bond_map.keys())

    exo_bonds, exo_bond_orders = [], []
    for aidx in atoms:
        atom = mol.GetAtomWithIdx(aidx)
        for bond in atom.GetBonds():
            bidx = bond.GetIdx()
            if bidx in bonds or bidx in exo_bonds:
                continue
            elif bond.GetOtherAtom(atom).GetAtomicNum() == 1:
                continue
            exo_bonds.append(bond.GetIdx())
            if bond.GetBondType() == 2:
                exo_bond_orders.append(1)
            else:
                exo_bond_orders.append(0)
    n_exo_bond = len(exo_bonds)

    n_atom = len(atoms)

    A_eq = np.zeros((n_atom, n_ring + n_ring_bond + n_exo_bond))

    for i, atom in enumerate(atoms):
        A_eq[i, list(atom_ring_map[atom] | atom_bond_map[atom])] = 1

    c = np.array([1] * n_ring + [0] * (n_ring_bond + n_exo_bond))

    bounds = [(0, 1)] * (n_ring + n_ring_bond) + [(1, 1)] * n_exo_bond

    solutions = _solve_clar_lp(num_rings=n_ring, c=c, A_eq=A_eq, bounds=bounds)

    return solutions, aromatic_rings, bonds


def _solve_clar_lp(num_rings, c, A_eq, bounds, constraints=None, max_num=None):
    if constraints is None:
        A_ub, b_ub = None, None
    else:
        A_ub = np.array([new_a for new_a, _ in constraints])
        b_ub = np.array([new_b for _, new_b in constraints])

    result = linprog(
        c=-c,  # Note: negative to maximize
        A_eq=A_eq,
        b_eq=np.ones(A_eq.shape[0]),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        integrality=1,
    )

    if result.status != 0:
        raise RuntimeError("Optimization could not find a valid solution.")

    obj_val, solution = -result.fun, result.x

    # Check that we the result contains at least one aromatic sextet
    if obj_val == 0:
        return []

    # Check that the solution contains the maximum number of sextets possible
    if max_num is None:
        max_num = obj_val  # This is the first solution, so the result should be an upper limit
    elif obj_val < max_num:
        raise RuntimeError("Optimization obtained a sub-optimal solution.")

    if np.any(np.logical_and(solution != 1, solution != 0)):
        raise Exception("Optimization obtained a non-integer solution.")

    # Generate constraints based on the solution obtained
    y = solution[0:num_rings]
    new_a = np.hstack([y, [0] * (c.shape[0] - num_rings)])
    new_b = sum(y) - 1
    if constraints is not None:
        constraints.append((new_a, new_b))
    else:
        constraints = [(new_a, new_b)]

    # Run optimization with additional constraints
    try:
        inner_solutions = _solve_clar_lp(
            num_rings, c, A_eq, bounds, constraints, max_num
        )
    except RuntimeError as e:
        inner_solutions = []

    return inner_solutions + [solution]


def _clar_transformation(mol, aromatic_rings, ring_assign):
    """
    Performs Clar transformation for given ring in a molecule, ie. conversion to aromatic sextet.

    Args:
        mol             a :class:`Molecule` object
        aromaticRing    a list of :class:`Atom` objects corresponding to an aromatic ring in mol

    This function directly modifies the input molecule and does not return anything.
    """
    for index, is_arom in enumerate(ring_assign):
        if is_arom == 1:
            for bidx in aromatic_rings[index]:
                bond = mol.GetBondWithIdx(bidx)
                bond.SetBondType(Chem.rdchem.BondType.AROMATIC)
                bond.SetIsAromatic(True)
                bond.GetBeginAtom().SetIsAromatic(True)
                bond.GetEndAtom().SetIsAromatic(True)
