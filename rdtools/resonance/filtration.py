#!/usr/bin/env python3

###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2023 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_dev@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################
"""Module for filtering resonance structures.

This module contains functions for filtering a list of Molecules representing a single Species,
keeping only the representative structures. Relevant for filtration of negligible mesomerism contributing structures.

The rules this module follows are (by order of importance):

    1. Minimum overall deviation from the Octet Rule (elaborated for Dectet for sulfur as a third row element)
    2. Additional charge separation is only allowed for radicals if it makes a new radical site in the species
    3. If a structure must have charge separation, negative charges will be assigned to more electronegative atoms,
        whereas positive charges will be assigned to less electronegative atoms (charge stabilization)
    4. Opposite charges will be as close as possible to one another, and vice versa (charge stabilization)

(inspired by http://web.archive.org/web/20140310074727/http://www.chem.ucla.edu/~harding/tutorials/resonance/imp_res_str.html
which is quite like http://www.chem.ucla.edu/~harding/IGOC/R/resonance_contributor_preference_rules.html)
"""

import logging
from itertools import combinations, product
from typing import Any, Optional, Union

from rdkit import Chem

from rdtools.resonance.utils import (
    get_charge_span,
    get_electronegativity,
    get_lone_pair,
    get_order_str,
    get_radical_count,
    get_shortest_path,
    get_total_bond_order,
    is_aromatic,
    is_identical,
)

logger = logging.getLogger(__name__)


# Pure RDKit
def filter_structures(
    mol_list: list[Chem.Mol],
    allow_expanded_octet: bool = True,
    features: Optional[dict[str, bool]] = None,
    **kwargs: Any,
) -> list[Chem.Mol]:
    """Filter a list of molecules to keep only the representative structures.

    This function filters them out by minimizing the number of C/N/O/S atoms without
    a full octet, non-preferred charge separation, and non-preferred aromatic
    structures.

    Args:
        mol_list (list[Chem.Mol]): The list of molecules to filter.
        allow_expanded_octet (bool, optional): Whether to allow expanded octets for third row elements.
            Default is ``True``.
        features (Optional[dict[str, bool]], optional): A list of features of the species. Default is ``None``.
        **kwargs (Any): Additional keyword arguments. They are ignored, but included for compatibility.

    Returns:
        list[Chem.Mol]: The filtered list of molecules.

    Raises:
        RuntimeError: If no representative structures are found.
    """
    logger.debug(f"Filter_structures: {len(mol_list)} structures are fed in.")

    # 1. Remove structures with different multiplicities generated
    filtered_list = multiplicity_filtration(
        mol_list,
        ref_idx=0,
    )
    logger.debug(
        f"Filter_structures: {len(filtered_list)} structures after removing ones with different multiplicities."
    )

    # 2. Filter mol_list using the octet rule and the respective octet deviation list
    filtered_list = octet_filtration(
        mol_list, allow_expanded_octet=allow_expanded_octet
    )
    logger.debug(
        f"Filter_structures: {len(mol_list)} structures after octet filtration."
    )

    # 3. Filter by charge
    filtered_list = charge_filtration(filtered_list)
    logger.debug(
        f"Filter_structures: {len(mol_list)} structures after charge filtration."
    )

    # 4. Filter aromatic structures
    if features is not None and features["is_aromatic"]:
        filtered_list = aromaticity_filtration(
            filtered_list, features["isPolycyclicAromatic"]
        )
        logger.debug(
            f"Filter_structures: {len(mol_list)} structures after aromaticity filtration."
        )

    if not filtered_list:
        raise RuntimeError(
            "Could not determine representative localized structures for the input molecules."
        )

    # Originally RMG checks reactivity here, it is removed since it is not used in RDMC

    # Make sure that the (first) original structure is always first in the list.
    for index, filtered in enumerate(filtered_list):
        if is_identical(mol_list[0], filtered):
            filtered_list.insert(0, filtered_list.pop(index))
            break
    else:
        # Append the original structure to list
        filtered_list.insert(0, mol_list[0])

    return filtered_list


# RDKit / RDMC compatible
def multiplicity_filtration(
    mol_list: list[Chem.Mol],
    ref_idx: int = 0,
) -> list[Chem.Mol]:
    """Filter a list of molecules based on their multiplicity.

    Returns a filtered list based on the multiplicity of the species. The
    multiplicity of the species is determined by the number of radical electrons in the
    species and only the one with the same multiplicity as the reference species (the
    first by default) is kept.

    Args:
        mol_list (list[Chem.Mol]): The list of molecules to filter. Can be either RDKit Mol or RDMC RDKitMol.
        ref_idx (int, optional): The index of the reference molecule in ``mol_list``. Default is ``0``.

    Returns:
        list[Chem.Mol]: The filtered list of molecules.
    """
    ref_radical_count = get_radical_count(mol_list[ref_idx])
    return [mol for mol in mol_list if get_radical_count(mol) == ref_radical_count]


# RDKit / RDMC compatible
def get_octet_deviation_list(
    mol_list: list[Chem.Mol], allow_expanded_octet: bool = True
) -> list[float]:
    """Get the octet deviations for a list of molecules.

    Args:
        mol_list (list[Chem.Mol]): The list of molecules to get the octet deviations for.
        allow_expanded_octet (bool, optional): Whether to allow expanded octets for third row elements.
            Default is ``True``.

    Returns:
        list[float]: The octet deviations for the molecules in ``mol_list``.
    """
    return [
        get_octet_deviation(mol, allow_expanded_octet=allow_expanded_octet)
        for mol in mol_list
    ]


# RDKit / RDMC compatible
def get_octet_deviation(
    mol: Chem.Mol,
    allow_expanded_octet: bool = True,
) -> float:
    """Returns the octet deviation for a molecule.

    Args:
        mol (Chem.Mol): The molecule to get the octet deviation for.
        allow_expanded_octet (bool, optional): Whether to allow expanded octets for third row elements.
            if `allow_expanded_octet` is ``True`` (by default),
            then the function also considers dectet for third row elements.
            Default is ``True``.

    Returns:
        float: The octet deviation for the molecule.
    """
    # The overall "score" for the molecule, summed across all non-H atoms
    octet_deviation: Union[int, float] = 0
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 1:
            continue
        num_lone_pair = get_lone_pair(atom)
        num_rad_elec = atom.GetNumRadicalElectrons()
        val_electrons = (
            2 * (int(get_total_bond_order(atom)) + num_lone_pair) + num_rad_elec
        )
        if atomic_num in [6, 7, 8]:
            # expecting C/N/O to be near octet
            octet_deviation += abs(8 - val_electrons)
        elif atomic_num == 16:
            if not allow_expanded_octet:
                # If allow_expanded_octet is False, then adhere to the octet rule for sulfur as well.
                # This is in accordance with J. Chem. Educ., 1995, 72 (7), p 583, DOI: 10.1021/ed072p583
                # This results in O=[:S+][:::O-] as a representative structure for SO2 rather than O=S=O,
                # and in C[:S+]([:::O-])C as a representative structure for DMSO rather than CS(=O)C.
                octet_deviation += abs(8 - val_electrons)
            else:
                # If allow_expanded_octet is True, then do not adhere to the octet rule for sulfur
                # and allow dectet structures (but don't prefer duedectet).
                # This is in accordance with:
                # -  J. Chem. Educ., 1972, 49 (12), p 819, DOI: 10.1021/ed049p819
                # -  J. Chem. Educ., 1986, 63 (1), p 28, DOI: 10.1021/ed063p28
                # -  J. Chem. Educ., 1992, 69 (10), p 791, DOI: 10.1021/ed069p791
                # -  J. Chem. Educ., 1999, 76 (7), p 1013, DOI: 10.1021/ed076p1013
                # This results in O=S=O as a representative structure for SO2 rather than O=[:S+][:::O-],
                # and in CS(=O)C as a representative structure for DMSO rather than C[:S+]([:::O-])C.
                if num_lone_pair <= 1:
                    octet_deviation += min(
                        abs(8 - val_electrons),
                        abs(10 - val_electrons),
                        abs(12 - val_electrons),
                    )  # octet/dectet on S p[0,1]
                    # eg [O-][S+]=O, O[S]=O, OS([O])=O, O=S(=O)(O)O
                elif num_lone_pair >= 2:
                    octet_deviation += abs(8 - val_electrons)  # octet on S p[2,3]
                    # eg [S][S], OS[O], [NH+]#[N+][S-][O-], O[S-](O)[N+]#N, S=[O+][O-]
            for bond in atom.GetBonds():
                atom2 = bond.GetOtherAtom(atom)
                if atom2.GetAtomicNum() == 16 and bond.GetBondType() == 3:
                    # penalty for S#S substructures. Often times sulfur can have a triple
                    # bond to another sulfur in a structure that obeys the octet rule, but probably shouldn't be a
                    # correct resonance structure. This adds to the combinatorial effect of resonance structures
                    # when generating reactions, yet probably isn't too important for reactivity. The penalty value
                    # is 0.5 since S#S substructures are captured twice (once for each S atom).
                    # Examples: CS(=O)SC <=> CS(=O)#SC;
                    # [O.]OSS[O.] <=> [O.]OS#S[O.] <=> [O.]OS#[S.]=O; N#[N+]SS[O-] <=> N#[N+]C#S[O-]
                    octet_deviation += 0.5
        # Penalize birad sites only if they theoretically substitute a lone pair.
        # E.g., O=[:S..] is penalized, but [C..]=C=O isn't.
        if num_rad_elec >= 2 and (
            (atomic_num == 7 and num_lone_pair == 0)
            or (atomic_num == 8 and num_lone_pair in [0, 1, 2])
            or (atomic_num == 16 and num_lone_pair in [0, 1, 2])
        ):
            octet_deviation += 3

    return octet_deviation


# RDKit / RDMC compatible
def octet_filtration(
    mol_list: list[Chem.Mol],
    allow_expanded_octet: bool = True,
) -> list[Chem.Mol]:
    """Filter unrepresentative mol by the octet deviation criterion.

    Args:
        mol_list (list[Chem.Mol]): The list of molecules to filter.
        allow_expanded_octet (bool, optional): Whether to allow expanded octets for third row elements.

    Returns:
        list[Chem.Mol]: The filtered list of molecules.
    """
    octet_deviation_list = get_octet_deviation_list(
        mol_list, allow_expanded_octet=allow_expanded_octet
    )
    min_octet_deviation = min(octet_deviation_list)
    return [
        mol
        for mol, octet_deviation in zip(mol_list, octet_deviation_list)
        if octet_deviation == min_octet_deviation
    ]


# Pure RDKit
def get_charge_span_list(mol_list: list[Chem.Mol]) -> list[float]:
    """Get the list of charge spans for a list of molecules.

    This is also calculated in
    the octet_filtration() function along with the octet filtration process.

    Args:
        mol_list (list[Chem.Mol]): The list of molecules to get the charge spans for.

    Returns:
        list[float]: The charge spans for the molecules in `mol_list`.
    """
    return [get_charge_span(mol) for mol in mol_list]


# Pure RDKit
def charge_filtration(mol_list: list[Chem.Mol]) -> list[Chem.Mol]:
    """Filtered based on charge_span, electronegativity and proximity.

    If structures with an additional charge layer introduce
    new reactive sites (i.e., radicals or multiple bonds) they will also be considered.
    For example:

        - Both of NO2's resonance structures will be kept: [O]N=O <=> O=[N+.][O-]
        - NCO will only have two resonance structures [N.]=C=O <=> N#C[O.], and will loose the third structure which has
            the same octet deviation, has a charge separation, but the radical site has already been considered: [N+.]#C[O-]
        - CH2NO keeps all three structures, since a new radical site is introduced: [CH2.]N=O <=> C=N[O.] <=> C=[N+.][O-]
        - NH2CHO has two structures, one of which is charged since it introduces a multiple bond: NC=O <=> [NH2+]=C[O-]

    However, if the species is not a radical, or multiple bonds do not alter, we only keep the structures with the
    minimal charge span. For example:

        - NSH will only keep the N#S form and not [N-]=[SH+]
        - The following species will loose two thirds of its resonance structures, which are charged: CS(=O)SC <=>
            CS(=O)#SC <=> C[S+]([O-]SC <=> CS([O-])=[S+]C <=> C[S+]([O-])#SC <=> C[S+](=O)=[S-]C
        - Azide is know to have three resonance structures: [NH-][N+]#N <=> N=[N+]=[N-] <=> [NH+]#[N+][N-2];

    Args:
        mol_list (list[Chem.Mol]): The list of molecules to filter.

    Returns:
        list[Chem.Mol]: The filtered list of molecules.
    """
    charge_span_list = get_charge_span_list(mol_list)
    min_charge_span = min(charge_span_list)

    filtered_list = mol_list

    if len(set(charge_span_list)) > 1:
        # Proceed if there are structures with different charge spans
        charged_list = [
            filtered_mol
            for index, filtered_mol in enumerate(filtered_list)
            if charge_span_list[index] == min_charge_span + 1
        ]  # save the 2nd charge span layer
        filtered_list = [
            filtered_mol
            for index, filtered_mol in enumerate(filtered_list)
            if charge_span_list[index] == min_charge_span
        ]  # the minimal charge span layer
        rad_idxs, mul_bond_idxs = set(), set()
        for mol in filtered_list:
            for atom in mol.GetAtoms():
                if atom.GetNumRadicalElectrons():
                    rad_idxs.add(atom.GetIdx())
            for bond in mol.GetBonds():
                if bond.GetBondType() in [2, 3]:
                    mul_bond_idxs.add(
                        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
                    )
        unique_charged_list = [
            mol
            for mol in charged_list
            if has_unique_sites(mol, rad_idxs, mul_bond_idxs)
        ]

        # Charge stabilization considerations for the case where there are several charge span layers
        # are checked here for filtered_list and unique_charged_list separately.
        if min_charge_span:
            filtered_list = stabilize_charges_by_electronegativity(filtered_list)
            filtered_list = stabilize_charges_by_proximity(filtered_list)
        if unique_charged_list:
            unique_charged_list = stabilize_charges_by_electronegativity(
                unique_charged_list, allow_empty_list=True
            )
            unique_charged_list = stabilize_charges_by_proximity(unique_charged_list)
            filtered_list.extend(unique_charged_list)

    if min_charge_span:
        filtered_list = stabilize_charges_by_electronegativity(filtered_list)
        filtered_list = stabilize_charges_by_proximity(filtered_list)

    return filtered_list


# RDKit / RDMC Compatible
def has_unique_sites(
    mol: Chem.Mol,
    rad_idxs: set[int],
    mul_bond_idxs: set[tuple[int, int]],
) -> bool:
    """Check if a resonance structure has unique sites.

    Check if a resonance structure has unique radical and multiple bond sites that
    are not present in other structures.

    Args:
        mol (Chem.Mol): The molecule to check.
        rad_idxs (set[int]): The set of radical sites in the other structures.
        mul_bond_idxs (set[tuple[int, int]]): The set of multiple bond sites in the other structures.

    Returns:
        bool: ``True`` if the structure has unique radical and multiple bond sites, ``False`` otherwise.
    """
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() and atom.GetIdx() not in rad_idxs:
            return True
    for bond in mol.GetBonds():
        bond_idx = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        if (
            (bond.GetBondType() in [2, 3])
            and bond_idx not in mul_bond_idxs
            and not (
                bond.GetBeginAtom().GetAtomicNum()
                == bond.GetEndAtom().GetAtomicNum()
                == 16
            )
        ):
            # We check that both atoms aren't S, otherwise we get [S.-]=[S.+] as a structure of S2 triplet
            return True
    return False


# Oxonium template for electronegativity considerations
template = Chem.MolFromSmarts("[O+X{1-3};!$([O+]-F)]")


# RDKit / RDMC compatible
def stabilize_charges_by_electronegativity(
    mol_list: list[Chem.Mol],
    allow_empty_list: bool = False,
) -> list[Chem.Mol]:
    """Only keep structures that obey the electronegativity rule.

    If a structure must
    have charge separation, negative charges will be assigned to more electronegative
    atoms, and vice versa.

    Args:
        mol_list (list[Chem.Mol]): The list of molecules to filter.
        allow_empty_list (bool, optional): Whether to allow an empty list to be returned. Default is ``False``.
            If allow_empty_list is set to ``False``, and all structures in `mol_list` violate the
            electronegativity heuristic, this function will return the original ``mol_list``.
            (examples: [C-]#[O+], CS, [NH+]#[C-], [OH+]=[N-], [C-][S+]=C violate this heuristic).

    Returns:
        list[Chem.Mol]: The filtered list of molecules.
    """
    mol_list_copy = []
    for mol in mol_list:
        X_positive = X_negative = 0
        for atom in mol.GetAtoms():
            charge = atom.GetFormalCharge()
            if charge > 0:
                X_positive += get_electronegativity(atom) * abs(charge)
            elif charge < 0:
                X_negative += get_electronegativity(atom) * abs(charge)
        # The following treatment is introduced in RMG
        # However, the condition is weird (asking for O-[F-] which is not valid)
        # The current implementation loosen the condition to [O+]-F and use substructure matching
        # The following is a comment from RMG along with the original code:
        # as in [N-2][N+]#[O+], [O-]S#[O+], OS(S)([O-])#[O+], [OH+]=S(O)(=O)[O-], [OH.+][S-]=O.
        # [C-]#[O+] and [O-][O+]=O, which are correct structures, also get penalized here, but that's OK
        # since they are still eventually selected as representative structures according to the rules here
        X_positive += len(mol.GetSubstructMatches(template))

        if X_positive <= X_negative:
            # Filter structures in which more electronegative atoms are positively charged.
            # This condition is NOT hermetic: It is possible to think of a situation where one structure has
            # several pairs of formally charged atoms, where one of the pairs isn't obeying the
            # electronegativity rule, while the sum of the pairs does.
            mol_list_copy.append(mol)

    if mol_list_copy or allow_empty_list:
        return mol_list_copy
    return mol_list


pos_atom_pattern = Chem.MolFromSmarts("[+]")
neg_atom_pattern = Chem.MolFromSmarts("[-]")


# Pure RDKit
def get_charge_distance(mol: Chem.Mol) -> tuple[int, int]:
    """Get the cumulated charge distance for similar charge and difference charge pairs.

    Args:
        mol (Chem.Mol): The molecule to check.

    Returns:
        tuple[int, int]: The cumulated charge distance for similar charge and difference charge pairs, respectively.
    """
    pos_atoms = [a[0] for a in mol.GetSubstructMatches(pos_atom_pattern)]
    neg_atoms = [a[0] for a in mol.GetSubstructMatches(neg_atom_pattern)]

    cumulative_similar_charge_distance = sum([
        len(get_shortest_path(mol, a1, a2)) for a1, a2 in combinations(pos_atoms, 2)
    ])
    cumulative_similar_charge_distance += sum([
        len(get_shortest_path(mol, a1, a2)) for a1, a2 in combinations(neg_atoms, 2)
    ])
    cumulative_opposite_charge_distance = sum([
        len(get_shortest_path(mol, a1, a2)) for a1, a2 in product(pos_atoms, neg_atoms)
    ])
    return cumulative_opposite_charge_distance, cumulative_similar_charge_distance


# Pure RDKit
def stabilize_charges_by_proximity(mol_list: list[Chem.Mol]) -> list[Chem.Mol]:
    """Only keep structures that obey the charge proximity rule.

    Opposite charges will be as close as possible to one another, and vice versa.

    Args:
        mol_list (list[Chem.Mol]): The list of molecules to filter.

    Returns:
        list[Chem.Mol]: The filtered list of molecules.
    """
    if not mol_list:
        return mol_list

    charge_distance_list = [get_charge_distance(mol) for mol in mol_list]
    min_cumulative_opposite_charge_distance = min(
        [distances[0] for distances in charge_distance_list],
        default=0,
    )
    # The stepwise filtering is based on the RMG original implementation
    mol_list, charge_distance_list = zip(  # type: ignore
        *[
            (mol_list[i], dist)
            for i, dist in enumerate(charge_distance_list)
            if dist[0] <= min_cumulative_opposite_charge_distance
        ]
    )
    max_cumulative_similar_charge_distance = max(
        [distances[1] for distances in charge_distance_list],
        default=0,
    )
    return [
        mol_list[i]
        for i, dist in enumerate(charge_distance_list)
        if dist[0] >= max_cumulative_similar_charge_distance
    ]


# RDKit / RDMC compatible
def aromaticity_filtration(
    mol_list: list[Chem.Mol],
    is_polycyclic_aromatic: bool = False,
) -> list[Chem.Mol]:
    """Filter molecules by heuristics.

    For monocyclic aromatics, Kekule structures are removed, with the
    assumption that an equivalent aromatic structure exists. Non-aromatic
    structures are maintained if they present new radical sites. Instead of
    explicitly checking the radical sites, we only check for the SDSDSD bond
    motif since radical delocalization will disrupt that pattern.

    For polycyclic aromatics, structures without any benzene bonds are removed.
    The idea is that radical delocalization into the aromatic pi system is
    unfavorable because it disrupts aromaticity. Therefore, structures where
    the radical is delocalized so far into the molecule such that none of the
    rings are aromatic anymore are not representative. While this isn't strictly
    true, it helps reduce the number of representative structures by focusing
    on the most important ones.

    Args:
        mol_list (list[Chem.Mol]): The list of molecules to filter.
        is_polycyclic_aromatic (bool, optional): Whether the species is polycyclic aromatic. Default is ``False``.

    Returns:
        list[Chem.Mol]: The filtered list of molecules.
    """
    # Start by selecting all aromatic resonance structures
    filtered_list = []
    other_list = []
    for mol in mol_list:
        if is_aromatic(mol):
            filtered_list.append(mol)
        else:
            other_list.append(mol)

    if not is_polycyclic_aromatic:
        # Look for structures that don't have standard SDSDSD bond orders
        for mol in other_list:
            # Check all 6 membered rings
            # rings = [ring for ring in mol.get_relevant_cycles() if len(ring) == 6]
            # RDKit doesn't have a support to get all relevant cycles...
            # Temporarily use the BondRings as a rough fix
            # TODO: Implement pyrdl to get all relevant cycles which doesn't have full support
            # TODO: for different python versions and different OS
            # TODO: Another workaround maybe temporarily ignore polycyclic aromatics
            bond_lists = [
                ring for ring in mol.GetRingInfo().BondRings() if len(ring) == 6
            ]
            for bond_list in bond_lists:
                bond_orders = "".join([
                    get_order_str(mol.GetBondWithIdx(bond)) for bond in bond_list
                ])
                if bond_orders == "SDSDSD" or bond_orders == "DSDSDS":
                    break
            else:
                filtered_list.append(mol)

    return filtered_list
