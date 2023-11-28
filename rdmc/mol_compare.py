#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides methods for comparing molecules.
"""

from typing import List, Tuple, Union

from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from rdmc.resonance import generate_resonance_structures


def get_resonance_structure_match(
    mol1_res: List["RDKitMol"],
    mol2_res: List["RDKitMol"],
) -> tuple:
    """
    Get the match between two lists of resonance structures.

    Args:
        mol1_res (List['RDKitMol']): The first list of resonance structures.
        mol2_res (List['RDKitMol']): The second list of resonance structures.

    Returns:
        tuple: The match between the two lists of resonance structures. Empty tuple if no match is found.
    """
    for m1 in mol1_res:
        for m2 in mol2_res:
            match = m1.GetSubstructMatch(m2)
            if match:
                return match
    return tuple()


def get_unique_mols(
    mols: List["RDKitMol"],
    consider_atommap: bool = False,
    same_formula: bool = False,
):
    """
    Find the unique molecules from a list of molecules.

    Args:
        mols (list): The molecules to be processed.
        consider_atommap (bool, optional): If treat chemically equivalent molecules with
                                           different atommap numbers as different molecules.
                                           Defaults to ``False``.
        same_formula (bool, opional): If the mols has the same formula you may set it to ``True``
                                      to save computational time. Defaults to ``False``.

    Returns:
        list: A list of unique molecules.
    """
    # Dictionary:
    # Keys: chemical formula;
    # Values: list of mols with same formula
    # Use chemical formula to reduce the call of the more expensive graph substructure check
    unique_formula_mol = {}

    for mol in mols:
        # Get the molecules with the same formula as the query molecule
        form = "same" if same_formula else CalcMolFormula(mol._mol)
        unique_mol_list = unique_formula_mol.get(form)

        if unique_mol_list and has_matched_mol(
            mol, unique_mol_list, consider_atommap=consider_atommap
        ):
            continue
        elif unique_mol_list:
            unique_formula_mol[form].append(mol)
        else:
            unique_formula_mol[form] = [mol]

    return sum(unique_formula_mol.values(), [])


def has_matched_mol(
    mol: "RDKitMol",
    mols: List["RDKitMol"],
    consider_atommap: bool = False,
) -> bool:
    """
    Check if a molecule has a structure match in a list of molecules.

    Args:
        mol (RDKitMol): The target molecule.
        mols (List[RDKitMol]): The list of molecules to be processed.
        consider_atommap (bool, optional): If treat chemically equivalent molecules with
                                           different atommap numbers as different molecules.
                                           Defaults to ``False``.

    Returns:
        bool: if a matched molecules if found.
    """
    for mol_in_list in mols:
        mapping = mol_in_list.GetSubstructMatch(
            mol
        )  # A tuple of atom indexes if matched
        if mapping and not consider_atommap:
            return True
        elif mapping and mapping == tuple(range(len(mapping))):
            # if identical, the mapping is always as 1,2,...,N
            return True
    return False


def is_same_complex(
    complex1: Union["RDKitMol", Union[List["RDKitMol"], Tuple["RDKitMol"]]],
    complex2: Union["RDKitMol", Union[List["RDKitMol"], Tuple["RDKitMol"]]],
    resonance: bool = False,
) -> bool:
    """
    Check if two complexes are the same regardless of the sequence of the molecules
    and the atom mapping.

    Args:
        complex1 (Union['RDKitMol', list['RDKitMol']]): The first complex.
        complex2 (Union['RDKitMol', list['RDKitMol']]): The second complex.
        resonance (bool, optional): Whether to consider resonance structures. Defaults to ``False``.

    Returns:
        bool: Whether the two complexes are the same.
    """
    if not isinstance(complex1, (list, tuple)):
        complex1 = list(complex1.GetMolFrags(asMols=True))
    if not isinstance(complex2, (list, tuple)):
        complex2 = list(complex2.GetMolFrags(asMols=True))

    if len(complex1) != len(complex2):
        return False

    mol1s = sorted([(m, m.GetNumAtoms()) for m in complex1], key=lambda x: x[1])
    mol2s = sorted([(m, m.GetNumAtoms()) for m in complex2], key=lambda x: x[1])

    matched = []
    mol2_res_dict = {}

    for mol1 in mol1s:
        mol1_res = (
            generate_resonance_structures(mol1[0])
            if resonance
            else [mol1[0]]
        )
        for i, mol2 in enumerate(mol2s):
            if mol1[1] > mol2[1] or i in matched:
                continue
            if mol1[1] < mol2[1]:
                return False

            mol2_res = mol2_res_dict.get(i)
            if mol2_res is None:
                mol2_res = (
                    generate_resonance_structures(mol2[0])
                    if resonance
                    else [mol2[0]]
                )
                mol2_res_dict[i] = mol2_res

            match = get_resonance_structure_match(mol1_res, mol2_res)

            if match:
                matched.append(i)
                break
        else:
            return False
    return True


def is_equivalent_reaction(
    rxn1: "Reaction",
    rxn2: "Reaction",
) -> bool:
    """
    If two reactions has equivalent transformation regardless of atom mapping. This can be useful when filtering
    out duplicate reactions that are generated from different atom mapping.

    Args:
        rxn1 (Reaction): The first reaction.
        rxn2 (Reaction): The second reaction.

    Returns:
        bool: Whether the two reactions are equivalent.
    """
    equiv = (rxn1.num_formed_bonds == rxn2.num_formed_bonds) \
        and (rxn1.num_broken_bonds == rxn2.num_broken_bonds) \
        and (rxn1.num_changed_bonds == rxn2.num_changed_bonds)

    if not equiv:
        return False

    match_r, recipe_r = rxn1.reactant_complex.GetSubstructMatchAndRecipe(
        rxn2.reactant_complex
    )
    match_p, recipe_p = rxn1.product_complex.GetSubstructMatchAndRecipe(
        rxn2.product_complex
    )

    if not match_r or not match_p:
        # Reactant and product not matched
        return False

    elif recipe_r == recipe_p:
        # Both can match and can be recovered with the same recipe,
        # meaning we can simply recover the other reaction by swapping the index of these reactions
        # Note, this also includes the case recipes are empty, meaning things are perfectly matched
        # and no recovery operation is needed.
        return True

    elif not recipe_r:
        # The reactants are perfectly matched
        # Then we need to see if r/p match after the r/p are renumbered based on the product recipe,
        new_rxn2_rcomplex = rxn2.reactant_complex.RenumberAtoms(recipe_p)
        new_rxn2_pcomplex = rxn2.product_complex.RenumberAtoms(recipe_p)

    elif not recipe_p:
        # The products are perfectly matched
        # Then we need to see if r/p match after the r/p are renumbered based on the reactant recipe,
        new_rxn2_rcomplex = rxn2.reactant_complex.RenumberAtoms(recipe_r)
        new_rxn2_pcomplex = rxn2.product_complex.RenumberAtoms(recipe_r)

    else:
        # TODO: this condition hasn't been fully investigated and tested yet.
        # TODO: Usually this will results in a non-equivalent reaction.
        # TODO: for now, we just renumber based on recipe_r
        new_rxn2_rcomplex = rxn2.reactant_complex.RenumberAtoms(recipe_r)
        new_rxn2_pcomplex = rxn2.product_complex.RenumberAtoms(recipe_r)

    # To check if not recipe is the same
    _, recipe_r = rxn1.reactant_complex.GetSubstructMatchAndRecipe(new_rxn2_rcomplex)
    _, recipe_p = rxn1.product_complex.GetSubstructMatchAndRecipe(new_rxn2_pcomplex)

    return recipe_r == recipe_p
