#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides methods for comparing molecules.
"""


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

    match_r, recipe_r = rxn1.reactant_complex.GetMatchAndRecoverRecipe(
        rxn2.reactant_complex
    )
    match_p, recipe_p = rxn1.product_complex.GetMatchAndRecoverRecipe(
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
    _, recipe_r = rxn1.reactant_complex.GetMatchAndRecoverRecipe(new_rxn2_rcomplex)
    _, recipe_p = rxn1.product_complex.GetMatchAndRecoverRecipe(new_rxn2_pcomplex)

    return recipe_r == recipe_p
