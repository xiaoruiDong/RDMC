#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the helper functions for fixing parsed molecules.
"""

from functools import reduce
from typing import List

from rdmc import RDKitMol
from rdkit.Chem import rdChemReactions, rdmolops


DEFAULT_REMEDIES = [
    # Remedy 1 - Carbon monoxide: [C]=O to [C-]#[O+]
    rdChemReactions.ReactionFromSmarts(
        "[O+0-0v2X1:1]=[C+0-0v2X1:2]>>[O+1v3X1:1]#[C-1v3X1:2]"
    ),
    # Remedy 2 - Oxygen Molecule: O=O to [O]-[O]
    rdChemReactions.ReactionFromSmarts(
        "[O+0-0v2X1:1]=[O+0-0v2X1:2]>>[O+0-0v1X1:1]-[O+0-0v1X1:2]"
    ),
    # Remedy 3 - isocyanide: R[N]#[C] to R[N+]#[C-]
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v4X2:1]#[C+0-0v3X1:2]>>[N+v4X2:1]#[C-v3X1:2]"
    ),
    # Remedy 4 - amine radical: RC(R)-N(R)(R)R to R[C-](R)-[N+](R)(R)R
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v4X4:1]-[C+0-0v3X3:2]>>[N+1v4X4:1]-[C-1v3X3:2]"
    ),
    # Remedy 5 - amine radical: RN(R)=C to RN(R)-[C]
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v4X3:1]=[C+0-0v4X3:2]>>[N+0-0v3X3:1]-[C+0-0v3X3:2]"
    ),
    # Remedy 5 - quintuple C bond, usually due to RC(=O)=O: R=C(R)=O to R=[C+](R)-[O-]
    rdChemReactions.ReactionFromSmarts(
        "[C+0-0v5X3:1]=[O+0-0v2X1:2]>>[C+0-0v4X3:1]-[O+0-0v1X1:2]"
    ),
    # Remedy 6 - amine oxide: RN(R)(R)-O to R[N+](R)(R)-[O-]
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v4X4:1]-[O+0-0v1X1:2]>>[N+1v4X4:1]-[O-1v1X1:2]"
    ),
    # Remedy 7 - criegee like molecule: RN(R)(R)-C(R)(R)=O to R[N+](R)(R)-[C](R)(R)-[O-]
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v4X4:1]-[C+0-0v4X3:2]=[O+0-0v2X1:3]>>[N+1v4X4:1]-[C+0-0v3X3:2]-[O-1v1X1:3]"
    ),
    # Remedy 8 - criegee like molecule: RN(R)(R)-C(R)(R)=O to R[N+](R)(R)-[C](R)(R)-[O-]
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v4X4:1]-[C+0-0v3X3:2]-[O+0-0v1X1:3]>>[N+1v4X4:1]-[C+0-0v3X3:2]-[O-1v1X1:3]"
    ),
]


def update_product_atom_map_after_reaction(
    mol: "RDKitMol",
    ref_mol: "RDKitMol",
    clean_rxn_props: bool = True,
):
    """
    Update the atom map number of the product molecule after reaction according to the reference molecule (usually the reactant).
    The operation is in-place.

    Args:
        mol (RDKitMol): The product molecule after reaction.
        ref_mol (RDKitMol): The reference molecule (usually the reactant).
        clean_rxn_props (bool, optional): Whether to clean the reaction properties.
                                          RDKit add `"old_mapno"` and `"react_atom_idx"`
                                          to atoms. Defaults to ``True``.
    """
    map_dict = {str(a.GetIdx()): a.GetAtomMapNum() for a in ref_mol.GetAtoms()}

    for atom in mol.GetAtoms():
        if atom.HasProp("old_mapno"):
            # atom map number will zeroed out in the reaction
            atom.SetAtomMapNum(int(map_dict[atom.GetProp("react_atom_idx")]))
        if clean_rxn_props:
            atom.ClearProp("react_atom_idx")
            atom.ClearProp("old_mapno")


def fix_mol_by_remedy(
    mol: "RDKitMol",
    remedy: "ChemicalReaction",
    max_attempts: int = 10,
    sanitize: bool = True,
) -> "RDKitMol":
    """
    Fix the molecule according to the given remedy defined as an RDKit ChemicalReaction.

    Args:
        mol (RDKitMol): The molecule to be fixed.
        remedy (ChemicalReaction): The functional group transformation as the remedy to fix the molecule,
                                   defined as an RDKit ChemicalReaction.
        max_attempts (int, optional): The maximum number of attempts to fix the molecule.
                                      Defaults to ``10``.
        sanitize (bool, optional): Whether to sanitize the molecule after the fix. Defaults to ``True``.

    Returns:
        RDKitMol: The fixed molecule.
    """
    tmp_mol = mol.ToRWMol()
    fix_flag = False

    for _ in range(max_attempts):
        tmp_mol.UpdatePropertyCache(False)
        try:
            # Remedy are designed to be unimolecular (group transformation), so the product will be unimolecular as well
            # If no match, RunReactants will return an empty tuple and thus cause an IndexError.
            # If there is a match, then there is always a single product being generated, and we can
            # query the product by the second index `[0]`
            fix_mol = remedy.RunReactants([tmp_mol], maxProducts=1)[0][0]
            update_product_atom_map_after_reaction(fix_mol, tmp_mol)
        except IndexError:
            break

        if fix_mol.GetNumAtoms() < tmp_mol.GetNumAtoms():
            # If the input molecule contains multiple fragments (i.e., isolated graphs),
            # RDKit will only keep the fragment matching the reaction pattern.
            # Therefore we need to append the other fragments back to the molecule.
            frag_assign = []
            frags = list(
                rdmolops.GetMolFrags(
                    tmp_mol, asMols=True, sanitizeFrags=False, frags=frag_assign
                )
            )
            atom_map_num_in_fix_mol = fix_mol.GetAtomWithIdx(0).GetAtomMapNum()
            for i in range(tmp_mol.GetNumAtoms()):
                if tmp_mol.GetAtomWithIdx(i).GetAtomMapNum() == atom_map_num_in_fix_mol:
                    frag_idx = frag_assign[i]
                    break
            frags[frag_idx] = fix_mol
            tmp_mol = reduce(rdmolops.CombineMols, frags)
        else:
            tmp_mol = fix_mol

        fix_flag = True

    else:
        raise RuntimeError(
            "The fix may be incomplete, as the maximum number of attempts has been reached."
        )

    if not fix_flag:
        return mol

    tmp_mol = RDKitMol(tmp_mol)
    if sanitize:
        tmp_mol.Sanitize()

    return tmp_mol


def fix_mol_by_remedies(
    mol: "RDKitMol",
    remedies: List["ChemicalReaction"],
    max_attempts: int = 10,
    sanitize: bool = True,
) -> "RDKitMol":
    """
    Fix the molecule according to the given remedy defined as an RDKit ChemicalReaction.

    Args:
        mol (RDKitMol): The molecule to be fixed.
        remedy (ChemicalReaction): The functional group transformation as the remedy to fix the molecule,
                                   defined as an RDKit ChemicalReaction.
        max_attempts (int, optional): The maximum number of attempts to fix the molecule.
                                      Defaults to ``10``.
        sanitize (bool, optional): Whether to sanitize the molecule after the fix. Defaults to ``True``.

    Returns:
        RDKitMol: The fixed molecule.
    """
    for remedy in remedies:
        mol = fix_mol_by_remedy(mol, remedy, max_attempts=max_attempts, sanitize=False)

    if sanitize:
        mol.Sanitize()

    return mol


def fix_mol_spin_multiplicity(
    mol: "RDKitMol",
    mult: int,
):
    """
    Fix the molecule by saturating the radical sites to full fill the desired spin multiplicity.
    It is worth noting that there isn't always a solution to the issue.

    Args:
        mol (RDKitMol): The molecule to be fixed.
        mult (int): The desired spin multiplicity.
    """
    mol.SaturateMol(mult)
    return mol


def fix_mol(
    mol: "RDKitMol",
    remedies: List["ChemicalReaction"] = DEFAULT_REMEDIES,
    max_attempts: int = 10,
    sanitize: bool = True,
    fix_spin_multiplicity: bool = False,
    mult: int = 1,
) -> "RDKitMol":
    """
    Fix the molecule by applying the given remedies and saturating the radical sites to full fill the desired spin multiplicity.

    Args:
        mol (RDKitMol): The molecule to be fixed.
        remedies (List[ChemicalReaction], optional): The list of remedies to fix the molecule,
                                                     defined as RDKit ChemicalReaction.
                                                     Defaults to ``rdmc.fix.DEFAULT_REMEDIES``.
        max_attempts (int, optional): The maximum number of attempts to fix the molecule.
                                        Defaults to ``10``.
        sanitize (bool, optional): Whether to sanitize the molecule after the fix. Defaults to ``True``.
        fix_spin_multiplicity (bool, optional): Whether to fix the spin multiplicity of the molecule.
                                                 Defaults to ``False``.
        mult (int, optional): The desired spin multiplicity. Defaults to ``1``.
                              Only used when ``fix_spin_multiplicity`` is ``True``.

    Returns:
        RDKitMol: The fixed molecule.
    """
    mol = fix_mol_by_remedies(
        mol,
        remedies,
        max_attempts=max_attempts,
        sanitize=sanitize,
    )

    if fix_spin_multiplicity:
        mol = fix_mol_spin_multiplicity(mol, mult)

    return mol
