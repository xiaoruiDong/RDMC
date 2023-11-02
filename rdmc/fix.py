#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the helper functions for fixing parsed molecules.
"""

from functools import reduce
from typing import List

import numpy as np

from rdmc import RDKitMol
from rdkit.Chem import BondType, rdChemReactions, rdmolops


RECOMMEND_REMEDIES = [
    # Remedy 1 - Carbon monoxide: [C]=O to [C-]#[O+]
    rdChemReactions.ReactionFromSmarts(
        "[O+0-0v2X1:1]=[C+0-0v2X1:2]>>[O+1v3X1:1]#[C-1v3X1:2]"
    ),
    # Remedy 2 - Carbon monoxide: [C]=O to [C-]#[O+]
    rdChemReactions.ReactionFromSmarts(
        "[O+0-0v3X1:1]#[C+0-0v3X1:2]>>[O+1v3X1:1]#[C-1v3X1:2]"
    ),
    # Remedy 3 - Oxygen Molecule: O=O to [O]-[O]
    rdChemReactions.ReactionFromSmarts(
        "[O+0-0v2X1:1]=[O+0-0v2X1:2]>>[O+0-0v1X1:1]-[O+0-0v1X1:2]"
    ),
    # Remedy 4 - isocyanide: R[N]#[C] to R[N+]#[C-]
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v4X2:1]#[C+0-0v3X1:2]>>[N+v4X2:1]#[C-v3X1:2]"
    ),
    # Remedy 5 - azide: RN=N=[N] to RN=[N+]=[N-]
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v3X2:1]=[N+0-0v4X2:2]=[N+0-0v2X1:3]>>[N+0-0v3X2:1]=[N+1v4X2:2]=[N-1v2X1:3]"
    ),
    # Remedy 6 - amine oxide: RN(R)(R)-O to R[N+](R)(R)-[O-]
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v4X4:1]-[O+0-0v1X1:2]>>[N+1v4X4:1]-[O-1v1X1:2]"
    ),
    # Remedy 7 - amine radical: R[C](R)-N(R)(R)R to R[C-](R)-[N+](R)(R)R
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v4X4:1]-[C+0-0v3X3:2]>>[N+1v4X4:1]-[C-1v3X3:2]"
    ),
    # Remedy 8 - amine radical: RN(R)=C to RN(R)-[C]
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v4X3:1]=[C+0-0v4X3:2]>>[N+0-0v3X3:1]-[C+0-0v3X3:2]"
    ),
    # Remedy 9 - quintuple C bond, usually due to RC(=O)=O: R=C(R)=O to R=C(R)-[O]
    rdChemReactions.ReactionFromSmarts(
        "[C+0-0v5X3:1]=[O+0-0v2X1:2]>>[C+0-0v4X3:1]-[O+0-0v1X1:2]"
    ),
    # Remedy 10 - sulphuric bi-radicals: R[S](R)(-[O])-[O] to R[S](R)(=O)(=O)
    rdChemReactions.ReactionFromSmarts(
        "[S+0-0v4X4:1](-[O+0-0v1X1:2])-[O+0-0v1X1:3]>>[S+0-0v6X4:1](=[O+0-0v2X1:2])=[O+0-0v2X1:3]"
    ),
    # Remedy 11 - Triazinane: C1=N=C=N=C=N=1 to c1ncncn1
    rdChemReactions.ReactionFromSmarts(
        "[C+0-0v5X3:1]1=[N+0-0v4X2:2]=[C+0-0v5X3:3]=[N+0-0v4X2:4]=[C+0-0v5X3:5]=[N+0-0v4X2:6]=1"
        ">>[C+0-0v5X3:1]1[N+0-0v4X2:2]=[C+0-0v5X3:3][N+0-0v4X2:4]=[C+0-0v5X3:5][N+0-0v4X2:6]=1"
    ),
]


ZWITTERION_REMEDIES = [
    # Remedy 1 - criegee Intermediate: R[C](R)O[O] to RC=(R)[O+][O-]
    rdChemReactions.ReactionFromSmarts(
        "[C+0-0v3X3:1]-[O+0-0v2X2:2]-[O+0-0v1X1:3]>>[C+0-0v4X3:1]=[O+1v3X2:2]-[O-1v1X1:3]"
    ),
    # Remedy 2 - criegee Intermediate: [C]-C=C(R)O[O] to C=C-C=(R)[O+][O-]
    rdChemReactions.ReactionFromSmarts(
        "[C+0-0v3X3:1]-[C:2]=[C+0-0v4X3:3]-[O+0-0v2X2:4]-[O+0-0v1X1:5]>>[C+0-0v4X3:1]=[C:2]-[C+0-0v4X3:3]=[O+1v3X2:4]-[O-1v1X1:5]"
    ),
    # Remedy 3 - criegee like molecule: RN(R)(R)-C(R)(R)=O to R[N+](R)(R)-[C](R)(R)-[O-]
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v4X4:1]-[C+0-0v4X3:2]=[O+0-0v2X1:3]>>[N+1v4X4:1]-[C+0-0v3X3:2]-[O-1v1X1:3]"
    ),
    # Remedy 4 - criegee like molecule: R[N+](R)(R)-[C-](R)(R)[O] to R[N+](R)(R)-[C](R)(R)-[O-]
    rdChemReactions.ReactionFromSmarts(
        "[N+1v4X4:1]-[C-1v3X3:2]-[O+0-0v1X1:3]>>[N+1v4X4:1]-[C+0-0v3X3:2]-[O-1v1X1:3]"
    ),
    # Remedy 5 - ammonium + carboxylic: ([N]R4.C(=O)[O]) to ([N+]R4.C(=O)[O-])
    rdChemReactions.ReactionFromSmarts(
        "([N+0-0v4X4:1].[O+0-0v2X1:2]=[C+0-0v4X3:3]-[O+0-0v1X1:4])>>([N+1v4X4:1].[O+0-0v2X1:2]=[C+0-0v4X3:3]-[O-1v1X1:4])"
    ),
    # Remedy 6 - ammonium + phosphoric: ([N]R4.P(=O)[O]) to ([N+]R4.P(=O)[O-])
    rdChemReactions.ReactionFromSmarts(
        "([N+0-0v4X4:1].[P+0-0v5X4:2]-[O+0-0v1X1:3])>>([N+1v4X4:1].[P+0-0v5X4:2]-[O-1v1X1:3])"
    ),
    # Remedy 7 - ammonium + sulphuric: ([N]R4.S(=O)(=O)[O]) to ([N+]R4.S(=O)(=O)[O-])
    rdChemReactions.ReactionFromSmarts(
        "([N+0-0v4X4:1].[S+0-0v6X4:2]-[O+0-0v1X1:3])>>([N+1v4X4:1].[S+0-0v6X4:2]-[O-1v1X1:3])"
    ),
    # Remedy 8 - ammonium + carbonyl in ring: ([N]R4.C=O) to ([N+]R4.[C.]-[O-])
    rdChemReactions.ReactionFromSmarts(
        "([N+0-0v4X4:1].[C+0-0v4X3R:2]=[O+0-0v2X1:3])>>([N+1v4X4:1].[C+0-0v3X3R:2]-[O-1v1X1:3])"
    ),
]


RING_REMEDIES = [
    # The first four elements' sequence matters
    # TODO: Find a better solution to avoid the impact of sequence
    # Remedy 1 - quintuple C in ring: R1=C(R)=N-R1 to R1=C(R)[N]-R1
    rdChemReactions.ReactionFromSmarts(
        "[C+0-0v5X3R:1]=[N+0-0v3X2R:2]>>[C+0-0v4X3R:1]-[N+0-0v2X2R:2]"
    ),
    # Remedy 2 - quadruple N in ring: R1=N=C(R)R1 to R1=N-[C](R)R1
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v4X2R:1]=[C+0-0v4X3R:2]>>[N+0-0v3X2R:1]-[C+0-0v3X3R:2]"
    ),
    # Remedy 3 - ring =C(R)=N-[C]: R1=C(R)=N-[C](R)R1 to R1=C(R)-N=C(R)R1
    rdChemReactions.ReactionFromSmarts(
        "[C+0-0v5X3R:1]=[N+0-0v3X2R:2]-[C+0-0v3X3:3]>>[C+0-0v4X3R:1]-[N+0-0v3X2R:2]=[C+0-0v4X3:3]"
    ),
    # Remedy 4 - ring -N-N-: R1-N-N-R1 to R1-N=N-R1
    rdChemReactions.ReactionFromSmarts(
        "[N+0-0v2X2R:1]-[N+0-0v2X2R:2]>>[N+0-0v3X2R:1]=[N+0-0v3X2R:2]"
    ),
    # Remedy 5 - bicyclic radical
    rdChemReactions.ReactionFromSmarts(
        "[C+0-0v4:1]1[C+0-0v4X4:2]23[C+0-0v4:3][N+0-0v4X4:4]12[C+0-0v4:5]3>>[C+0-0v4:1]1[C+0-0v3X3:2]2[C+0-0v4:3][N+0-0v3X3:4]1[C+0-0v4:5]2"
    ),
]


DEFAULT_REMEDIES = RECOMMEND_REMEDIES
ALL_REMEDIES = RECOMMEND_REMEDIES + ZWITTERION_REMEDIES + RING_REMEDIES


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
        tmp_mol.UpdatePropertyCache(False)  # Update connectivity
        rdmolops.GetSymmSSSR(tmp_mol)  # Update ring information
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
    mult: int = 0,
    renumber_atoms: bool = True,
) -> "RDKitMol":
    """
    Fix the molecule by applying the given remedies and saturating bi-radical or carbene to fix spin multiplicity.

    Args:
        mol (RDKitMol): The molecule to be fixed.
        remedies (List[ChemicalReaction], optional): The list of remedies to fix the molecule,
                                                     defined as RDKit ChemicalReaction.
                                                     Defaults to ``rdmc.fix.DEFAULT_REMEDIES``.
        max_attempts (int, optional): The maximum number of attempts to fix the molecule.
                                        Defaults to ``10``.
        sanitize (bool, optional): Whether to sanitize the molecule after the fix. Defaults to ``True``.
                                   Using ``False`` is only recommended for debugging and testing.
        fix_spin_multiplicity (bool, optional): Whether to fix the spin multiplicity of the molecule.
                                                 Defaults to ``False``.
        mult (int, optional): The desired spin multiplicity. Defaults to ``0``, which means the lowest possible
                                spin multiplicity will be inferred from the number of unpaired electrons.
                              Only used when ``fix_spin_multiplicity`` is ``True``.
        renumber_atoms (bool, optional): Whether to renumber the atoms after the fix. Defaults to ``True``.
                                         Turn this off when the atom map number is not important.

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
        if mult == 0:
            # Infer the possible lowest spin multiplicity from the number of unpaired electrons
            mult = 1 if mol.GetSpinMultiplicity() % 2 else 2
        mol = fix_mol_spin_multiplicity(mol, mult)

    if renumber_atoms:
        mol = mol.RenumberAtoms()

    return mol


def find_oxonium_bonds(
    mol: "RDKitMol",
    threshold: float = 1.65,
) -> List[tuple]:
    """
    Find the potential oxonium atom.

    Args:
        mol (RDKitMol): The molecule to be fixed.
        threshold (float, optional): The threshold to determine if two atoms are connected.

    Returns:
        List[tuple]: a list of (oxygen atom index, the other atom index).
    """
    heavy_idxs = [atom.GetIdx() for atom in mol.GetHeavyAtoms()]
    oxygen_idxs = [
        atom.GetIdx() for atom in mol.GetHeavyAtoms() if atom.GetAtomicNum() == 8
    ]

    if len(oxygen_idxs) == 0:
        return []

    dist_mat = mol.GetDistanceMatrix()
    dist_mat[oxygen_idxs, oxygen_idxs] = 100  # Set the self distance to a large number

    # A detailed check may be done by element type
    # for now we will use the threshold based on the longest C-O bond 1.65 A
    infer_conn_mat = (dist_mat[oxygen_idxs][:, heavy_idxs] <= threshold).astype(int)
    actual_conn_mat = mol.GetAdjacencyMatrix()[oxygen_idxs][:, heavy_idxs]

    # Find potentially missing bonds
    raw_miss_bonds = np.transpose(np.where((infer_conn_mat - actual_conn_mat) == 1))
    miss_bonds = np.unique(raw_miss_bonds, axis=0).tolist()

    return [
        (oxygen_idxs[miss_bond[0]], heavy_idxs[miss_bond[1]])
        for miss_bond in miss_bonds
    ]


def fix_oxonium_bonds(
    mol: "RDKitMol",
    threshold: float = 1.65,
    sanitize: bool = True,
) -> 'RDKitMol':
    """
    Fix the oxonium atom. Openbabel and Jensen perception algorithm do not perceive the oxonium atom correctly.
    This is a fix to detect if the molecule contains oxonium atom and fix it.

    Args:
        mol (RDKitMol): The molecule to be fixed.
        threshold (float, optional): The threshold to determine if two atoms are connected.
        sanitize (bool, optional): Whether to sanitize the molecule after the fix. Defaults to ``True``.
                                   Using ``False`` is only recommended for debugging and testing.

    Returns:
        RDKitMol: The fixed molecule.
    """
    oxonium_bonds = find_oxonium_bonds(mol, threshold=threshold)

    if len(oxonium_bonds) == 0:
        return mol

    mol = mol.Copy()
    for miss_bond in oxonium_bonds:
        try:
            mol.AddBond(*miss_bond, order=BondType.SINGLE)
        except RuntimeError:
            # Oxygen may get double counted
            continue

        # Usually the connected atom is a radical site
        # So, update the number of radical electrons afterward
        rad_atom = mol.GetAtomWithIdx(miss_bond[1])
        if rad_atom.GetNumRadicalElectrons() > 0:
            rad_atom.SetNumRadicalElectrons(rad_atom.GetNumRadicalElectrons() - 1)

    # This remedy is only used for oxonium
    remedies = [
        # Remedy 1 - R[O](R)[O] to R[O+](R)[O-]
        # This is a case combining two radicals R-O-[O] and [R]
        rdChemReactions.ReactionFromSmarts(
            "[O+0-0v3X3:1]-[O+0-0v1X1:2]>>[O+1v3X3:1]-[O-1v1X1:2]"
        ),
        # Remedy 2 - R[O](R)C(R)=O to R[O+](R)[C](R)[O-]
        # This is a case combining a closed shell ROR with a radical R[C]=O
        rdChemReactions.ReactionFromSmarts(
            "[O+0-0v3X3:1]-[C+0-0v4X3:2]=[O+0-0v2X1:3]>>[O+1v3X3:1]-[C+0-0v3X3:2]-[O-1v1X1:3]"
        ),
    ]

    return fix_mol(mol, remedies=remedies, sanitize=sanitize)
