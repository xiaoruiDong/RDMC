from typing import List, Tuple, Union
import traceback

import numpy as np

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from rdmc.rdtools.dist import get_shortest_path, get_adjacency_matrix
from rdmc.rdtools.resonance import generate_resonance_structures
from rdmc.rdtools.conversion.xyz import mol_to_xyz, mol_from_xyz


def has_matched_mol(
    mol: "Mol",
    mols: List["Mol"],
    consider_atommap: bool = False,
) -> bool:
    """
    Check if a molecule has a isomorphic match in a list of molecules.

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


# Todo: Think of a more generic name for this function
def get_resonance_structure_match(
    mol1_res: List["Mol"],
    mol2_res: List["Mol"],
) -> tuple:
    """
    Get the match between two lists of resonance structures.

    Args:
        mol1_res (List['Mol']): The first list of resonance structures.
        mol2_res (List['Mol']): The second list of resonance structures.

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
    mols: List["Mol"],
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
        form = "same" if same_formula else CalcMolFormula(mol)
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


def is_same_complex(
    complex1: Union["Mol", Union[List["Mol"], Tuple["Mol"]]],
    complex2: Union["Mol", Union[List["Mol"], Tuple["Mol"]]],
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
        mol1_res = generate_resonance_structures(mol1[0]) if resonance else [mol1[0]]
        for i, mol2 in enumerate(mol2s):
            if mol1[1] > mol2[1] or i in matched:
                continue
            if mol1[1] < mol2[1]:
                return False

            mol2_res = mol2_res_dict.get(i)
            if mol2_res is None:
                mol2_res = (
                    generate_resonance_structures(mol2[0]) if resonance else [mol2[0]]
                )
                mol2_res_dict[i] = mol2_res

            match = get_resonance_structure_match(mol1_res, mol2_res)

            if match:
                matched.append(i)
                break
        else:
            return False
    return True


def get_match_and_recover_recipe(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
) -> Tuple[tuple, dict]:
    """
    Get the isomorphism match between two molecules and the recipe to recover
    mol2 to mol1. If swapping the atom indices in mol2 according to the recipe,
    mol2 should be the same as mol1.

    Args:
        mol1 (RWMol): The first molecule.
        mol2 (RWMol): The second molecule.

    Returns:
        tuple: The substructure match.
        dict: A truncated atom mapping of mol2 to mol1.
    """
    if mol1.GetNumAtoms() != mol2.GetNumAtoms():
        return (), {}
    match = mol1.GetSubstructMatch(mol2)
    recipe = {i: j for i, j in enumerate(match) if i != j}

    if len(recipe) == 0:
        # Either mol1 and mol2 has identical graph or no match at all
        return match, recipe

    # The default GetSubstructMatch may not always return the simplest mapping
    # The following implements a naive algorithm fixing the issue caused by equivalent
    # hydrogens. The idea is that if two hydrogens are equivalent, they are able to
    # be mapped to the same atom in mol1.

    # Find equivalent hydrogens
    hs = [i for i in recipe.keys() if mol1.GetAtomWithIdx(i).GetAtomicNum() == 1]
    equivalent_hs = []
    checked_hs = set()

    for i in range(len(hs)):
        if i in checked_hs:
            continue
        equivalent_hs.append([hs[i]])
        checked_hs.add(i)
        for j in range(i + 1, len(hs)):
            if j in checked_hs:
                continue
            path = get_shortest_path(mol2, hs[i], hs[j])
            if len(path) == 3:  # H1-X2-H3
                equivalent_hs[-1].append(hs[j])
                checked_hs.add(j)

    # Clean up the recipe based on the equivalent hydrogens
    # E.g. {2: 13, 12: 2, 13: 12} -> {2: 13, 12: 13}
    match = list(match)
    for group in equivalent_hs:
        for i in group:
            j = recipe.get(i)
            if j is not None and j in group:
                recipe[i] = recipe[j]
                match[i], match[j] = match[j], j
                del recipe[j]

    return tuple(match), recipe


def is_same_connectivity_mol(mol1: "Mol", mol2: "Mol") -> bool:
    """
    Check whether the two molecules has the same connectivity. Note, this is not an
    isomorphic check, and different atom ordering will be treated as "different connectivity".

    Args:
        mol1 (RDKitMol): The first molecule.
        mol2 (RDKitMol): The second molecule.

    Returns:
        bool: Whether the two molecules has the same connectivity.
    """
    return np.array_equal(
        get_adjacency_matrix(mol1),
        get_adjacency_matrix(mol2)
    )


def is_same_connectivity_conf(
    mol: Chem.Mol,
    conf_id: int = 0,
    backend: str = "openbabel",
    **kwargs,
) -> bool:
    """
    Check whether the conformer of the molecule (defined by its spacial coordinates)
    has the same connectivity as the molecule. Useful sanity check when coordinates are changed.

    Args:
        conf_id (int, optional): The conformer ID. Defaults to ``0``.
        backend (str, optional): The backend to use for the comparison. Defaults to ``'openbabel'``.
        **kwargs: The keyword arguments to pass to the backend.

    Returns:
        bool: Whether the conformer has the same connectivity as the molecule.
    """
    # Get the connectivity of ith conformer
    try:
        xyz_str = mol_to_xyz(mol, conf_id, header=True)
        # Sanitization is not applied to account for
        # special cases like zwitterionic molecules
        # or molecule complexes
        new_mol = mol_from_xyz(
            xyz_str,
            **{
                **dict(header=True, backend=backend, sanitize=False),
                **kwargs,
            },
        )
    except Exception as exc:
        # Error in preserving the molecule
        print(f"Error in preserving the molecule: {exc}")
        traceback.print_exc()
        return False

    else:
        return is_same_connectivity_mol(mol, new_mol)
