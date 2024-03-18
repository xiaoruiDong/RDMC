from collections import Counter
from itertools import product as cartesian_product
from typing import List, Optional, Tuple

import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Geometry.rdGeometry import Point3D

from rdmc.rdtools.atom import get_element_symbol, get_atom_mass
from rdmc.rdtools.conf import (
    embed_multiple_null_confs,
    reflect as _reflect,
    set_conformer_coordinates,
)


def get_spin_multiplicity(mol: Chem.Mol) -> int:
    """
    Get spin multiplicity of a molecule. The spin multiplicity is calculated
    using Hund's rule of maximum multiplicity defined as 2S + 1.

    Args:
        mol (Chem.Mol): The molecule to get spin multiplicity.

    Returns:
        int : Spin multiplicity.
    """
    return 1 + Descriptors.NumRadicalElectrons(mol)


def get_formal_charge(mol: Chem.Mol) -> int:
    """
    Get formal charge of a molecule.

    Args:
        mol (Chem.Mol): The molecule to get formal charge.

    Returns:
        int : Formal charge.
    """
    return Chem.GetFormalCharge(mol)


def get_mol_weight(
    mol: Chem.Mol,
    exact: bool = False,
    heavy_atoms: bool = False,
) -> float:
    """
    Get the molecule weight.

    Args:
        mol (Chem.Mol): The molecule to get the weight.
        exact (bool, optional): If ``True``, the exact weight is returned.
            Otherwise, the average weight is returned. Defaults to ``False``.
        heavy_atoms (bool, optional): If ``True``, the weight is calculated using only heavy atoms.
            Otherwise, the weight is calculated using all atoms. Defaults to ``False``.

    Returns:
        float: The weight of the molecule.
    """
    if heavy_atoms:
        return Descriptors.HeavyAtomMolWt(mol)
    if exact:
        return Descriptors.ExactMolWt(mol)
    return Descriptors.MolWt(mol)


def get_heavy_atoms(mol: Chem.Mol) -> list:
    """
    Get heavy atoms of a molecule.

    Args:
        mol (Chem.Mol): The molecule to get heavy atoms.

    Returns:
        list: the list of heavy atoms.
    """
    return [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]


def get_element_symbols(mol: Chem.Mol) -> List[str]:
    """
    Get element symbols of a molecule.

    Args:
        mol (Chem.Mol): The molecule to get element symbols.

    Returns:
        List[str] : List of element symbols (e.g. ``["H", "C", "O",]`` etc.)
    """
    return [get_element_symbol(atom) for atom in mol.GetAtoms()]


def get_atomic_nums(mol: Chem.Mol) -> List[int]:
    """
    Get atomic numbers of a molecule.

    Args:
        mol (Chem.Mol): The molecule to get atomic numbers.

    Returns:
        List[int] : List of atomic numbers (e.g. ``[1, 6, 8, ]`` etc.)
    """
    return [atom.GetAtomicNum() for atom in mol.GetAtoms()]

def get_atom_masses(mol: Chem.Mol) -> List[float]:
    """
    Get atomic masses of a molecule.

    Args:
        mol (Chem.Mol): The molecule to get atomic masses.

    Returns:
        List[float] : List of atomic masses (e.g. ``[1.008, 12.01, 15.999, ]`` etc.)
    """
    return [get_atom_mass(atom) for atom in mol.GetAtoms()]


def get_element_counts(mol: Chem.Mol) -> dict:
    """
    Get element counts of a molecule.

    Args:
        mol (Chem.Mol): The molecule to get element counts.

    Returns:
        dict: {"element_symbol": count}
    """
    return dict(Counter(get_element_symbols(mol)))


def combine_mols(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    offset: Optional[np.ndarray] = None,
    c_product: bool = False,
):
    """
    Combine two molecules (``mol1`` and ``mol2``).
    A new object instance will be created and changes are not made to the current molecule.

    Args:
        mol1 (Chem.Mol): The current molecule.
        mol2 (Chem.Mol): The molecule to be combined.
        offset (np.ndarray, optional): The offset to be added to the coordinates of ``mol2``. It should be a length-3 array.
                                       This is not used when any of the molecules has 0 conformer. Defaults to ``None``.
        c_product (bool, optional): If ``True``, generate conformers for every possible combination
                                    between the current molecule and the ``molFrag``. E.g.,
                                    (1,1), (1,2), ... (1,n), (2,1), ...(m,1), ... (m,n). :math:`N(conformer) = m \\times n.`

                                    Defaults to ``False``, meaning only generate conformers according to
                                    (1,1), (2,2), ... When ``c_product`` is set to ``False``, if the current
                                    molecule has 0 conformer, conformers will be embedded to the current molecule first.
                                    The number of conformers of the combined molecule will be equal to the number of conformers
                                    of ``molFrag``. Otherwise, the number of conformers of the combined molecule will be equal
                                    to the number of conformers of the current molecule. Some coordinates may be filled by 0s,
                                    if the current molecule and ``molFrag`` have different numbers of conformers.

    Returns:
        Chem.Mol: The combined molecule.
    """
    vector = Point3D()
    if not c_product and offset is not None:
        for i, coord in enumerate("xyz"):
            vector.__setattr__(coord, float(offset[i]))

    combined_mol = Chem.CombineMols(mol1, mol2, vector)
    if c_product:
        c1s, c2s = mol1.GetConformers(), mol2.GetConformers()
        pos_list = [
            [c1.GetPositions(), c2.GetPositions() + offset]
            for c1, c2 in cartesian_product(c1s, c2s)
        ]
        if len(pos_list) > 0:
            embed_multiple_null_confs(combined_mol, len(pos_list), random=False)
            for i, pos in enumerate(pos_list):
                conf = combined_mol.GetConformer(i)
                set_conformer_coordinates(conf, np.concatenate(pos))

    return combined_mol


def force_no_implicit(mol: Chem.Mol):
    """
    Set no implicit hydrogen for atoms without implicit/explicit hydrogens. When
    manipulating molecules by changing number of radical electrons / charges and then updating the cached properties,
    additional hydrogens may be added to the molecule. This function helps avoid this problem.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1 and not atom.GetTotalNumHs():
            atom.SetNoImplicit(True)


def reflect(mol: Chem.Mol, conf_id: int = 0):
    """
    Reflect the coordinates of the conformer of the molecule.

    Args:
        mol (Chem.Mol): The molecule to reflect.
        conf_id (int, optional): The conformer ID to reflect.
    """
    conf = mol.GetConformer(conf_id)
    _reflect(conf)


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
            path = Chem.rdmolops.GetShortestPath(mol2, hs[i], hs[j])
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
