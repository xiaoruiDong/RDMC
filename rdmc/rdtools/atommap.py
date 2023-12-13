from typing import Iterable, List, Optional

import numpy as np

from rdkit import Chem


def get_atom_map_numbers(mol: Chem.Mol) -> List[int]:
    """
    Get atom map numbers of the molecule.

    Args:
        mol (Chem.Mol): The molecule to get atom map numbers.

    Returns:
        np.ndarray: The atom map numbers of the molecule.
    """
    return [atom.GetAtomMapNum() for atom in mol.GetAtoms()]


def has_atom_map_numbers(mol: Chem.Mol) -> bool:
    """
    Check whether the molecule has atom map numbers.

    Args:
        mol (Chem.Mol): The molecule to check.

    Returns:
        bool: Whether the molecule has atom map numbers.
    """
    return any(atom.GetAtomMapNum() for atom in mol.GetAtoms())


def clear_atom_map_numbers(mol: Chem.Mol) -> List[int]:
    """
    Clear atom map numbers of the molecule

    Args:
        mol (Chem.Mol): The molecule to clear atom map numbers.

    Returns:
        np.ndarray: The atom map numbers of the molecule.
    """
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]


def needs_renumber(mol: Chem.Mol) -> bool:
    """
    Check whether the molecule needs renumbering. Expect atom map numbers to be non-zero and
    monotonically increasing but not necessarily continuous.

    Args:
        mol (Chem.Mol): The molecule to check.

    Returns:
        bool: Whether the molecule needs renumbering.
    """
    cur_atom_map_number = 0
    for atom in mol.GetAtoms():
        atom_map_num = atom.GetAtomMapNum()
        if atom_map_num <= cur_atom_map_number:
            return True
        cur_atom_map_number = atom_map_num
    return False


def _renumber_atoms(
    mol: Chem.Mol,
    new_order: Iterable,
    update_atom_map: bool = True,
):
    """
    Renumber the atoms of the molecule.

    Args:
        mol (Chem.Mol): The molecule to renumber atoms.
        new_order (Iterable): The new ordering the atoms.
        update_atom_map (bool, optional): Whether to update the atom map numbers of the molecule.
                                          Defaults to ``True``.

    Returns:
        Chem.Mol: The molecule with renumbered atoms.
    """
    mol = Chem.RenumberAtoms(mol, new_order)
    if update_atom_map:
        reset_atom_map_numbers(mol)
    return mol


def renumber_atoms(
    mol: Chem.Mol,
    new_order: Optional[Iterable] = None,
    update_atom_map: bool = True,
):
    """
    Renumber the atoms of the molecule.

    Args:
        mol (Chem.Mol): The molecule to renumber atoms.
        new_order (Optional[Iterable], optional): The new ordering the atoms.
                                                   - If provided as a list, it should a list of atom indexes
                                                   and should have a length of the number of atoms.
                                                   E.g., if newOrder is ``[3,2,0,1]``, then atom ``3``
                                                   in the original molecule will be atom ``0`` in the new one.
                                                   - If provided as a dict, it should be a mapping between atoms. E.g.,
                                                   if newOrder is ``{0: 3, 1: 1, 2: 2, 3: 0}``, then atom ``0`` in the
                                                   original molecule will be atom ``3`` in the new one. Unlike the list case,
                                                   the newOrder can be a partial mapping, but one should make sure all the pairs
                                                   are consistent. E.g.,``{0: 3}`` and ``{0: 3, 3: 0}`` are also acceptable
                                                   input for ``{0: 3, 1: 1, 2: 2, 3: 0}``, but you can't have inconsistent ones like
                                                   ``{0: 3, 3: 2}``.
                                                   - If no value provided (default), then the molecule
                                                   will be renumbered based on the current atom map numbers. It is helpful
                                                   when the sequence of atom map numbers and atom indexes are inconsistent.
        update_atom_map (bool, optional): Whether to update the atom map numbers of the molecule.
                                          Defaults to ``True``.

    Returns:
        Chem.Mol: The molecule with renumbered atoms.
    """
    if new_order is None:
        return renumber_atoms_by_atom_map_numbers(mol, update_atom_map)
    elif isinstance(new_order, dict):
        return renumber_atoms_by_map_dict(mol, new_order, update_atom_map)
    else:
        return _renumber_atoms(mol, new_order, update_atom_map)


def renumber_atoms_by_atom_map_numbers(
    mol: Chem.Mol,
    update_atom_map: bool = True,
) -> Chem.Mol:
    """
    Renumber the atoms of the molecule according to the atom map numbers in the molecule.
    It is okay if the atom map number is not continuous, as the renumbering is based on the ordering of the atom map number.

    Args:
        mol (Chem.Mol): The molecule to renumber atoms.
        update_atom_map (bool, optional): Whether to update the atom map numbers of the molecule.
                                          Defaults to ``True``.
                                          If ``False``, the atom map numbers will be kept.
                                          If ``True``, the atom map numbers will be reset be consistent with atomic number.

    Returns:
        Chem.Mol: The molecule with renumbered atoms.
    """
    mapping = reverse_map(get_atom_map_numbers(mol))
    return _renumber_atoms(mol, mapping, update_atom_map)


def renumber_atoms_by_substruct_match_result(
    mol: Chem.Mol,
    substruct_match_result: Iterable,
    as_ref: bool = True,
    update_atom_map: bool = True,
):
    """
    Renumber the atoms of the molecule according to the substruct match result.

    Args:
        mol (Chem.Mol): The molecule to renumber atoms.
        substruct_match_result (Iterable): The substruct match result. it should be a tuple of
                                           atom indices. E.g., if the substruct match result is
                                           ``(0, 2, 3)``, then atom ``0`` in the original molecule
                                           will be atom ``0`` in the new one, atom ``2`` in the
                                           original molecule will be atom ``1`` in the new one, and
                                           atom ``3`` in the original molecule will be atom ``2`` in
                                           the new one.
        as_ref (bool, optional): The molecule to renumber is used as the reference molecule during
                                 the substructure match (i.e., ``mol`` is the query molecule).
        update_atom_map (bool, optional): Whether to update the atom map numbers of the molecule.
                                           Defaults to ``True``.
    """
    mapping = substruct_match_result if as_ref else reverse_map(substruct_match_result)
    return _renumber_atoms(mol, mapping, update_atom_map)


def renumber_atoms_by_map_dict(
    mol: Chem.Mol,
    new_order: dict,
    update_atom_map: bool = True,
) -> Chem.Mol:
    """
    Renumber the atoms of the molecule according to a dict-based mapping.

    Args:
        mol (Chem.Mol): The molecule to renumber atoms.
        new_order(dict): The dict-based mapping, it should be a mapping between atoms. E.g.,
                         if newOrder is ``{0: 3, 1: 1, 2: 2, 3: 0}``, then atom ``0`` in the
                         original molecule will be atom ``3`` in the new one. Unlike the list case,
                         the newOrder can be a partial mapping, but one should make sure all the pairs
                         are consistent. E.g.,``{0: 3}`` and ``{0: 3, 3: 0}`` are also acceptable
                         input for ``{0: 3, 1: 1, 2: 2, 3: 0}``, but you can't have inconsistent ones like
                         ``{0: 3, 3: 2}``.
        update_atom_map (bool, optional): Whether to update the atom map numbers of the molecule.
                                          Defaults to ``True``.

    Returns:
        Chem.Mol: The molecule with renumbered atoms.
    """
    new_order_ = list(range(mol.GetNumAtoms()))
    for pair in new_order.items():
        new_order_[pair[1]] = pair[0]
        new_order_[pair[0]] = pair[1]

    return _renumber_atoms(mol, new_order_, update_atom_map)


def reset_atom_map_numbers(mol: Chem.Mol):
    """
    Reset atom map numbers according to the atom indices.

    Args:
        mol (Chem.Mol): The molecule to reset atom map numbers.
    """
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        atom.SetAtomMapNum(idx + 1)


def reverse_map(map: Iterable, as_list: bool = True):
    """
    Inverse-transform the index and value relationship in a mapping.
    E.g., when doing a subgraph match, RDKit will returns a list
    that the indexes correspond to the reference molecule and the values
    correspond to the probing molecule. To recover the reference molecule, one
    needs to get the reverse map and renumber the atoms of probing molecule with
    the reverse map.

    Args:
        map (Iterable): An atom mapping as a list (1D array).
        as_list (bool, optional): Output result as a `list` object. Otherwise,
                                  the output is a np.ndarray.

    Returns:
        An inverted atom map from the given ``match`` atom map
    """
    return np.argsort(map).tolist() if as_list else np.argsort(map)


def update_product_atom_map_after_reaction(
    mol: Chem.Mol,
    ref_mol: Chem.Mol,
) -> List[Chem.Atom]:
    """
    Update the atom map number of the product molecule after reaction according to the reference molecule (usually the reactant).
    The operation is in-place.

    Args:
        mol (Chem.Mol): The product molecule after reaction.
        ref_mol (Chem.Mol): The reference molecule (usually the reactant).
        clean_rxn_props (bool, optional): Whether to clean the reaction properties (`"old_mapno"` and `"react_atom_idx"`).
                                          Defaults to ``True``.

    Returns:
        list: atoms that are updated.
    """
    updated_atoms = []
    for atom in mol.GetAtoms():
        if atom.HasProp("old_mapno"):
            updated_atoms.append(atom)
            # atom map number of the product will zeroed out during the reaction
            react_atom_idx = int(atom.GetProp("react_atom_idx"))
            atom.SetAtomMapNum(ref_mol.GetAtomWithIdx(react_atom_idx).GetAtomMapNum())

    return updated_atoms
