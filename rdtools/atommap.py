"""A module of atom map number operations."""

from itertools import permutations
from typing import Iterable, List, Literal, Optional, Tuple, Union, cast, overload

import numpy as np
import numpy.typing as npt
from rdkit import Chem

from rdtools.bond import get_formed_and_broken_bonds


def get_atom_map_numbers(mol: Chem.Mol) -> list[int]:
    """Get atom map numbers of the molecule.

    Args:
        mol (Chem.Mol): The molecule to get atom map numbers.

    Returns:
        list[int]: The atom map numbers of the molecule.
    """
    return [atom.GetAtomMapNum() for atom in mol.GetAtoms()]


def has_atom_map_numbers(mol: Chem.Mol) -> bool:
    """Check whether the molecule has atom map numbers.

    Args:
        mol (Chem.Mol): The molecule to check.

    Returns:
        bool: Whether the molecule has atom map numbers.
    """
    return any(atom.GetAtomMapNum() for atom in mol.GetAtoms())


def clear_atom_map_numbers(mol: Chem.Mol) -> None:
    """Clear atom map numbers of the molecule.

    Args:
        mol (Chem.Mol): The molecule to clear atom map numbers.
    """
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]


def needs_renumber(mol: Chem.Mol) -> bool:
    """Check whether the molecule needs renumbering.

    Expect atom map numbers to be non-zero and monotonically
    increasing but not necessarily continuous.

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
    new_order: Iterable[int],
    update_atom_map: bool = True,
) -> Chem.Mol:
    """Renumber the atoms of the molecule.

    Args:
        mol (Chem.Mol): The molecule to renumber atoms.
        new_order (Iterable[int]): The new ordering the atoms.
        update_atom_map (bool, optional): Whether to update the atom map numbers of the molecule.
            Defaults to ``True``.

    Returns:
        Chem.Mol: The molecule with renumbered atoms.
    """
    new_mol = Chem.RenumberAtoms(mol, new_order)
    if isinstance(mol, Chem.RWMol):
        new_mol = mol.__class__(new_mol)
    if update_atom_map:
        reset_atom_map_numbers(new_mol)
    return new_mol


def renumber_atoms(
    mol: Chem.Mol,
    new_order: Optional[Union[Iterable[int], dict[int, int]]] = None,
    update_atom_map: bool = True,
) -> Chem.Mol:
    """Renumber the atoms of the molecule.

    Args:
        mol (Chem.Mol): The molecule to renumber atoms.
        new_order (Optional[Union[Iterable[int], dict[int, int]]], optional):
            The new ordering the atoms.
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
    """Renumber the atoms according to the atom map numbers in the molecule.

    It is okay if the atom map number is not continuous, as the renumbering is
    based on the ordering of the atom map number.

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
    substruct_match_result: Iterable[int],
    as_ref: bool = True,
    update_atom_map: bool = True,
) -> Chem.Mol:
    """Renumber the atoms of the molecule according to the substruct match result.

    Args:
        mol (Chem.Mol): The molecule to renumber atoms.
        substruct_match_result (Iterable[int]): The substruct match result. it should be a tuple of
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

    Returns:
        Chem.Mol: The molecule with renumbered atoms.
    """
    mapping = substruct_match_result if as_ref else reverse_map(substruct_match_result)
    return _renumber_atoms(mol, mapping, update_atom_map)


def renumber_atoms_by_map_dict(
    mol: Chem.Mol,
    new_order: dict[int, int],
    update_atom_map: bool = True,
) -> Chem.Mol:
    """Renumber the atoms of the molecule according to a dict-based mapping.

    Args:
        mol (Chem.Mol): The molecule to renumber atoms.
        new_order(dict[int, int]): The dict-based mapping, it should be a mapping between atoms. E.g.,
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


def reset_atom_map_numbers(mol: Chem.Mol) -> None:
    """Reset atom map numbers according to the atom indices.

    Args:
        mol (Chem.Mol): The molecule to reset atom map numbers.
    """
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        atom.SetAtomMapNum(idx + 1)


@overload
def reverse_map(map: Iterable[int], as_list: Literal[True]) -> List[int]: ...


@overload
def reverse_map(
    map: Iterable[int], as_list: Literal[False]
) -> npt.NDArray[np.int_]: ...


@overload
def reverse_map(map: Iterable[int]) -> List[int]: ...


def reverse_map(
    map: Iterable[int], as_list: bool = True
) -> Union[npt.NDArray[np.int_], List[int]]:
    """Inverse-transform the index and value relationship in a mapping.

    E.g., when doing
    a subgraph match, RDKit will returns a list that the indexes correspond to the
    reference molecule and the values correspond to the probing molecule. To recover the
    reference molecule, one needs to get the reverse map and renumber the atoms of
    probing molecule with the reverse map.

    Args:
        map (Iterable[int]): An atom mapping as a list (1D array).
        as_list (bool, optional): Output result as a `list` object. Otherwise,
            the output is a np.ndarray.

    Returns:
        Union[npt.NDArray[np.int_], List[int]]: An inverted atom map from the given ``match`` atom map
    """
    if as_list:
        return cast(list[int], np.argsort(map).tolist())  # type: ignore[arg-type]
    return np.argsort(map)  # type: ignore[arg-type]


def update_product_atom_map_after_reaction(
    mol: Chem.Mol,
    ref_mol: Chem.Mol,
) -> list[Chem.Atom]:
    """Update the atom map number of the product molecule after reaction.

    The renumbering is according to
    the reference molecule (usually the reactant). The operation is in-place.

    Args:
        mol (Chem.Mol): The product molecule after reaction.
        ref_mol (Chem.Mol): The reference molecule (usually the reactant).

    Returns:
        list[Chem.Atom]: atoms that are updated.
    """
    updated_atoms = []
    for atom in mol.GetAtoms():
        if atom.HasProp("old_mapno"):
            updated_atoms.append(atom)
            # atom map number of the product will zeroed out during the reaction
            react_atom_idx = int(atom.GetProp("react_atom_idx"))
            atom.SetAtomMapNum(ref_mol.GetAtomWithIdx(react_atom_idx).GetAtomMapNum())

    return updated_atoms


def move_atommaps_to_notes(mol: Chem.Mol, clear_atommap: bool = True) -> None:
    """Move atom map numbers to the `atomNote` property.

    This helps to make the display slightly cleaner.

    Args:
        mol (Chem.Mol): The molecule to move atom map numbers to the `atomNote` property.
        clear_atommap (bool, optional): Whether to clear the atom map numbers after moving.
            Defaults to ``True``.
            If ``True``, the atom map numbers will be cleared.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum():
            atom.SetProp("atomNote", str(atom.GetAtomMapNum()))
            if clear_atommap:
                atom.SetAtomMapNum(0)


def move_notes_to_atommaps(mol: Chem.Mol) -> None:
    """Move atom map numbers from `atomNote` to the `atomMap` property.

    This is only valid if there is no other info in atomNote.

    Args:
        mol (Chem.Mol): The molecule to move atom map numbers to the `atomMap` property.
    """
    for atom in mol.GetAtoms():
        if atom.HasProp("atomNote"):
            atom.SetAtomMapNum(int(atom.GetProp("atomNote")))
            atom.ClearProp("atomNote")


def map_h_atoms_in_reaction(
    rmol: Chem.Mol,
    pmol: Chem.Mol,
) -> Tuple[Chem.Mol, Chem.Mol]:
    """Map H atoms in a reaction given that heavy atoms are mapped already.

    The function
    only applies to the case where each atom in rmol paired to an atom in pmol and vise
    verse. Atom map numbers are assumed to start from 1. This is originally work for the
    results from rxnmapper, and viability to other input is unknown.

    Args:
        rmol (Chem.Mol): The reactant molecule.
        pmol (Chem.Mol): The product molecule.

    Returns:
        Tuple[Chem.Mol, Chem.Mol]: The reactant and product molecule with mapped H atoms.
    """
    # Even though the heavy atoms atoms are mapped,
    # there can be cases with unequal number of atoms labeled with atom map index. Known cases:
    # * H will not be labeled in most case but will be labeled if it has no neighbors, [H][H] or [H]

    r_atommap_nums, p_atommap_nums = (
        get_atom_map_numbers(rmol),
        get_atom_map_numbers(pmol),
    )

    # Step 1: pair the heavy atoms
    ridx_to_pidx = {}
    for ridx, map_num in enumerate(r_atommap_nums):
        if map_num != 0:
            try:
                pidx = p_atommap_nums.index(map_num)
            except ValueError:
                continue
            else:
                ridx_to_pidx[ridx] = pidx

    # Step 2: map non-reacting H atoms
    unused_ridxs = []
    unset_pidxs = []

    idxs_to_add = {}
    for ridx, pidx in ridx_to_pidx.items():
        ratom = rmol.GetAtomWithIdx(ridx)
        patom = pmol.GetAtomWithIdx(pidx)

        rH_nbs, pH_nbs = [], []
        for rnb in ratom.GetNeighbors():
            if rnb.GetAtomicNum() == 1:
                rH_nbs.append(rnb.GetIdx())
        for pnb in patom.GetNeighbors():
            if pnb.GetAtomicNum() == 1:
                pH_nbs.append(pnb.GetIdx())
        for rH_idx, pH_idx in zip(rH_nbs, pH_nbs):
            idxs_to_add[rH_idx] = pH_idx

        if len(rH_nbs) > len(pH_nbs):
            unused_ridxs.extend(rH_nbs[len(pH_nbs) :])
        elif len(rH_nbs) < len(pH_nbs):
            unset_pidxs.extend(pH_nbs[len(rH_nbs) :])

    ridx_to_pidx.update(idxs_to_add)

    # Step 3: map reacting H atoms
    n_atoms = len(r_atommap_nums)

    if unused_ridxs or unset_pidxs:
        if not unused_ridxs:
            unmapped_ridxs = set(range(n_atoms)) - set(ridx_to_pidx.keys())
            for ridx, pidx in zip(unmapped_ridxs, unset_pidxs):
                ridx_to_pidx[ridx] = pidx

        elif not unset_pidxs:
            unmapped_pidxs = set(range(n_atoms)) - set(ridx_to_pidx.values())
            for ridx, pidx in zip(unused_ridxs, unmapped_pidxs):
                ridx_to_pidx[ridx] = pidx

        elif len(unused_ridxs) == len(unset_pidxs) == 1:
            ridx_to_pidx[unused_ridxs[0]] = unset_pidxs[0]

        else:
            min_num_bond_change = np.inf
            opt_pairs: tuple[tuple[int, int], ...] = ()
            for pairs in permutations(zip(unused_ridxs, unset_pidxs)):
                for pair in pairs:
                    ridx_to_pidx[pair[0]] = pair[1]
                formed, broken = get_formed_and_broken_bonds(
                    renumber_atoms_by_map_dict(rmol, ridx_to_pidx),
                    pmol,
                )
                if len(formed) + len(broken) < min_num_bond_change:
                    opt_pairs = pairs
                    min_num_bond_change = len(formed) + len(broken)

            for pair in opt_pairs:
                ridx_to_pidx[pair[0]] = pair[1]

    # Step 4: update atom map numbers
    # Assign atom map number missing in reactant
    remaining_atom_map_nums = []
    for i in range(1, max(len(r_atommap_nums), len(p_atommap_nums)) + 1):
        if i not in r_atommap_nums and i not in p_atommap_nums:
            # discontinuity
            remaining_atom_map_nums.append(i)
        elif i not in r_atommap_nums:  # must be in pmol
            pidx = p_atommap_nums.index(i)
            ridx = [ridx for ridx, pid in ridx_to_pidx.items() if pid == pidx][0]
            rmol.GetAtomWithIdx(ridx).SetAtomMapNum(i)
            r_atommap_nums[ridx] = i

    # assign H atom map numbers
    cur_atommap_idx = 0
    for i, mapnum in enumerate(r_atommap_nums):
        if mapnum == 0:
            rmol.GetAtomWithIdx(i).SetAtomMapNum(
                remaining_atom_map_nums[cur_atommap_idx]
            )
            cur_atommap_idx += 1

    # assign pmol atom map numbers
    for ridx, pidx in ridx_to_pidx.items():
        pmol.GetAtomWithIdx(pidx).SetAtomMapNum(
            rmol.GetAtomWithIdx(ridx).GetAtomMapNum()
        )

    return renumber_atoms(rmol, update_atom_map=False), renumber_atoms(
        pmol, update_atom_map=False
    )
