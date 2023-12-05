from typing import List

from rdkit import Chem

# The rotational bond definition in RDkit
# It is the same as rdkit.Chem.Lipinski import RotatableBondSmarts
ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
ROTATABLE_BOND_SMARTS_WO_METHYL = Chem.MolFromSmarts(
    "[!$(*#*)&!D1!H3]-&!@[!$(*#*)&!D1&!H3]"
)


def determine_smallest_atom_index_in_torsion(
    atom1: Chem.Atom,
    atom2: Chem.Atom,
) -> int:
    """
    Determine the smallest atom index in mol connected to ``atom1`` which is not ``atom2``.
    Returns a heavy atom if available, otherwise a hydrogen atom.
    Useful for deterministically determining the indices of four atom in a torsion.
    This function assumes there ARE additional atoms connected to ``atom1``, and that ``atom2`` is not a hydrogen atom.

    Args:
        atom1 (Atom): The atom who's neighbors will be searched.
        atom2 (Atom): An atom connected to ``atom1`` to exclude (a pivotal atom).

    Returns:
        int: The smallest atom index (1-indexed) connected to ``atom1`` which is not ``atom2``.
    """
    neighbor = [a for a in atom1.GetNeighbors() if a.GetIdx() != atom2.GetIdx()]
    atomic_num_list = sorted([nb.GetAtomicNum() for nb in neighbor])
    try:
        min_atomic, max_atomic = atomic_num_list[0], atomic_num_list[-1]
    except IndexError:
        raise ValueError(
            f"Invalid torsion: "
            f"Atom {atom1.GetIdx()} has no neighbors except {atom2.GetIdx()}."
        )
    if min_atomic == max_atomic or min_atomic > 1:
        return min([nb.GetIdx() for nb in neighbor])
    else:
        return min([nb.GetIdx() for nb in neighbor if nb.GetAtomicNum() != 1])


def find_internal_torsions(
    mol: Chem.Mol,
    exclude_methyl: bool = False,
) -> List[List[int]]:
    """
    Find the internal torsions from RDkit molecule.

    Args:
        mol (Chem.Mol): RDKit molecule.
        exclude_methyl (bool): Whether exclude the torsions with methyl groups.

    Returns:
        list: A list of internal torsions.
    """
    query = (
        ROTATABLE_BOND_SMARTS if not exclude_methyl else ROTATABLE_BOND_SMARTS_WO_METHYL
    )
    rot_atom_pairs = mol.GetSubstructMatches(query, uniquify=False)

    torsions = []
    for atoms_idxs in rot_atom_pairs:
        # Remove duplicates due to smart matching, e.g., (2,3) and (3,2)
        if atoms_idxs[0] > atoms_idxs[1]:
            continue
        pivots = [mol.GetAtomWithIdx(i) for i in atoms_idxs]
        first_atom_ind = determine_smallest_atom_index_in_torsion(*pivots)
        pivots.reverse()
        last_atom_ind = determine_smallest_atom_index_in_torsion(*pivots)
        torsions.append([first_atom_ind, *atoms_idxs, last_atom_ind])
    return torsions


def find_ring_torsions(mol: Chem.Mol) -> List[List[int]]:
    """
    Find the ring from RDkit molecule.

    Args:
        mol (Chem.Mol): RDKit molecule.

    Returns:
        list: A list of ring torsions.
    """
    # Originally uses CalculateTorsionLists
    # Replace by the implementation athttps://github.com/rdkit/rdkit/
    # blob/abbad1689982c899797db3b1636792da5ff0429a/rdkit/Chem/TorsionFingerprints.py#L266C2-L280C36
    ring_torsions = []
    rings = Chem.GetSymmSSSR(mol)
    for ring in rings:
        ring_size = len(ring)
        for i in range(ring_size):
            tor = [ring[(i + j) % ring_size] for j in range(4)]
            if tor[1] > tor[2]:
                tor.reverse()
            ring_torsions.append(tor)
    return ring_torsions


def get_torsional_modes(
    mol: Chem.Mol,
    exclude_methyl: bool = False,
    include_ring: bool = True,
) -> List[List[int]]:
    """
    Get the torsional modes from RDkit molecule.

    Args:
        mol (Chem.Mol): RDKit molecule.
        exclude_methyl (bool): Whether exclude the torsions with methyl groups. defaults to ``False``.
        include_ring (bool): Whether include the ring torsions. Defaults to ``True``.

    Returns:
        list: A list of torsional modes.
    """
    internal_torsions = find_internal_torsions(mol, exclude_methyl)
    ring_torsions = find_ring_torsions(mol) if include_ring else []

    return internal_torsions + ring_torsions
