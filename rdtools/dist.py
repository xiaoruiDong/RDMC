# -*- coding: utf-8 -*-
"""A module contains functions to manipulate distances in a molecule."""

from functools import lru_cache
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
from rdkit import Chem

from rdtools.element import get_bond_radius, get_covalent_radius, get_vdw_radius


def get_adjacency_matrix(mol: Chem.Mol) -> npt.NDArray[np.int_]:
    """Get the adjacency matrix of the molecule.

    Args:
        mol (Chem.Mol): The molecule to get the adjacency matrix of.

    Returns:
        npt.NDArray[np.int_]: A square adjacency matrix of the molecule, where a `1` indicates that atoms are bonded
            and a `0` indicates that atoms aren't bonded.
    """
    return Chem.GetAdjacencyMatrix(mol)


def get_distance_matrix(
    mol: Chem.Mol,
    conf_id: int = -1,
    balaban: bool = False,
) -> npt.NDArray[np.float64]:
    """Get the distance matrix of the molecule.

    Args:
        mol (Chem.Mol): The molecule to get the distance matrix of.
        conf_id (int, optional): The conformer ID of the molecule to get the distance matrix of. Defaults to ``-1``.
        balaban (bool, optional): Whether to use the Balaban distance. Defaults to ``False``.
            If ``True``, the distance matrix will be calculated using the Balaban distance.

    Returns:
        npt.NDArray[np.float64]: A square distance matrix of the molecule, where the value of each element is the distance
            between the atoms.
    """
    return Chem.Get3DDistanceMatrix(mol, confId=conf_id, useAtomWts=balaban)


def _create_matrix_with_radii_values(
    radii: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Create a matrix with the radii values of the atoms.

    Args:
        radii (npt.NDArray[np.float64]): A 1 x N array of the radii of the atoms.

    Returns:
        npt.NDArray[np.float64]: A N x N square matrix with the radii values of the atoms.
    """
    num_atom = len(radii)
    mat = np.zeros((num_atom, num_atom))
    return mat + radii + radii.T


@lru_cache()
def get_vdw_distance_matrix(
    mol: Chem.Mol,
) -> npt.NDArray[np.float64]:
    """Generate a van der Waals matrix of the molecule.

    Args:
        mol (Chem.Mol): The molecule to generate the derived van der Waals matrix of.

    Returns:
        npt.NDArray[np.float64]: A square van der Waals matrix of the molecule, where the value of each element is the van der
            Waals radius of the atoms.
    """
    vdw_radii = np.array([
        [
            get_vdw_radius(mol.GetAtomWithIdx(i).GetAtomicNum())
            for i in range(mol.GetNumAtoms())
            # TODO: Update this to mol.GetAtoms() when RDKit 2023.09 is widely used
        ]
    ])
    return _create_matrix_with_radii_values(vdw_radii)


@lru_cache()
def get_covalent_distance_matrix(mol: Chem.Mol) -> npt.NDArray[np.float64]:
    """Generate a covalent matrix of the molecule.

    Args:
        mol (Chem.Mol): The molecule to generate the derived covalent matrix of.

    Returns:
        npt.NDArray[np.float64]: A square covalent matrix of the molecule, where the value of each element is the covalent
            radius of the atoms.
    """
    covalent_radii = np.array([
        [
            get_covalent_radius(mol.GetAtomWithIdx(i).GetAtomicNum())
            for i in range(mol.GetNumAtoms())
        ]
    ])
    return _create_matrix_with_radii_values(covalent_radii)


@lru_cache()
def get_bond_distance_matrix(mol: Chem.Mol) -> npt.NDArray[np.float64]:
    """Get the bond distance matrix of the molecule.

    Args:
        mol (Chem.Mol): The molecule to get the bond distance matrix of.

    Returns:
        npt.NDArray[np.float64]: A square bond distance matrix of the molecule, where the value of each element is the
            bond distance between the atoms.
    """
    bond_radii = np.array([
        [
            get_bond_radius(mol.GetAtomWithIdx(i).GetAtomicNum())
            for i in range(mol.GetNumAtoms())
        ]
    ])
    return _create_matrix_with_radii_values(bond_radii)


_ref_mat = {
    "vdw": get_vdw_distance_matrix,
    "covalent": get_covalent_distance_matrix,
    "bond": get_bond_distance_matrix,
}


def _get_close_atoms(
    mol: Chem.Mol,
    conf_id: int = -1,
    threshold: float = 0.4,
    reference: Literal["vdw", "covalent", "bond"] = "vdw",
) -> npt.NDArray[np.int_]:
    """Get a upper triangular matrix indicating close atoms.

    If the distance between two atoms <= threshold * reference distance, the atoms are
    considered to be too close. The reference distance is based on the van der Waals
    radius, covalent radius or bond radius.

    Args:
        mol (Chem.Mol): The molecule.
        conf_id (int, optional): The conformer ID of the molecule to get the distance matrix of. Defaults to ``-1``.
        threshold (float, optional): A multiplier applied on the reference matrix . Defaults to ``0.4``.
        reference (Literal["vdw", "covalent", "bond"], optional): The reference matrix to use. Defaults to ``"vdw"``.
            Options:
            - ``"vdw"`` for reference distance based on van de Waals radius
            - ``"covalent"`` for reference distance based on covalent radius
            - ``"bond"`` for reference distance based on bond radius

    Returns:
        npt.NDArray[np.int_]: A upper triangular matrix indicating if the distance between two atoms <= threshold * reference distance are too close
    """
    # Filter out the bonded atoms, as they are expected to be within the threshold
    adjacency_mat = get_adjacency_matrix(mol)
    # Remove the diagonal and lower triangle
    dist_mat = np.triu(get_distance_matrix(mol, conf_id=conf_id))
    ref_mat = _ref_mat[reference](mol)
    # only compare the elements that are not zero
    return (adjacency_mat == 0) & (dist_mat != 0) & (dist_mat <= threshold * ref_mat)


def has_colliding_atoms(
    mol: Chem.Mol,
    conf_id: int = -1,
    threshold: float = 0.4,
    reference: Literal["vdw", "covalent", "bond"] = "vdw",
) -> bool:
    """Check whether the molecule has colliding atoms.

    If the distance between two atoms <= threshold * reference distance,
    the atoms are considered to be colliding.

    Args:
        mol (Chem.Mol): The molecule to check for colliding atoms.
        conf_id (int, optional): The conformer ID of the molecule to get the distance matrix of. Defaults to ``-1``.
        threshold (float, optional): A multiplier applied on the reference matrix . Defaults to ``0.4``.
        reference (Literal["vdw", "covalent", "bond"], optional): The reference matrix to use. Defaults to ``"vdw"``.
            Options:
            - ``"vdw"`` for reference distance based on van de Waals radius
            - ``"covalent"`` for reference distance based on covalent radius
            - ``"bond"`` for reference distance based on bond radius

    Returns:
        bool: Whether the molecule has colliding atoms.
    """
    return bool(np.any(_get_close_atoms(mol, conf_id, threshold, reference)))


def get_colliding_atoms(
    mol: Chem.Mol,
    conf_id: int = -1,
    threshold: float = 0.4,
    reference: Literal["vdw", "covalent", "bond"] = "vdw",
) -> list[tuple[int, int]]:
    """Get the atom pairs that are considered colliding.

    The atoms are considered to be colliding if the distance between two atoms
    <= threshold * reference distance. The reference distance is based on the
    van der Waals radius, covalent radius or bond radius.

    Args:
        mol (Chem.Mol): The molecule to check for colliding atoms.
        conf_id (int, optional): The conformer ID of the molecule to get the distance matrix of. Defaults to ``-1``.
        threshold (float, optional): A multiplier applied on the reference matrix . Defaults to ``0.4``.
        reference (Literal["vdw", "covalent", "bond"], optional): The reference matrix to use. Defaults to ``"vdw"``.
            Options:
            - ``"vdw"`` for reference distance based on van de Waals radius
            - ``"covalent"`` for reference distance based on covalent radius
            - ``"bond"`` for reference distance based on bond radius

    Returns:
        list[tuple[int, int]]: A list of tuples of the atom indices that are potentially colliding.
    """
    return list(zip(*np.nonzero(_get_close_atoms(mol, conf_id, threshold, reference))))


def get_missing_bonds(
    mol: Chem.Mol,
    conf_id: int = -1,
    threshold: float = 1.5,
    reference: Literal["vdw", "covalent", "bond"] = "covalent",
) -> list[tuple[int, int]]:
    """Check whether the molecule has missing bonds heuristically.

    If the distance between two atoms <= threshold * reference distance, the bond
    between the atoms are considered to be missing.

    Args:
        mol (Chem.Mol): The molecule to check for missing bonds.
        conf_id (int, optional): The conformer ID of the molecule to get the distance matrix of. Defaults to ``-1``.
        threshold (float, optional): A multiplier applied on the reference matrix . Defaults to ``1.5``.
        reference (Literal["vdw", "covalent", "bond"], optional): The reference matrix to use. Defaults to ``"covalent"``.
            Options:
            - ``"vdw"`` for reference distance based on van de Waals radius
            - ``"covalent"`` for reference distance based on covalent radius
            - ``"bond"`` for reference distance based on bond radius

    Returns:
        list[tuple[int, int]]: A list of tuples of the atom indices that are potentially missing bonds.
    """
    return get_colliding_atoms(mol, conf_id, threshold, reference)


def _find_shortest_path(
    start: Chem.Atom,
    end: Chem.Atom,
    path: Optional[list[Chem.Atom]] = None,
    path_idxs: Optional[list[int]] = None,
) -> Optional[list[Chem.Atom]]:
    """Get the shortest path between two atoms in a molecule.

    Args:
        start (Chem.Atom): The starting atom.
        end (Chem.Atom): The ending atom.
        path (Optional[list[Chem.Atom]], optional): The current path. Defaults to None.
        path_idxs (Optional[list[int]], optional): The current path indexes. Defaults to None.

    Returns:
        Optional[list[Chem.Atom]]: A list of atoms in the shortest path between the two atoms.
    """
    path = path if path else []
    path_idxs = path_idxs if path_idxs else []
    path = path + [start]
    path_idx = path_idxs + [start.GetIdx()]
    if path_idx[-1] == end.GetIdx():
        return path

    shortest = None
    for node in start.GetNeighbors():
        if node.GetIdx() not in path_idx:
            newpath = _find_shortest_path(node, end, path, path_idx)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest


def get_shortest_path(mol: Chem.Mol, idx1: int, idx2: int) -> tuple[int, ...]:
    """Get the shortest path between two atoms in a molecule.

    The RDKit ``GetShortestPath``
    has a very long setup time ~ 0.5ms (on test machine) regardless of the size of the
    molecule. As a comparison, on the same machine, a naive python implementation of DFS
    (`_find_shortest_path`) takes ~0.5 ms for a 100-C normal alkane end to end.
    Therefore, it make more sense to use a method with a shorter setup time though
    scaling worse for smaller molecules while using GetShortestPath for larger
    molecules.

    Args:
        mol (Chem.Mol): The molecule to be checked.
        idx1 (int): The index of the first atom.
        idx2 (int): The index of the second atom.

    Returns:
        tuple[int, ...]: A list of atoms in the shortest path between the two atoms.
    """
    if mol.GetNumHeavyAtoms() > 100:  # An empirical cutoff
        return Chem.GetShortestPath(mol, idx1, idx2)

    shortest_path = _find_shortest_path(
        mol.GetAtomWithIdx(idx1), mol.GetAtomWithIdx(idx2)
    )
    if shortest_path is not None:
        return tuple(atom.GetIdx() for atom in shortest_path)
    return ()
