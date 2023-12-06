from typing import List, Tuple

import numpy as np

from rdkit import Chem
from rdmc.rdtools.element import get_vdw_radius, get_covalent_radius, get_bond_radius


def get_adjacency_matrix(mol: Chem.Mol) -> np.ndarray:
    """
    Get the adjacency matrix of the molecule.

    Args:
        mol (rdkit.Chem.Mol): The molecule to get the adjacency matrix of.

    Returns:
        numpy.ndarray: A square adjacency matrix of the molecule, where a `1` indicates that atoms are bonded
                        and a `0` indicates that atoms aren't bonded.
    """
    return Chem.GetAdjacencyMatrix(mol)


def get_distance_matrix(
    mol: Chem.Mol,
    conf_id: int = -1,
) -> np.ndarray:
    """
    Get the distance matrix of the molecule.

    Args:
        mol (rdkit.Chem.Mol): The molecule to get the distance matrix of.
        conf_id (int, optional): The conformer ID of the molecule to get the distance matrix of. Defaults to ``-1``.

    Returns:
        numpy.ndarray: A square distance matrix of the molecule, where the value of each element is the distance
                        between the atoms.
    """
    return Chem.Get3DDistanceMatrix(mol, confId=conf_id)


def _create_matrix_with_radii_values(radii: np.array) -> np.array:
    """
    Create a matrix with the radii values of the atoms.

    Args:
        radii (numpy.ndarray): A 1 x N array of the radii of the atoms.

    Returns:
        numpy.ndarray: A N x N square matrix with the radii values of the atoms.
    """
    num_atom = len(radii)
    mat = np.zeros((num_atom, num_atom))
    return mat + radii + radii.T


def get_vdw_distance_matrix(
    mol: Chem.Mol,
) -> np.array:
    """
    Generate a van der Waals matrix of the molecule.

    Args:
        mol (rdkit.Chem.Mol): The molecule to generate the derived van der Waals matrix of.

    Raises:
        ValueError: Invalid threshold is supplied.
    """
    vdw_radii = np.array(
        [
            [
                get_vdw_radius(mol.GetAtomWithIdx(i).GetAtomicNum())
                for i in range(mol.GetNumAtoms())
                # TODO: Update this to mol.GetAtoms() when RDKit 2023.09 is widely used
            ]
        ]
    )
    return _create_matrix_with_radii_values(vdw_radii)


def get_covalent_distance_matrix(mol: Chem.Mol) -> np.array:
    """
    Generate a covalent matrix of the molecule.

    Args:
        mol (rdkit.Chem.Mol): The molecule to generate the derived covalent matrix of.

    Returns:
        numpy.ndarray: A square covalent matrix of the molecule, where the value of each element is the covalent
                       radius of the atoms.
    """
    covalent_radii = np.array(
        [
            [
                get_covalent_radius(mol.GetAtomWithIdx(i).GetAtomicNum())
                for i in range(mol.GetNumAtoms())
            ]
        ]
    )
    return _create_matrix_with_radii_values(covalent_radii)

def get_bond_distance_matrix(mol: Chem.Mol) -> np.ndarray:
    """
    Get the bond distance matrix of the molecule.

    Args:
        mol (rdkit.Chem.Mol): The molecule to get the bond distance matrix of.

    Returns:
        numpy.ndarray: A square bond distance matrix of the molecule, where the value of each element is the
                        bond distance between the atoms.
    """
    bond_radii = np.array(
        [
            [
                get_bond_radius(mol.GetAtomWithIdx(i).GetAtomicNum())
                for i in range(mol.GetNumAtoms())
            ]
        ]
    )
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
    reference: str = "vdw",
) -> np.array:
    """
    Get a upper triangular matrix indicating if the distance between two atoms <= threshold * reference distance are too close
    for non-bonding atoms.

    Args:
        mol (rdkit.Chem.Mol): The molecule.
        conf_id (int, optional): The conformer ID of the molecule to get the distance matrix of. Defaults to ``-1``.
        threshold (float): A multiplier applied on the reference matrix . Defaults to ``0.4``.
        reference (str): The reference matrix to use. Defaults to ``"vdw"``.
                         Options:
                         - ``"vdw"`` for reference distance based on van de Waals radius
                         - ``"covalent"`` for reference distance based on covalent radius
                         - ``"bond"`` for reference distance based on bond radius

    Returns:
        np.array: A upper triangular matrix indicating if the distance between two atoms <= threshold * reference distance are too close
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
    reference: str = "vdw",
) -> bool:
    """
    Check whether the molecule has colliding atoms. If the distance between two atoms <= threshold * reference distance,
    the atoms are considered to be colliding.

    Args:
        mol (rdkit.Chem.Mol): The molecule to check for colliding atoms.
        conf_id (int, optional): The conformer ID of the molecule to get the distance matrix of. Defaults to ``-1``.
        threshold (float): A multiplier applied on the reference matrix . Defaults to ``0.4``.
        reference (str): The reference matrix to use. Defaults to ``"vdw"``.
                         Options:
                         - ``"vdw"`` for reference distance based on van de Waals radius
                         - ``"covalent"`` for reference distance based on covalent radius
                         - ``"bond"`` for reference distance based on bond radius

    Returns:
        bool: Whether the molecule has colliding atoms.
    """
    return np.any(_get_close_atoms(mol, conf_id, threshold, reference))


def get_colliding_atoms(
    mol: Chem.Mol,
    conf_id: int = -1,
    threshold: float = 0.4,
    reference: str = "vdw",
) -> List[Tuple[int,int]]:
    """
    Get the atom pairs that are potentially colliding (if the distance between two atoms <= threshold * reference distance).

    Args:
        mol (rdkit.Chem.Mol): The molecule to check for colliding atoms.
        conf_id (int, optional): The conformer ID of the molecule to get the distance matrix of. Defaults to ``-1``.
        threshold (float): A multiplier applied on the reference matrix . Defaults to ``0.4``.
        reference (str): The reference matrix to use. Defaults to ``"vdw"``.
                         Options:
                            - ``"vdw"`` for reference distance based on van de Waals radius
                            - ``"covalent"`` for reference distance based on covalent radius
                            - ``"bond"`` for reference distance based on bond radius

    Returns:
        list: A list of tuples of the atom indices that are potentially colliding.
    """
    return list(zip(*np.nonzero(_get_close_atoms(mol, conf_id, threshold, reference))))


def get_missing_bonds(
    mol: Chem.Mol,
    conf_id: int = -1,
    threshold: float = 1.5,
    reference: str = "covalent",
) -> List[Tuple[int,int]]:
    """
    Check whether the molecule has missing bonds. If the distance between two atoms <= threshold * reference distance,
    the bond between the atoms are considered to be missing.

    Args:
        mol (rdkit.Chem.Mol): The molecule to check for missing bonds.
        conf_id (int, optional): The conformer ID of the molecule to get the distance matrix of. Defaults to ``-1``.
        threshold (float): A multiplier applied on the reference matrix . Defaults to ``1.5``.
        reference (str): The reference matrix to use. Defaults to ``"covalent"``.
                         Options:
                            - ``"vdw"`` for reference distance based on van de Waals radius
                            - ``"covalent"`` for reference distance based on covalent radius
                            - ``"bond"`` for reference distance based on bond radius

    Returns:
        list: A list of tuples of the atom indices that are potentially missing bonds.
    """
    return get_colliding_atoms(mol, conf_id, threshold, reference)
