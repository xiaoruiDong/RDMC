# -*- coding: utf-8 -*-
"""A module contains functions to manipulate conformers in a molecule."""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms as rdMT

from rdtools.bond import add_bonds, get_bonds_as_tuples


def add_conformer(
    mol: Chem.RWMol,
    conf: Optional[Chem.Conformer] = None,
    coords: Optional[npt.NDArray[np.float64]] = None,
    conf_id: Optional[int] = None,
) -> int:
    """Add a conformer to the current `RDKitMol`.

    Args:
        mol (Chem.RWMol): The molecule to be set.
        conf (Optional[Chem.Conformer], optional): A user may provide a predefined ``conformer`` and add it to the ``mol``. Defaults to ``None``,
            but either ``conf`` or ``coords`` must be provided.
        coords (Optional[npt.NDArray[np.float64]], optional): Instead of feeding a conformer, a user may provide the coordinates of the conformer to be added.
            Defaults to ``None``, but either ``conf`` or ``coords`` must be provided.
        conf_id (Optional[int], optional): Which ID to set for the conformer (will be added as the last conformer by default). Defaults to ``None``.

    Returns:
        int: The conformer ID added.

    Raises:
        ValueError: If both ``conf`` and ``coords`` are ``None``.
    """
    assign_id = True if conf_id is None else False
    if conf is None:
        if coords is not None:
            conf = create_conformer(coords, conf_id)
        else:
            raise ValueError("conf and coords cannot be both None.")
    elif conf_id is not None:
        conf.SetId(conf_id)
    return mol.AddConformer(conf, assign_id)


def add_null_conformer(
    mol: Chem.RWMol,
    conf_id: Optional[int] = None,
    random: bool = True,
) -> int:
    """Embed a conformer with null atom coordinates.

    The coordinates are either set to random numbers or all zero coordinates.

    Args:
        mol (Chem.RWMol): The molecule to be set.
        conf_id (Optional[int], optional): Which ID to set for the conformer (will be added as the last conformer by default).
        random (bool, optional): Whether set coordinates to random numbers. Otherwise, set to all-zero
            coordinates. Defaults to ``True``.

    Returns:
        int: The conformer ID added.
    """
    num_atoms = mol.GetNumAtoms()
    # Constructing a conformer with number of atoms will create a null conformer
    conf = Chem.Conformer(num_atoms)
    if random:
        set_conformer_coordinates(conf, np.random.rand(num_atoms, 3))
    return add_conformer(mol, conf=conf, conf_id=conf_id)


def create_conformer(
    coords: npt.NDArray[np.float64],
    conf_id: Optional[int] = None,
) -> Chem.Conformer:
    """Create a conformer with the given coordinates.

    Args:
        coords (npt.NDArray[np.float64]): The coordinates to be set.
        conf_id (Optional[int], optional): Which ID to set for the conformer. Defaults to ``0``.

    Returns:
        Chem.Conformer: The conformer created.
    """
    conf = Chem.Conformer()
    set_conformer_coordinates(conf, coords)
    if conf_id is not None:
        conf.SetId(conf_id)
    return conf


def embed_multiple_null_confs(
    mol: Chem.Mol,
    n: int = 10,
    random: bool = False,
) -> None:
    """Embed conformers with null or random atom coordinates.

    This helps the cases where
    a conformer can not be successfully embedded. You can choose to generate all zero
    coordinates or random coordinates. You can set to all-zero coordinates, if you will
    set coordinates later; You should set to random coordinates, if you want to optimize
    this molecule by force fields (RDKit force field cannot optimize all-zero
    coordinates).

    Args:
        mol (Chem.Mol): The molecule to be set.
        n (int, optional): The number of conformers to be embedded. Defaults to ``10``.
        random (bool, optional): Whether set coordinates to random numbers. Otherwise, set to all-zero
            coordinates. Defaults to ``True``.
    """
    mol.RemoveAllConformers()
    for i in range(n):
        add_null_conformer(mol, conf_id=i, random=random)


def reflect(
    conf: Chem.Conformer,
) -> None:
    """Reflect the coordinates of the conformer.

    Args:
        conf (Chem.Conformer): The conformer to be set.
    """
    coords = conf.GetPositions()
    coords[:, 0] *= -1
    set_conformer_coordinates(conf, coords)


def set_conformer_coordinates(
    conf: Chem.Conformer,
    coords: (
        tuple[tuple[float, float, float], ...]
        | list[list[float]]
        | npt.NDArray[np.float64]
    ),
) -> None:
    """Set the Positions of atoms of the conformer.

    Args:
        conf (Chem.Conformer): The conformer to be set.
        coords (tuple[tuple[float, float, float], ...] | list[list[float]] | npt.NDArray[np.float64]): The coordinates to be set.
    """
    if isinstance(coords, (tuple, list)):
        coords = np.array(coords)

    for i in range(len(coords)):
        conf.SetAtomPosition(i, coords[i, :])


def embed_conformer(
    mol: Chem.Mol,
    allow_null: bool = True,
    **kwargs: Any,
) -> None:
    """Embed a conformer to the molecule object.

    This will overwrite current conformers.
    By default, it will first try embedding a 3D conformer; if fails, it then try to
    compute 2D coordinates and use that for the conformer structure; if both approaches
    fail, and embedding a null conformer is allowed, a conformer with all random
    coordinates will be embedded. The last one is helpful for the case where you can use
    `SetPositions` to set their positions afterward, or if you want to optimize the
    geometry using force fields.

    Args:
        mol (Chem.Mol): The molecule object to embed.
        allow_null (bool, optional): If embedding 3D and 2D coordinates fails, whether to embed a conformer
            with all null coordinates, ``(0, 0, 0)``, for each atom. Defaults to ``True``.
        **kwargs (Any): The keyword arguments to pass to the embedding function.

    Raises:
        RuntimeError: If embedding fails and ``allow_null`` is ``False``.
    """
    try:
        return_code = AllChem.EmbedMolecule(mol, **kwargs)  # type: ignore[attr-defined]
    except Chem.AtomValenceException:
        try:
            AllChem.Compute2DCoords(mol)  # type: ignore[attr-defined]
            return_code = 0
        except BaseException:  # To be replaced by a more specific error type
            return_code = -1

    if return_code == -1:
        if allow_null:
            embed_multiple_null_confs(mol, n=1, random=True)
        else:
            raise RuntimeError("Cannot embed conformer for this molecule.")


def embed_multiple_confs(
    mol: Chem.Mol,
    n: int = 10,
    allow_null: bool = True,
    **kwargs: Any,
) -> None:
    """Embed multiple conformers to the ``RDKitMol``.

    This will overwrite current
    conformers. By default, it will first try embedding a 3D conformer; if fails, it
    then try to compute 2D coordinates and use that for the conformer structure; if both
    approaches fail, and embedding a null conformer is allowed, a conformer with all
    random coordinates will be embedded. The last one is helpful for the case where you
    can use `SetPositions` to set their positions afterward, or if you want to optimize
    the geometry using force fields.

    Args:
        mol (Chem.Mol): The molecule object to embed.
        n (int, optional): The number of conformers to be embedded. The default is ``1``.
        allow_null (bool, optional): If embedding fails, whether to embed null conformers. Defaults to ``True``.
        **kwargs (Any): The keyword arguments to pass to the embedding function.

    Raises:
        RuntimeError: If embedding fails and ``allow_null`` is ``False``.
    """
    try:
        AllChem.EmbedMultipleConfs(mol, numConfs=n, **kwargs)  # type: ignore[attr-defined]
    except Chem.AtomValenceException:
        if allow_null:
            embed_multiple_null_confs(mol, n=n, random=True)
        else:
            raise RuntimeError("Cannot embed conformer for this molecule!")


def edit_conf_by_add_bonds(
    conf: Chem.Conformer,
    function_name: str,
    atoms: tuple[int, ...],
    value: int | float,
) -> None:
    """Add bonds to the conformer and modify the internal coordinates.

    This is a workaround for RDKit forbidding modifying internal coordinates with
    non-bonding atoms. The function will add the bonds, modify the internal coordinates,
    and then remove the bonds. The conformer will be modified in place.
    Note, the function will not work for ring systems.

    Args:
        conf (Chem.Conformer): The conformer to be modified.
        function_name (str): The function name of the edit, should be a method provided in rdMolTransforms.
        atoms (tuple[int, ...]): A list of atoms representing the internal coordinates.
        value (int | float): Value to be set.
    """
    parent_mol = conf.GetOwningMol()
    all_bonds = get_bonds_as_tuples(parent_mol)
    bonds_to_add = [
        (atoms[i], atoms[i + 1])
        for i in range(len(atoms) - 1)
        if (atoms[i], atoms[i + 1]) not in all_bonds
    ]
    tmp_mol = add_bonds(parent_mol, bonds_to_add, inplace=False)

    add_conformer(tmp_mol, conf.GetPositions(), conf_id=0)
    tmp_conf = tmp_mol.GetConformer()
    getattr(rdMT, function_name)(tmp_conf, *atoms, value)

    set_conformer_coordinates(conf, tmp_conf.GetPositions())


def get_bond_length(
    conf: Chem.Conformer,
    atom_ids: tuple[int, int],
) -> float:
    """Get the distance between two atoms.

    Note, the two atoms can be non-bonded.

    Args:
        conf (Chem. Conformer): The conformer to be set.
        atom_ids (tuple[int, int]): The atom IDs to be set.

    Returns:
        float: The bond length between two atoms.
    """
    return rdMT.GetBondLength(conf, *atom_ids)


def get_angle_deg(
    conf: Chem.Conformer,
    atom_ids: tuple[int, int, int],
) -> float:
    """Get the angle between three atoms in degrees.

    Args:
        conf (Chem.Conformer): The conformer to be set.
        atom_ids (tuple[int, int, int]): The atom IDs to be set.

    Returns:
        float: The angle between three atoms in degrees.
    """
    return rdMT.GetAngleDeg(conf, *atom_ids)


def get_torsion_deg(
    conf: Chem.Conformer,
    atom_ids: tuple[int, int, int, int],
) -> float:
    """Get the torsion angle between four atoms in degrees.

    Args:
        conf (Chem.Conformer): The conformer to be set.
        atom_ids (tuple[int, int, int, int]): The atom IDs to be set.

    Returns:
        float: The torsion angle between four atoms in degrees.
    """
    return rdMT.GetDihedralDeg(conf, *atom_ids)


def set_bond_length(
    conf: Chem.Conformer,
    atom_ids: tuple[int, int],
    value: int | float,
) -> None:
    """Set the distance between two atoms.

    If the two atoms are not bonded, the function
    will give it a try by forming the bonds, adjusting the bond length, and then
    removing the bonds. This is a workaround for RDKit forbidding modifying internal
    coordinates with non-bonding atoms.

    Args:
        conf (Chem.Conformer): The conformer to be set.
        atom_ids (tuple[int, int]): The atom IDs to be set.
        value (int | float): The value to be set.

    Raises:
        NotImplementedError: If the approach for modifying the bond length is not available.
    """
    try:
        rdMT.SetBondLength(conf, *atom_ids, value)
    except ValueError:
        try:
            edit_conf_by_add_bonds(conf, "SetBondLength", atom_ids, value)
        except ValueError:
            # RDKit doesn't allow change bonds for atoms in a ring
            # A workaround hasn't been proposed
            raise NotImplementedError(
                f"Approach for modifying the bond length of {atom_ids} is not available."
            )


def set_angle_deg(
    conf: Chem.Conformer,
    atom_ids: tuple[int, int, int],
    value: int | float,
) -> None:
    """Set the angle between three atoms in degrees.

    If the three atoms are not bonded,
    the function will give it a try by forming the bonds, adjusting the angle, and then
    removing the bonds. This is a workaround for RDKit forbidding modifying internal
    coordinates with non-bonding atoms.

    Args:
        conf (Chem.Conformer): The conformer to be set.
        atom_ids (tuple[int, int, int]): The atom IDs to be set.
        value (int | float): The value to be set.

    Raises:
        NotImplementedError: If the approach for modifying the angle is not available.
    """
    try:
        rdMT.SetAngleDeg(conf, *atom_ids, value)
    except ValueError:
        try:
            edit_conf_by_add_bonds(conf, "SetAngleDeg", atom_ids, value)
        except ValueError:
            # RDKit doesn't allow change bonds for atoms in a ring
            # A workaround hasn't been proposed
            raise NotImplementedError(
                f"Approach for modifying the angle of {atom_ids} is not available."
            )


def set_torsion_deg(
    conf: Chem.Conformer,
    atom_ids: tuple[int, int, int, int],
    value: int | float,
) -> None:
    """Set the torsion angle between four atoms in degrees.

    If the four atoms are not
    bonded, the function will give it a try by forming the bonds, adjusting the torsion,
    and then removing the bonds. This is a workaround for RDKit forbidding modifying
    internal coordinates with non-bonding atoms.

    Args:
        conf (Chem.Conformer): The conformer to be set.
        atom_ids (tuple[int, int, int, int]): The atom IDs to be set.
        value (int | float): The value to be set.

    Raises:
        NotImplementedError: If the approach for modifying the torsion is not available.
    """
    try:
        rdMT.SetDihedralDeg(conf, *atom_ids, value)
    except ValueError:
        try:
            edit_conf_by_add_bonds(conf, "SetDihedralDeg", atom_ids, value)
        except ValueError:
            # RDKit doesn't allow change bonds for atoms in a ring
            # A workaround hasn't been proposed
            raise NotImplementedError(
                f"Approach for modifying the torsion of {atom_ids} is not available."
            )
