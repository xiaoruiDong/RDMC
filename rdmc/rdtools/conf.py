from typing import Optional, Union

import numpy as np

from rdkit import Chem


def add_conformer(
    mol: Chem.RWMol,
    conf: Optional[Chem.Conformer] = None,
    coords: Optional[np.ndarray] = None,
    conf_id: Optional[int] = None,
):
    """
    Add a conformer to the current `RDKitMol`.

    Args:
        mol (Chem.RWMol): The molecule to be set.
        conf (Conformer, optional): A user may provide a predefined ``conformer`` and add it to the ``mol``. Defaults to ``None``,
                                    but either ``conf`` or ``coords`` must be provided.
        coords (np.ndarray, optional): Instead of feeding a conformer, a user may provide the coordinates of the conformer to be added.
                                       Defaults to ``None``, but either ``conf`` or ``coords`` must be provided.
        conf_id (int, optional): Which ID to set for the conformer (will be added as the last conformer by default). Defaults to ``None``.
    """
    assign_id = True if conf_id is None else False
    if conf is None:
        try:
            conf = create_conformer(coords, conf_id)
        except IndexError:  # Pass coords as None causing IndexError
            raise ValueError("conf and coords cannot be both None.")
    elif conf_id is not None:
        conf.SetId(conf_id)
    mol.AddConformer(conf, assign_id)


def add_null_conformer(
    mol: Chem.RWMol,
    conf_id: Optional[int] = None,
    random: bool = True,
):
    """
    Embed a conformer with atoms' coordinates of random numbers or with all atoms
    located at the origin to the current `RDKitMol`.

    Args:
        mol (Chem.RWMol): The molecule to be set.
        conf_id (int, optional): Which ID to set for the conformer (will be added as the last conformer by default).
    """
    num_atoms = mol.GetNumAtoms()
    # Constructing a conformer with number of atoms will create a null conformer
    conf = Chem.Conformer(num_atoms)
    if random:
        set_conformer_coordinates(conf, np.random.rand(num_atoms, 3))
    add_conformer(mol, conf=conf, conf_id=conf_id)


def create_conformer(
    coords: np.ndarray,
    conf_id: Optional[int] = None,
):
    """
    Create a conformer with the given coordinates.

    Args:
        coords (np.ndarray): The coordinates to be set.
        conf_id (int, optional): Which ID to set for the conformer. Defaults to ``0``.

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
):
    """
    Embed conformers with null or random atom coordinates. This helps the cases where a conformer
    can not be successfully embedded. You can choose to generate all zero coordinates or random coordinates.
    You can set to all-zero coordinates, if you will set coordinates later; You should set to random
    coordinates, if you want to optimize this molecule by force fields (RDKit force field cannot optimize
    all-zero coordinates).

    Args:
        mol (Chem.Mol): The molecule to be set.
        n (int): The number of conformers to be embedded. Defaults to ``10``.
        random (bool, optional): Whether set coordinates to random numbers. Otherwise, set to all-zero
                                 coordinates. Defaults to ``True``.
    """
    mol.RemoveAllConformers()
    for i in range(n):
        add_null_conformer(mol, conf_id=i, random=random)


def reflect(
    conf: Chem.Conformer,
):
    """
    Reflect the coordinates of the conformer.

    Args:
        conf (Chem.Conformer): The conformer to be set.
    """
    coords = conf.GetPositions()
    coords[:, 0] *= -1
    set_conformer_coordinates(conf, coords)


def set_conformer_coordinates(
    conf: Chem.Conformer, coords: Union[tuple, list, np.ndarray]
):
    """
    Set the Positions of atoms of the conformer.

    Args:
        conf (Union[Conformer, 'RDKitConf']): The conformer to be set.
        coords (Union[tuple, list, np.ndarray]): The coordinates to be set.

    Raises:
        ValueError: Not a valid ``coords`` input, when giving something else.
    """
    for i in range(len(coords)):
        conf.SetAtomPosition(i, coords[i, :])
