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


def embed_conformer(
    mol: Chem.Mol,
    allow_null: bool = True,
    **kwargs,
):
    """
    Embed a conformer to the molecule object. This will overwrite current conformers. By default, it
    will first try embedding a 3D conformer; if fails, it then try to compute 2D coordinates
    and use that for the conformer structure; if both approaches fail, and embedding a null
    conformer is allowed, a conformer with all random coordinates will be embedded. The last one is
    helpful for the case where you can use `SetPositions` to set their positions afterward, or if you want to
    optimize the geometry using force fields.

    Args:
        mol (Mol): The molecule object to embed.
        allow_null (bool, optional): If embedding 3D and 2D coordinates fails, whether to embed a conformer
                                     with all null coordinates, ``(0, 0, 0)``, for each atom. Defaults to ``True``.
    """
    try:
        return_code = Chem.AllChem.EmbedMolecule(mol, **kwargs)
    except Chem.AtomValenceException:
        try:
            Chem.AllChem.Compute2DCoords(mol)
            return_code = 0
        except BaseException:  # To be replaced by a more specific error type
            return_code = -1

    if return_code == -1:
        if allow_null:
            embed_multiple_null_confs(mol, n=1, random=True)
        else:
            raise RuntimeError("Cannot embed conformer for this molecule.")


def embed_multiple_confs(mol: Chem.Mol, n: int = 10, allow_null: bool = True, **kwargs):
    """
    Embed multiple conformers to the ``RDKitMol``. This will overwrite current conformers. By default, it
    will first try embedding a 3D conformer; if fails, it then try to compute 2D coordinates
    and use that for the conformer structure; if both approaches fail, and embedding a null
    conformer is allowed, a conformer with all random coordinates will be embedded. The last one is
    helpful for the case where you can use `SetPositions` to set their positions afterward, or if you want to
    optimize the geometry using force fields.

    Args:
        n (int): The number of conformers to be embedded. The default is ``1``.
        allow_null (bool): If embedding fails, whether to embed null conformers. Defaults to ``True``.
    """
    try:
        Chem.AllChem.EmbedMultipleConfs(mol, numConfs=n, **kwargs)
    except Chem.AtomValenceException:
        if allow_null:
            embed_multiple_null_confs(mol, n=n, random=True)
        else:
            raise RuntimeError("Cannot embed conformer for this molecule!")
