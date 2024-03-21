from typing import List
import logging

from rdkit import Chem
from rdmc.rdtools.view.base import base_viewer, animation_viewer
from rdmc.rdtools.view.utils import clean_ts

logger = logging.getLogger(__name__)


def mol_viewer(
    mol: 'Mol',
    conf_id: int = 0,
    **kwargs,
) -> 'py3Dmol.view':
    """
    Create a viewer for viewing the RDKit Molecule. This viewer
    accepts additional keyword arguments for viewer specs,
    following the same way base_viewer is implemented.

    Args:
        mol (Mol): The RDKit Molecule.
        conf_id (int): The ID of the conformer to view.

    Returns:
        py3Dmol.view: The viewer.
    """
    try:
        obj, model = Chem.MolToMolBlock(mol, confId=conf_id), 'sdf'
    except BaseException as e:
        logger.debug(f"Failed to convert mol to MolBlock when using mol_viewer. Got: {e}")
        obj, model = Chem.MolToXYZBlock(mol, confId=conf_id), 'xyz'

    return base_viewer(obj, model, **kwargs)


def mol_animation(
    mols: List["Mol"],
    conf_ids: List[int] = None,
    interval: int = 1000,  # more proper to look at different molecules
    **kwargs,
) -> "py3Dmol.view":
    """
    Create an animation viewer for viewing the RDKit Molecules. This
    viewer accepts additional keyword arguments for viewer specs,
    following the same way animation_viewer is implemented.

    Args:
        mols (list): A list of RDKit Molecules, each assumes to contain at least one conformer.
        conf_ids (list, optional): A list of IDs of the conformers to view. If ``None``, the first conformer of
                         each molecule will be displayed. Defaults to None.
        interval (int, optional): The time interval between each frame in millisecond. Defaults to ``1000``.

    Returns:
        py3Dmol.view: The viewer.
    """
    conf_ids = conf_ids or [0] * len(mols)

    assert len(mols) == len(conf_ids), \
        f"The number of molecules ({len(mols)}) and the " \
        f"number of conformers ({len(conf_ids)} must be the same."

    xyz = "".join([Chem.MolToXYZBlock(mol, confId=i) for mol, i in zip(mols, conf_ids)])

    return animation_viewer(xyz, "xyz", interval=interval, **kwargs)


def ts_viewer(
    mol: "Mol",
    broken_bonds: list = [],
    formed_bonds: list = [],
    broken_bond_color: str = "red",
    formed_bond_color: str = "green",
    broken_bond_width: float = 0.1,
    formed_bond_width: float = 0.1,
    **kwargs,
):
    mol = clean_ts(mol, broken_bonds, formed_bonds)

    viewer = mol_viewer(mol, **kwargs)

    coords = mol.GetConformer(id=kwargs.get("conf_id", 0)).GetPositions()
    for bond in broken_bonds:
        start, end = coords[bond, :]
        viewer.addCylinder(
            {
                "start": dict(x=start[0], y=start[1], z=start[2]),
                "end": dict(x=end[0], y=end[1], z=end[2]),
                "color": broken_bond_color,
                "radius": broken_bond_width,
                "dashed": True,
            },
            **{"viewer": kwargs.get('viewer_loc')},
        )
    for bond in formed_bonds:
        start, end = coords[bond, :]
        viewer.addCylinder(
            {
                "start": dict(x=start[0], y=start[1], z=start[2]),
                "end": dict(x=end[0], y=end[1], z=end[2]),
                "color": formed_bond_color,
                "radius": formed_bond_width,
                "dashed": True,
            },
            **{"viewer": kwargs.get("viewer_loc")},
        )
    return viewer
