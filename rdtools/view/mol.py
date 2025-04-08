"""Module for viewing RDKit Molecules using py3Dmol."""

import logging
from typing import Any

from rdkit import Chem

from rdtools.view.base import animation_viewer, base_viewer, py3Dmol
from rdtools.view.utils import clean_ts

logger = logging.getLogger(__name__)


def mol_viewer(
    mol: Chem.Mol,
    conf_id: int = 0,
    **kwargs: Any,
) -> py3Dmol.view:
    """Create a viewer for viewing the RDKit Molecule.

    This viewer accepts additional keyword arguments for viewer specs,
    following the same way base_viewer is implemented.

    Args:
        mol (Chem.Mol): The RDKit Molecule.
        conf_id (int, optional): The ID of the conformer to view.
        **kwargs (Any): Additional keyword arguments to be passed to the viewer.

    Returns:
        py3Dmol.view: The viewer.
    """
    try:
        obj, model = Chem.MolToMolBlock(mol, confId=conf_id), "sdf"
    except BaseException as e:
        logger.debug(
            f"Failed to convert mol to MolBlock when using mol_viewer. Got: {e}"
        )
        obj, model = Chem.MolToXYZBlock(mol, confId=conf_id), "xyz"

    return base_viewer(obj, model, **kwargs)


def mol_animation(
    mols: list[Chem.Mol],
    conf_ids: list[int] | None = None,
    interval: int = 1000,  # more proper to look at different molecules
    **kwargs: Any,
) -> py3Dmol.view:
    """Create an animation viewer for viewing the RDKit Molecules.

    This viewer accepts additional keyword arguments for viewer specs, following the same way
    animation_viewer is implemented.

    Args:
        mols (list[Chem.Mol]): A list of RDKit Molecules, each assumes to contain at least one conformer.
        conf_ids (list[int] | None, optional): A list of IDs of the conformers to view. If ``None``, the first conformer of
            each molecule will be displayed. Defaults to None.
        interval (int, optional): The time interval between each frame in millisecond. Defaults to ``1000``.
        **kwargs (Any): Additional keyword arguments to be passed to the viewer.

    Returns:
        py3Dmol.view: The viewer.
    """
    conf_ids = conf_ids or [0] * len(mols)

    assert len(mols) == len(conf_ids), (
        f"The number of molecules ({len(mols)}) and the "
        f"number of conformers ({len(conf_ids)} must be the same."
    )

    xyz = "".join([Chem.MolToXYZBlock(mol, confId=i) for mol, i in zip(mols, conf_ids)])

    return animation_viewer(xyz, "xyz", interval=interval, **kwargs)


def ts_viewer(
    mol: Chem.Mol,
    broken_bonds: list[tuple[int, int]] = [],
    formed_bonds: list[tuple[int, int]] = [],
    broken_bond_color: str = "red",
    formed_bond_color: str = "green",
    broken_bond_width: float = 0.1,
    formed_bond_width: float = 0.1,
    **kwargs: Any,
) -> py3Dmol.view:
    """Create a viewer for transition state.

    Args:
        mol (Chem.Mol): The RDKit Molecule.
        broken_bonds (list[tuple[int, int]], optional): The broken bonds. Defaults to [].
        formed_bonds (list[tuple[int, int]], optional): The formed bonds. Defaults to [].
        broken_bond_color (str, optional): The color of the broken bond. Defaults to "red".
        formed_bond_color (str, optional): The color of the formed bond. Defaults to "green".
        broken_bond_width (float, optional): The width of the broken bond. Defaults to 0.1.
        formed_bond_width (float, optional): The width of the formed bond. Defaults to 0.1.
        **kwargs (Any): Additional keyword arguments to be passed to the viewer.

    Returns:
        py3Dmol.view: The viewer.
    """
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
            **{"viewer": kwargs.get("viewer_loc")},
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
