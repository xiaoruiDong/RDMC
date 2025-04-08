"""Module for visualizing chemical reactions using RDKit and Py3Dmol."""

from typing import Any, Literal, Optional

from rdkit import Chem

from rdtools.view.base import grid_viewer, py3Dmol
from rdtools.view.mol import mol_viewer, ts_viewer
from rdtools.view.utils import get_broken_formed_bonds

_ts_viewer_keys = [
    "broken_bonds",
    "formed_bonds",
    "broken_bond_color",
    "formed_bond_color",
    "broken_bond_width",
    "formed_bond_width",
]


def reaction_viewer(
    r_mol: Chem.Mol,
    p_mol: Chem.Mol,
    ts_mol: Optional[Chem.Mol] = None,
    alignment: Literal["horizontal", "vertical"] = "horizontal",
    **kwargs: Any,
) -> py3Dmol.view:
    """View reactant, product and ts of a given reaction.

    The broken bonds in the TS will be shown with red
    lines while the formed bonds in the TS will be shown with green lines. Uses keywords from mol_viewer
    function, so all arguments of that function are available here.

    Args:
        r_mol (Chem.Mol): The reactant complex.
        p_mol (Chem.Mol): The product complex.
        ts_mol (Optional[Chem.Mol], optional): The TS corresponding to r_mol and p_mol. It will be placed in between.
        alignment (Literal["horizontal", "vertical"], optional): Indicate if geometries are displayed horizontally (``horizontal``)
            or vertically (``vertical``). Defaults to ``horizontal``.
        **kwargs (Any): Additional keyword arguments to be passed to the viewer.

    Returns:
        py3Dmol.view: The viewer.
    """
    # Check if TS is provided
    grid = ["r", "p"] if ts_mol is None else ["r", "ts", "p"]
    mols = {"r": r_mol, "p": p_mol}
    if ts_mol is not None:
        mols["ts"] = ts_mol

    ts_kwargs = {}
    # Remove the ts_viewer's key from kwargs
    for key in _ts_viewer_keys:
        if key in kwargs:
            ts_kwargs[key] = kwargs.pop(key)

    # Set up grid viewer
    if "viewer" in kwargs:
        viewer = kwargs["viewer"]
    else:
        viewer_grid = (1, len(grid)) if alignment == "horizontal" else (len(grid), 1)
        if "viewer_size" in kwargs:
            indivial_viewer_size: tuple[int, int] = kwargs["viewer_size"]
            viewer_size = (
                viewer_grid[0] * indivial_viewer_size[0],
                viewer_grid[1] * indivial_viewer_size[1],
            )  # Find out the correct viewer size for the whole grid
        if "linked" in kwargs:
            linked: bool = kwargs.pop("linked")
        else:
            linked = True
        viewer = grid_viewer(viewer_grid, linked, viewer_size)

    # Clean up TS by removing formed and broken bonds
    ts_kwargs["broken_bonds"], ts_kwargs["formed_bonds"] = get_broken_formed_bonds(
        r_mol, p_mol
    )

    for i, label in enumerate(grid):
        mol = mols[label]
        viewer_loc = (0, i) if alignment == "horizontal" else (i, 0)
        plot_kwargs = {**kwargs, **{"viewer": viewer, "viewer_loc": viewer_loc}}
        if label == "ts":
            ts_viewer(
                mol,
                **ts_kwargs,
                **plot_kwargs,
            )
        else:
            mol_viewer(mol, **plot_kwargs)

        viewer.zoomTo(viewer=viewer_loc)

    return viewer
