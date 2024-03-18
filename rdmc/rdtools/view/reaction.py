from typing import Optional, Tuple, Union

from rdmc.rdtools.view.base import grid_viewer
from rdmc.rdtools.view.mol import mol_viewer, ts_viewer
from rdmc.rdtools.view.utils import get_broken_formed_bonds


_ts_viewer_keys = [
    "broken_bonds", "formed_bonds",
    "broken_bond_color", "formed_bond_color",
    "broken_bond_width", "formed_bond_width"
]


def reaction_viewer(
    r_mol: "Mol",
    p_mol: "Mol",
    ts_mol: Optional["Mol"] = None,
    alignment: str = 'horizontal',
    **kwargs
) -> "py3Dmol.view":
    """
    View reactant, product and ts of a given reaction. The broken bonds in the TS will be shown with red
    lines while the formed bonds in the TS will be shown with green lines. Uses keywords from mol_viewer
    function, so all arguments of that function are available here.

    Args:
        r_mol (Mol): The reactant complex.
        p_mol (Mol): The product complex.
        ts_mol (Mol, optional): The ts corresponding to r_mol and p_mol. It will be placed in between.
        alignment (list, optional): Indicate if geometries are displayed horizontally (``horizontal``)
                                    or vertically (``vertical``). Defaults to ``horizontal``.
        alignment_direction (str optional): Reactant, product, and TS are vertically aligned. Defaults to `True`.
                                            Only valid when `only_ts` is `False`.
    """
    # Check if TS is provided
    grid = ['r', 'p'] if ts_mol is None else ['r', 'ts', 'p'] 

    ts_kwargs = {}
    # Remove the ts_viewer's key from kwargs
    for key in _ts_viewer_keys:
        if key in kwargs:
            ts_kwargs[key] = kwargs.pop(key)

    # Set up grid viewer
    if "viewer" in kwargs:
        viewer = kwargs["viewer"]
    else:
        # Set up a grid viewer if not provided
        grid_arguments = {}
        grid_arguments["viewer_grid"] = (1, len(grid)) if alignment == 'horizontal' else (len(grid), 1)
        if "viewer_size" in kwargs:
            grid_arguments["viewer_size"] = (
                grid_arguments["viewer_grid"][0] * kwargs["viewer_size"][0],
                grid_arguments["viewer_grid"][1] * kwargs["viewer_size"][1],
            )  # Find out the correct viewer size for the whole grid
        grid_arguments["linked"] = kwargs.pop("linked") if "linked" in kwargs else True
        viewer = grid_viewer(**grid_arguments)

    # Clean up TS by removing formed and broken bonds
    ts_kwargs['broken_bonds'], ts_kwargs['formed_bonds'] = get_broken_formed_bonds(r_mol, p_mol)

    mols = {"r": r_mol, "p": p_mol, "ts": ts_mol}
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
