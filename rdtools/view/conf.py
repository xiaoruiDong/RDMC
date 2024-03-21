from typing import Optional

from rdkit import Chem
from rdtools.view.base import base_viewer, default_style_spec, animation_viewer


def conformer_viewer(
    mol: "Mol",
    conf_ids: Optional[list] = None,
    highlight_ids: Optional[list] = None,
    conf_opacity: float = 0.5,
    **kwargs,
) -> "py3Dmol.view":
    """
    Create a viewer for viewing multiple overlaid conformers. This viewer
    accepts additional keyword arguments following the same way base_viewer is implemented.
    It is recommended aligning the conformers before viewing the structures for a better
    visualization.

    Args:
        mol (RDKitMol): An RDKitMol object with embedded conformers.
        conf_ids (list, optional): A list of conformer ids (as ``int``) to be overlaid and viewed.
                                   If not provided, all embedded conformers will be used.
        highlight_ids (list, optional): It is possible to highlight some of the conformers while greying out
                                        other conformers by providing the conformer IDs you want to highlight.
        conf_opacity (float, optional): Set the opacity of the non-highlighted conformers and is only used with the highlighting feature.
                                        the value should be a ``float`` between ``0.0`` to ``1.0``. The default value is ``0.5``.
                                        Values below ``0.3`` may be hard to see.

    Returns:
        py3Dmol.view: The conformer viewer.
    """
    if not conf_ids:
        conf_ids = list(range(mol.GetNumConformers()))

    obj = [Chem.MolToMolBlock(mol, confId=i) for i in conf_ids]
    model = "sdf"

    # Turn off atom_index, unless user explicitly set it
    atom_index = kwargs.pop("atom_index") if "atom_index" in kwargs else False

    viewer = base_viewer(obj, model, atom_index=atom_index, **kwargs)

    if highlight_ids is None:  # No need extra setting
        return viewer

    # to highlight a few conformers, we need to set styles in two steps
    # step 1 to set the highlighted ones and step 2 to set the non-highlighted ones
    # let's do step 1 first here
    style_spec = kwargs.get("style_spec") or default_style_spec
    highlight_seq = [conf_ids.index(i) for i in highlight_ids]
    viewer.setStyle(
        {"model": highlight_seq}, style_spec, viewer=kwargs.get("viewer_loc")
    )
    viewer.setStyle(
        {"model": [i for i in range(len(conf_ids)) if i not in highlight_seq]},
        {
            key: {**value, **{"opacity": conf_opacity}}
            for key, value in style_spec.items()
        },
        viewer=kwargs.get("viewer_loc"),
    )
    return viewer


def conformer_animation(
    mol: "Mol",
    conf_ids: Optional[list] = None,
    **kwargs,
) -> "py3Dmol.view":
    """
    Create an animation viewer for viewing the RDKit Molecules. This
    viewer accepts additional keyword arguments for viewer specs,
    following the same way base_viewer is implemented.

    Args:
        mols (list): A list of RDKit Molecules, each assumes to contain at least one conformer.
        conf_ids (list): A list of IDs of the conformers to view. If None, all conformers will be shown.
                         Default to None.
        **kwargs

    Returns:
        py3Dmol.view: The viewer.
    """
    conf_ids = conf_ids or list(range(mol.GetNumConformers()))

    xyz = "".join([Chem.MolToXYZBlock(mol, confId=i) for i in conf_ids])

    return animation_viewer(xyz, "xyz", **kwargs)
