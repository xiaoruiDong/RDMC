"""Interactive viewer for conformers using ipywidgets.

This viewer requires ipywidgets to be installed and is designed to work in Jupyter
notebooks.
"""

from typing import Any

import py3Dmol
from rdkit import Chem

from rdtools.view.mol import mol_viewer

try:
    import ipywidgets
    from ipywidgets.widgets.interaction import interactive
except ImportError:
    ipywidgets = None
    interactive = Any


def interactive_conformer_viewer(mol: Chem.Mol, **kwargs: Any) -> interactive:
    """View individual conformers using an ipython slider widget.

    Args:
        mol (Chem.Mol): An Mol object with embedded conformers.
        **kwargs (Any): Additional keyword arguments to be passed to the viewer.

    Returns:
        interactive: The molecule viewer with slider to view different conformers.

    Raises:
        ImportError: If ipywidgets is not installed.
    """
    if not ipywidgets:
        raise ImportError(
            "This function requires ipywidgets to be installed. You can install it by pip or conda"
        )

    if isinstance(mol, (list, tuple)):

        def viewer(conf_id: int) -> py3Dmol.view:
            return mol_viewer(mol[conf_id], conf_id=0, **kwargs)

        return ipywidgets.interact(
            viewer,
            conf_id=ipywidgets.IntSlider(
                value=0, min=0, max=len(mol) - 1, step=1, description="Molecule ID:"
            ),
        )

    else:

        def viewer(conf_id: int) -> py3Dmol.view:
            return mol_viewer(mol, conf_id=conf_id, **kwargs)

        return ipywidgets.interact(
            viewer,
            conf_id=ipywidgets.IntSlider(
                value=0,
                min=0,
                max=mol.GetNumConformers() - 1,
                step=1,
                description="Conformer ID:",
            ),
        )
