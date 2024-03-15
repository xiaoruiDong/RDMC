try:
    import ipywidgets
except ImportError:
    ipywidgets = None


from rdmc.rdtools.view.mol import mol_viewer


def interactive_conformer_viewer(mol: "Mol", **kwargs):
    """
    This is a viewer for individually viewing multiple conformers using an ipython slider widget.

    Args:
        mol (RDKitMol): An RDKitMol object with embedded conformers.

    Returns:
        py3Dmol.view: The molecule viewer with slider to view different conformers.
    """
    if not ipywidgets:
        raise ImportError(
            "This function requires ipywidgets to be installed. You can install it by pip or conda"
        )

    if isinstance(mol, (list, tuple)):

        def viewer(conf_id):
            return mol_viewer(mol[conf_id], conf_id=0, **kwargs)

        return ipywidgets.interact(
            viewer,
            conf_id=ipywidgets.IntSlider(min=0, max=len(mol) - 1,step=1),
        )

    else:

        def viewer(conf_id):
            return mol_viewer(mol, conf_id=conf_id, **kwargs)

        return ipywidgets.interact(
            viewer,
            conf_id=ipywidgets.IntSlider(min=0, max=mol.GetNumConformers() - 1, step=1),
        )
