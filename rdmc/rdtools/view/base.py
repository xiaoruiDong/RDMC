from typing import Optional

try:
    import py3Dmol
except ImportError:
    py3DMol = None


default_style_spec = {
    'stick': {'radius': 0.05, 'color': '#f2f2f2'},
    'sphere': {'scale': 0.25},
}

def base_viewer(
    obj: str,
    model: str = 'xyz',
    model_extra: Optional[dict] = None,
    animate: Optional[dict] = None,
    atom_index: bool = True,
    style_spec: Optional[dict] = None,
    viewer: Optional['py3Dmol.view'] = None,
    viewer_size: tuple = (400, 400),
    viewer_loc: Optional[tuple] = None,
) -> 'py3Dmol.view':
    """
    This is the most general function to view a molecule powered by py3Dmol and its backend 3Dmol.js.
    This allows you to visualize molecules in 3D with a javascript object or in IPython notebooks. This
    function is also used to build up more complicated viewers, e.g., freq_viewer.

    Args:
        obj (str): A string representation of the molecule can be xyz string,
                   sdf string, etc.
        model (str, optional): The format of the molecule representation, e.g., ``'xyz'``.
                               Defaults to ``'xyz'``.
        model_extra (dict, optional): Extra specs for the model (format). E.g., frequency specs.
                                      Default to ``None``
        animate (dict, optional): Specs for animation. E.g., ``{'loop': 'backAndForth'}``.
        atom_index (bool, optional): Whether to show atom index. Defaults to ``True``.
                                     Otherwise, atom index can be viewed by hovering the mouse
                                     onto the atom and stay a while.
        style_spec (dict, Optional): Style of the shown molecule. The default is showing atom as spheres and
                                     bond as rods. Other specs can be found in the documentation of 3Dmol.js.
        viewer (py3Dmol.view, optional): Provide an existing viewer, instead of create a new one.
        viewer_size (tuple, optional): Set the viewer size. Only useful if ``viewer`` is not provided.
                                       Defaults to ``(400, 400)``.
        viewer_loc (tuple, optional): The location of the viewer in the grid. E.g., (0, 1). Defaults to None.

    Returns:
        py3Dmol.view: The molecule viewer.
    """
    if not viewer:
        try:
            viewer = py3Dmol.view(width=viewer_size[0], height=viewer_size[1])
        except TypeError:
            raise RuntimeError("py3Dmol is not installed. Please install py3Dmol with conda or pip. ")

    if not model_extra:
        viewer.addModel(obj, model, viewer=viewer_loc)
    else:
        viewer.addModel(obj, model, model_extra, viewer=viewer_loc)

    style_spec = style_spec or default_style_spec
    viewer.setStyle(default_style_spec, viewer=viewer_loc)

    if animate:
        viewer.animate(animate, viewer=viewer_loc)

    if atom_index:
        viewer.addPropertyLabels(
            "index",  # property name
            {},  # AtomSelectionSpec
            {
                'fontSize': 15,
                'fontColor': 'white',
                'alignment': 'center',
                'showBackground': True,
                'backgroundOpacity': 0.2,  # I found adding a background is slightly better
                'backgroundColor': 'black',
            },  # LabelSpec
            viewer=viewer_loc
        )
    else:
        set_atom_index_hoverable(viewer)

    viewer.zoomTo(viewer=viewer_loc)

    return viewer


def set_atom_index_hoverable(viewer):

    viewer.setHoverable(
        # AtomSelectionSpec
        {},
        # is Hoverable (Boolean)
        True,
        # hover_callback
        """function(atom,viewer,event,container) {
            if(!atom.label) {
            atom.label = viewer.addLabel(
                atom.atom+":"+atom.index,
                {
                    position: atom,
                    backgroundColor: 'black',
                    fontColor: 'white',
                    alignment: 'center',
                    showBackground: true,
                    backgroundOpacity: 0.2
                }
            );
        }}""",
        # unhover_callback
        """function(atom,viewer) {
            if(atom.label) {
            viewer.removeLabel(atom.label);
            delete atom.label;
            }
        }""",
    )
