"""Base viewer functions built on py3Dmol."""

from typing import Any, Optional

try:
    import py3Dmol
except ImportError:
    from rdtools.utils import get_fake_module

    py3Dmol = get_fake_module("py3Dmol")


default_style_spec = {
    "stick": {"radius": 0.05, "color": "#f2f2f2"},
    "sphere": {"scale": 0.25},
}


default_label_spec = {
    "fontSize": 15,
    "fontColor": "white",
    "alignment": "center",
    "showBackground": True,
    "backgroundOpacity": 0.2,  # I found adding a background is slightly better
    "backgroundColor": "black",
}


def _set_atom_index(
    viewer: py3Dmol.view,
    viewer_loc: Optional[tuple[int, int]] = None,
) -> None:
    """Set atom index as labels.

    Args:
        viewer (py3Dmol.view): The py3Dmol viewer.
        viewer_loc (Optional[tuple[int, int]], optional): The location of the viewer in the grid.
    """
    viewer.addPropertyLabels(
        "index",  # property name
        {},  # AtomSelectionSpec
        default_label_spec,
        viewer=viewer_loc,
    )


def _set_atom_index_hoverable(viewer: py3Dmol.view) -> None:
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


def base_viewer(
    obj: str | list[str],
    model: str = "xyz",
    model_extra: Optional[dict[str, Any]] = None,
    animate: Optional[dict[str, Any]] = None,
    atom_index: bool = True,
    style_spec: Optional[dict[str, Any]] = None,
    viewer: Optional[py3Dmol.view] = None,
    viewer_size: tuple[int, int] = (400, 400),
    viewer_loc: Optional[tuple[int, int]] = None,
    as_frames: bool = False,
) -> py3Dmol.view:
    """General function to view a molecule.

    Powered by py3Dmol and its backend 3Dmol.js. This allows you to visualize molecules
    in 3D with a javascript object or in IPython notebooks. This function is also used to build up more
    complicated viewers, e.g., freq_viewer.

    Args:
        obj (str | list[str]): A string representation of the molecule can be xyz string,
            sdf string, etc.
        model (str, optional): The format of the molecule representation, e.g., ``'xyz'``.
            Defaults to ``'xyz'``.
        model_extra (Optional[dict[str, Any]], optional): Extra specs for the model (format). E.g., frequency specs.
            Default to ``None``
        animate (Optional[dict[str, Any]], optional): Specs for animation. E.g., ``{'loop': 'backAndForth'}``.
        atom_index (bool, optional): Whether to show atom index persistently. Defaults to ``True``.
            Otherwise, atom index can be viewed by hovering the mouse
            onto the atom and stay a while.
        style_spec (Optional[dict[str, Any]], optional): Style of the shown molecule. The default is showing atom as spheres and
            bond as rods. The default setting is:

            .. code-block:: javascript

                {'stick': {'radius': 0.05,
                        'color': '#f2f2f2'},
                'sphere': {'scale': 0.25},}

            which set both bond width/color and atom sizes. For more details, please refer to the
            original APIs in `3DMol.js <https://3dmol.org/doc/tutorial-home.html>`_.
        viewer (Optional[py3Dmol.view], optional): Provide an existing viewer, instead of create a new one.
        viewer_size (tuple[int, int], optional): Set the viewer size. Only useful if ``viewer`` is not provided.
            Defaults to ``(400, 400)``.
        viewer_loc (Optional[tuple[int, int]], optional): The location of the viewer in the grid. E.g., (0, 1). Defaults to None.
        as_frames (bool, optional): If add object as frames of an animation. Defaults to ``False``.

    Returns:
        py3Dmol.view: The molecule viewer.

    Raises:
        NotImplementedError: If passing multiple objects with ``model_extra``.
    """
    if not viewer:
        viewer = py3Dmol.view(width=viewer_size[0], height=viewer_size[1])

    if as_frames:
        if model_extra:
            viewer.addModelsAsFrames(obj, model, model_extra, viewer=viewer_loc)
        else:
            viewer.addModelsAsFrames(obj, model, viewer=viewer_loc)
    else:
        if model_extra:
            if not isinstance(obj, str):
                raise NotImplementedError(
                    "Passing multiple objs with model_extra is currently not supported."
                )
            viewer.addModel(obj, model, model_extra, viewer=viewer_loc)
        else:  # Most common usage
            obj = [obj] if isinstance(obj, str) else obj
            for object in obj:
                viewer.addModel(object, model, viewer=viewer_loc)

    style_spec = style_spec or default_style_spec
    viewer.setStyle(default_style_spec, viewer=viewer_loc)

    if animate:
        viewer.animate(animate, viewer=viewer_loc)

    if atom_index:
        _set_atom_index(viewer, viewer_loc)
    else:
        _set_atom_index_hoverable(viewer)

    viewer.zoomTo(viewer=viewer_loc)

    return viewer


def grid_viewer(
    viewer_grid: tuple[int, int],
    linked: bool = False,
    viewer_size: Optional[tuple[int, int]] = None,
) -> py3Dmol.view:
    """Create a empty grid viewer.

    You can then fill in each blank by passing this viewer and ``viewer_loc`` to desired viewer functions.

    Args:
        viewer_grid (tuple[int, int]): The layout of the grid, e.g., (1, 4) or (2, 2).
        linked (bool, optional): Whether changes in different sub viewers are linked. Defaults to ``False``.
        viewer_size (Optional[tuple[int, int]], optional): The size of the viewer in (width, height). By Default, each block
            is 250 width and 400 height.

    Returns:
        py3Dmol.view: The empty grid viewer.
    """
    if viewer_size:
        width, height = viewer_size
    else:
        width = viewer_grid[1] * 250
        height = viewer_grid[0] * 400

    return py3Dmol.view(
        width=width, height=height, linked=linked, viewergrid=viewer_grid
    )


def animation_viewer(
    obj: str,
    model: str = "xyz",
    loop: str = "forward",
    reps: int = 0,
    step: int = 1,
    interval: int = 60,
    atom_index: bool = False,
    **kwargs: Any,
) -> py3Dmol.view:
    """Create a viewer for molecule animation.

    The only viable input the RDMC authors know is a xyz string of multiple molecules.

    Args:
        obj (str): A string representation of molecules in xyz format.
        model (str, optional): The model (format) of the molecule representation, e.g., ``'xyz'``.
            Defaults to ``'xyz'``.
        loop (str, optional): The direction of looping. Available options are ``'forward'``, ``'backward'``,
            or ``'backAndForth'``. Defaults to ``'forward'``.
        reps (int, optional): The number of times the animation is repeated. Defaults to ``0``, for infinite loop.
        step (int, optional): The number of steps between frames. Defaults to ``1``, showing all the frames,
        interval (int, optional): The time interval between each frame in millisecond. Defaults to ``60``.
            To slow down the animation, you may want to use a larger number.
        atom_index (bool, optional): Whether to show atom index persistently. Defaults to ``False``.
            Currently, the label is only created based on the first frame, so we
            suggest turning it off.
        **kwargs (Any, optional): Additional arguments for the viewer. E.g., ``{'viewer_size': (400, 400)}``.
            See `base_viewer <#base_viewer>`_ for more details.

    Returns:
        py3Dmol.view: The molecule viewer.
    """
    animate = {"loop": loop, "reps": reps, "step": step, "interval": interval}

    return base_viewer(
        obj,
        model,
        animate=animate,
        as_frames=True,
        atom_index=atom_index,
        **kwargs,
    )
