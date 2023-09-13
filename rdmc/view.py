#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A module contains py3Dmol functions to draw molecules
"""

from typing import Optional, Tuple, Union

import py3Dmol

from rdmc.mol import RDKitMol
from rdmc.ts import clean_ts
from ipywidgets import interact, IntSlider


def mol_viewer(obj: str,
               model: str = 'xyz',
               model_extra: Optional[dict] = None,
               animate: Optional[dict] = None,
               atom_index: bool = True,
               style_spec: Optional[dict] = None,
               viewer: Optional[py3Dmol.view] = None,
               viewer_size: tuple = (400, 400),
               viewer_loc: Optional[tuple] = None,
               gv_background: bool = False,
               confId: int = 0,
               ) -> py3Dmol.view:
    """
    This is the most general function to view a molecule powered by py3Dmol

    Args:
        obj (str): A string representation of the molecule can be xyz string,
                   sdf string, etc. Or an RDKitMol object.
        model (str, optional): The model (format) of the molecule representation, e.g., ``'xyz'``.
                     Defaults to ``'xyz'``. If RDKitMol is feed in, the model will forced to be
                     ``sdf``.
        model_extra (dict, optional): Extra specs for the model. E.g., frequency specs
                     Can be provided here. Default to ``None``
        animate (dict, optional): Specs for animation. E.g., ``{'loop': 'backAndForth'}``.
        atom_index (bool, optional): Whether to show atom index. Defaults to True.
        style_spec (dict, Optional): Style of the shown molecule. The default is showing both
                                     both atoms and bonds.
        viewer (py3Dmol.view, optional): Provide an existing viewer, instead of create a new one.
        viewer_size (tuple, optional): Set the viewer size. Only useful if ``viewer`` is not provided.
                                       Defaults to (400, 400).
        viewer_loc (tuple, optional): The location of the viewer in the grid. E.g., (0, 1). Defaults to None.
        gv_background (optional, bool): To use a background that is similar to the style of GaussView.
                                        Defaults to ``False``.
        confId (int): Integer of embedded conformer to view

    Returns:
        py3Dmol.view: The molecule viewer.
    """
    if isinstance(obj, RDKitMol):
        obj, model = obj.ToMolBlock(confId=confId), 'sdf'
    if not viewer:
        viewer = py3Dmol.view(width=viewer_size[0], height=viewer_size[1])

    if not model_extra:
        viewer.addModel(obj, model, viewer=viewer_loc)
    else:
        viewer.addModel(obj, model, model_extra, viewer=viewer_loc)

    if style_spec is None:
        viewer.setStyle({'stick': {'radius': 0.05, 'color': '#f2f2f2'},
                         'sphere': {'scale': 0.25}, }, viewer=viewer_loc)
    else:
        viewer.setStyle(style_spec, viewer=viewer_loc)

    if animate:
        viewer.animate(animate, viewer=viewer_loc)

    if atom_index:
        viewer.addPropertyLabels("index",
                                 "",
                                 {'fontSize': 15,
                                  'fontColor': 'white',
                                  'alignment': 'center',
                                  'showBackground': True,
                                  'backgroundOpacity': 0.2,
                                  'backgroundColor': 'black',
                                  },
                                 viewer=viewer_loc)
    if gv_background:
        viewer.setBackgroundColor('#e5e5ff')
    viewer.zoomTo(viewer=viewer_loc)
    return viewer


def conformer_viewer(mol: 'RDKitMol',
                     conf_ids: Optional[list] = None,
                     highlight_ids: Optional[list] = None,
                     opacity: float = 0.5,
                     style_spec: Optional[dict] = None,
                     viewer: Optional[py3Dmol.view] = None,
                     viewer_size: tuple = (400, 400),
                     viewer_loc: Optional[tuple] = None,
                     ) -> py3Dmol.view:
    """
    This is a viewer for viewing multiple overlaid conformers.

    Args:
        mol (RDKitMol): An RDKitMol object with embedded conformers.
        conf_ids (list, optional): A list of conformer ids (as ``int``) to be overlaid and viewed.
                                   If not provided, all embedded conformers will be used.
        highlight_ids (list, optional): It is possible to highlight some of the conformers while greying out
                                        other conformers by providing the conformer IDs you want to highlight.
        opacity (float, optional): Set the opacity of the non-highlighted conformers and is only used with the highlighting feature.
                                   the value should be a ``float`` between ``0.0`` to ``1.0``. The default value is ``0.5``.
                                   Values below ``0.3`` may be hard to see.
        style_spec (dict, Optional): Style of the shown molecule. The default setting is:

                                     .. code-block:: javascript

                                        {'stick': {'radius': 0.05,
                                                   'color': '#f2f2f2'},
                                         'sphere': {'scale': 0.25},}

                                     which set both bond width/color and atom sizes. For more details, please refer to the
                                     original APIs in `3DMol.js <https://3dmol.org/doc/tutorial-home.html>`_.
        viewer (py3Dmol.view, optional): Provide an existing viewer, instead of create a new one.
        viewer_size (tuple, optional): Set the viewer size. Only useful if ``viewer`` is not provided.
                                       Defaults to ``(400, 400)``.
        viewer_loc (tuple, optional): The location of the viewer in the grid. E.g., ``(0, 1)``. Only useful if ``viewer`` is not provided.

    Returns:
        py3Dmol.view: The conformer viewer.
    """
    if not viewer:
        viewer = py3Dmol.view(width=viewer_size[0], height=viewer_size[1])
    if not conf_ids:
        conf_ids = list(range(mol.GetNumConformers()))
    for i in range(len(conf_ids)):
        viewer.addModel(mol.ToMolBlock(confId=conf_ids[i]), 'sdf')

    if style_spec is None:
        style_spec = {'stick': {'radius': 0.05, 'color': '#f2f2f2'},
                      'sphere': {'scale': 0.25}, }
    if highlight_ids is None:
        viewer.setStyle(style_spec, viewer=viewer_loc)
    else:
        # We need to figure out the sequence number of the highlighted IDs,
        # since it is the actual input for setting styles
        highlight_seq = [conf_ids.index(i) for i in highlight_ids]
        viewer.setStyle({'model': highlight_seq},
                        style_spec, viewer=viewer_loc)
        viewer.setStyle({'model': [i for i in range(len(conf_ids)) if i not in highlight_seq]},
                        {key: {**value, **{'opacity': opacity}} for key, value in style_spec.items()},
                        viewer=viewer_loc)
    viewer.zoomTo(viewer=viewer_loc)
    return viewer


def interactive_conformer_viewer(mol, **kwargs):
    """
    This is a viewer for individually viewing multiple conformers using an ipython slider widget.
    Uses keywords from mol_viewer function, so all arguments of that function are available here.

    Args:
        mol (RDKitMol): An RDKitMol object with embedded conformers.
    Returns:
        py3Dmol.view: The molecule viewer with slider to view different conformers.
    """
    if isinstance(mol, list) or isinstance(mol, tuple):
        def viewer(confId):
            return mol_viewer(obj=mol[confId], confId=0, **kwargs)
        return interact(
            viewer,
            confId=IntSlider(min=0, max=len(mol) - 1, step=1)
        )

    else:
        def viewer(confId):
            return mol_viewer(obj=mol, confId=confId, **kwargs)
        return interact(
            viewer,
            confId=IntSlider(min=0, max=mol.GetNumConformers() - 1, step=1)
        )


def freq_viewer(obj: str,
                model: str = 'xyz',
                frames: int = 10,
                amplitude: float = 1.,
                atom_index: bool = True,
                style_spec: Optional[dict] = None,
                viewer: Optional[py3Dmol.view] = None,
                viewer_size: tuple = (400, 400),
                viewer_loc: Optional[tuple] = None,
                ) -> py3Dmol.view:
    """
    This is the viewer for frequency.

    Args:
        obj (str): A string representation of the molecule can be xyz string,
                   sdf string, etc.
        model (str, optional): The model (format) of the molecule representation, e.g., ``'xyz'``.
                     Defaults to ``'xyz'``.
        frames (int, optional): Number of frames to be created.
        amplitude (float, optional): amplitude of distortion.
        atom_index (bool, optional): Whether to show atom index. Defaults to True.
        style_spec (dict, Optional): Style of the shown molecule. The default is showing both
                                     both atoms and bonds.
        viewer (py3Dmol.view, optional): Provide an existing viewer, instead of create a new one.
        viewer_size (tuple, optional): Set the viewer size. Only useful if ``viewer`` is not provided.
                                       Defaults to (400, 400).
        viewer_loc (tuple, optional): The location of the viewer in the grid. E.g., (0, 1). Defaults to None.

    Returns:
        py3Dmol.view: The molecule frequency viewer.
    """
    model_extra = {'vibrate': {'frames': frames, 'amplitude': amplitude}}
    animate = {'loop': 'backAndForth'}
    viewer = mol_viewer(obj, model, model_extra, animate, atom_index,
                        style_spec, viewer, viewer_size, viewer_loc)
    viewer.vibrate(frames, amplitude, True, {'color': 'black',
                                             'radius': 0.08, })
    return viewer


def grid_viewer(viewer_grid: tuple,
                linked: bool = False,
                viewer_size: Optional[tuple] = None,
                ) -> py3Dmol.view:
    """
    Create a empty grid viewer.

    Args:
        viewer_grid (tuple): The layout of the grid, e.g., (1, 4) or (2, 2).
        linked (bool, optional): Whether changes in different sub viewers are linked. Defaults to False.
        viewer_size (tuple, optional): The size of the viewer in (width, height). By Default, each block
                                       is 250 width and 400 height.
    """
    if viewer_size:
        width, height = viewer_size
    else:
        width = viewer_grid[1] * 250
        height = viewer_grid[0] * 400

    return py3Dmol.view(width=width, height=height, linked=linked, viewergrid=viewer_grid)


def ts_viewer(r_mol: 'RDKitMol',
              p_mol: 'RDKitMol',
              ts_mol: 'RDKitMol',
              only_ts: bool = False,
              alignment: list = ['r', 'p', 'ts'],
              vertically_aligned: bool = True,
              ts_bond_color: Union[str, Tuple[str]] = ("red", "green"),
              ts_bond_width: float = 0.1,
              **kwargs):
    """
    View reactant, product and ts of a given reaction. The broken bonds in the TS will be shown with red
    lines while the formed bonds in the TS will be shown with green lines. Uses keywords from mol_viewer
    function, so all arguments of that function are available here.

    Args:
        r_mol (RDKitMol): The reactant complex.
        p_mol (RDKitMol): The product complex.
        ts_mol (RDKitMol): The ts corresponding to r_mol and p_mol.
        only_ts (bool, optional): Only shows TS. Defaults to showing reactant, product, and TS. Defaults to False.
        alignment (list, optional): The alignment of reactant, product, and TS. Defaults to `['r', 'p', 'ts']`.
                                    It can be changed to `['r', 'ts', 'p']` to make TS in the middle. You can even remove
                                    'r' or 'p' to only show TS and one side of the reaction, e.g., `['r', 'ts']`.
                                    This argument is only valid when `only_ts` is False.
        vertically_aligned(bool, optional): Reactant, product, and TS are vertically aligned. Defaults to `True`.
                                            Only valid when `only_ts` is `False`.
        ts_bond_color (tuple or str, optional): The TS bond color for broken bonds (defaults to `"red"`) and formed bonds (defaults to `"green"`).
                                                If a single str is input, the same color will be used. If a length-2 tuple is provided, then the
                                                first color will be used as the broken bond color. As a reference, normal bond color is '#f2f2f2'.
        ts_bond_width (float): The width of the TS bonds. Defaults to 0.1. As a reference, normal bond width is 0.05.
    """
    # Create grid viewer
    if kwargs.get('viewer'):
        viewer = kwargs['viewer']
    else:
        grid_arguments = {}
        if only_ts:
            grid_arguments['viewer_grid'] = (1, 1)
        elif vertically_aligned:
            grid_arguments['viewer_grid'] = (len(alignment), 1)
        else:
            grid_arguments['viewer_grid'] = (1, len(alignment))
        if kwargs.get('viewer_size'):
            grid_arguments['viewer_size'] = (grid_arguments['viewer_grid'][0] * kwargs['viewer_size'][0],
                                             grid_arguments['viewer_grid'][1] * kwargs['viewer_size'][1],)
        grid_arguments['linked'] = kwargs.get('linked', False)
        viewer = grid_viewer(**grid_arguments)

    if only_ts:
        alignment = ['ts']

    if isinstance(ts_bond_color, str):
        ts_bond_color = (ts_bond_color,) * 2

    # Clean up TS by removing formed and broken bonds
    edited_ts_mol, broken_bonds, formed_bonds = clean_ts(r_mol, p_mol, ts_mol)
    mols = {'r': r_mol, 'p': p_mol, 'ts': edited_ts_mol}

    i = 0
    for label in alignment:

        mol = mols[label]
        viewer_loc = kwargs.get('viewer_loc') or ((i, 0) if vertically_aligned else (0, i))
        mol_viewer(mol, **{**kwargs, **{'viewer': viewer, 'viewer_loc': viewer_loc}})
        if label == 'ts':
            # Recreate broken bonds and formed bonds
            for bbs in broken_bonds:
                start, end = mol.GetPositions()[bbs, :]
                viewer.addCylinder({"start": dict(x=start[0], y=start[1], z=start[2]),
                                    "end": dict(x=end[0], y=end[1], z=end[2]),
                                    "color": ts_bond_color[0],
                                    "radius": ts_bond_width,
                                    "dashed": True,
                                    }, **{**kwargs, **{'viewer': viewer_loc}})
            for fbs in formed_bonds:
                start, end = mol.GetPositions()[fbs, :]
                viewer.addCylinder({"start": dict(x=start[0], y=start[1], z=start[2]),
                                    "end": dict(x=end[0], y=end[1], z=end[2]),
                                    "color": ts_bond_color[1],
                                    "radius": ts_bond_width,
                                    "dashed": True,
                                    }, **{**kwargs, **{'viewer': viewer_loc}})

        viewer.zoomTo(viewer=viewer_loc)
        i += 1

    return viewer


def mol_animation(obj: str,
                  model: str = 'xyz',
                  model_extra: Optional[dict] = None,
                  animate: Optional[dict] = None,
                  atom_index: bool = True,
                  style_spec: Optional[dict] = None,
                  viewer: Optional[py3Dmol.view] = None,
                  viewer_size: tuple = (400, 400),
                  viewer_loc: Optional[tuple] = None,
                  ) -> py3Dmol.view:
    """
    This is the most general function to view a molecule powered by py3Dmol

    Args:
        obj (str): A string representation of the molecule can be xyz string,
                   sdf string, etc.
        model (str, optional): The model (format) of the molecule representation, e.g., ``'xyz'``.
                     Defaults to ``'xyz'``.
        model_extra (dict, optional): Extra specs for the model. E.g., frequency specs
                     Can be provided here. Default to ``None``
        animate (dict, optional): Specs for animation. E.g., ``{'loop': 'backAndForth'}``.
        atom_index (bool, optional): Whether to show atom index. Defaults to True.
        style_spec (dict, Optional): Style of the shown molecule. The default is showing both
                                     both atoms and bonds.
        viewer (py3Dmol.view, optional): Provide an existing viewer, instead of create a new one.
        viewer_size (tuple, optional): Set the viewer size. Only useful if ``viewer`` is not provided.
                                       Defaults to (400, 400).
        viewer_loc (tuple, optional): The location of the viewer in the grid. E.g., (0, 1). Defaults to None.

    Returns:
        py3Dmol.view: The molecule viewer.
    """
    if not viewer:
        viewer = py3Dmol.view(width=viewer_size[0], height=viewer_size[1])

    if not model_extra:
        viewer.addModelsAsFrames(obj, model, viewer=viewer_loc)
    else:
        viewer.addModelsAsFrames(obj, model, model_extra, viewer=viewer_loc)

    if style_spec is None:
        viewer.setStyle({'stick': {'radius': 0.05, 'color': '#f2f2f2'},
                         'sphere': {'scale': 0.25}, },)

    else:
        viewer.setStyle(style_spec, viewer=viewer_loc)

    if not animate:
        animate = {"loop": "forward", "reps": 0, "step": 1, "interval": 60}
    viewer.animate(animate, viewer=viewer_loc)

    if atom_index:
        viewer.addPropertyLabels("index",
                                 "",
                                 {'fontSize': 15,
                                  'fontColor': 'white',
                                  'alignment': 'center',
                                  'showBackground': True,
                                  'backgroundOpacity': 0.2,
                                  'backgroundColor': 'black',
                                  },
                                 viewer=viewer_loc)
    viewer.zoomTo(viewer=viewer_loc)
    return viewer
