#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
A module contains py3Dmol functions to draw molecules
"""

from typing import Optional

import py3Dmol


def mol_viewer(obj: str,
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
        viewer.addModel(obj, model, viewer=viewer_loc)
    else:
        viewer.addModel(obj, model, model_extra, viewer=viewer_loc)

    if style_spec is None:
        viewer.setStyle({'stick': {'radius': 0.2},
                        'sphere': {'scale': 0.25},},)

    else:
        viewer.setStyle(style_spec, viewer=viewer_loc)

    if animate:
        viewer.animate(animate, viewer=viewer_loc)

    if atom_index:
        viewer.addPropertyLabels("index",
                                 "",
                                {'fontSize':15,
                                  'fontColor':'white',
                                  'showBackground': False,
                                  'alignment': 'center',
                                  'showBackground': True,
                                  'backgroundOpacity': 0.2,
                                  'backgroundColor': 'black',
                                 }, viewer=viewer_loc)
    viewer.zoomTo(viewer=viewer_loc)
    return viewer


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
        py3Dmol.view: The molecule frequence viewer.
    """
    model_extra = {'vibrate': {'frames': frames,'amplitude': amplitude}}
    animate = {'loop': 'backAndForth'}
    return mol_viewer(obj, model, model_extra, animate, atom_index,
                      style_spec, viewer, viewer_size, viewer_loc)


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
