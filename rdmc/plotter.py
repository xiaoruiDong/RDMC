#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides class and methods for plotting curves.
"""

from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_curve(y: Union[list,tuple,'np.array'],
               x: Optional[Union[list,tuple,'np.array']] = None,
               periodic_x: Optional[Union[float,tuple,list]] = None,
               relative_x: Optional[Union[int, str]] = None,
               xlabel: str = '',
               relative_y: Optional[Union[int, str]] = None,
               ylabel: str = '',
               highlight_index: Optional[int] = None,
               ax: 'matplotlib.pyplot.axes' = None,
               ):
    """
    An universal function to plot the energy curve.

    Args:
        y (list-like): A list-like object to be plotted as y variable.
        x (list-like): A list-like object to be plotted as y variable.
    """
    # Modify x variables
    if relative_x != None:
        if isinstance(relative_x, int):
            # Reference based on the index
            try:
                x_ref = x[relative_x]
            except KeyError:
                raise ValueError('Invalid x_baseline. If an int is given, it should be within'
                                    'the range of indices.')
        elif isinstance(relative_x, str):
            if relative_x == 'min':
                x_ref = np.min(x)
            elif relative_x == 'max':
                x_ref = np.max(x)
            else:
                raise NotImplementedError('The relative_x method is not supported.')
        x = np.array(x) - x_ref

    if periodic_x:
        if isinstance(periodic_x, (int, float)):
            # the periodic range is 0 to the value
            periodic_x = [0, periodic_x]
        try:
            periodicity = periodic_x[1] - periodic_x[0]
        except:
            raise ValueError(f'Invalid periodic_x value {periodic_x}')
        too_small = x < periodic_x[0]
        too_large = x > periodic_x[1]
        while any(too_small) or any(too_large):
            x[too_small] += periodicity
            x[too_large] -= periodicity
            too_small = x < periodic_x[0]
            too_large = x > periodic_x[1]

    # Modify y variables
    if relative_y != None:
        if isinstance(relative_y, int):
            # Reference based on the index
            try:
                y_ref = y[relative_y]
            except KeyError:
                raise ValueError('Invalid relative_y. If an int is given, it should be within'
                                    'the range of indices.')
        elif isinstance(relative_y, str):
            if relative_y == 'min':
                y_ref = np.min(y)
            elif relative_y == 'max':
                y_ref = np.max(y)
            else:
                raise NotImplementedError('The x_baseline method is not supported.')
        y = np.array(y) - y_ref

    ax = ax or plt.axes()
    if x is None:
        x = np.arange(y.shape[0])
    ax.plot(x, y, '.-')
    ax.set(xlabel=xlabel, ylabel=ylabel)

    if highlight_index and highlight_index < x.shape[0]:
        ax.plot(x[highlight_index], y[highlight_index], 'ro')
    return ax
