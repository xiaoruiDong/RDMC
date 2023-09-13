#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides class and methods for dealing with 3D geometry operations.
"""

from typing import Iterable, Optional

import numpy as np
from scipy.spatial.transform import Rotation


def get_centroid(coords: np.array, keepdims=False) -> np.array:
    """
    Get the centroid given the coords of each item.

    Args:
        coords (np.array): The 3D coordinates in numpy array with a size of N x 3.
        keepdims (bool): Defaults to False (export 1D array); otherwise output an array of a size of 1 x 3.
    Returns:
        np.array: 1D array indicate the coordinates of the centroid.
    """
    return np.mean(coords, axis=0, keepdims=keepdims)


def get_weighted_center(coords: np.array,
                        weights: Iterable,
                        keepdims=False,
                        ) -> np.array:
    """
    Get the mass center given the coords of each item and its corresponding weight.

    Args:
        coords (np.array): The 3D coordinates in numpy array with a size of N x 3.
        weights (Iterable): A list or an array of weights corresponding to each element.
        keepdims (bool): Defaults to False (export 1D array); otherwise output an array of a size of 1 x 3.

    Returns:
        np.array: 1D array indicate the coordinates of the mass center.
    """
    return np.mean(np.array(weights).reshape(-1, 1) * coords, axis=0, keepdims=keepdims)


def get_mass_center(coords: np.array,
                    atommasses: Iterable,
                    keepdims=False,
                    ) -> np.array:
    """
    Get the mass center given the coords of each item and its corresponding mass.

    Args:
        coords (np.array): The 3D coordinates in numpy array with a size of N x 3.
        atommasses (Iterable): A list or an array of mass values.
        keepdims (bool): Defaults to False (export 1D array); otherwise output an array of a size of 1 x 3.

    Returns:
        np.array: 1D array indicate the coordinates of the mass center.
    """
    return get_weighted_center(coords, atommasses, keepdims)


def translate(coords: np.array,
              tran_vec: np.array) -> np.array:
    """
    Translate the coordinates according to the `tran_vec` vector.

    Args:
        coords (np.array): The 3D coordinates in numpy array with a size of N x 3.
        tran_vec (np.array): A vector indicate the direction and the magnitude of the translational operation.
                             It should be a numpy array with a size of (3,) or (1,3).

    Returns:
        np.array: An numpy array with the same size as the original coords.
    """
    return coords + tran_vec.reshape(1, -1)


def translate_centroid(coords: np.array,
                       new_ctr: np.array = np.zeros(3)):
    """
    Translate the coordinates according to the `tran_vec` vector.

    Args:
        coords (np.array): The 3D coordinates in numpy array with a size of N x 3.
        new_ctr (np.array): A vector indicate the new position of the centroid.
                            It should be a numpy array with a size of (3,) or (1,3).
                            By defaults, the centroid will be moved to the origin.

    Returns:
        np.array: An numpy array with the same size as the original coords.
    """
    return coords + new_ctr - get_centroid(coords)


def get_distances_from_a_point(coords: np.array,
                               pos: np.array,
                               keepdims: bool = False):
    """
    Get the Euclidiean distance to a point for all elements.

    Args:
        coords (np.array): The 3D coordinates in numpy array with a size of N x 3.
        pos (np.array): The coordinates of the point.
        keepdims (bool): Defaults to False (export 1D array); otherwise output an array of a size of 1 x 3.

    Returns:
        np.array: 1D array indicate the distances.
    """
    return np.sqrt(np.sum((coords - pos.reshape(1, -1)) ** 2, axis=1, keepdims=keepdims))


def get_max_distance_from_center(coords: np.array, ) -> float:
    """
    Get the maximum distance from the center of a set of coordinates

    Args:
        coords (np.array): The 3D coordinates in numpy array with a size of N x 3.

    Returns:
        float: The distance between the center and the farthest item.
    """
    center = get_centroid(coords)
    return np.max(get_distances_from_a_point(coords, center))


def rotate(coords: np.array,
           angles: np.array,
           degrees: bool = False,
           about_center: bool = False,
           about: Optional[np.array] = None,):
    """
    Rotate the coordinates according to the angles about the x, y, and z axes. The rotation is
    about the origin, but there are a few options about choosing the ``about`` location.

    Args:
        coords (np.array): The 3D coordinates in numpy array with a size of :math:`N \\times 3`.
        angles (np.array): An array with a size of ``(1,3)`` indicates the rotation angles about the
                           x, y, and z axes, respectively.
        degrees (bool): If the angles are defined as degrees. Defaults to ``False``.
        about_center (bool): Whether to rotate the coordinates about their center.
                             Defaults to ``False``. Note ``about_center`` cannot be assigned simultaneously with ``about``.
        about (np.array): The coordinate that the rotation is about. Should be a vector with a length of 3.
                          It is defaults to ``None``, rotating about the origin.
                          ``about`` cannot be specified along with ``about_center``.
    Returns:
        np.array: coordinates after the rotation.
    """
    if about_center and about is not None:
        raise ValueError('about and about_center cannot be specified simultaneously.')
    elif about_center:
        about = get_centroid(coords)
    elif about is None:
        about = np.array([0., 0., 0.,])

    rot = Rotation.from_euler('xyz', np.array(angles).reshape(1, -1), degrees=degrees)
    coords_tmp = translate(coords, -about)
    coords_tmp = rot.apply(coords_tmp)
    return translate(coords_tmp, about)
