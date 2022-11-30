#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for finding local minima on the scanned potential energy surface by a greedy algorithm.
The search is done on a discrete space potentially with periodic boundaries. A parallel version will
be implemented in future.
"""

from itertools import product
from typing import Iterable, List, Tuple

import numpy as np


def get_step_to_adjacent_points(
        fsize: int,
        dim: int = 2,
        cutoff: float = np.inf,
        ) -> Iterable:
    """
    Get a generator containing the adjacent points to a explored point. By default, it will find all
    the neighbors in an hypercubic space centered at the explored point.

    Args:
        fsize: The furthest distance to the explored point. E.g., 2 refers to a filter x=[-2, -1, 0, 1, 2] on a 1D mesh.
        dim (optional): The dimension of the space. Default to 2.
        cutoff (optional): A cut-off used to exclude neighbors that are too far away. This is equivalent to only consider
                           neighbors in an hyperspheric space centered at the explored point.

    Returns:
        Iterable: coordinates relative to the explored point.
    """
    one_d_points = list(range(-fsize, fsize + 1))
    var_combinations = product(*[one_d_points] * dim)
    for points in var_combinations:
        dist = np.linalg.norm(np.array(points))
        if dist <= cutoff:
            yield points


def get_adjacent_energy(coord: Tuple,
                        energies: np.ndarray,
                        periodic: bool = True,
                        ) -> float:
    """
    Get the energies of adjacent points.

    Args:
        coord (tuple): The coordinate of the point.
        energies (np.ndarray): An array of all PES energies.
        periodic (bool): If the space has a periodic boundary. Defaults to True.

    Returns:
        energies (np.ndarray): A 1D array with energies queried.
    """
    try:
        return energies[coord]
    except IndexError:
        if periodic:
            new_coord = tuple(
                x if x < energies.shape[i] else x - energies.shape[i]
                for i, x in enumerate(coord)
            )
            return energies[new_coord]
        else:
            return np.inf


def compare_to_adjacent_point(
        coord: Tuple,
        energies: np.ndarray,
        unchecked_points: List[Tuple],
        filters: List[Tuple],
        ) -> Tuple:
    """
    Compare the energy of the current point to its adjacent points.

    Args:
        coord (tuple): The coordinate of the current explored point.
        energies (np.ndarray): A matrix of energies for each point of the space.
        unchecked_points (list): A list book-keeping points that haven't be checked.
        filters (list): A filter used to define adjacent points.

    Returns:
        tuple: the coordinate to explore next. If the value is the same as the current coordinate,
               it means the current point is the minimum.
    """
    # Generate the actual coordinates of the adjacent points.
    new_coords = [tuple(x + var_x for x, var_x in zip(coord, var)) for var in filters]

    # Get the energies of the adjacent points.
    energies = [get_adjacent_energy(new_coord, energies) for new_coord in new_coords]

    # Sort the coordinates according to the corresponding energies.
    energies, new_coords = zip(*sorted(zip(energies, new_coords)))

    # Find the current point index and points that has higher energy than this point.
    # These points will be removed from unchecked points list
    cur_point_ind = new_coords.index(coord)
    for new_coord in new_coords[cur_point_ind:]:
        try:
            unchecked_points.remove(new_coord)
        except ValueError:
            # ValueError if coord_min is not in unchecked_points
            pass
    return new_coords[0]


def search_for_a_minimum(
        coord: Tuple,
        energies: np.ndarray,
        unchecked_points: List[Tuple],
        filters: List[Tuple],
        ) -> Tuple:
    """
    Search a minimum on a given PES. This will only identify a single minimum identified by the greedy always-go-downward algorithm.

    Args:
        coord (tuple): The starting point coordinates.
        energies (np.ndarray): A matrix of energies.
        unchecked_points (list): A list of coordinates of the unchecked points.
        filters: (list): A filter used to identify adjacent points to an explored point.

    Returns:
        tuple: The coordinate of the minimum point.
    """
    while True:
        next_point = compare_to_adjacent_point(
            coord=coord,
            energies=energies,
            unchecked_points=unchecked_points,
            filters=filters,
        )
        next_point = tuple(
            x if x >= 0 else energies.shape[i] + x
            for i, x in enumerate(next_point)
        )

        if next_point == coord:
            # The next point is the same point as the current point,
            # indicating a local minimum is found
            return coord
        elif next_point not in unchecked_points:
            # The next point has been checked already
            return
        else:
            # Initiate another iteration with a new coordinate
            coord = next_point


def search_minimum(
        energies: np.ndarray,
        fsize: int,
        cutoff: float = np.inf,
        ) -> List[Tuple]:
    """
    Search all the minimums on a given PES by the greedy always-go-downward algorithm.

    Args:
        energies (np.ndarray): A matrix of energies.
        fsize (int): A filter to identify adjacent points defined by the furthest distance to the explored point.
        cutoff (int): A cut-off used to exclude neighbors that are too far away. This is equivalent to only consider
                      neighbors in an hyperspheric space centered at the explored point.

    Returns:
        list: The coordinates of the minimum point.
    """
    minimum = []

    # Generate a filter to identify the adjacent points of an explored point
    dim = len(energies.shape)
    filters = list(get_step_to_adjacent_points(fsize, dim, cutoff))

    # Generate all coordinates as the unchecked points
    oned_points = [list(range(energies.shape[i])) for i in range(dim)]
    unchecked_points = list(product(*oned_points))

    while True:
        if not unchecked_points:
            break
        # Randomly pick up an unexplored point
        coord = unchecked_points[np.random.randint(len(unchecked_points))]
        new_min = search_for_a_minimum(coord, energies, unchecked_points, filters)
        if new_min:
            minimum.append(new_min)
    return minimum
