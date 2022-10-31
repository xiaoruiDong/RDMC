#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for finding local minima on the scanned potential energy surface by greedy algorithm
"""

from itertools import combinations, product
from typing import List, Tuple

import numpy as np


def get_step_to_adjacent_points(
    fsize: int, dim: int = 2, cutoff: float = np.inf
) -> "generator":
    """Get a generator containig the adjacent points."""
    one_d_points = list(range(-fsize, fsize + 1))
    var_combinations = product(*[one_d_points] * dim)
    for points in var_combinations:
        dist = np.linalg.norm(np.array(points))
        if dist <= cutoff:
            yield points


def get_adjacent_energy(coord: List[Tuple], energies: np.ndarray) -> float:
    """Get the energies of adjacent points."""
    try:
        return energies[coord]
    except IndexError:
        new_coord = tuple(
            x if x < energies.shape[i] else x - energies.shape[i]
            for i, x in enumerate(coord)
        )
        return energies[new_coord]


def compare_to_adjacent_point(
    coord: List[Tuple],
    energies: np.ndarray,
    unchecked_points: List[Tuple],
    filters: List[Tuple],
) -> Tuple:
    """Compare the energy of current point and those of other points."""
    # each element is a coordinate
    new_coords = [tuple(x + var_x for x, var_x in zip(coord, var)) for var in filters]

    # Get the energies of adjacent points
    energies = [get_adjacent_energy(new_coord, energies) for new_coord in new_coords]

    # Sorted
    energies, new_coords = zip(*sorted(zip(energies, new_coords)))

    # Find the current point index and points that has higher energy than this point
    # Will be removed from unchecked points list
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
    """Search a minimum on a given PES."""
    while True:
        next_point = compare_to_adjacent_point(
            coord, energies, unchecked_points, filters
        )
        next_point = tuple(
            x if x >= 0 else energies.shape[i] + x for i, x in enumerate(next_point)
        )
        if next_point == coord:
            return coord
        elif next_point not in unchecked_points:
            return
        else:
            coord = next_point


def search_minimum(
    energies: np.ndarray, fsize: int, cutoff: float = np.inf
) -> List[Tuple]:
    """Search all the minimums on a given PES."""
    minimum = []

    dim = len(energies.shape)
    filters = list(get_step_to_adjacent_points(fsize, dim, cutoff))

    oned_points = [list(range(energies.shape[i])) for i in range(dim)]
    unchecked_points = list(product(*oned_points))

    while True:
        if not unchecked_points:
            break
        coord = unchecked_points[np.random.randint(len(unchecked_points))]
        new_min = search_for_a_minimum(coord, energies, unchecked_points, filters)
        if new_min:
            minimum.append(new_min)
    return minimum
