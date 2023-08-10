#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modules for finding local minima on the scanned potential energy surface
by a greedy algorithm. Need to parallelize this.
"""

from itertools import product
import random
from typing import List, Tuple

import numpy as np


def get_step_to_adjacent_points(fsize: int,
                                dim: int = 2,
                                cutoff: float = np.inf,
                                ) -> "generator":
    """
    Get a generator containing the adjacent points.

    Args:
        fsize: The filter size as positive integer.
        dim: The dimension of the PES. Default is 2.
        cutoff: The cutoff distance. only consider steps with
                distance smaller than the cutoff. Default is
                infinity, meaning no cutoff is applied.

    Returns:
        A generator containing the adjacent coordinates.
    """
    one_d_points = list(range(-fsize, fsize + 1))
    var_combinations = product(*[one_d_points] * dim)
    for points in var_combinations:
        dist = np.linalg.norm(np.array(points))
        if dist <= cutoff:
            yield points


def get_energy(coord: List[Tuple],
               energies: np.ndarray,
               ) -> float:
    """
    Get the energies of adjacent points considering periodic boundary condition.

    Args:
        coord: The coordinate of a point.
        energies: The energies of all points.

    Returns:
        float: The energy of the point.
    """
    try:
        return energies[coord]
    except IndexError:
        # Periodic boundary condition
        new_coord = tuple(
            x if x < energies.shape[i] else x - energies.shape[i]
            for i, x in enumerate(coord)
        )
        return energies[new_coord]


def compare_to_adjacent_point(coord: List[Tuple],
                              energies: np.ndarray,
                              unchecked_points: List[Tuple],
                              filters: List[Tuple],
                              ) -> Tuple:
    """
    Compare the energy of current point and those of other points.

    Args:
        coord (list of tuples): The coordinate of the current point.
        energies (np.ndarray): The energies of all points.
        unchecked_points (list of tuples): The points that have not been checked.
        filters (list of tuples): The filters for searching adjacent points.

    Returns:
        The coordinate of the adjacent point with the lowest energy.
    """
    # The coordinates of the adjacent points
    new_coords = [tuple(x + var_x for x, var_x in zip(coord, var))
                  for var in filters]

    # Get the energies of adjacent points
    energies = [get_energy(new_coord, energies)
                for new_coord in new_coords]

    # Sort the coordinates by energy
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


def search_for_a_minimum(coord: tuple,
                         energies: np.ndarray,
                         unchecked_points: List[Tuple],
                         filters: List[Tuple],
                         ) -> Tuple:
    """
    Search a local minimum on a given PES.

    Args:
        coord (tuple): The coordinate of the current point.
        energies (np.ndarray): The energies of all points.
        unchecked_points (list of tuples): The points that have not been checked.
        filters (list of tuples): The filters for searching adjacent points.

    Args:
        The coordinate of a local minimum.
    """
    while True:
        # Given current coordinates,
        # find the adjacent point with the lowest energy
        next_point = compare_to_adjacent_point(coord=coord,
                                               energies=energies,
                                               unchecked_points=unchecked_points,
                                               filters=filters)
        next_point = tuple(x if x >= 0 else energies.shape[i] + x
                           for i, x in enumerate(next_point))
        if next_point == coord:
            # The current point is a minimum
            return coord
        elif next_point not in unchecked_points:
            # The adjacent point has been checked
            return
        else:
            # Another iteration
            coord = next_point


def search_minimum(energies: np.ndarray,
                   fsize: int,
                   cutoff: float = np.inf,
                   ) -> List[Tuple]:
    """
    Search all the minimums on a given PES.

    Args:
        energies (np.ndarray): The energies of all points.
        fsize (int): The filter size as positive integer.
        cutoff (float): The cutoff distance. only consider steps with
                        distance smaller than the cutoff. Default is
                        infinity, meaning no cutoff is applied.

    Returns:
        A list of coordinates of the minimums.
    """
    # Initiate a list to store the minima
    minima = []

    # Generate the filter masks
    dim = len(energies.shape)
    filters = list(get_step_to_adjacent_points(fsize, dim, cutoff))

    # Generate the coordinates of all points
    oned_points = [list(range(energies.shape[i])) for i in range(dim)]
    unchecked_points = list(product(*oned_points))

    while unchecked_points:
        # Randomly choose a point from unchecked points
        coord = random.choice(unchecked_points)
        # Find the local minimum traced from this point
        new_min = search_for_a_minimum(coord,
                                       energies,
                                       unchecked_points,
                                       filters)
        if new_min:
            minima.append(new_min)
    return minima
