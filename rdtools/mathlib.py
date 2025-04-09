# -*- coding: utf-8 -*-
"""A module contains math functions."""

import numpy as np
import numpy.typing as npt


def solve_quadratic(a: float, b: float, c: float) -> tuple[float, float]:
    """Solve the quadratic equation ax^2 + bx + c = 0.

    Return the two solutions as a tuple (x1, x2).

    Args:
        a (float): Coefficient of x^2.
        b (float): Coefficient of x.
        c (float): Constant term.

    Returns:
        tuple[float, float]: Two solutions of the quadratic equation.
    """
    delta = b**2 - 4 * a * c
    assert delta >= 0
    return (-b + delta**0.5) / (2 * a), (-b - delta**0.5) / (2 * a)


def get_magnitude(
    x: npt.NDArray[np.float64],
    dx: npt.NDArray[np.float64],
    target: float,
) -> tuple[float, float]:
    """Get the magnitude of a vector.

    Given a vector x and a modification vector dx, return the magnitude a, so that
    the 2-norm of x + a * dx is equal to target_dist. Two solutions are expected.

    Args:
        x (npt.NDArray[np.float64]): The original vector.
        dx (npt.NDArray[np.float64]): The modification vector.
        target (float): The target distance.

    Returns:
        tuple[float, float]: Two solutions for the magnitude.
    """
    l_x = np.linalg.norm(x, ord=2)
    l_dx = np.linalg.norm(dx, ord=2)

    cosine = np.dot(x, dx) / (l_x * l_dx)

    a = float(l_dx**2)
    b = -2 * l_x * l_dx * cosine
    c = float(l_x**2 - target**2)

    x1, x2 = solve_quadratic(a, b, c)
    return x1, x2
