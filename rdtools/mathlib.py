import numpy as np


def solve_quadratic(a: float, b: float, c: float):
    """
    Solve the quadratic equation ax^2 + bx + c = 0.
    Return the two solutions as a tuple (x1, x2).
    """
    delta = b ** 2 - 4 * a * c
    assert delta >= 0
    return (-b + delta ** 0.5) / (2 * a), (-b - delta ** 0.5) / (2 * a)


def get_magnitude(
    x: np.ndarray,
    dx: np.ndarray,
    target: float
) -> tuple:
    """
    Given a vector x and a modification vector dx, return the magnitude a, so that
    the 2-norm of x + a * dx is equal to target_dist. Two solutions are expected
    """
    l_x = np.linalg.norm(x, ord=2)
    l_dx = np.linalg.norm(dx, ord=2)

    cosine = np.dot(x, dx) / (l_x * l_dx)

    a = l_dx ** 2
    b = -2 * l_x * l_dx * cosine
    c = l_x ** 2 - target ** 2

    x1, x2 = solve_quadratic(a, b, c)
    return x1, x2
