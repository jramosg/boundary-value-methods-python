"""Shooting methods for y'' = f(x, y, y') on [a, b].

The left condition fixes the initial data up to one parameter t (see
Robin.initial_values). Each method looks for the t that makes the
residual of the right condition, F(t), vanish.
"""

from dataclasses import dataclass, field

import numpy as np

from .ivp import rk4


@dataclass
class ShootingResult:
    x: np.ndarray
    y: np.ndarray
    dy: np.ndarray
    t: float
    iterations: int
    converged: bool
    history: list = field(default_factory=list)  # [(t_k, F(t_k)), ...]


def _result(x, u, t, iterations, converged, history):
    return ShootingResult(x, u[:, 0], u[:, 1], t, iterations, converged,
                          history)


def linear_shooting(p, q, r, a, b, left, right, n=10):
    """Solve y'' = p(x) y' + q(x) y + r(x) with Robin conditions.

    The problem is linear, so y(x; t) = u(x) + t v(x) and F(t) is affine:
    one solve of the full equation (u) and one of the homogeneous
    equation (v) give the exact t.
    """
    def full(x, w):
        return np.array([w[1], p(x) * w[1] + q(x) * w[0] + r(x)])

    def homogeneous(x, w):
        return np.array([w[1], p(x) * w[1] + q(x) * w[0]])

    u_init, v_init = left.initial_values(0.0)
    x, u = rk4(full, a, b, np.array(u_init), n)
    _, v = rk4(homogeneous, a, b, np.array(v_init), n)

    slope = right.alpha * v[-1, 0] + right.beta * v[-1, 1]
    if slope == 0:
        raise ValueError("the boundary value problem has no unique "
                         "solution")
    t = -right.residual(u[-1, 0], u[-1, 1]) / slope
    w = u + t * v
    return _result(x, w, t, 1, True,
                   [(t, right.residual(w[-1, 0], w[-1, 1]))])
