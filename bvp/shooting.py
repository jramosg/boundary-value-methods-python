"""Shooting methods for y'' = f(x, y, y') on [a, b].

The left condition fixes the initial data up to one parameter t (see
Robin.initial_values). Each method looks for the t that makes the
residual of the right condition, F(t), vanish.
"""

import functools
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


def _quiet_overflow(fun):
    """Silence overflow warnings: a shot that blows up returns inf/nan."""
    @functools.wraps(fun)
    def wrapper(*args):
        with np.errstate(over="ignore", invalid="ignore"):
            return fun(*args)
    return wrapper


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


def secant_shooting(f, a, b, left, right, t0, t1, n=10, tol=1e-6,
                    max_iter=50):
    """Nonlinear shooting with the secant method.

    Stops when |F(t_k)| < tol. Iterations counts secant updates.
    """
    def system(x, w):
        return np.array([w[1], f(x, w[0], w[1])])

    @_quiet_overflow
    def evaluate(t):
        init, _ = left.initial_values(t)
        x, u = rk4(system, a, b, np.array(init), n)
        return right.residual(u[-1, 0], u[-1, 1]), x, u

    t_prev, t = t0, t1
    F_prev, _, _ = evaluate(t_prev)
    F, x, u = evaluate(t)
    history = [(t_prev, F_prev), (t, F)]
    iterations = 0
    while abs(F) >= tol and iterations < max_iter:
        if not np.isfinite(F) or F == F_prev:
            break
        t_prev, t = t, t - F * (t - t_prev) / (F - F_prev)
        F_prev = F
        F, x, u = evaluate(t)
        history.append((t, F))
        iterations += 1
    return _result(x, u, t, iterations, bool(abs(F) < tol), history)


def newton_shooting(f, f_y, f_dy, a, b, left, right, t0, n=10, tol=1e-6,
                    max_iter=50):
    """Nonlinear shooting with Newton's method.

    F'(t) comes from the variational equation for z = dy/dt,
        z'' = f_y(x, y, y') z + f_dy(x, y, y') z',
    integrated together with the original equation.
    Stops when |F(t_k)| < tol. Iterations counts Newton updates.
    """
    def system(x, w):
        y, dy, z, dz = w
        return np.array([dy, f(x, y, dy),
                         dz, f_y(x, y, dy) * z + f_dy(x, y, dy) * dz])

    @_quiet_overflow
    def evaluate(t):
        (y0, dy0), (z0, dz0) = left.initial_values(t)
        x, w = rk4(system, a, b, np.array([y0, dy0, z0, dz0]), n)
        F = right.residual(w[-1, 0], w[-1, 1])
        dF = right.alpha * w[-1, 2] + right.beta * w[-1, 3]
        return F, dF, x, w

    t = t0
    F, dF, x, w = evaluate(t)
    history = [(t, F)]
    iterations = 0
    while abs(F) >= tol and iterations < max_iter:
        if not np.isfinite(F) or not np.isfinite(dF) or dF == 0:
            break
        t = t - F / dF
        F, dF, x, w = evaluate(t)
        history.append((t, F))
        iterations += 1
    return _result(x, w, t, iterations, bool(abs(F) < tol), history)
