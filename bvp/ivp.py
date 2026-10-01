"""Fixed-step integrators for initial value problems."""

import numpy as np


def rk4(fun, a, b, u0, n):
    """Integrate u' = fun(x, u) on [a, b] with n steps of classic RK4.

    Returns the nodes x, shape (n + 1,), and the approximations u,
    shape (n + 1, len(u0)), with u[i] ~ u(x[i]).
    """
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    u = np.empty((n + 1, len(u0)))
    u[0] = u0
    for i in range(n):
        xi, ui = x[i], u[i]
        k1 = fun(xi, ui)
        k2 = fun(xi + h / 2, ui + h / 2 * k1)
        k3 = fun(xi + h / 2, ui + h / 2 * k2)
        k4 = fun(xi + h, ui + h * k3)
        u[i + 1] = ui + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x, u
