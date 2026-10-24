"""Second-order central differences with Dirichlet, Neumann or Robin data.

Scalar callbacks use f(x, y, dy), f_y(x, y, dy), f_dy(x, y, dy).
Robin ghost values are eliminated before differentiating the residual.
A local Newton iteration is not a proof of existence or uniqueness.
"""

from dataclasses import dataclass, field

import numpy as np

from .linalg import solve_tridiagonal


@dataclass
class FiniteDifferenceResult:
    x: np.ndarray
    y: np.ndarray
    iterations: int
    converged: bool
    residual_norm: float
    history: list = field(default_factory=list)


def _mesh(a, b, n):
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer >= 2 (number of subintervals)")
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        raise ValueError("expected finite endpoints with a < b")
    return np.linspace(a, b, n+1)


def finite_difference_system(f, f_y, f_dy, x, y, left, right):
    """Return scaled residual F and Jacobian diagonals (a, b, c).

    Interior: F_i = 2 y_i - y_{i-1} - y_{i+1} + h**2 f(x_i,y_i,Dy_i).
    At Robin ends use Dy = (gamma-beta*y)/alpha and eliminate the ghost.
    Dirichlet rows are h**2 * (y-value), so ||F||inf/h**2 includes their
    value residual. Robin endpoint rows use the eliminated ODE, not an
    independent boundary residual. All n+1 endpoint/interior values are
    retained; Dirichlet corrections are constrained by their rows.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 3:
        raise ValueError("x and y must be matching vectors of length >= 3")
    h = x[1]-x[0]
    if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or h <= 0
            or not np.allclose(np.diff(x), h, rtol=1e-10, atol=1e-14*h)):
        raise ValueError("expected a finite, increasing, uniform mesh and finite y")

    m = len(x)
    F, main = np.empty(m), np.empty(m)
    upper, lower = np.zeros(m-1), np.zeros(m-1)
    for i in range(1, m-1):
        dy = (y[i+1]-y[i-1])/(2*h)
        F[i] = 2*y[i]-y[i-1]-y[i+1] + h*h*f(x[i], y[i], dy)
        main[i] = 2 + h*h*f_y(x[i], y[i], dy)
        upper[i] = -1 + h/2*f_dy(x[i], y[i], dy)
        lower[i-1] = -1 - h/2*f_dy(x[i], y[i], dy)

    for i, cond, side in ((0, left, -1), (m-1, right, 1)):
        if cond.alpha == 0:
            F[i] = h*h*(y[i]-cond.gamma/cond.beta)
            main[i] = h*h
            continue
        slope = (cond.gamma-cond.beta*y[i])/cond.alpha
        # side=-1: left ghost=y[1]-2h*slope; side=+1: right ghost.
        neighbor = y[1] if i == 0 else y[-2]
        F[i] = 2*y[i]-2*neighbor-side*2*h*slope + h*h*f(x[i], y[i], slope)
        main[i] = (2+side*2*h*cond.beta/cond.alpha
                   + h*h*(f_y(x[i], y[i], slope)
                          - cond.beta/cond.alpha*f_dy(x[i], y[i], slope)))
        if i == 0:
            upper[0] = -2
        else:
            lower[-1] = -2
    if not all(np.all(np.isfinite(v)) for v in (F, main, upper, lower)):
        raise FloatingPointError("non-finite residual or Jacobian")
    return F, main, upper, lower


def linear_finite_difference(p, q, r, a, b, left, right, *, n=80,
                             method="crout"):
    """Solve y''=p(x)y'+q(x)y+r(x) on a uniform n-subinterval mesh."""
    x = _mesh(a, b, n)
    f = lambda x, y, dy: p(x)*dy + q(x)*y + r(x)
    fy = lambda x, y, dy: q(x)
    fdy = lambda x, y, dy: p(x)
    F, main, upper, lower = finite_difference_system(
        f, fy, fdy, x, np.zeros(n+1), left, right)
    y = solve_tridiagonal(main, upper, lower, -F, method=method)
    final = finite_difference_system(f, fy, fdy, x, y, left, right)[0]
    residual = float(np.linalg.norm(final, np.inf)/(x[1]-x[0])**2)
    return FiniteDifferenceResult(x, y, 1, True, residual)


def nonlinear_finite_difference(f, f_y, f_dy, a, b, left, right, initial,
                                *, n=80, method="crout", tol=1e-10,
                                residual_tol=1e-9, max_iter=50):
    """Solve y''=f(x,y,y') by full-step Newton and a chosen linear solver.

    initial is an n+1 vector or a callable initial(x) evaluated on the mesh.
    A successful iteration requires both a small correction and small
    scaled residual. Failure at max_iter returns converged=False; unsafe
    pivots/non-finite values raise exceptions rather than report success.
    Newton has no global convergence guarantee or automatic line search.
    """
    x = _mesh(a, b, n)
    if (not np.isfinite(tol) or not np.isfinite(residual_tol)
            or tol <= 0 or residual_tol <= 0):
        raise ValueError("tolerances must be finite and positive")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be an integer >= 1")
    y = np.array(initial(x) if callable(initial) else initial, dtype=float, copy=True)
    if y.shape != x.shape:
        raise ValueError("initial must supply n+1 values")
    h = x[1]-x[0]
    history = []
    for k in range(1, max_iter+1):
        F, main, upper, lower = finite_difference_system(f, f_y, f_dy, x, y, left, right)
        step = solve_tridiagonal(main, upper, lower, -F, method=method)
        y += step
        F = finite_difference_system(f, f_y, f_dy, x, y, left, right)[0]
        residual = float(np.linalg.norm(F, np.inf)/h**2)
        increment = float(np.linalg.norm(step, np.inf))
        history.append({"iteration": k, "step_norm": increment,
                        "residual_norm": residual})
        if increment <= tol*(1+np.linalg.norm(y, np.inf)) and residual <= residual_tol:
            return FiniteDifferenceResult(x, y, k, True, residual, history)
    return FiniteDifferenceResult(x, y, max_iter, False, residual, history)
