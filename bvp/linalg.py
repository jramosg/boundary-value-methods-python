"""Linear algebra for tridiagonal finite-difference systems."""

import numpy as np


def _diagonals(a, b, c, d):
    arrays = tuple(np.asarray(v, dtype=float) for v in (a, b, c, d))
    if any(v.ndim != 1 for v in arrays):
        raise ValueError("diagonals and right-hand side must be vectors")
    a, b, c, d = arrays
    if not len(a) or (len(b), len(c), len(d)) != (len(a)-1, len(a)-1, len(a)):
        raise ValueError("expected lengths m, m-1, m-1, m")
    if not all(np.all(np.isfinite(v)) for v in arrays):
        raise ValueError("system contains non-finite values")
    return arrays


def tridiagonal_matvec(a, b, c, x):
    """Return A @ x without assembling the full matrix."""
    a, b, c, x = _diagonals(a, b, c, x)
    out = a * x
    out[:-1] += b * x[1:]
    out[1:] += c * x[:-1]
    return out


def solve_tridiagonal(a, b, c, d, *, method="crout"):
    """Solve A x = d; a main, b upper, c lower diagonal.

    Crout factors A = L U with unit diagonal in U, in O(m) work and
    storage. It does not pivot. A zero or unsafe pivot raises LinAlgError;
    use method='dense' for NumPy's general LU solver with pivoting.
    The dense path includes matrix assembly and needs O(m**2) storage.
    Input arrays are not modified. A matrix can be invertible yet fail
    the no-pivot Crout factorization.
    """
    a, b, c, d = _diagonals(a, b, c, d)
    if method == "dense":
        matrix = np.diag(a) + np.diag(b, 1) + np.diag(c, -1)
        return np.linalg.solve(matrix, d)
    if method != "crout":
        raise ValueError("method must be 'crout' or 'dense'")

    m = len(a)
    diagonal = np.empty(m)
    upper = np.empty(m-1)
    scale = max(np.max(np.abs(v), initial=0.0) for v in (a, b, c))
    threshold = np.finfo(float).eps * scale

    # Factorization: subdiagonal(L) = c and diagonal(U) = 1.
    for i in range(m):
        diagonal[i] = a[i] if i == 0 else a[i] - c[i-1] * upper[i-1]
        if not np.isfinite(diagonal[i]) or abs(diagonal[i]) <= threshold:
            raise np.linalg.LinAlgError("unsafe Crout pivot; try method='dense'")
        if i < m-1:
            upper[i] = b[i] / diagonal[i]

    # Forward substitution L z = d.
    z = np.empty(m)
    for i in range(m):
        rhs = d[i] if i == 0 else d[i] - c[i-1] * z[i-1]
        z[i] = rhs / diagonal[i]

    # Back substitution U x = z.
    result = z.copy()
    for i in range(m-2, -1, -1):
        result[i] -= upper[i] * result[i+1]
    return result
