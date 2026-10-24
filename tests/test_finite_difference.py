import numpy as np
import pytest

from bvp import (Robin, dirichlet, neumann, finite_difference_system,
                 linear_finite_difference, nonlinear_finite_difference,
                 solve_tridiagonal, tridiagonal_matvec)


def cubic_f(x, y, dy):
    return x*y*dy - 3*y*y + 6*x


def cubic_fy(x, y, dy):
    return x*dy - 6*y


def cubic_fdy(x, y, dy):
    return x*y


LEFT, RIGHT = Robin(-2, 3, 0), Robin(-1/3, 2, 1)


def test_crout_known_solution_and_pivoting_difference():
    rng = np.random.default_rng(7)
    a = np.full(30, 4.0)
    b, c = rng.uniform(-1, 1, (2, 29))
    expected = rng.normal(size=30)
    d = tridiagonal_matvec(a, b, c, expected)
    for method in ('crout', 'dense'):
        np.testing.assert_allclose(solve_tridiagonal(a, b, c, d, method=method), expected)
    assert solve_tridiagonal([2], [], [], [6])[0] == 3
    with pytest.raises(np.linalg.LinAlgError):
        solve_tridiagonal([0, 1], [1], [1], [1, 2])
    np.testing.assert_allclose(
        solve_tridiagonal([0, 1], [1], [1], [1, 2], method='dense'), [1, 1])


@pytest.mark.parametrize('left,right', [(LEFT, RIGHT),
    (dirichlet(0), RIGHT), (LEFT, dirichlet(1)), (neumann(0), RIGHT)])
def test_analytic_jacobian_including_boundary_chain_rule(left, right):
    x = np.linspace(0, 1, 13)
    y = 0.2 + x*x
    F, a, b, c = finite_difference_system(cubic_f, cubic_fy, cubic_fdy, x, y, left, right)
    analytic = np.diag(a)+np.diag(b, 1)+np.diag(c, -1)
    numeric = np.empty_like(analytic)
    for j in range(len(y)):
        delta = np.zeros_like(y)
        delta[j] = 1e-6
        plus = finite_difference_system(cubic_f, cubic_fy, cubic_fdy, x, y+delta, left, right)[0]
        minus = finite_difference_system(cubic_f, cubic_fy, cubic_fdy, x, y-delta, left, right)[0]
        numeric[:, j] = (plus-minus)/(2e-6)
    np.testing.assert_allclose(analytic, numeric, atol=1e-9)


@pytest.mark.parametrize('left,right', [
    (dirichlet(1), dirichlet(np.e)),
    (dirichlet(1), neumann(np.e)),
    (Robin(-2, 3, 1), Robin(-1/3, 2, 5*np.e/3)),
])
def test_linear_manufactured_solution_second_order(left, right):
    errors = []
    for n in (20, 40, 80):
        sol = linear_finite_difference(lambda x: x, lambda x: 3,
            lambda x: -(x+2)*np.exp(x), 0, 1, left, right, n=n)
        assert sol.converged and sol.residual_norm < 1e-9
        errors.append(np.max(np.abs(sol.y-np.exp(sol.x))))
    assert 3.8 < errors[0]/errors[1] < 4.2
    assert 3.8 < errors[1]/errors[2] < 4.2


def test_nonlinear_crout_dense_and_mesh_convergence():
    errors = []
    for n in (20, 40, 80):
        sol = nonlinear_finite_difference(cubic_f, cubic_fy, cubic_fdy,
            0, 1, LEFT, RIGHT, lambda x: .75*x*x, n=n)
        assert sol.converged and sol.residual_norm < 1e-9
        errors.append(np.max(np.abs(sol.y-sol.x**3)))
    dense = nonlinear_finite_difference(cubic_f, cubic_fy, cubic_fdy,
        0, 1, LEFT, RIGHT, lambda x: .75*x*x, method='dense')
    np.testing.assert_allclose(dense.y, sol.y, atol=1e-11)
    assert 3.9 < errors[1]/errors[2] < 4.1
    assert errors[-1] == pytest.approx(7.024513e-5, rel=1e-6)
    assert sol.iterations == 4


def test_newton_failure_and_bad_input():
    result = nonlinear_finite_difference(cubic_f, cubic_fy, cubic_fdy,
        0, 1, LEFT, RIGHT, lambda x: .75*x*x, max_iter=1)
    assert not result.converged
    for n in (0, 1, 2.5):
        with pytest.raises(ValueError):
            linear_finite_difference(lambda x: 0, lambda x: 0, lambda x: 0,
                0, 1, dirichlet(0), dirichlet(1), n=n)
    with pytest.raises(ValueError):
        solve_tridiagonal([1, 2], [1, 2], [1], [1, 2])
    with pytest.raises(np.linalg.LinAlgError):
        linear_finite_difference(lambda x: 0, lambda x: 0, lambda x: 0,
            0, 1, neumann(0), neumann(0))
