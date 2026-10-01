import numpy as np
import pytest

from bvp import Robin, dirichlet, linear_shooting, secant_shooting


# y'' = -2/x y' + 2/x^2 y + sin(ln x)/x^2, y(1) = 1, y(2) = 2
# (Burden & Faires, linear shooting example)
def burden_linear_exact(x):
    c2 = (8 - 12 * np.sin(np.log(2)) - 4 * np.cos(np.log(2))) / 70
    c1 = 11 / 10 - c2
    return (c1 * x + c2 / x**2 - 3 / 10 * np.sin(np.log(x))
            - 1 / 10 * np.cos(np.log(x)))


# y'' = (32 + 2x^3 - y y') / 8, y(1) = 17, y(3) = 43/3, y = x^2 + 16/x
# (Burden & Faires, nonlinear shooting example)
def burden_f(x, y, dy):
    return (32 + 2 * x**3 - y * dy) / 8


# y'' = -(y')^2, y(0) + y'(0) = 1, y(1) - y'(1) = ln 2 - 1/2,
# y = ln(1 + x), unique solution
def log_f(x, y, dy):
    return -dy**2


LOG_LEFT = Robin(1, 1, 1)
LOG_RIGHT = Robin(1, -1, np.log(2) - 0.5)


def test_linear_shooting_dirichlet():
    res = linear_shooting(lambda x: -2 / x, lambda x: 2 / x**2,
                          lambda x: np.sin(np.log(x)) / x**2,
                          1, 2, dirichlet(1), dirichlet(2), n=10)
    assert np.max(np.abs(res.y - burden_linear_exact(res.x))) < 1e-6


def test_linear_shooting_robin():
    # y'' = -y, y(0) + y'(0) = 2, 2 y(pi/2) + y'(pi/2) = 1,
    # y = sin x + cos x
    res = linear_shooting(lambda x: 0, lambda x: -1, lambda x: 0,
                          0, np.pi / 2, Robin(1, 1, 2), Robin(2, 1, 1),
                          n=10)
    exact = np.sin(res.x) + np.cos(res.x)
    assert np.max(np.abs(res.y - exact)) < 1e-5


def test_secant_dirichlet():
    res = secant_shooting(burden_f, 1, 3, dirichlet(17), dirichlet(43 / 3),
                          -4 / 3, -1, n=20, tol=1e-6)
    assert res.converged
    exact = res.x**2 + 16 / res.x
    assert np.max(np.abs(res.y - exact)) < 1e-4
    assert res.t == pytest.approx(-14, abs=1e-3)


def test_secant_robin():
    res = secant_shooting(log_f, 0, 1, LOG_LEFT, LOG_RIGHT, 0.5, 0.4,
                          n=10, tol=1e-8)
    assert res.converged
    assert np.max(np.abs(res.y - np.log1p(res.x))) < 1e-5
    assert np.max(np.abs(res.dy - 1 / (1 + res.x))) < 1e-5


def test_secant_divergent_start_reports_failure():
    # t > 2 gives y'(0) < -1 and the solution blows up before x = 1
    res = secant_shooting(log_f, 0, 1, LOG_LEFT, LOG_RIGHT, 5.0, 6.0)
    assert not res.converged
