import numpy as np

from bvp import Robin, dirichlet, linear_shooting


# y'' = -2/x y' + 2/x^2 y + sin(ln x)/x^2, y(1) = 1, y(2) = 2
# (Burden & Faires, linear shooting example)
def burden_linear_exact(x):
    c2 = (8 - 12 * np.sin(np.log(2)) - 4 * np.cos(np.log(2))) / 70
    c1 = 11 / 10 - c2
    return (c1 * x + c2 / x**2 - 3 / 10 * np.sin(np.log(x))
            - 1 / 10 * np.cos(np.log(x)))


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
