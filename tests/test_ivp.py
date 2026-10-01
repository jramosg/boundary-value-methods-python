import numpy as np

from bvp import rk4


def test_rk4_is_fourth_order():
    def errors(n):
        x, u = rk4(lambda x, u: u, 0, 1, np.array([1.0]), n)
        return abs(u[-1, 0] - np.e)

    ratio = errors(10) / errors(20)
    assert 14 < ratio < 18
