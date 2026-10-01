import pytest

from bvp import Robin, dirichlet, neumann


def test_robin_initial_values():
    (y, dy), (dy_dt, ddy_dt) = Robin(2, 1, 0.25).initial_values(0.5)
    assert (y, dy) == (0.5, -0.75)
    assert (dy_dt, ddy_dt) == (1.0, -2.0)
    assert dirichlet(3).initial_values(7) == ((3, 7), (0.0, 1.0))
    assert neumann(3).initial_values(7)[0] == (7, 3)


def test_robin_rejects_empty_condition():
    with pytest.raises(ValueError):
        Robin(0, 0, 1)
