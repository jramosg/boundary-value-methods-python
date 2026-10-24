"""Shooting and finite differences for two-point boundary value problems."""

from .boundary import Robin, dirichlet, neumann
from .ivp import rk4
from .shooting import (ShootingResult, linear_shooting, newton_shooting,
                       secant_shooting)

__all__ = ["Robin", "dirichlet", "neumann", "rk4", "ShootingResult",
           "linear_shooting", "newton_shooting", "secant_shooting"]

from .finite_difference import (FiniteDifferenceResult, finite_difference_system,
                                linear_finite_difference,
                                nonlinear_finite_difference)
from .linalg import solve_tridiagonal, tridiagonal_matvec

__all__ += ["FiniteDifferenceResult", "finite_difference_system",
            "linear_finite_difference", "nonlinear_finite_difference",
            "solve_tridiagonal", "tridiagonal_matvec"]
