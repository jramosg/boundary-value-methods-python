"""Boundary conditions of the form alpha * y + beta * y' = gamma."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Robin:
    """Boundary condition alpha * y + beta * y' = gamma.

    Dirichlet (beta = 0) and Neumann (alpha = 0) are special cases.
    """

    alpha: float
    beta: float
    gamma: float

    def __post_init__(self):
        if self.alpha == 0 and self.beta == 0:
            raise ValueError("alpha and beta cannot both be zero")

    def residual(self, y, dy):
        """Zero when (y, y') satisfies the condition."""
        return self.alpha * y + self.beta * dy - self.gamma

    def initial_values(self, t):
        """Initial data (y, y') at the left end for shooting parameter t.

        If beta != 0, t is y(a) and y'(a) follows from the condition.
        If beta == 0 (Dirichlet), y(a) is fixed and t is y'(a).
        Also returns the derivatives of (y, y') with respect to t.
        """
        if self.beta != 0:
            dy = (self.gamma - self.alpha * t) / self.beta
            return (t, dy), (1.0, -self.alpha / self.beta)
        return (self.gamma / self.alpha, t), (0.0, 1.0)


def dirichlet(value):
    """Condition y = value."""
    return Robin(1.0, 0.0, value)


def neumann(value):
    """Condition y' = value."""
    return Robin(0.0, 1.0, value)
