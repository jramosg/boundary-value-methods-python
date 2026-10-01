"""Boundary conditions of the form alpha * y' + beta * y = gamma."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Robin:
    """Natural (Robin) condition alpha * y' + beta * y = gamma.

    Dirichlet (alpha = 0) and Neumann (beta = 0) are special cases.
    """

    alpha: float
    beta: float
    gamma: float

    def __post_init__(self):
        if self.alpha == 0 and self.beta == 0:
            raise ValueError("alpha and beta cannot both be zero")

    def residual(self, y, dy):
        """Zero when (y, y') satisfies the condition."""
        return self.alpha * dy + self.beta * y - self.gamma

    def initial_values(self, t):
        """Initial data (y, y') at the left end for shooting parameter t.

        If alpha != 0, t is y(a) and y'(a) follows from the condition.
        If alpha == 0 (Dirichlet), y(a) is fixed and t is y'(a).
        Also returns the derivatives of (y, y') with respect to t.
        """
        if self.alpha != 0:
            dy = (self.gamma - self.beta * t) / self.alpha
            return (t, dy), (1.0, -self.beta / self.alpha)
        return (self.gamma / self.beta, t), (0.0, 1.0)

    def linear_initial_values(self):
        """Initial data (y, y') of y1 and y2 for linear shooting.

        y1 satisfies the condition and y2 its homogeneous version, so
        y1 + s y2 satisfies it for every s.
        """
        if self.alpha != 0:
            return (0.0, self.gamma / self.alpha), (self.alpha, -self.beta)
        return (self.gamma / self.beta, 0.0), (0.0, 1.0)


def dirichlet(value):
    """Condition y = value."""
    return Robin(0.0, 1.0, value)


def neumann(value):
    """Condition y' = value."""
    return Robin(1.0, 0.0, value)
