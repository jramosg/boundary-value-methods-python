# Problemas de contorno: métodos de disparo

Métodos numéricos para problemas de contorno de segundo orden

$$
y'' = f(x, y, y'), \qquad x \in [a, b], \qquad
\alpha_1 y(a) + \beta_1 y'(a) = \gamma_1, \quad
\alpha_2 y(b) + \beta_2 y'(b) = \gamma_2,
$$

con condiciones de Dirichlet, Neumann o Robin. Explicaciones en español,
código en Python. *English summary below.*

## Contenido

| Cuaderno | Qué cubre |
|---|---|
| [01_disparo_lineal](notebooks/01_disparo_lineal.ipynb) | Disparo lineal por superposición, condiciones de Robin, orden de convergencia de RK4 |
| [02_disparo_no_lineal](notebooks/02_disparo_no_lineal.ipynb) | Disparo no lineal con secante y con Newton (ecuación variacional), raíces dobles |
| [03_convergencia](notebooks/03_convergencia.ipynb) | Convergencia según los valores iniciales, problemas con dos soluciones, comparación con `scipy.integrate.solve_bvp` |

Todos los ejemplos tienen solución exacta conocida, para poder medir el
error.

El paquete `bvp` contiene las implementaciones:

- `rk4`: Runge-Kutta clásico de orden 4 con paso fijo.
- `Robin`, `dirichlet`, `neumann`: condiciones de contorno
  $\alpha y + \beta y' = \gamma$.
- `linear_shooting`: disparo lineal para
  $y'' = p(x) y' + q(x) y + r(x)$.
- `secant_shooting`, `newton_shooting`: disparo no lineal.

```python
import numpy as np
from bvp import Robin, newton_shooting

# y'' = -(y')^2,  y(0) + y'(0) = 1,  y(1) - y'(1) = ln 2 - 1/2
res = newton_shooting(
    f=lambda x, y, dy: -dy**2,
    f_y=lambda x, y, dy: 0.0,
    f_dy=lambda x, y, dy: -2 * dy,
    a=0, b=1,
    left=Robin(1, 1, 1), right=Robin(1, -1, np.log(2) - 0.5),
    t0=1.0, n=10, tol=1e-6,
)
print(res.converged, res.iterations, res.t)   # True 3 2.5e-06
```

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[notebooks,test]'
pytest
jupyter lab notebooks/
```

## English summary

Shooting methods for two-point boundary value problems
$y'' = f(x, y, y')$ with Dirichlet, Neumann or Robin conditions:
linear shooting by superposition, and nonlinear shooting with the secant
method and with Newton's method (using the variational equation for
$F'(t)$). The notebooks (in Spanish) study convergence with respect to the
initial guesses, double roots, problems with two solutions, and compare
against `scipy.integrate.solve_bvp`. The `bvp` package and its docstrings
are in English.

## Licencia / License

[MIT](LICENSE)
