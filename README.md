# Métodos de disparo en Python

Métodos de disparo para problemas de contorno de segundo orden

$$
y'' = f(x, y, y'), \qquad x \in [a, b], \qquad
\alpha_a y'(a) + \beta_a y(a) = \gamma_a, \quad
\alpha_b y'(b) + \beta_b y(b) = \gamma_b,
$$

con condiciones de Dirichlet, Neumann o naturales (Robin). Explicaciones en
español, código en Python. *English summary below.*

Es el código que acompaña a la sección de
[problemas de frontera](https://metodosnumericos.jonramos.dev/es/articulos/frontera-introduccion/) de
[metodosnumericos.jonramos.dev](https://metodosnumericos.jonramos.dev/es/), donde están la teoría, las
deducciones y los ejercicios resueltos con la misma notación.

## Contenido

| Cuaderno | Qué cubre | Teoría |
|---|---|---|
| [01_disparo_lineal](notebooks/01_disparo_lineal.ipynb) | Disparo lineal $y = y_1 + s\,y_2$, condiciones naturales, orden de convergencia de RK4 | [Disparo lineal](https://metodosnumericos.jonramos.dev/es/articulos/frontera-disparo-lineal/), [condiciones naturales y mixtas](https://metodosnumericos.jonramos.dev/es/articulos/frontera-disparo-lineal-condiciones-generales/) |
| [02_disparo_no_lineal](notebooks/02_disparo_no_lineal.ipynb) | Disparo no lineal con secante y con Newton (ecuación variacional), raíces dobles | [Disparo no lineal](https://metodosnumericos.jonramos.dev/es/articulos/frontera-disparo-no-lineal/), [Newton](https://metodosnumericos.jonramos.dev/es/articulos/frontera-disparo-newton/) |
| [03_convergencia](notebooks/03_convergencia.ipynb) | Convergencia según los valores iniciales, problemas con dos soluciones, comparación con `scipy.integrate.solve_bvp` | [Disparo no lineal](https://metodosnumericos.jonramos.dev/es/articulos/frontera-disparo-no-lineal/) |

Todos los ejemplos tienen solución exacta conocida, para poder medir el
error.

El paquete `bvp` contiene las implementaciones:

- `rk4`: Runge-Kutta clásico de orden 4 con paso fijo.
- `Robin`, `dirichlet`, `neumann`: condiciones de contorno
  $\alpha y' + \beta y = \gamma$ (Dirichlet: $\alpha = 0$; Neumann:
  $\beta = 0$).
- `linear_shooting`: disparo lineal $y = y_1 + s\,y_2$ para
  $y'' = p(x) y' + q(x) y + r(x)$.
- `secant_shooting`, `newton_shooting`: disparo no lineal.

```python
import numpy as np
from bvp import Robin, newton_shooting

# y'' = -(y')^2,  y'(0) + y(0) = 1,  -y'(1) + y(1) = ln 2 - 1/2
res = newton_shooting(
    f=lambda x, y, dy: -dy**2,
    f_y=lambda x, y, dy: 0.0,
    f_dy=lambda x, y, dy: -2 * dy,
    a=0, b=1,
    left=Robin(1, 1, 1), right=Robin(-1, 1, np.log(2) - 0.5),
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
$y'' = f(x, y, y')$ with Dirichlet, Neumann or Robin conditions
$\alpha y' + \beta y = \gamma$:
linear shooting by superposition, and nonlinear shooting with the secant
method and with Newton's method (using the variational equation for
$F'(t)$). The notebooks (in Spanish) study convergence with respect to the
initial guesses, double roots, problems with two solutions, and compare
against `scipy.integrate.solve_bvp`. The `bvp` package and its docstrings
are in English. Theory, derivations and solved exercises, in English too,
are on [metodosnumericos.jonramos.dev](https://metodosnumericos.jonramos.dev/en/articulos/frontera-introduccion/).

## Licencia / License

[MIT](LICENSE)
