# Boundary value problems in Python

[![Python tests](https://github.com/jramosg/boundary-value-methods-python/actions/workflows/tests.yml/badge.svg)](https://github.com/jramosg/boundary-value-methods-python/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Learn **shooting methods and finite differences** for second-order boundary
value problems. Work through equations, run Python notebooks, and measure
accuracy against exact solutions. The NumPy-based `bvp` package implements
linear shooting, secant and Newton shooting, central differences, Newton for
systems, and a tridiagonal Crout factorization.

**[Leer en español](README.es.md)** ·
[Theory and worked examples](https://metodosnumericos.jonramos.dev/en/articulos/frontera-diferencias-finitas/) ·
[Español](https://metodosnumericos.jonramos.dev/es/articulos/frontera-diferencias-finitas/) ·
[Euskara](https://metodosnumericos.jonramos.dev/eu/articulos/frontera-diferencias-finitas/)

Formerly `shooting-methods-python`. Existing `from bvp import ...` imports
and the three shooting notebooks retain their names.

## The problem

We solve two-point boundary value problems (BVPs)

$$y''=f(x,y,y'),\quad x\in[a,b],\qquad
\alpha_a y'(a)+\beta_a y(a)=\gamma_a,\quad
\alpha_b y'(b)+\beta_b y(b)=\gamma_b.$$

These include **Dirichlet**, **Neumann** and **Robin** conditions. Shooting
adjusts initial data and integrates an IVP. Finite differences approximate
all mesh values together and solve an algebraic system. For Robin conditions
we eliminate ghost nodes before constructing the Jacobian.

## Install and run

Python 3.10 or later:

```sh
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[notebooks,test]'
python -m pytest -q
jupyter lab
```

Open `notebooks/04_finite_differences.ipynb` and run all cells. Start Jupyter
from the repository root, using the environment above. Installation is from
this checkout; the commands do not assume a published PyPI release.

## Learning path

| Notebook | Language | What you learn |
|---|---|---|
| [01 Disparo lineal](notebooks/01_disparo_lineal.ipynb) | Spanish | Superposition, general boundary conditions and RK4 convergence |
| [02 Disparo no lineal](notebooks/02_disparo_no_lineal.ipynb) | Spanish | Secant, Newton and the variational equation |
| [03 Convergencia](notebooks/03_convergencia.ipynb) | Spanish | Initial guesses, multiple solutions and `solve_bvp` |
| [04 Finite differences](notebooks/04_finite_differences.ipynb) | English | Taylor formulas, Robin ghost nodes, Jacobian, Crout and mesh refinement |
| [05 Comparing BVP methods](notebooks/05_comparing_methods.ipynb) | English | Linear/nonlinear problems, dense LU, shooting and SciPy with timing methodology |

Prerequisites: derivatives, Taylor series, linear systems and basic NumPy.
The [Spanish finite-difference guide](docs/diferencias-finitas.es.md) explains
the derivation and Python API. The linked wiki covers the theory in Spanish,
Basque and English at equal depth.

## First example: nonlinear Robin conditions

The exact solution is $y=x^3$ for
$y''=xyy'-3y^2+6x$, $3y(0)-2y'(0)=0$,
$2y(1)-y'(1)/3=1$. Compute it without using the exact solution as the initial guess:

```python
import numpy as np
from bvp import Robin, nonlinear_finite_difference

result = nonlinear_finite_difference(
    f=lambda x, y, dy: x*y*dy - 3*y*y + 6*x,
    f_y=lambda x, y, dy: x*dy - 6*y,
    f_dy=lambda x, y, dy: x*y,
    a=0, b=1,
    left=Robin(-2, 3, 0), right=Robin(-1/3, 2, 1),
    initial=lambda x: 0.75*x*x, n=80,
)
assert result.converged
print(result.iterations)  # 4
print(np.max(np.abs(result.y - result.x**3)))  # about 7.02e-5
```

`n` counts subintervals, so the result contains `n+1` nodes. `Robin(alpha,
beta, gamma)` always means **alpha multiplies the derivative**. The solver
accepts scalar callbacks; `initial` accepts the mesh array or an `n+1` vector.

## Algorithms and API

| API | Purpose |
|---|---|
| `Robin`, `dirichlet`, `neumann` | Boundary data shared by both method families |
| `rk4` | Fixed-step fourth-order initial-value integration |
| `linear_shooting` | Linear BVP by superposition |
| `secant_shooting`, `newton_shooting` | Scalar shooting residual; Newton uses sensitivities |
| `linear_finite_difference` | $y''=p(x)y'+q(x)y+r(x)$ with central differences |
| `nonlinear_finite_difference` | Full-step Newton with an analytic tridiagonal Jacobian |
| `finite_difference_system` | Inspect the residual and Jacobian diagonals |
| `solve_tridiagonal` | Crout, or `method="dense"` for NumPy's pivoted LU |

Both finite-difference solvers accept `method="crout"` or `method="dense"`.
Crout uses linear work/storage and does not pivot. A zero pivot can make
Crout fail even when the matrix is invertible; the dense solver can handle
that case. Pure Neumann problems may be singular or incompatible. Newton
is local and can fail or reach different roots from different starts. Check
`result.converged`; no method proves uniqueness by numerical convergence.

## Compare accuracy and runtime

```sh
python examples/compare_methods.py --n 80 --repeat 7 --csv results/comparison.csv
```

The experiment covers the nonlinear cubic example and the linear equation
$y''=xy'+3y$ with the same Robin conditions. It reports errors on a common
query mesh, iterations, mesh sizes, median runtime and quartiles, and saves
environment metadata beside the CSV. Plotting and warm-up are excluded;
assembly and the complete solve are included. The linear reference comes
from a separate high-accuracy IVP integration.

Equal initial mesh sizes do **not** imply equal accuracy or equal work:
finite differences have expected global order two under smoothness and
stability, shooting uses fixed-step RK4, and `scipy.integrate.solve_bvp`
uses adaptive fourth-order collocation. Its residual tolerance differs
from Newton's stopping tolerances. NumPy's compiled dense solver may beat
a Python Crout loop at small sizes despite the higher asymptotic cost.
Timings depend on the computer and numerical libraries.

## Tests and contributions

Tests cover manufactured solutions, second-order mesh convergence,
Dirichlet/Neumann/Robin combinations, boundary Jacobians, unsafe pivots and
non-convergence. CI runs Python 3.10, 3.12 and 3.14 and executes notebooks.
See [CONTRIBUTING.md](CONTRIBUTING.md) to add a method or reproducible example.
Educational implementations prioritize readable equations and diagnostics;
for demanding applications compare against a maintained numerical solver.

## References

- [NumPy linear solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html)
- [SciPy boundary value solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html)
- [Boundary value theory and derivations](https://metodosnumericos.jonramos.dev/en/articulos/frontera-introduccion/)

## License

[MIT](LICENSE).
