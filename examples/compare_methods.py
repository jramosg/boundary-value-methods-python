"""Reproducible accuracy/timing comparison; run from an editable installation."""

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy import __version__ as scipy_version
from scipy.integrate import solve_bvp, solve_ivp

from bvp import (Robin, linear_finite_difference, linear_shooting,
                 newton_shooting, nonlinear_finite_difference)

LEFT, RIGHT = Robin(-2, 3, 0), Robin(-1/3, 2, 1)


def f(x, y, dy):
    return x*y*dy-3*y*y+6*x


def fy(x, y, dy):
    return x*dy-6*y


def fdy(x, y, dy):
    return x*y


def initial(x):
    return .75*x*x


def reference_bvp(n, *, linear=False):
    x = np.linspace(0, 1, n+1)
    def ode(x, v):
        return np.vstack((v[1], x*v[1]+3*v[0] if linear else f(x, v[0], v[1])))
    def bc(a, b):
        return np.array((LEFT.residual(*a), RIGHT.residual(*b)))
    result = solve_bvp(ode, bc, x, np.vstack((initial(x), 1.5*x)),
                       tol=1e-9, max_nodes=10000)
    if not result.success:
        raise RuntimeError(result.message)
    return result


def run_comparison(n=80, repeat=7):
    """Return rows; medians include setup, assembly and complete solution.

    Warm-up is excluded. Accuracy is measured on the same n+1 query nodes.
    Finite differences use full-step Newton; shooting uses fixed-step RK4;
    solve_bvp adapts its mesh with a residual-based tolerance. Equal n is
    not equal accuracy or equal work. No speed ranking is asserted.
    """
    if n < 2 or repeat < 1:
        raise ValueError("n >= 2 and repeat >= 1 are required")
    x = np.linspace(0, 1, n+1)
    # Independent high-accuracy linear reference: y = u / (2u(1)-u'(1)/3).
    fundamental = solve_ivp(lambda x, v: [v[1], x*v[1]+3*v[0]],
        (0, 1), [1, 1.5], rtol=1e-12, atol=1e-14, dense_output=True)
    if not fundamental.success:
        raise RuntimeError(fundamental.message)
    denominator = 2*fundamental.y[0, -1]-fundamental.y[1, -1]/3
    exact_linear = fundamental.sol(x)[0]/denominator
    cases = []
    for method in ("crout", "dense"):
        cases.append(("nonlinear", "FD + Newton + " + method,
            lambda method=method: nonlinear_finite_difference(
                f, fy, fdy, 0, 1, LEFT, RIGHT, initial, n=n, method=method)))
        cases.append(("linear", "FD + " + method,
            lambda method=method: linear_finite_difference(
                lambda x: x, lambda x: 3, lambda x: 0,
                0, 1, LEFT, RIGHT, n=n, method=method)))
    cases += [
        ("nonlinear", "Newton shooting (RK4)", lambda: newton_shooting(
            f, fy, fdy, 0, 1, LEFT, RIGHT, t0=.1, n=n, tol=1e-10)),
        ("linear", "Linear shooting (RK4)", lambda: linear_shooting(
            lambda x: x, lambda x: 3, lambda x: 0, 0, 1, LEFT, RIGHT, n=n)),
        ("nonlinear", "solve_bvp", lambda: reference_bvp(n)),
        ("linear", "solve_bvp", lambda: reference_bvp(n, linear=True)),
    ]
    rows = []
    for problem, name, solve in cases:
        solve()  # Warm-up, excluded from timings.
        times = []
        for _ in range(repeat):
            start = perf_counter()
            result = solve()
            times.append(perf_counter()-start)
        if name == "solve_bvp":
            values, nodes = result.sol(x)[0], len(result.x)
            iterations = result.niter
        else:
            if not result.converged:
                raise RuntimeError(f"{name} did not converge")
            values, nodes, iterations = result.y, len(result.x), result.iterations
        reference = x**3 if problem == "nonlinear" else exact_linear
        rows.append({"problem": problem, "method": name,
            "nodes": nodes, "iterations": iterations,
            "max_error": float(np.max(np.abs(values-reference))),
            "median_ms": 1000*median(times),
            "q25_ms": 1000*float(np.quantile(times, .25)),
            "q75_ms": 1000*float(np.quantile(times, .75))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = run_comparison(args.n, args.repeat)
    environment = {"python": sys.version.split()[0], "numpy": np.__version__,
                   "scipy": scipy_version, "platform": platform.platform(),
                   "n": args.n, "repeat": args.repeat}
    print(json.dumps(environment, indent=2))
    print("problem | method | nodes | iterations | max error | median ms [Q25,Q75]")
    for r in rows:
        print(f"{r['problem']} | {r['method']} | {r['nodes']} | {r['iterations']} | "
              f"{r['max_error']:.3e} | {r['median_ms']:.3f} "
              f"[{r['q25_ms']:.3f}, {r['q75_ms']:.3f}]")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        args.csv.with_suffix('.environment.json').write_text(json.dumps(environment, indent=2)+'\n')


if __name__ == "__main__":
    main()
