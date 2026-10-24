# Problemas de contorno en Python

Aprende a resolver problemas de frontera mediante **disparo y diferencias
finitas**. El paquete `bvp` incluye disparo lineal, secante, Newton con ecuación
variacional, diferencias centrales, Newton para sistemas y Crout tridiagonal.

[English README](README.md) ·
[Teoría en la wiki](https://metodosnumericos.jonramos.dev/es/articulos/frontera-diferencias-finitas/) ·
[Guía de diferencias finitas](docs/diferencias-finitas.es.md)

Antes el repositorio se llamaba `shooting-methods-python`. Conservamos el
paquete `bvp`, sus importaciones de disparo y los tres cuadernos originales.

## Instalación

Con Python 3.10 o posterior, desde la raíz del repositorio:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebooks,test]'
python -m pytest -q
jupyter lab
```

En PowerShell activa el entorno con `.venv\Scripts\Activate.ps1`.
La instalación usa este repositorio; no presupone una publicación en PyPI.

## Recorrido para aprender

| Cuaderno | Contenido | Idioma |
|---|---|---|
| [01](notebooks/01_disparo_lineal.ipynb) | Disparo lineal, superposición y condiciones generales | Español |
| [02](notebooks/02_disparo_no_lineal.ipynb) | Disparo con secante y Newton | Español |
| [03](notebooks/03_convergencia.ipynb) | Valores iniciales, varias soluciones y `solve_bvp` | Español |
| [04](notebooks/04_finite_differences.ipynb) | Taylor, nodos ficticios, jacobiana, Crout y convergencia | Inglés |
| [05](notebooks/05_comparing_methods.ipynb) | Disparo, diferencias finitas, LU densa y SciPy | Inglés |

La guía en español desarrolla las fórmulas de los nuevos cuadernos. La wiki
ofrece teoría, deducciones y ejemplos en español, euskera e inglés.

## Ejecutar las comparaciones

```sh
python examples/compare_methods.py --n 80 --repeat 7 --csv results/comparison.csv
```

Comparamos el problema no lineal $y''=xyy'-3y^2+6x$, con solución $x^3$,
y el lineal $y''=xy'+3y$, ambos con las condiciones
$3y(0)-2y'(0)=0$ y $2y(1)-y'(1)/3=1$. Para el caso lineal calculamos una
referencia mediante un PVI con tolerancias estrictas.

Mostramos errores en una malla común, tamaños de malla, iteraciones,
medianas de tiempo y cuartiles. Cada medición incluye la preparación y la
resolución completa, y excluye las gráficas. Guardamos las versiones del
entorno junto al CSV. Igualar el número inicial de nodos no iguala precisión
ni trabajo: RK4, diferencias finitas y colocación adaptativa usan órdenes
y controles de error distintos.

Crout no usa pivoteo y puede fallar incluso con una matriz invertible.
Newton es local; comprueba `result.converged` y no deduzcas unicidad de una
sola ejecución. Las pruebas verifican soluciones fabricadas, convergencia,
jacobianas y casos de fallo. Consulta la [API](README.md#algorithms-and-api)
y las [normas para contribuir](CONTRIBUTING.md).

Licencia [MIT](LICENSE).
