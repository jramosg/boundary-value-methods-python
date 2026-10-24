# Diferencias finitas para problemas de frontera

Buscamos $y$ en $[a,b]$ con $y''=f(x,y,y')$ y una condición
$\alpha y'+\beta y=\gamma$ en cada extremo. En esta notación $\alpha$
multiplica a la derivada. Con $n$ subintervalos tenemos $n+1$ nodos:
$x_i=a+ih$, $h=(b-a)/n$.

## De Taylor al sistema

Al desarrollar en $x_i+h$ y $x_i-h$ hasta el término cúbico, con resto
$O(h^4)$, sumamos para cancelar las potencias impares:

$$y(x_i+h)+y(x_i-h)=2y(x_i)+h^2y''(x_i)+O(h^4).$$

Restamos $2y(x_i)$ en ambos miembros y dividimos por $h^2$. Para la primera
derivada restamos los desarrollos y dividimos por $2h$. Obtenemos

$$y'_i\approx\frac{y_{i+1}-y_{i-1}}{2h},\qquad
y''_i\approx\frac{y_{i+1}-2y_i+y_{i-1}}{h^2}.$$

En cada nodo interior escribimos

$$F_i=2y_i-y_{i-1}-y_{i+1}+h^2 f(x_i,y_i,(y_{i+1}-y_{i-1})/(2h))=0.$$

## Condiciones Robin

Introducimos un nodo ficticio a cada lado. Las condiciones proporcionan

$$y_{-1}=y_1+2h(\beta_a/\alpha_a)y_0-2h\gamma_a/\alpha_a,$$
$$y_{n+1}=y_{n-1}-2h(\beta_b/\alpha_b)y_n+2h\gamma_b/\alpha_b.$$

Sustituimos esos valores en las ecuaciones de los extremos. No quedan como
incógnitas adicionales. Para Dirichlet ($\alpha=0$) fijamos el valor del
extremo con su propia fila. El código conserva los $n+1$ valores del vector.

## Newton y Crout

Para una EDO no lineal resolvemos $J\Delta y=-F$ y actualizamos
$y\leftarrow y+\Delta y$. Las tres diagonales interiores de la jacobiana son

$$J_{i,i-1}=-1-\frac h2 f_{y'},\quad
J_{i,i}=2+h^2f_y,\quad J_{i,i+1}=-1+\frac h2f_{y'}.$$

Evaluamos las parciales en $(x_i,y_i,(y_{i+1}-y_{i-1})/(2h))$.
En los extremos derivamos después de eliminar los nodos ficticios. Allí la
pendiente vale $(\gamma-\beta y)/\alpha$ y su derivada respecto de $y$ es
$-\beta/\alpha$; necesitamos incluir ese factor con la regla de la cadena.

Crout factoriza la matriz tridiagonal como $LU$, con diagonal unidad en
$U$. Primero calculamos los factores, después resolvemos $Lz=-F$ y al final
$U\Delta y=z$. No calculamos la inversa. Su coste y memoria son lineales,
pero el algoritmo no usa pivoteo.

## Ejemplo ejecutable

```python
import numpy as np
from bvp import Robin, nonlinear_finite_difference

sol = nonlinear_finite_difference(
    lambda x, y, dy: x*y*dy-3*y*y+6*x,
    lambda x, y, dy: x*dy-6*y,
    lambda x, y, dy: x*y,
    0, 1, Robin(-2, 3, 0), Robin(-1/3, 2, 1),
    initial=lambda x: .75*x*x, n=80,
)
assert sol.converged
print(sol.iterations)
print(np.max(np.abs(sol.y-sol.x**3)))
```

El problema tiene la solución $y=x^3$: al sustituir cancelamos $-3x^6$ y
$3x^6$, y obtenemos $6x$. Las condiciones dan cero en el extremo izquierdo
y $2-1=1$ en el derecho. Desde $3x^2/4$, Newton converge en cuatro
iteraciones; el error máximo en los 81 nodos ronda $7.02\cdot10^{-5}$.

`sol.residual_norm` contiene $\|F\|_\infty/h^2$, que mide el cumplimiento
del sistema discreto. No es el error frente a la solución continua. Con
`n=20,40,80,160` el error se reduce por un factor próximo a cuatro al
reducir $h$ a la mitad. Esa observación concuerda con orden global dos.

Para la EDO lineal $y''=p(x)y'+q(x)y+r(x)$ usamos
`linear_finite_difference(p,q,r,a,b,left,right,n=80)`. Resolvemos una vez
el sistema; no necesitamos iteraciones de Newton. Ambos solvers aceptan
`method='dense'` para comparar con LU general.

## Referencias

- [Teoría, deducciones y ejemplos](https://metodosnumericos.jonramos.dev/es/articulos/frontera-diferencias-finitas/).
- [Cuaderno ejecutable](../notebooks/04_finite_differences.ipynb).
- [Comparación de métodos](../notebooks/05_comparing_methods.ipynb).
