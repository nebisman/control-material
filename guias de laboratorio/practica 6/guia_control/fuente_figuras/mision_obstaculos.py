"""Figura de misión con obstáculos (boceto mision3.jpeg).

Recorrido, en el marco de la misión (x adelante al inicio, y a la izquierda):
  1. avanzar 1 m
  2. girar 90° a la derecha en el sitio
  3. semicircunferencia a la izquierda de radio 0.5 m alrededor del primer obstáculo
  4. avanzar 0.5 m
  5. semicircunferencia a la derecha de radio 0.5 m (centro a 0.5 m del segundo obstáculo)
  6. avanzar 1 m
  7. girar 90° a la izquierda en el sitio
  8. avanzar 1 m y parar
Los tres obstáculos (cajas de 30 cm, a 1 m entre sí) están alineados con el primer tramo.
Solo se rotulan el inicio y el fin; ANOTACIONES = True agrega los números de los pasos,
la leyenda, los radios y la nota de la cuadrícula.
Usa la proyección y los sprites en picada de mision_sprites.py.

Uso:  python mision_obstaculos.py [salida.pdf]
"""
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import mision_sprites as ms                                  # noqa: E402

SALIDA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, "..", "Images", "mision_obstaculos.pdf")

CELDA = 0.25            # cuadrícula del piso [m]
R = 0.5                 # radio de las semicircunferencias [m]
LADO, ALTO = 0.30, 0.30  # obstáculos [m]
ms.PHI = 10             # orientación del recorrido respecto a la cámara
ms.AMPLIA = 1.5
AZUL, NARANJA, VERDE, ROJO = ms.AZUL, ms.NARANJA, ms.VERDE, ms.ROJO

# puntos del recorrido (x adelante al inicio, y a la izquierda)
P_INI = (-1.0, 0.0)
C1 = (0.0, 0.0)                 # fin del primer tramo: giro a la derecha
O1 = (0.5, 0.0)                 # centro de la primera semicircunferencia (= primer obstáculo)
D1 = (1.0, 0.0)
E1 = (1.0, 0.5)
O2 = (1.5, 0.5)                 # centro de la segunda semicircunferencia
F = (2.0, 0.5)
G = (2.0, -0.5)                 # giro a la izquierda
P_FIN = (3.0, -0.5)
OBSTACULOS = ((0.5, 0.0), (1.5, 0.0), (2.5, 0.0))      # a 1 m entre sí
ANOTACIONES = False


def arco(centro, a0, a1, n=120):
    a = np.radians(np.linspace(a0, a1, n))
    return [(centro[0] + R * math.cos(t), centro[1] + R * math.sin(t)) for t in a]


def recorrido():
    """Puntos del recorrido completo en el marco de la misión."""
    pts = [P_INI, C1]
    pts += arco(O1, 180, 360)          # semicircunferencia a la izquierda (antihoraria)
    pts += [E1]
    pts += arco(O2, 180, 0)            # semicircunferencia a la derecha (horaria)
    pts += [G, P_FIN]
    return pts


def caja(ax, c, z0):
    """Obstáculo: caja de LADO x LADO x ALTO con las caras visibles sombreadas."""
    h = LADO / 2
    base = [(c[0] - h, c[1] - h), (c[0] + h, c[1] - h), (c[0] + h, c[1] + h), (c[0] - h, c[1] + h)]
    caras = []
    for k in range(4):
        a, b = base[k], base[(k + 1) % 4]
        n = np.array((b[1] - a[1], -(b[0] - a[0])))              # normal hacia afuera (vértices antihorarios)
        X, Y = ms.mundo(*n)
        if X * ms.MIRADA[0] + Y * ms.MIRADA[1] < 0:              # la cara mira hacia la cámara
            luz = 0.78 if abs(n[0]) > 0 else 0.62
            caras.append(([ms.proy(*a), ms.proy(*b), ms.proy(*b, ALTO), ms.proy(*a, ALTO)], luz))
    # sombra suave en el piso
    sombra = [ms.proy(x + 0.04, y - 0.04) for x, y in base]
    ax.add_patch(Polygon(sombra, closed=True, fc="black", alpha=0.12, lw=0, zorder=z0))
    tono = np.array((0.80, 0.63, 0.40))
    for poli, luz in caras:
        ax.add_patch(Polygon(poli, closed=True, fc=tono * luz, ec="0.25", lw=0.6, zorder=z0 + 0.1))
    ax.add_patch(Polygon([ms.proy(x, y, ALTO) for x, y in base], closed=True, fc=tono, ec="0.25", lw=0.6,
                         zorder=z0 + 0.2))


def giro(ax, centro, a0, a1, radio=0.24):
    """Flecha curva de giro en el sitio (ángulos polares en el marco de la misión)."""
    pts = np.array([ms.proy(centro[0] + radio * math.cos(t), centro[1] + radio * math.sin(t))
                    for t in np.radians(np.linspace(a0, a1, 40))])
    ax.plot(pts[:-1, 0], pts[:-1, 1], color=NARANJA, lw=2.2, zorder=5)
    ms.flecha(ax, pts[-3], pts[-1], NARANJA, lw=2.2, z=5)


def numero(ax, p, n, color):
    ax.text(*ms.proy(*p), r"\textbf{%d}" % n, fontsize=11.5, color=color, ha="center", va="center", zorder=22,
            bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec=color, lw=1.3))


def radio(ax, centro, ang, texto_lado):
    c = ms.proy(*centro)
    b = ms.proy(centro[0] + R * math.cos(math.radians(ang)), centro[1] + R * math.sin(math.radians(ang)))
    ax.plot(*c, marker="+", ms=9, mew=1.4, color="0.25", zorder=6)
    ax.annotate("", b, c, arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.0, mutation_scale=10,
                                          shrinkA=0, shrinkB=0), zorder=6)
    ax.text(*((c + b) / 2 + texto_lado), r"$R=0.5$\,m", fontsize=11, color="0.15", ha="center", va="center",
            zorder=7, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))


def main():
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}"})
    camino = np.array([ms.proy(*p) for p in recorrido()])
    x0, y0 = camino.min(axis=0) - (0.40, 0.95 if ANOTACIONES else 0.55)
    x1, y1 = camino.max(axis=0) + (0.40, 0.40)
    ancho_cm = 16.0
    fig = plt.figure(figsize=(ancho_cm / 2.54, ancho_cm / 2.54 * (y1 - y0) / (x1 - x0)))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#f3f1ed", ec="none", zorder=0))

    # cuadrícula del piso (líneas de 1 m más marcadas)
    for k in range(-24, 25):
        c = k * CELDA
        mayor = abs(c - round(c)) < 1e-9
        estilo = dict(color="0.55" if mayor else "0.72", lw=0.9 if mayor else 0.45, zorder=1)
        ax.plot(*zip(ms.proy(c, -6), ms.proy(c, 6)), **estilo)
        ax.plot(*zip(ms.proy(-6, c), ms.proy(6, c)), **estilo)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="0.35", lw=1.0, zorder=30))
    if ANOTACIONES:
        ax.text(x1 - 0.05, y1 - 0.05,
                r"cuadr\'icula: %d\,cm\quad robot ampliado %.1f$\times$" % (round(CELDA * 100), ms.AMPLIA),
                ha="right", va="top", fontsize=10, color="0.2", zorder=31,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.5))

    # recorrido y flechas de sentido
    ax.plot(camino[:, 0], camino[:, 1], color=AZUL, lw=2.8, solid_joinstyle="round", zorder=2)
    for a, b in ((P_INI, C1), (D1, E1), (F, G), (G, P_FIN)):
        m = np.array(a) + 0.55 * (np.array(b) - np.array(a))
        u = (np.array(b) - np.array(a)) / np.linalg.norm(np.array(b) - np.array(a))
        ms.flecha(ax, ms.proy(*(m - 0.12 * u)), ms.proy(*(m + 0.02 * u)), AZUL, lw=2.8)
    for centro, a in ((O1, 270), (O2, 90)):
        s = 1 if centro is O1 else -1                         # antihoraria / horaria
        p = [ms.proy(centro[0] + R * math.cos(math.radians(t)), centro[1] + R * math.sin(math.radians(t)))
             for t in (a - s * 8, a + s * 2)]
        ms.flecha(ax, p[0], p[1], AZUL, lw=2.8)

    # giros en el sitio
    giro(ax, C1, 150, 20)               # 90° a la derecha (horario)
    giro(ax, G, 110, 330)               # 90° a la izquierda (antihorario)

    if ANOTACIONES:
        # radios
        radio(ax, O1, 315, (0.0, 0.0))
        radio(ax, O2, 140, (0.0, 0.0))

    # obstáculos y robots, el más lejano primero
    objetos = [("caja", o, None) for o in OBSTACULOS]
    objetos += [("robot", P_INI, 0), ("robot", (O2[0], O2[1] + R), 0), ("robot", P_FIN, 0)]
    objetos.sort(key=lambda q: -ms.profundidad(*q[1]))
    for i, (tipo, p, d) in enumerate(objetos):
        if tipo == "caja":
            caja(ax, p, z0=10 + i)
        else:
            ms.robot(ax, p[0], p[1], d, z0=10 + i)

    if ANOTACIONES:
        # números de los pasos
        numero(ax, (-0.5, 0.28), 1, AZUL)
        numero(ax, (-0.05, 0.42), 2, NARANJA)
        numero(ax, (O1[0], O1[1] - R - 0.3), 3, AZUL)
        numero(ax, (0.78, 0.25), 4, AZUL)
        numero(ax, (O2[0] + 0.7 * math.cos(math.radians(40)), O2[1] + 0.7 * math.sin(math.radians(40))), 5, AZUL)
        numero(ax, (2.25, 0.05), 6, AZUL)
        numero(ax, (1.68, -0.82), 7, NARANJA)
        numero(ax, (2.55, -0.8), 8, AZUL)

    # rótulos
    caja_txt = dict(boxstyle="round,pad=0.3", lw=1.0)
    p_ini, p_fin = ms.proy(*P_INI), ms.proy(*P_FIN)
    ax.annotate(r"\textbf{Inicio}", p_ini, xytext=p_ini + (-0.05, 0.42), fontsize=12, color=VERDE, ha="center",
                va="center", zorder=25, bbox=dict(fc="white", ec=VERDE, **caja_txt),
                arrowprops=dict(arrowstyle="-", color=VERDE, lw=1.0, shrinkB=12))
    ax.annotate(r"\textbf{Fin}", p_fin, xytext=p_fin + (0.05, -0.42), fontsize=12, color=ROJO, ha="center",
                va="center", zorder=25, bbox=dict(fc="white", ec=ROJO, **caja_txt),
                arrowprops=dict(arrowstyle="-", color=ROJO, lw=1.0, shrinkB=12))
    if ANOTACIONES:
        p_obs = ms.proy(*OBSTACULOS[2], ALTO)
        ax.annotate(r"\begin{tabular}{@{}c@{}}obst\'aculos\\(cajas de 30\,cm)\end{tabular}", p_obs,
                    xytext=p_obs + (0.16, 0.50), fontsize=11, color="0.15",
                    ha="center", va="center", zorder=25, bbox=dict(fc="white", ec="0.4", **caja_txt),
                    arrowprops=dict(arrowstyle="-", color="0.3", lw=0.9))

        pasos = (r"\textbf{1} avanzar 1\,m", r"\textbf{2} girar 90$^\circ$ a la derecha",
                 r"\textbf{3} semicircunferencia a la izquierda", r"\textbf{4} avanzar 0.5\,m",
                 r"\textbf{5} semicircunferencia a la derecha", r"\textbf{6} avanzar 1\,m",
                 r"\textbf{7} girar 90$^\circ$ a la izquierda", r"\textbf{8} avanzar 1\,m y parar")
        leyenda = (r"\begin{tabular}{@{}l@{\hspace{1.5em}}l@{}}"
                   + r" \\[2pt] ".join(r"%s & %s" % (pasos[k], pasos[k + 4]) for k in range(4))
                   + r"\end{tabular}")
        ax.text(x0 + 0.05, y0 + 0.05, leyenda, fontsize=10.5, color="0.15", ha="left", va="bottom",
                zorder=25, bbox=dict(fc="white", ec="0.4", **caja_txt))

    os.makedirs(os.path.dirname(os.path.abspath(SALIDA)), exist_ok=True)
    fig.savefig(SALIDA, dpi=600)
    print("figura:", os.path.abspath(SALIDA))


if __name__ == "__main__":
    main()
