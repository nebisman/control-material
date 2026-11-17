"""Figura de misión: el robot describe sin parar un 8 formado por dos circunferencias
tangentes en el centro del 8: la de giro a la derecha (tramos 1 y 4) de radio 0.5 m y la
de giro a la izquierda (tramos 2 y 3) de radio 1 m.

El inicio está en el extremo izquierdo del 8 (en la figura). El robot recorre el trazo
azul (1: media vuelta girando a la derecha, 2: media vuelta girando a la izquierda), cruza
el centro, recorre el trazo naranja (3: a la izquierda, 4: a la derecha), vuelve al inicio
y repite.
Usa la proyección y los sprites en picada de mision_sprites.py.

Uso:  python mision_ocho.py [salida.pdf]
"""
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import mision_sprites as ms                                  # noqa: E402

SALIDA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, "..", "Images", "mision_ocho.pdf")

R_DER = 0.5             # radio de la circunferencia de giro a la derecha (tramos 1 y 4) [m]
R_IZQ = 1.0             # radio de la circunferencia de giro a la izquierda (tramos 2 y 3) [m]
CELDA = 0.25            # cuadrícula del piso [m]
ms.PHI = -50            # rumbo inicial respecto a la cámara (-65: eje largo del 8 horizontal)
ms.AMPLIA = 1.5         # robot ampliado para que se distinga
AZUL, NARANJA, VERDE, MORADO = ms.AZUL, ms.NARANJA, ms.VERDE, "#7d3c98"
# Los dos trazos del 8 se cruzan en el inicio. Cada circunferencia se parte en la mitad
# que queda adelante del inicio (s de 0 a pi) y la de atrás (s de pi a 2 pi):
#   trazo A: mitad de atrás de la derecha (2) + mitad de adelante de la izquierda (3)
#   trazo B: mitad de atrás de la izquierda (4) + mitad de adelante de la derecha (1)
TRAZO_A, TRAZO_B = AZUL, NARANJA
# (sentido, tramo s0-s1, color, número en el orden del recorrido)
# El inicio es el punto más lejano de la circunferencia de la derecha (s = pi), que en
# la figura queda en el extremo izquierdo del 8: desde ahí se recorre todo el trazo A y
# después todo el trazo B.
MITADES = ((+1, math.pi, 2 * math.pi, TRAZO_A, 1), (-1, 0.0, math.pi, TRAZO_A, 2),
           (-1, math.pi, 2 * math.pi, TRAZO_B, 3), (+1, 0.0, math.pi, TRAZO_B, 4))
INICIO = (+1, math.pi)


def radio(sentido):
    return R_DER if sentido > 0 else R_IZQ


def centro_y(sentido):
    return -R_DER if sentido > 0 else R_IZQ


def punto_circ(_, sentido, s):
    """Posición y rumbo (delta, grados, + a la derecha) tras recorrer el ángulo s (rad)
    de la circunferencia. sentido = +1: derecha (horario), -1: izquierda (antihorario)."""
    # La circunferencia de la derecha tiene centro (0, -R_DER) y empieza en el origen
    # (ángulo polar +90°); la de la izquierda tiene centro (0, +R_IZQ) y empieza en -90°.
    r, cy = radio(sentido), centro_y(sentido)
    a = math.pi / 2 - s if sentido > 0 else -math.pi / 2 + s
    p = (r * math.cos(a), cy + r * math.sin(a))
    return p, sentido * math.degrees(s)


def curva(sentido, s0, s1, n=80):
    return np.array([ms.proy(*punto_circ(None, sentido, s)[0]) for s in np.linspace(s0, s1, n)])


def main():
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}"})

    der, izq = curva(+1, 0, 2 * math.pi, 240), curva(-1, 0, 2 * math.pi, 240)
    pts = np.vstack([der, izq])
    x0, y0 = pts.min(axis=0) - (0.30, 0.52)
    x1, y1 = pts.max(axis=0) + (0.30, 0.40)
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
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="0.35", lw=1.0, zorder=7))
    ax.text(x1 - 0.05, y1 - 0.05,
            r"cuadr\'icula: %d\,cm\quad robot ampliado %.1f$\times$" % (round(CELDA * 100), ms.AMPLIA),
            ha="right", va="top", fontsize=10, color="0.2", zorder=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.5))

    # trayectoria: cada semicircunferencia con el color de su trazo, su flecha de sentido
    # y su número en el orden del recorrido (el número va por fuera de la circunferencia)
    for sentido, s0, s1, color, num in MITADES:
        c = curva(sentido, s0, s1, 120)
        ax.plot(c[:, 0], c[:, 1], color=color, lw=3.0, solid_capstyle="butt", zorder=2)
        sm = (s0 + s1) / 2
        f = curva(sentido, sm - 0.1, sm + 0.02, 6)
        ms.flecha(ax, f[-2], f[-1], color, lw=3.0, z=4)
        p, _ = punto_circ(None, sentido, sm)
        cy = centro_y(sentido)
        afuera = np.array((p[0], p[1] - cy)) * 0.14 / radio(sentido)
        ax.text(*ms.proy(p[0] + afuera[0], p[1] + afuera[1]), r"\textbf{%d}" % num, fontsize=12,
                color=color, ha="center", va="center", zorder=21,
                bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec=color, lw=1.3))

    # centros y radios
    for sentido, ang in ((+1, -150), (-1, 150)):
        cy, r = centro_y(sentido), radio(sentido)
        cpx = ms.proy(0, cy)
        ax.plot(*cpx, marker="+", ms=9, mew=1.4, color="0.25", zorder=3)
        borde = ms.proy(r * math.cos(math.radians(ang)), cy + r * math.sin(math.radians(ang)))
        ax.annotate("", borde, cpx, arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.0,
                                                    mutation_scale=10, shrinkA=0, shrinkB=0), zorder=3)
        m = (cpx + borde) / 2
        ax.text(*m, r"$R=%g$\,m" % r, fontsize=11, color="0.15", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    # robots: inicio (en el punto de tangencia) y a mitad de cada circunferencia; el más lejano primero
    poses = [punto_circ(None, +1, 0.0), punto_circ(None, +1, math.pi), punto_circ(None, -1, math.pi)]
    for i, (p, d) in enumerate(sorted(poses, key=lambda q: -ms.profundidad(*q[0]))):
        ms.robot(ax, p[0], p[1], d, z0=10 + i)

    # rótulos
    caja = dict(boxstyle="round,pad=0.3", lw=1.0)
    p_ini = ms.proy(*punto_circ(None, *INICIO)[0])
    ax.annotate(r"\textbf{Inicio}", p_ini, xytext=p_ini + (0.0, -0.40), fontsize=12, color=VERDE,
                ha="center", va="center", zorder=20, bbox=dict(fc="white", ec=VERDE, **caja),
                arrowprops=dict(arrowstyle="-", color=VERDE, lw=1.0, shrinkB=14))
    ax.text(x0 + 0.05, y0 + 0.05,
            r"\textbf{1}, \textbf{4}: girando a la derecha (horario) \quad \textbf{2}, \textbf{3}: girando a la izquierda (antihorario)"
            "\n" r"repetir sin parar: 1 $\rightarrow$ 2 $\rightarrow$ 3 $\rightarrow$ 4 $\rightarrow$ 1 $\rightarrow$ \dots",
            fontsize=11.5, color="0.15", ha="left", va="bottom", linespacing=1.5, zorder=20,
            bbox=dict(fc="white", ec="0.4", **caja))

    os.makedirs(os.path.dirname(os.path.abspath(SALIDA)), exist_ok=True)
    fig.savefig(SALIDA, dpi=600)
    print("figura:", os.path.abspath(SALIDA))


if __name__ == "__main__":
    main()
