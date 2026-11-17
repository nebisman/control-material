"""Figura de una misión del robot con los sprites en picada (vista oblicua).

Misión: avanzar 2 m, girar 90° a la derecha, avanzar 2 m, girar 90° a la izquierda,
avanzar 1 m y parar.
Piso con cuadrícula de 25 cm (líneas de 1 m más marcadas); el recorrido está a escala
y el robot se dibuja ampliado AMPLIA veces para que se distinga.

Uso:  python mision_sprites.py [carpeta_sprites] [salida.pdf]
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

AQUI = os.path.dirname(os.path.abspath(__file__))
SPR = sys.argv[1] if len(sys.argv) > 1 else "/home/leonardo/datos/videos/animaciones/robot/video_modos/sprites"
SALIDA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(AQUI, "..", "Images", "mision_avanzar_girar2.pdf")

V = json.load(open(os.path.join(SPR, "sprites.json")))["vistas"]["picada"]
RX, RZ = math.radians(V["rx_grados"]), math.radians(V["rz_grados"])
ANCHO = V["ancho_m"]                      # ancho real que cubre cada sprite [m]
PASO = V["paso_grados"]

PHI = 50                                  # rumbo inicial respecto a la cámara (grados, antihorario)
L1, L2, L3 = 2.0, 2.0, 1.0                # tramos de la misión [m]
CELDA = 0.5                              # cuadrícula del piso [m]
AMPLIA = 1.8                              # el robot se dibuja ampliado para que se distinga

AZUL, NARANJA, VERDE, ROJO = "#0b5394", "#e67300", "#2e8b57", "#c0392b"


def mundo(xm, ym):
    """Marco de la misión (x adelante, y a la izquierda) -> piso, girado PHI."""
    c, s = math.cos(math.radians(PHI)), math.sin(math.radians(PHI))
    return c * xm - s * ym, s * xm + c * ym


def proy(xm, ym, z=0.0):
    """Proyección ortográfica de la cámara en picada (la misma de los sprites)."""
    X, Y = mundo(xm, ym)
    return np.array((X * math.cos(RZ) + Y * math.sin(RZ),
                     (-X * math.sin(RZ) + Y * math.cos(RZ)) * math.cos(RX) + z * math.sin(RX)))


MIRADA = (-math.sin(RZ) * math.sin(RX), math.cos(RZ) * math.sin(RX))   # hacia dónde mira la cámara en el piso


def profundidad(xm, ym):
    X, Y = mundo(xm, ym)
    return X * MIRADA[0] + Y * MIRADA[1]                               # mayor = más lejos


def sprite(nombre):
    return np.asarray(Image.open(os.path.join(SPR, "picada", nombre)).convert("RGBA"))


def robot(ax, xm, ym, delta, z0, alfa=1.0):
    """Robot en (xm, ym) del marco de la misión con rumbo delta (grados, + a la derecha)."""
    giro = int(round((PHI - delta) / PASO)) * PASO % 360
    g = math.radians(giro)
    izq_lejos = -math.sin(g) * MIRADA[0] + math.cos(g) * MIRADA[1] > 0
    ri, rd = sprite("ri_%03d_0.png" % giro), sprite("rd_%03d_0.png" % giro)
    capas = (ri, sprite("cuerpo_%03d.png" % giro), rd) if izq_lejos else (rd, sprite("cuerpo_%03d.png" % giro), ri)
    cx, cy = proy(xm, ym)
    a = ANCHO * AMPLIA
    ax.add_patch(Ellipse((cx, cy), 0.23 * AMPLIA, 0.23 * AMPLIA * math.cos(RX), color="black",
                         alpha=0.22 * alfa, lw=0, zorder=z0))
    for k, im in enumerate(capas):
        ax.imshow(im, extent=(cx - a / 2, cx + a / 2, cy - a / 2, cy + a / 2),
                  zorder=z0 + 0.1 * (k + 1), alpha=alfa, interpolation="antialiased")


def flecha(ax, p, q, color, lw=2.6, z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z))


def cota(ax, a, b, desplaz, texto, lineas=(True, True)):
    """Línea de cota paralela al tramo a-b (marco de la misión), desplazada 'desplaz' [m].

    'lineas' indica en qué extremos se dibuja la línea de extensión (se omite donde
    coincidiría con el recorrido)."""
    a, b = np.array(a), np.array(b)
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array((-t[1], t[0])) * desplaz
    pa, pb = proy(*(a + n)), proy(*(b + n))
    ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="<|-|>", mutation_scale=10, color="0.25", lw=1.0,
                                 shrinkA=0, shrinkB=0, zorder=3))
    for p, dibujar in zip((a, b), lineas):
        if dibujar:
            ax.plot(*zip(proy(*(p + n * 0.25)), proy(*(p + n * 1.15))), color="0.25", lw=0.8, zorder=3)
    m, d = (pa + pb) / 2, pb - pa
    ang = math.degrees(math.atan2(d[1], d[0]))
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    off = proy(*(n / abs(desplaz) * 0.13)) - proy(0, 0)
    ax.text(*(m + off), texto, rotation=ang, rotation_mode="anchor", ha="center", va="center",
            fontsize=12, color="0.15", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def main():
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}"})
    P0, P1, P2, P3 = (0.0, 0.0), (L1, 0.0), (L1, -L2), (L1 + L3, -L2)

    # encuadre: recorrido y robots con margen
    pts = np.array([proy(*p) for p in (P0, P1, P2, P3)])
    x0, y0 = pts.min(axis=0) - (0.62, 0.42)
    x1, y1 = pts.max(axis=0) + (0.62, 0.74)
    # rótulos del último tramo, ubicados en el plano de la figura respecto a P2 y P3
    T4 = proy(*P2) + (-1.12, -0.24)
    T5 = proy(*P3) + (-0.08, 0.47)
    T6 = proy(*P3) + (0.70, -0.32)
    x1 = max(x1, T6[0] + 0.42)
    y0 = min(y0, T4[1] - 0.12, proy(*P2)[1] - 0.52)
    ancho_cm = 16.0
    fig = plt.figure(figsize=(ancho_cm / 2.54, ancho_cm / 2.54 * (y1 - y0) / (x1 - x0)))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#f3f1ed", ec="none", zorder=0))

    # cuadrícula del piso en el marco de la misión (cada 25 cm, cada 1 m más marcada)
    for k in range(-24, 25):
        c = k * CELDA
        mayor = abs(c - round(c)) < 1e-9
        estilo = dict(color="0.55" if mayor else "0.72", lw=0.9 if mayor else 0.45, zorder=1)
        ax.plot(*zip(proy(c, -6), proy(c, 6)), **estilo)
        ax.plot(*zip(proy(-6, c), proy(6, c)), **estilo)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="0.35", lw=1.0, zorder=7))
    ax.text(x0 + 0.06, y1 - 0.06, r"cuadr\'icula: %d\,cm\quad robot ampliado %.1f$\times$" % (round(CELDA * 100), AMPLIA),
            ha="left", va="top", fontsize=10,
            color="0.2", zorder=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.5))

    # recorrido
    ax.plot(*zip(*[proy(*p) for p in (P0, P1, P2, P3)]), color=AZUL, lw=2.6, solid_joinstyle="round", zorder=2)
    flecha(ax, proy(0.55, 0), proy(1.15, 0), AZUL)
    flecha(ax, proy(L1, -0.65), proy(L1, -1.25), AZUL)
    flecha(ax, proy(L1 + 0.34, -L2), proy(L1 + 0.66, -L2), AZUL)

    # giros de 90° en las esquinas: a la derecha (horario) en P1, a la izquierda (antihorario) en P2
    R = 0.42

    def giro(centro, a0, a1):
        angs = np.radians(np.linspace(a0, a1, 40))
        arco = np.array([proy(centro[0] + R * math.cos(a), centro[1] + R * math.sin(a)) for a in angs])
        ax.plot(arco[:-1, 0], arco[:-1, 1], color=NARANJA, lw=2.2, zorder=5)
        flecha(ax, arco[-3], arco[-1], NARANJA, lw=2.2, z=5)

    giro(P1, 165, -55)
    giro(P2, 110, 240)

    # robots: el más lejano primero
    poses = [(P0, 0, VERDE), (P1, 45, NARANJA), (P2, 45, NARANJA), (P3, 0, ROJO)]
    for i, (p, d, _) in enumerate(sorted(poses, key=lambda q: -profundidad(*q[0]))):
        robot(ax, p[0], p[1], d, z0=10 + i)

    # cotas
    cota(ax, P0, P1, 0.42, r"2\,m")
    cota(ax, P1, P2, 0.42, r"2\,m", lineas=(True, False))
    cota(ax, P2, P3, -0.42, r"1\,m")

    # rótulos de la misión
    caja = dict(boxstyle="round,pad=0.3", lw=1.0)
    ax.annotate(r"\textbf{Inicio}", proy(*P0), xytext=proy(-0.15, -0.95), fontsize=12, color=VERDE,
                ha="center", zorder=20, bbox=dict(fc="white", ec=VERDE, **caja),
                arrowprops=dict(arrowstyle="-", color=VERDE, lw=1.0, shrinkB=12))
    ax.text(*proy(1.05, -0.42), r"\textbf{1.} avanzar 2\,m", fontsize=12, color=AZUL, ha="center",
            zorder=20, bbox=dict(fc="white", ec=AZUL, **caja))
    ax.annotate(r"\textbf{2.} girar 90$^\circ$ a la derecha", proy(L1 - 0.33, R * 0.55), xytext=proy(L1 - 1.3, 0.95),
                fontsize=12, color=NARANJA, ha="center", zorder=20, bbox=dict(fc="white", ec=NARANJA, **caja),
                arrowprops=dict(arrowstyle="-", color=NARANJA, lw=1.0))
    ax.text(*proy(L1 - 0.48, -1.35), r"\textbf{3.} avanzar 2\,m", fontsize=12, color=AZUL, ha="center",
            zorder=20, bbox=dict(fc="white", ec=AZUL, **caja))
    ax.annotate(r"\textbf{4.} girar 90$^\circ$ a la izquierda", proy(L1 + R * math.cos(math.radians(200)),
                -L2 + R * math.sin(math.radians(200))), xytext=T4,
                fontsize=12, color=NARANJA, ha="center", zorder=20, bbox=dict(fc="white", ec=NARANJA, **caja),
                arrowprops=dict(arrowstyle="-", color=NARANJA, lw=1.0))
    ax.text(*T5, r"\textbf{5.} avanzar 1\,m", fontsize=12, color=AZUL, ha="center",
            zorder=20, bbox=dict(fc="white", ec=AZUL, **caja))
    ax.annotate(r"\textbf{6.} parar", proy(*P3), xytext=T6, fontsize=12,
                color=ROJO, ha="center", zorder=20, bbox=dict(fc="white", ec=ROJO, **caja),
                arrowprops=dict(arrowstyle="-", color=ROJO, lw=1.0, shrinkB=12))

    os.makedirs(os.path.dirname(os.path.abspath(SALIDA)), exist_ok=True)
    fig.savefig(SALIDA, dpi=600)
    print("figura:", os.path.abspath(SALIDA))


if __name__ == "__main__":
    main()
