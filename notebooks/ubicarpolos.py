"""
asigne_polos.py
Diseño de controladores por asignación de polos usando la matriz de Sylvester.

Equivalente en Python de la función asigne_polos.jl (Julia/ControlSystems).
Dependencias: numpy, control (python-control)
"""

import numpy as np
import control as ctrl


def asigne_polos(planta, polos):
    """
    Calcula un controlador K(s) por asignación de polos con realimentación unitaria.

    Parámetros
    ----------
    planta : control.TransferFunction
        Función de transferencia de la planta.
    polos : array_like
        Vector con los polos deseados del sistema en lazo cerrado.

    Retorna
    -------
    K  : control.TransferFunction  – Controlador.
    T  : control.TransferFunction  – Función de transferencia en lazo cerrado.
    Gur: control.TransferFunction  – Transferencia de control  u/r = T/P.
    S  : control.TransferFunction  – Función de sensibilidad  S = 1 − T.
    ind_error : int                – 0 = éxito, 1 = polos insuficientes.
    """

    # --- Datos de la planta ---------------------------------------------------
    num_P = np.array(planta.num[0][0], dtype=float)
    den_P = np.array(planta.den[0][0], dtype=float)

    n = len(den_P) - 1                     # orden de la planta
    l = len(den_P) - len(num_P)            # diferencia de grados (rel. degree)

    # --- Orden del controlador ------------------------------------------------
    nMm = len(polos)                       # orden del lazo cerrado deseado
    m = nMm - n                            # orden del controlador

    # --- Polinomio deseado (coefs descendentes) --------------------------------
    DT = np.real(np.poly(polos))

    # --- Numerador con relleno de ceros (mismo largo que den_P) ----------------
    Na = np.zeros(len(den_P))
    Na[l:] = num_P

    D = den_P

    # --- Construir la matriz de Sylvester -------------------------------------
    filas = n + m + 1
    cols = 2 * (m + 1)
    Sm = np.zeros((filas, cols))

    for k in range(m + 1):
        Sm[k:k + n + 1, k] = D
        Sm[k:k + n + 1, k + m + 1] = Na

    # --- Resolver el sistema  Sm · [Y ; X] = DT ------------------------------
    ind_error = 0

    if m > n - 1:
        # Intentar diseño con rechazo de perturbaciones (quitar columna m)
        indices = list(range(cols))
        indices.remove(m)
        Smj = Sm[:, indices]

        if np.linalg.matrix_rank(Smj) >= filas:
            XY = np.linalg.solve(Smj, DT)
            Y = np.concatenate([XY[:m], [0.0]])
            X = XY[m:]
        else:
            XY = np.linalg.lstsq(Sm, DT, rcond=None)[0]
            Y = XY[:m + 1]
            X = XY[m + 1:]

    elif m == n - 1:
        XY = np.linalg.solve(Sm, DT)
        Y = XY[:m + 1]
        X = XY[m + 1:2 * m + 2]

    else:
        ind_error = 1
        raise ValueError(
            f"asigne_polos: número de polos insuficientes. "
            f"Se requieren al menos {2 * n - 1} polos (se dieron {nMm})."
        )

    # --- Construir funciones de transferencia ---------------------------------
    K = ctrl.tf(X, Y)
    T = ctrl.feedback(K * planta)
    T = ctrl.minreal(T)
    Gur = ctrl.minreal(T / planta)
    S = ctrl.minreal(ctrl.tf(1, 1) - T)

    return K, T, Gur, S, ind_error
