"""
asigne_polos.py
Diseño de controladores por asignación de polos usando la matriz de Sylvester.

Equivalente en Python de la función asigne_polos.jl (Julia/ControlSystems).
Dependencias: numpy, control (python-control)
Leonardo Bermeo
"""

import numpy as np
import control as ctrl


def _poly_to_latex(coefs):
    """Convierte una lista de coeficientes [a_n, a_{n-1}, ..., a_0] a string LaTeX."""
    n = len(coefs) - 1
    terms = []
    for i, c in enumerate(coefs):
        potencia = n - i
        if c == 0:
            continue
        # Formatear coeficiente
        if c == int(c):
            c_str = str(int(c))
        else:
            c_str = f'{c:g}'
        # Formatear término
        if potencia == 0:
            term = c_str
        elif potencia == 1:
            term = f'{c_str}\\,s' if c != 1 else 's'
        else:
            term = f'{c_str}\\,s^{{{potencia}}}' if c != 1 else f's^{{{potencia}}}'
        terms.append(term)
    return ' + '.join(terms)

def tf_to_latex(T):
    """Convierte una función de transferencia de la librería control a string LaTeX."""
    num = T.num[0][0].tolist()
    den = T.den[0][0].tolist()
    num_str = _poly_to_latex(num)
    den_str = _poly_to_latex(den)
    return f'T(s) = \\dfrac{{{num_str}}}{{{den_str}}}'


def calcular_itae(orden=3, omega=1, tipo="p"):
    """
    Calcula la función de transferencia ITAE óptima.

    Parámetros:
        orden : int
            Orden de la función de transferencia (1-6 para tipo 'p', 2-6 para tipo 'v'). Default 3.
        omega : float
            Frecuencia natural del sistema. Default 1.
        tipo : str
            'p' para cero error de posición (entrada escalón).
            'v' para cero error de velocidad (entrada rampa).
            Default 'p'.

    Retorna:
        T : control.TransferFunction
            Función de transferencia ITAE óptima.
    """

    # Coeficientes ITAE óptimos para entrada escalón (tipo "p")
    # Denominador: s^n + a[0]*omega*s^{n-1} + a[1]*omega^2*s^{n-2} + ... + omega^n
    coef_p = {
        1: [],
        2: [1.4],
        3: [1.75, 2.15],
        4: [2.1, 3.4, 2.7],
        5: [2.8, 5.0, 5.5, 3.4],
        6: [3.25, 6.60, 8.60, 7.45, 3.95]
    }

    # Coeficientes ITAE óptimos para entrada rampa (tipo "v")
    coef_v = {
        2: [3.2],
        3: [1.75, 3.25],
        4: [2.41, 4.93, 5.14],
        5: [2.19, 6.50, 6.30, 5.24],
        6: [3.58, 8.55, 13.0, 11.7, 6.60]
    }

    if tipo == "p":
        if orden not in coef_p:
            raise ValueError(f"Orden {orden} no soportado para tipo 'p'. Use 1-6.")
        coefs = coef_p[orden]
    elif tipo == "v":
        if orden not in coef_v:
            raise ValueError(f"Orden {orden} no soportado para tipo 'v'. Use 2-6.")
        coefs = coef_v[orden]
    else:
        raise ValueError("Tipo debe ser 'p' o 'v'")

    # Construir denominador:
    # s^n + a[0]*omega*s^{n-1} + a[1]*omega^2*s^{n-2} + ... + omega^n
    den = [1.0]
    for i, a in enumerate(coefs):
        den.append(a * omega**(i + 1))
    den.append(omega**orden)

    # Construir numerador
    if tipo == "p":
        # Numerador = omega^n (ganancia unitaria en DC)
        num = [omega**orden]
    else:
        # Para tipo "v": num = a1*omega^{n-1}*s + omega^n
        # donde a1 es el coeficiente del término en s del denominador
        a1 = coefs[-1]
        num = [a1 * omega**(orden - 1), omega**orden]

    T = ctrl.tf(num, den)
    return T



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
