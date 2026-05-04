"""
asigne_polos.py
Diseño de controladores por asignación de polos usando la matriz de Sylvester.

Equivalente en Python de la función asigne_polos.jl (Julia/ControlSystems).
Dependencias: numpy, control (python-control)
Leonardo Bermeo 2026
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
    Calcula un controlador C(s) por asignación de polos con realimentación unitaria.

    Parámetros
    ----------
    planta : control.TransferFunction
        Función de transferencia de la planta.
    polos : array_like
        Vector con los polos deseados del sistema en lazo cerrado.

    Retorna
    -------
    C  : control.TransferFunction  – Controlador.
    T  : control.TransferFunction  – Función de transferencia en lazo cerrado.
    Gur: control.TransferFunction  – Transferencia de control  U/R = T/G.

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
    C = ctrl.tf(X, Y)
    T = ctrl.feedback(C * planta)
    T = ctrl.minreal(T)
    Gur = ctrl.minreal(T / planta)

    return C, T, Gur 








def dise_2p(P, T, polos_obs):
    """
    Calcula un controlador de dos parámetros.

    Parámetros
    ----------
    P : control.TransferFunction
        Función de transferencia de la planta.
    T : control.TransferFunction
        Función de transferencia de lazo cerrado deseada.
    polos_obs : array_like
        Polos del observador.

    Retorna
    -------
    C2p : control.TransferFunction
        Controlador de dos parametros

    T : control.TransferFunction
        Función de transferencia de lazo cerrado resultante.
    Gur : control.TransferFunction
        Función de transferencia de la señal de control (T / G).
    """
    polos_obs = np.array(polos_obs, dtype=complex)
    m = len(polos_obs)

    # Extraer numerador y denominador de la planta
    N = np.array(P.num[0][0], dtype=float)
    D = np.array(P.den[0][0], dtype=float)
    n = len(D) - 1

    if m < n - 1:
        raise ValueError("dise_2p: número de polos del observador insuficientes")

    # Polinomio del observador
    Dhb = np.poly(polos_obs)

    # H = T / N  (equivalente a T * (1/N))
    Nt = np.array(T.num[0][0], dtype=float)
    Dt = np.array(T.den[0][0], dtype=float)
    # T / tf(N) => num=Nt, den=conv(Dt, N)
    Nh1 = Nt
    Dh1 = np.convolve(Dt, N)

    # Simplificar H cancelando factores comunes
    H = ctrl.minreal(ctrl.tf(Nh1, Dh1))
    Nh, Dh = ctrl.tfdata(H)
    Nh=Nh[0][0]
    Dh=Dh[0][0]

    # L = conv(Nh, Dhb),  F = conv(Dh, Dhb)
    L = np.convolve(Nh, Dhb)
    F = np.convolve(Dh, Dhb)

    n = len(D) - 1
    m = len(polos_obs)

    # Na: versión de N rellenada con ceros a la izquierda para igualar longitud de D
    l = n - (len(N) - 1)
    Na = np.zeros(len(D))
    Na[l:] = N

    # Construir la matriz de Sylvester Sm de tamaño (n+m+1) x 2(m+1)
    filas = n + m + 1
    cols = 2 * (m + 1)
    Sm = np.zeros((filas, cols))

    for k in range(m + 1):
        Sm[k:k + n + 1, k] = D
        Sm[k:k + n + 1, k + m + 1] = Na

    if m > n - 1:
        # Eliminar la columna m (índice m, que es la columna m+1 en notación 1-based)
        Smj = np.delete(Sm, m, axis=1)

        if np.linalg.matrix_rank(Smj) >= n + m + 1:
            XY = np.linalg.lstsq(Smj, F, rcond=None)[0]
            A_c = np.concatenate([XY[:m], [0.0]])
            M_c = XY[m:]
        else:
            XY = np.linalg.lstsq(Sm, F, rcond=None)[0]
            A_c = XY[:m + 1]
            M_c = XY[m + 1:]
    elif m == n - 1:
        XY = np.linalg.lstsq(Sm, F, rcond=None)[0]
        A_c = XY[:m + 1]
        M_c = XY[m + 1:2 * m + 2]

    A_c = np.real(A_c)
    M_c = np.real(M_c)
    L = np.real(L)

    Kr = ctrl.tf(L.flatten(), A_c.flatten())
    Ky = ctrl.tf(M_c.flatten(), A_c.flatten())
    
    num = [[L, -M_c.flatten()]]
    den = [[A_c.flatten(), A_c.flatten()]]
    C2p = ctrl.tf(num, den)
    # Lazo cerrado: T = Kr * feedback(P, Ky)
    Tcl = ctrl.minreal(Kr * ctrl.feedback(P, Ky))

    # Señal de control: Gur = T / P
    Gur = ctrl.minreal(Tcl / P)
    return C2p, Tcl, Gur

# ──────────────────────────────────────────────
# Ejemplo de uso
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Planta de ejemplo: P = 1 / (s^2 + 3s + 2)
    P = ctrl.tf([400], [1, 4, 0])

    # Lazo cerrado deseado: T = 4 / (s^2 + 3s + 4)
    T = ctrl.tf([1], [1, 1.4, 1])

    # Polos del observador
    polos_obs = [-5, -6]

    C2p, Tcl, Gur = dise_2p(P, T, polos_obs)

    print("=== Controlador de dos parámetros ===\n")
    print(C2p)
    print("\nFunción de lazo cerrado Tcl:")
    print(Tcl)
    print("\nSeñal de control Gur:")
    print(Gur)
