


import numpy as np
from scipy.signal import StateSpace, cont2discrete
import control as ct


def float2hex(value):
    val_binary = struct.pack('>f', value)
    return val_binary.hex()


 
def generate_controller_code(controller, sampling_time):
    """
    Generates C code for a digital filter from a state-space representation.
 
    Parameters
    ----------
    A, B, C, D : array_like
        State-space matrices of the discrete-time system.
    """

    struct = len(controller.den[0])

    if struct == 1:
        con = ct.tf(ct.tf(controller.num[0][0], controller.den[0][0]))
        N1, D1 = ct.tfdata(con)
        N1 = N1[0][0]
        D1 = D1[0][0]
        N1 = N1 / D1[0]
        D1 = D1 / D1[0]

        if len(N1) == len(D1):
            d1 = N1[0]
            N1 = N1 - d1* D1
            N1 = N1[1:]
        else:
            d1 = 0

        DB = np.array([-D1[1:]])
        DB = DB.T
        size = len(D1)-2
        In_1 = np.eye(size)
        ZR = np.zeros((1,size))
        Acon = np.block([[In_1], [ZR]])
        Acon = np.block([DB, Acon])
        Bcon = np.array([N1]).T
        Ccon = np.append([1], ZR)
        Dcon = np.array([d1])



    elif struct == 2:
        con1 = ct.tf(ct.tf(controller.num[0][0], controller.den[0][0]))
        con2 = ct.tf(ct.tf(controller.num[0][1], controller.den[0][1]))
        N1, D1 = ct.tfdata(con1)
        N2, D2 = ct.tfdata(con2)
        N1 = N1[0][0]
        D1 = D1[0][0]
        N1 = N1 / D1[0]
        D1 = D1 / D1[0]
        N2 = N2[0][0]
        D2 = D2[0][0]
        N2 = N2 / D2[0]
        D2 = D2/ D2[0]

        if len(N1) == len(D1):
            d1 = N1[0]
            N1 = N1 - d1* D1
            N1 = N1[1:]
        else:
            d1 = 0

        if len(N2) == len(D2):
            d2 = N2[0]
            N2 = N2 - d2* D2
            N2 = N2[1:]
        else:
            d2 = 0

        DB = np.array([-D1[1:]])
        DB = DB.T
        size = len(D1)-2
        In_1 = np.eye(size)
        ZR = np.zeros((1,size))
        Acon = np.block([[In_1], [ZR]])
        Acon = np.block([DB, Acon])
        B1 = np.array([N1]).T
        B2 = np.array([N2]).T
        Bcon = np.block([B1,B2])
        Ccon = np.append([1], ZR)
        Dcon = np.block([d1, d2])

    Ad, Bd, Cd, Dd, dt = cont2discrete((Acon, Bcon, Ccon, Dcon), sampling_time, method='bilinear')


 
   
 
    # If the system has non-zero eigenvalues, rebuild from a state-space object
    sys = StateSpace(Ad, Bd, Cd, Dd, dt= dt)
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    


    A = A.astype(np.float32)
    B = B.astype(np.float32)
    C = C.astype(np.float32)
    D = D.astype(np.float32)

    n = A.shape[0]  # number of states
    d_val = D[0, 0]
 
    code = "float computeController(float e){\n"
    code += "    // This function computes the control law for a discretized controller Cd(z)\n"
    code += "    // e = ref - y is the current error in the controlled system\n"
    code += "\n\n"
 
    # ---- A matrix constants ----
    code += "    // The following constants define the A matrix\n"
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0 and A[i, j] != 1:
                code += f"    const float a{i+1}_{j+1} = {A[i, j]:.32f};\n"
    code += "\n"
 
    # ---- B matrix constants ----
    code += "    // The following constants define the B matrix\n"
    for i in range(n):
        if B[i, 0] != 0 and B[i, 0] != 1:
            code += f"    const float b{i+1} = {B[i, 0]:.32f};\n"
    code += "\n"
 
    # ---- C matrix constants ----
    code += "    // The following constants define the C matrix\n"
    for i in range(C.shape[1]):
        if C[0, i] != 0 and C[0, i] != 1:
            code += f"    const float c{i+1} = {C[0, i]:.32f};\n"
    code += "\n"
 
    # ---- D constant ----
    code += "    // The following constant define the D scalar\n"
    if d_val != 0 and d_val != 1:
        code += f"    const float d = {d_val:.32f};\n"
    code += "\n"
 
    # ---- Static state variables x[n] ----
    code += "    // The following variables represent the states x[n]\n"
    code += "    // in the state-space representation. They must be declared\n"
    code += "    // as static to retain their values between function calls.\n"
    for i in range(n):
        code += f"    static float x{i+1} = 0;\n"
    code += "\n"
 
    # ---- New state variables x[n+1] ----
    code += "    // The following variables are the new computed states x[n+1]\n"
    code += "    // of the state space representation\n"
    for i in range(n):
        code += f"    float x{i+1}_new = 0;\n"
    code += "\n"
 
    # ---- Output variable ----
    code += "    // The following variable is control signal u  \n"
    code += "    // it also must be declared as static to retain its value between function calls. \n"
    code += "    float u = 0;\n"
    code += "\n"
 
    # ---- Filter computation block ----
    code += "    /*************************************************\n"
    code += "                THIS IS THE CONTROLLER'S CODE\n"
    code += "    **************************************************/\n\n"
    code += "    // Compute the new state x[n+1] = A*x[n] + B*e[n]\n"
 
    for i in range(n):
        code += f"    x{i+1}_new = "
        terms = []
        for j in range(n):
            if A[i, j] != 0:
                if A[i, j] == 1:
                    terms.append(f"x{j+1}")
                else:
                    terms.append(f"a{i+1}_{j+1}*x{j+1}")
        if B[i, 0] != 0:
            if B[i, 0] == 1:
                terms.append("e")
            else:
                terms.append(f"b{i+1}*e")
        code += " + ".join(terms) + ";\n"
 
    code += "\n"
 
    # ---- Output equation ----
    code += "    // Compute the control output u[n] = C*x[n] + D*e[n]\n"
    code += "    u = "
    terms = []
    for i in range(C.shape[1]):
        if C[0, i] != 0:
            if C[0, i] == 1:
                terms.append(f"x{i+1}")
            else:
                terms.append(f"c{i+1}*x{i+1}")
    if d_val != 0:
        if d_val == 1:
            terms.append("u")
        else:
            terms.append("d*e")
    code += " + ".join(terms) + ";\n"
 
    code += "\n"
 
    # ---- Update states ----
    code += "    // Make the next state the current state: x[n] <- x[n+1]\n"
    for i in range(n):
        code += f"    x{i+1} = x{i+1}_new;\n"
    code += "\n"
    code += "    // now, the filtered signal is available to the main routine\n"
    code += "    return u;\n"
    code += "}\n"
 
    with open("filter.h", "w") as fh:
        fh.write(code)
 
    print("code for filter has been generated")
 
 
if __name__ == "__main__":
    # Example usage:
    # Example usage:
    from metodo_algebraico import asigne_polos
   
    P = ct.tf([400], [1, 4, 0])

    # Lazo cerrado deseado: T = 4 / (s^2 + 3s + 4)
    T = ct.tf([1], [1, 1.4, 1])

    # Polos del observador
    polos_obs = [-5, -6, -7, -8 ]

    K, T, Gur,  = asigne_polos(P, polos_obs)
    print(K)

    generate_controller_code(K,0.01)

