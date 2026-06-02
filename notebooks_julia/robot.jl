# ============================================================
#  parametros_LQR.jl
#  Modelo LQR para robot de equilibrio en dos ruedas
#  
# ============================================================

using LinearAlgebra    # rank, Diagonal, diagm
using Printf           # @printf
using ControlSystems   # ss(), ctrb(), lqr()
using Plots            # plot(), plot!(), savefig()
# Instalar dependencias (una sola vez):
#   using Pkg
#   Pkg.add(["ControlSystems", "Plots"])

# ------------------------------------------------------------
# 1. Parámetros físicos del sistema
# ------------------------------------------------------------
m        = 0.035                   # Masa de cada rueda [kg]
r        = 0.0672 / 2              # Radio de la rueda [m]
inercia  = 0.5 * m * r^2          # Momento de inercia de la rueda [kg·m²]

M        = 1.000 - 2*m            # Masa del cuerpo del vehículo [kg]  (masa total = 1.000 kg)
L        = 0.5 * 0.0766           # Distancia del centro de masa al centro del chasis [m]
                                   

J_centroide = (1/12) * M * (0.0766^2 + 0.0575^2)

# Momento de inercia del cuerpo girando alrededor de su centro de masa [kg·m²]
# 0.0766: altura total (desde la placa base)
# 0.0575: mitad de la longitud de la placa base

d        = 0.1612                  # Ancho de vía (distancia entre ruedas) [m]

J_Y_delta = (1/12) * M * (0.0766^2 + 0.0575^2)
# Momento de inercia del cuerpo girando alrededor del eje Y [kg·m²]

g        = 9.8                     # Aceleración gravitacional [m/s²]

# ------------------------------------------------------------
# 2. Términos auxiliares para las matrices A y B
# ------------------------------------------------------------
Q_aux = J_centroide * M + (J_centroide + M*L^2) * (2*m + 2*inercia/r^2)

A_23 = -(M^2 * L^2 * g) / Q_aux
A_43 =  M * L * g * (M + 2*m + 2*inercia/r^2) / Q_aux

B_21 = (J_centroide + M*L^2 + M*L*r) / (Q_aux * r)
B_22 = B_21

B_41 = -(M*L/r + M + 2*m + 2*inercia/r^2) / Q_aux
B_42 = B_41

B_61 =  1 / (r * (m*d + inercia*d/r^2 + 2*J_Y_delta/d))
B_62 = -B_61

# ------------------------------------------------------------
# 3. Matrices del sistema (espacio de estados)
#    Estado: x = [posición, vel. lineal, ángulo, vel. angular, ángulo giro, vel. giro]
# ------------------------------------------------------------
A = [0  1    0    0  0  0;
     0  0  A_23   0  0  0;
     0  0    0    1  0  0;
     0  0  A_43   0  0  0;
     0  0    0    0  0  1;
     0  0    0    0  0  0]

B = (inercia/r) .* [0    0  ;
                     B_21 B_22;
                     0    0  ;
                     B_41 B_42;
                     0    0  ;
                     B_61 B_62]

# Matrices de salida y transmisión directa (observamos todo el estado)
C = Matrix{Float64}(I, 6, 6)
D = zeros(6, 2)

# Sistema en espacio de estados (tiempo continuo)
sys = ss(A, B, C, D)

# ------------------------------------------------------------
# 4. Verificación de controlabilidad
#    Wr = [B, AB, A²B, ..., Aⁿ⁻¹B]  (matriz de controlabilidad)
# ------------------------------------------------------------
Wr = ctrb(sys)          


# --------------------------------------------------------
    # 5. Diseño del controlador LQR
    #    Minimiza: J = ∫ (xᵀQx + uᵀRu) dt
    # --------------------------------------------------------

Q_lqr = diagm([7700.0, 0.0, 0.0, 1600.0, 500.0, 0.0])

R_lqr = [1.0  0.0;
            0.0  1.0]

# lqr() de ControlSystems — idéntico al lqr(A,B,Q,R) de MATLAB
K = lqr(sys, Q_lqr, R_lqr)

println("\nMatriz de ganancias K:")
display(K)

  