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
rank_Wr = rank(Wr)

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

Acl = A - B * K                         # Matriz de lazo cerrado
 
C5  = [0.0  0.0  0.0  0.0  1.0  0.0]   # Salida: x₅ (ángulo de giro ψ)
 
# Ganancia DC del canal diferencial (u₁ − u₂) hacia x₅
   DC_dif = (C5 * inv(Acl) * (B[:, 1] - B[:, 2]))[1]
    kr     = -1.0 / DC_dif
 
    @printf "\nPrecompensador kr (seguimiento x₅) = %.6f\n" kr
 
    # --------------------------------------------------------
    # 7. Sistema de lazo cerrado con seguimiento de x₅
    #
    #    Entrada escalar r → acción diferencial [+kr; −kr]
    #    ẋ = (A−BK)·x  +  B·[+kr; −kr]·r
    #    y  = I₆·x   (observamos todos los estados)
    # --------------------------------------------------------
    B_r   = B * [kr; -kr]                  # Vector de entrada efectivo (6×1)
    C_all = Matrix{Float64}(I, 6, 6)       # Salida = estado completo
    D_r   = zeros(6, 1)
 
    sys_lc = ss(Acl, B_r, C_all, D_r)     # Sistema para lsim
 
    # --------------------------------------------------------
    # 8. Simulación con lsim — respuesta al escalón unitario en x₅
    # --------------------------------------------------------
    t_sim = 0.0:0.01:5.0                           # Vector de tiempo [s]
    ref   = ones(1, length(t_sim))                 # Escalón unitario (1 rad ≈ 57°)
 
    Y, t_out, X = lsim(sys_lc, ref, t_sim , [0.1, 0, 0, 0, 0, 0])        # Y: 6×N,  X: 6×N
 
    # Reconstruir señales de control: u = −K·x + [+kr; −kr]·r
    U = -K * X .+ [kr; -kr] * ref                 # 2×N
 
    # --------------------------------------------------------
    # 9. Graficación — 3×2: estados + controles
    # --------------------------------------------------------
    using LaTeXStrings
 
    # Fila 1 — Giro lateral: x₅ y x₆  (la salida seguida)
    p1 = plot(t_out, Y[5, :],
              label = L"x_5\ —\ \psi\ \mathrm{[rad]}",
              color = :purple, linewidth = 2,
              ylabel = "rad / rad·s⁻¹")
    plot!(p1, t_out, [1.0 for _ in t_out],
              label = "referencia",
              color = :black, linewidth = 1, linestyle = :dot)
    plot!(p1, t_out, Y[6, :],
              label = L"x_6\ —\ \dot{\psi}\ \mathrm{[rad/s]}",
              color = :magenta, linewidth = 2, linestyle = :dash)
    title!(p1, "Giro lateral ψ  (salida seguida)")
 
    # Fila 1 — Ángulo de inclinación: x₃ y x₄
    p2 = plot(t_out, Y[3, :] .* (180/π),
              label = L"x_3\ —\ \theta\ \mathrm{[°]}",
              color = :red, linewidth = 2,
              ylabel = "° / rad·s⁻¹")
    plot!(p2, t_out, Y[4, :],
              label = L"x_4\ —\ \dot{\theta}\ \mathrm{[rad/s]}",
              color = :orange, linewidth = 2, linestyle = :dash)
    title!(p2, "Ángulo de inclinación θ")
 
    # Fila 2 — Posición y velocidad lineal
    p3 = plot(t_out, Y[1, :],
              label = L"x_1\ —\ \mathrm{posicion\ [m]}",
              color = :blue, linewidth = 2,
              ylabel = "m / m·s⁻¹")
    plot!(p3, t_out, Y[2, :],
              label = L"x_2\ —\ v\ \mathrm{[m/s]}",
              color = :teal, linewidth = 2, linestyle = :dash)
    title!(p3, "Posición y velocidad lineal")
 
    # Fila 2 — Señales de control
    p4 = plot(t_out, U[1, :],
              label = L"u_1\ —\ \mathrm{motor\ izq.\ [N\cdot m]}",
              color = :blue, linewidth = 2,
              ylabel = "N·m")
    plot!(p4, t_out, U[2, :],
              label = L"u_2\ —\ \mathrm{motor\ der.\ [N\cdot m]}",
              color = :red, linewidth = 2, linestyle = :dash)
    title!(p4, "Señales de control")
 
    fig = plot(p1, p2, p3, p4,
               layout  = (2, 2),
               size    = (1000, 600),
               xlabel  = "Tiempo [s]",
               legend  = :topright,
               suptitle = "Respuesta al escalón en x₅ (ψ = 1 rad)  —  LQR con seguimiento",
               plot_title_fontsize = 11,
               margin  = 5Plots.mm)
 
    display(fig)
    savefig(fig, "respuesta_escalon_x5.png")
    println("\nGráfica guardada en: respuesta_escalon_x5.png")
 
    # --------------------------------------------------------
    # 10. Resumen numérico
    # --------------------------------------------------------
    println("\n--- Resumen de la simulación ---")
    @printf "  Referencia x₅       :  1.000000 rad\n"
    @printf "  x₅ final (t=%.1f s) : %+.6f rad\n"  t_out[end]  Y[5,end]
    @printf "  Error en régimen    : %+.6f rad\n"   1.0 - Y[5,end]
    @printf "  Sobreimpulso x₅     : %+.4f rad (%.2f %%)\n" maximum(Y[5,:])  (maximum(Y[5,:])-1)*100
    @printf "  x₃ max (inclinación): %+.4f °\n"    maximum(abs.(Y[3,:])) * (180/π)
    @printf "  |u| máximo          : %+.4f N·m\n"  maximum(abs.(U))
