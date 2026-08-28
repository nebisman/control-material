#= =========================================================================
   identificacion_motorreductor.jl
   Identificación experimental de un motorreductor DC con encoder
   (uxcell 12 V, reducción 21.3:1, encoder Hall de cuadratura)

   Estima por mínimos cuadrados:
     R      : resistencia de armadura            [Ω]
     Ke     : constante de fem (eje del motor)   [V·s/rad]
     Kt     : constante de par (Kt = Ke en SI)   [N·m/A]
     η      : eficiencia de la caja de engranajes [-]
     τ_f    : par de fricción referido al motor  [N·m]

   Modelo estacionario:
     V = R·i + Ke·ω_m                (eléctrico)
     τ_sal = N·η·(Kt·i − τ_f)        (mecánico, eje de salida)
     ω_sal = ω_m / N

   NOTA: los vectores de datos incluidos son SINTÉTICOS (ejemplo).
   Reemplácelos por sus mediciones antes de reportar resultados.
   ========================================================================= =#

using Plots, LaTeXStrings, Printf
using LinearAlgebra
pgfplotsx()

# Color primario del estilo de casa
const CIAN = colorant"#00aad4"

# -------------------------------------------------------------------------
# 1. Parámetros del fabricante (citables — hoja de datos uxcell)
# -------------------------------------------------------------------------
N       = 21.3      # relación de reducción [-]
V_nom   = 12.0      # voltaje nominal [V]
ppr_enc = 11        # pulsos por revolución del encoder (eje del motor) — verificar

# -------------------------------------------------------------------------
# 2. Datos experimentales (REEMPLAZAR por mediciones propias)
# -------------------------------------------------------------------------
# Ensayo 1 — rotor bloqueado (voltajes bajos para no sobrecalentar)
V_rb = [2.0, 3.0, 4.0, 5.0, 6.0]                    # [V]
I_rb = [0.1701, 0.2493, 0.3376, 0.4291, 0.4977]     # [A]

# Ensayo 2 — sin carga: voltaje, corriente en vacío y velocidad de salida
V_sc   = [6.0, 8.0, 10.0, 12.0]                     # [V]
i0_sc  = [0.0457, 0.0482, 0.0471, 0.0454]           # [A]
rpm_sc = [96.37, 129.88, 165.11, 201.90]            # [rpm] eje de SALIDA

# Ensayo 3 — con carga a V = 12 V (freno de Prony o peso con carrete)
#   τ_sal = m·g·r  si se levanta una masa m con un carrete de radio r
V_ld   = 12.0                                        # [V]
i_ld   = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40] # [A]
τ_ld   = [0.0220, 0.0425, 0.0645, 0.0847,
          0.1083, 0.1265, 0.1457]                    # [N·m] eje de salida
ω_ld   = [19.88, 18.78, 17.67, 16.57,
          15.47, 14.36, 13.26]                       # [rad/s] eje de salida

# -------------------------------------------------------------------------
# 3. Estimación de parámetros
# -------------------------------------------------------------------------
# 3.1 Resistencia de armadura: V = R·i  (mínimos cuadrados por el origen)
R̂ = dot(I_rb, V_rb) / dot(I_rb, I_rb)

# 3.2 Constante de fem: V − R·i0 = Ke·ω_m, con ω_m = N·ω_sal
ω_m_sc = N .* rpm_sc .* (2π/60)          # velocidad del motor [rad/s]
y_sc   = V_sc .- R̂ .* i0_sc
K̂e     = dot(ω_m_sc, y_sc) / dot(ω_m_sc, ω_m_sc)
K̂t     = K̂e                              # en unidades SI

# 3.3 Eficiencia y fricción: τ_sal = a·i + b, con a = N·η·Kt, b = −N·η·τ_f
A      = [i_ld ones(length(i_ld))]
a, b   = A \ τ_ld
η̂      = a / (N * K̂t)
τ̂_f    = -b / (N * η̂)

# 3.4 Eficiencia global por balance de potencias (incluye fricción)
#     η_glob = τ_sal·ω_sal / (V·i − i²R)
η_glob = (τ_ld .* ω_ld) ./ (V_ld .* i_ld .- i_ld.^2 .* R̂)

# 3.5 Predicciones derivadas a 12 V
I_par   = V_nom / R̂                              # corriente de parada [A]
τ_parada = N * η̂ * (K̂t * I_par - τ̂_f)            # par de parada, salida [N·m]
ω0_sal  = (V_nom - R̂ * τ̂_f / K̂t) / (K̂e * N)      # velocidad sin carga [rad/s]

# -------------------------------------------------------------------------
# 4. Resultados en consola
# -------------------------------------------------------------------------
println("═"^60)
println(" Parámetros estimados del motorreductor")
println("═"^60)
@printf("  R   = %.2f Ω\n",        R̂)
@printf("  Ke  = %.4f V·s/rad (eje del motor)\n", K̂e)
@printf("  Kt  = %.4f N·m/A\n",    K̂t)
@printf("  η   = %.3f  (caja de engranajes)\n",   η̂)
@printf("  τ_f = %.2e N·m (fricción, eje del motor)\n", τ̂_f)
println("─"^60)
@printf("  Par de parada (salida, 12 V): %.3f N·m = %.2f kg·cm\n",
        τ_parada, τ_parada * 100/9.81)
@printf("  Corriente de parada estimada: %.2f A\n", I_par)
println("═"^60)

# -------------------------------------------------------------------------
# 5. Gráficas
# -------------------------------------------------------------------------
# 5.1 Curva par–velocidad a 12 V: recta del modelo + datos
ω_grid = range(0, ω0_sal; length = 100)
i_grid = (V_nom .- K̂e * N .* ω_grid) ./ R̂
τ_grid = N * η̂ .* (K̂t .* i_grid .- τ̂_f)

p1 = plot(ω_grid, τ_grid;
    lw = 2, color = CIAN,
    label = "Modelo ajustado",
    xlabel = L"\omega_{sal}\ [\si{rad/s}]",
    #ylabel = L"\tau_{sal}\ [\si{N.m}]",
   # title  = L"Curva par--velocidad a $\si{12}{V}$",
    legend = :topright)
scatter!(p1, ω_ld, τ_ld;
    color = :black, marker = :circle, ms = 4,
    label = "Mediciones")
# savefig(p1, "par_velocidad.pdf")

# # 5.2 Ajuste par–corriente (recta de mínimos cuadrados)
# i_fit = range(0.9minimum(i_ld), 1.05maximum(i_ld); length = 50)
# p2 = plot(i_fit, a .* i_fit .+ b;
#     lw = 2, color = CIAN,
#     label = L"\tau_{\mathrm{sal}} = a\,i + b",
#     xlabel = L"i\ [\si{A}]",
#     ylabel = L"\tau_{\mathrm{sal}}\ [\si{N.m}]",
#     title  = "Ajuste par--corriente",
#     legend = :topleft)
# scatter!(p2, i_ld, τ_ld;
#     color = :black, marker = :circle, ms = 4,
#     label = "Mediciones")
# savefig(p2, "par_corriente.pdf")

# # 5.3 Eficiencia global por punto de operación
# p3 = scatter(τ_ld, η_glob;
#     color = CIAN, marker = :circle, ms = 5,
#     label = "Balance de potencias",
#     xlabel = L"\tau_{\mathrm{sal}}\ [\si{N.m}]",
#     ylabel = L"\eta_{\mathrm{global}}\ [-]",
#     title  = "Eficiencia global vs.\\ carga",
#     ylims  = (0, 1), legend = :bottomright)
# hline!(p3, [η̂]; lw = 2, ls = :dash, color = :black,
#     label = L"\hat{\eta}\ \textrm{(sin fricci\'on)}")
# savefig(p3, "eficiencia.pdf")

# println("\nFiguras guardadas: par_velocidad.pdf, par_corriente.pdf, eficiencia.pdf")
