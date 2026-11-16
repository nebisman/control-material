# ==============================================================================
# Figuras de la presentación del robot PIR: control de velocidad lineal,
# de velocidad de giro y de ambas, sobre el modelo LINEAL.
#   figuras/pir_velocidad.pdf   respuesta inversa (cero en s = +z)
#   figuras/pir_giro.pdf        P vs. PI ante un torque perturbador
#   figuras/pir_trayectoria.pdf trayectoria en el plano con v_ref y ω_ref
# Parámetros iguales a los de ../codigo/control_lqr.jl, pero con las entradas
# en unidades físicas (torques en N·m, sin el escalado J_r/r del código).
# Ejecutar con: julia figuras_control.jl
# ==============================================================================

using LinearAlgebra
using ControlSystems
using Plots
using PGFPlotsX
using LaTeXStrings
using Printf
pgfplotsx()

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage[utf8]{inputenc}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage[T1]{fontenc}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{siunitx}")

# ------------------------------------------------------------------------------
# 1. Parámetros y modelo lineal (notación de la presentación)
# ------------------------------------------------------------------------------
m  = 0.035;  r = 0.0672/2;  Jr = 0.5*m*r^2
M  = 1.000 + 2m;  l = 0.0766/2;  d = 0.1612;  g = 9.8
Jc = (1/12)*M*(0.0766^2 + 0.0575^2);  Jδ = Jc

Q1 = Jc + M*l^2
Q2 = M + 2m + 2Jr/r^2
Q3 = m*d + d*Jr/r^2 + 2Jδ/d
Δ0 = Q1*Q2 - M^2*l^2

a23 = -M^2*l^2*g/Δ0
a43 =  M*g*l*Q2/Δ0
b2  =  (Q1 + M*l*r)/(r*Δ0)
b4  = -(Q2 + M*l/r)/Δ0
b6  =  1/(r*Q3)

# Avance: x_v = [v, θc, ωc], entrada τΣ
Av = [0 a23 0; 0 0 1; 0 a43 0];  Bv = [b2; 0; b4];  Cv = [1 0 0]
Gv = ss(Av, Bv, Cv, 0)
z  = sqrt(M*g*l/(Q1 + M*l*r))
@printf("Polos de G_v: %s\n", string(round.(real.(poles(Gv)), digits = 2)))
@printf("Ceros de G_v: %s   (±z = ±%.2f rad/s)\n",
        string(round.(real.(tzeros(Gv)), digits = 2)), z)
@printf("det ctrb(A_v,B_v) = %.3e\n", det(ctrb(Av, Bv)))

# Avance con acción integral: x = [v, θc, ωc, ξv],  ξv' = v_ref - v
Ava = [Av zeros(3); -Cv 0];  Bva = [Bv; 0]
# Giro con acción integral: x = [ωδ, ξδ],  ξδ' = ω_ref - ωδ
Aδa = [0.0 0; -1 0];  Bδa = [b6; 0]

# ------------------------------------------------------------------------------
# 2. Estilo
# ------------------------------------------------------------------------------
azul = "#1D477B";  cian = "#00aad4";  rojo = "#C4161D"
estilo = (guidefontsize = 12, tickfontsize = 10, legendfontsize = 10,
          framestyle = :box, lw = 1.5, fg_color_legend = :gray)
tlabel = L"\text{Tiempo } t\ [\si{\second}]"
path   = joinpath(@__DIR__, "figuras", "")
mkpath(path)

# ------------------------------------------------------------------------------
# 3. Velocidad lineal: escalón de v_ref con un lazo lento y uno rápido
#    τΣ = -K x  (la referencia entra solo por el integrador)
# ------------------------------------------------------------------------------
vref = 0.3;  t0 = 0.2;  t = 0:0.001:2.5
soa  = Dict(:subplot => Dict("scale only axis" => nothing, "width" => "55mm"))  # ejes alineados
uref = [ti >= t0 ? vref : 0.0 for ti in t]'
Bref = [0; 0; 0; 1.0]

function lazo_v(qξ)
    K = lqr(Continuous, Ava, Bva, diagm([10.0, 100, 1, qξ]), fill(100.0, 1, 1))
    y, _, _ = lsim(ss(Ava - Bva*K, Bref, Matrix(1.0I, 4, 4), 0), uref, t)
    return y
end
yl = lazo_v(1e3);  yr = lazo_v(1e5)
@printf("v mínima: lento %.3f m/s, rápido %.3f m/s\n", minimum(yl[1, :]), minimum(yr[1, :]))

p1 = plot(t, uref'; lc = :gray, ls = :dash, lw = 1.25, label = L"v_{\mathrm{ref}}",
          ylabel = L"v\ [\si{\meter/\second}]", estilo..., legend = :bottomright,
          xformatter = _ -> "", bottom_margin = 1Plots.mm, extra_kwargs = soa)
plot!(p1, t, yl[1, :]; lc = cian, label = "lento")
plot!(p1, t, yr[1, :]; lc = azul, label = "rápido")
p2 = plot(t, rad2deg.(yl[2, :]); lc = cian, label = "",
          ylabel = L"\theta_c\ [^\circ]", xlabel = tlabel, estilo...,
          yticks = 0:5:10, extra_kwargs = soa)
plot!(p2, t, rad2deg.(yr[2, :]); lc = azul, label = "")
hline!(p2, [0]; lc = :gray, lw = 0.75, label = "")
fig_v = plot(p1, p2; layout = (2, 1), link = :x, size = (280, 250))
savefig(fig_v, path * "pir_velocidad.pdf")

# ------------------------------------------------------------------------------
# 4. Velocidad de giro: P vs. PI ante un torque perturbador en t = 1.5 s
#    ωδ' = b6 (τΔ + τp)
# ------------------------------------------------------------------------------
ωref = 1.0;  tp = 1.5;  τp = -1e-3;  t = 0:0.001:4
ρδ = 1e5
kP = lqr(Continuous, fill(0.0, 1, 1), fill(b6, 1, 1), fill(1.0, 1, 1), fill(ρδ, 1, 1))[1]
Kδ = lqr(Continuous, Aδa, Bδa, diagm([1.0, 25.0]), fill(ρδ, 1, 1))
@printf("Giro P: k = %.4g,  polo %.2f (= -b6*sqrt(q/ρ) = %.2f)\n", kP, -b6*kP, -b6*sqrt(1/ρδ))
wr = [ti >= t0 ? ωref : 0.0 for ti in t]
tq = [ti >= tp ? τp : 0.0 for ti in t]
# P: entradas [ω_ref, τp];   PI: estado [ωδ, ξδ], entradas [ω_ref, τp]
sysP  = ss(fill(-b6*kP, 1, 1), [b6*kP b6], fill(1.0, 1, 1), 0)
sysPI = ss(Aδa - Bδa*Kδ, [0 b6; 1 0], [1.0 0], 0)
yP, _, _  = lsim(sysP,  [wr tq]', t)
yPI, _, _ = lsim(sysPI, [wr tq]', t)
@printf("Giro: error final P = %.3f rad/s, PI = %.4f rad/s\n", ωref - yP[end], ωref - yPI[end])

fig_δ = plot(t, wr; lc = :gray, ls = :dash, lw = 1.25, label = L"\omega_{\delta,\mathrm{ref}}",
             ylabel = L"\omega_\delta\ [\si{\radian/\second}]", xlabel = tlabel,
             ylims = (0, 1.25), estilo..., legend = :bottomright)
plot!(fig_δ, t, yP[1, :];  lc = cian, label = "P")
plot!(fig_δ, t, yPI[1, :]; lc = azul, label = "PI")
vline!(fig_δ, [tp]; lc = rojo, ls = :dot, lw = 1, label = "perturbación")
plot!(fig_δ; size = (300, 220))
savefig(fig_δ, path * "pir_giro.pdf")

# ------------------------------------------------------------------------------
# 5. Ambas: LQR conjunto sobre (τi, τd) con R = ρI
#    x = [v, θc, ωc, ξv, ωδ, ξδ];  τΣ = τi + τd,  τΔ = τd - τi
# ------------------------------------------------------------------------------
A6 = [Ava zeros(4, 2); zeros(2, 4) Aδa]
B6 = [Bva Bva; -Bδa Bδa]                      # columnas: τi, τd
Q6 = diagm([10.0, 100, 1, 1e4, 1.0, 25.0])
ρ  = 100.0
K6 = lqr(Continuous, A6, B6, Q6, Matrix(ρ*I, 2, 2))

# Comprobación del desacople: K6 = ½ [Kv -Kδ; Kv Kδ] con diseños separados (ρ/2)
Kv = lqr(Continuous, Ava, Bva, Q6[1:4, 1:4], fill(ρ/2, 1, 1))
Kδ = lqr(Continuous, Aδa, Bδa, Q6[5:6, 5:6], fill(ρ/2, 1, 1))
@printf("‖K6 - ½[Kv -Kδ; Kv Kδ]‖ = %.2e\n", norm(K6 - 0.5*[Kv -Kδ; Kv Kδ]))

t  = 0:0.002:7
ω0 = 1.5;  t1 = 2.0;  t2 = t1 + π/ω0         # media vuelta de radio v_ref/ω0
vr = [ti >= t0 ? vref : 0.0 for ti in t]
wr = [t1 <= ti < t2 ? ω0 : 0.0 for ti in t]
Br6 = [zeros(3, 2); 1 0; 0 0; 0 1]
y6, _, _ = lsim(ss(A6 - B6*K6, Br6, Matrix(1.0I, 6, 6), 0), [vr wr]', t)

# Cinemática en el plano: δ' = ωδ,  X' = v cos δ,  Y' = v sin δ
function plano(v, ω, t)
    h = t[2] - t[1];  δ = cumsum(ω)*h
    return cumsum(v .* cos.(δ))*h, cumsum(v .* sin.(δ))*h
end
Xr, Yr = plano(vr, wr, t)
X,  Y  = plano(y6[1, :], y6[5, :], t)
τ = -K6*y6
@printf("Ambas: max(|τΣ|+|τΔ|)/2 = %.4f N·m\n",
        maximum(abs.(τ[1, :] + τ[2, :]) + abs.(τ[2, :] - τ[1, :]))/2)

fig_xy = plot(Xr, Yr; lc = :gray, ls = :dash, lw = 1.25, label = "referencia",
              xlabel = L"X\ [\si{\meter}]", ylabel = L"Y\ [\si{\meter}]",
              aspect_ratio = :equal, estilo..., legend = :left)
plot!(fig_xy, X, Y; lc = azul, label = "robot")
scatter!(fig_xy, [0], [0]; mc = rojo, ms = 3, msw = 0, label = "")
plot!(fig_xy; size = (300, 220))
savefig(fig_xy, path * "pir_trayectoria.pdf")

# ------------------------------------------------------------------------------
# 6. Trayectoria curvilínea planificada hasta un círculo objetivo de radio 0.5 m
#    en un cuarto de 10 m × 10 m. El robot solo tiene la odometría de los
#    encoders (no mide su posición), así que no hay lazo externo: v_ref(t) y
#    ω_ref(t) se planifican de antemano y el LQR con acción integral de la
#    sección 5 los sigue. Sale de (1, 1) m con el rumbo a -135° de la dirección
#    hacia el objetivo y recorre una curva de Bézier cúbica hasta el centro.
#    Diferencia entre motores: torque constante en el motor izquierdo.
# ------------------------------------------------------------------------------
Lc = 10.0;  Xc, Yc, Rc = Lc/2, Lc/2, 0.5        # cuarto; centro y radio del círculo
X0, Y0 = 1.0, 1.0
δ0 = atan(Yc - Y0, Xc - X0) - deg2rad(135)      # rumbo inicial: -135° del objetivo
τm = -1e-3                                      # torque perturbador en el motor izquierdo

# Camino: Bézier cúbica de P0 (tangente = rumbo inicial) a P3 (centro), llegando con rumbo ψf
L1, L2, ψf = 1.2, 5.0, deg2rad(120)
P0 = [X0, Y0];  P3 = [Xc, Yc]
P1 = P0 + L1*[cos(δ0), sin(δ0)];  P2 = P3 - L2*[cos(ψf), sin(ψf)]
bez(u)   = (1 - u)^3*P0 + 3(1 - u)^2*u*P1 + 3(1 - u)*u^2*P2 + u^3*P3
dbez(u)  = 3(1 - u)^2*(P1 - P0) + 6(1 - u)*u*(P2 - P1) + 3u^2*(P3 - P2)
d2bez(u) = 6(1 - u)*(P2 - 2P1 + P0) + 6u*(P3 - 2P2 + P1)
uu = range(0, 1, length = 20001)
db = dbez.(uu);  d2b = d2bez.(uu)
κ  = [(p1[1]*p2[2] - p1[2]*p2[1])/norm(p1)^3 for (p1, p2) in zip(db, d2b)]  # curvatura con signo
ds = [(norm(db[i]) + norm(db[i+1]))/2*step(uu) for i in 1:length(uu)-1]   # arco entre nodos

# Perfil de velocidad sobre el camino: v ≤ vmax, |ω| = |κ| v ≤ ωmax, |κ| v² ≤ alat,
# |dv/dt| ≤ at; arranca y para en reposo (pasada hacia adelante y hacia atrás)
vmax, ωmax, alat, at = 0.3, 1.5, 0.3, 0.2
vc = [min(vmax, ωmax/abs(k), sqrt(alat/abs(k))) for k in κ]
vc[1] = 0.0;  vc[end] = 0.0
for i in 1:length(vc)-1;  vc[i+1] = min(vc[i+1], sqrt(vc[i]^2 + 2at*ds[i]));  end
for i in length(vc)-1:-1:1;  vc[i] = min(vc[i], sqrt(vc[i+1]^2 + 2at*ds[i]));  end
tn = [0; cumsum(2ds ./ (vc[1:end-1] + vc[2:end]))]           # tiempo en cada nodo
Tp = tn[end]
function interp1(x, y, xq)
    i = clamp(searchsortedlast(x, xq), 1, length(x) - 1)
    w = (xq - x[i])/(x[i+1] - x[i]);  return (1 - w)*y[i] + w*y[i+1]
end

t1 = 0.5;  tfin = t1 + Tp                                     # inicio y fin del plan
h  = 0.002;  tt = 0:h:tfin + 5
plan(f, t) = t1 <= t <= tfin ? interp1(tn, f, t - t1) : 0.0
Rf = [plan.(Ref(vc), tt) plan.(Ref(κ .* vc), tt)]'            # [v_ref; ω_ref = κ v_ref]
# lazo interno (lineal): entradas [v_ref, ω_ref, τ perturbador en el motor izquierdo]
y_in, _, _ = lsim(ss(A6 - B6*K6, [Br6 B6[:, 1]], Matrix(1.0I, 6, 6), 0),
                  [Rf; fill(τm, 1, length(tt))], tt)

# Cinemática en el plano (regla del trapecio): δ' = ωδ,  X' = v cos δ,  Y' = v sin δ
integ(f) = [0; cumsum((f[1:end-1] + f[2:end])/2*h)]
function pose(v, ω)
    δ = δ0 .+ integ(ω)
    return X0 .+ integ(v .* cos.(δ)), Y0 .+ integ(v .* sin.(δ)), δ
end
Xp, Yp, δp = pose(Rf[1, :], Rf[2, :])                         # plan
X,  Y,  δ  = pose(y_in[1, :], y_in[5, :])                     # robot
@printf("Plan: camino de %.2f m en %.1f s; radio mínimo %.2f m; max|ω_ref| = %.2f rad/s; el plan termina en (%.3f, %.3f)\n",
        sum(ds), Tp, 1/maximum(abs.(κ)), maximum(abs.(Rf[2, :])), Xp[end], Yp[end])
@printf("      robot a %.3f m del centro; distancia mínima a las paredes %.2f m; max|θc| = %.2f°; max|τ| = %.4f N·m\n",
        hypot(X[end] - Xc, Y[end] - Yc), minimum([X; Y; Lc .- X; Lc .- Y]),
        rad2deg(maximum(abs.(y_in[2, :]))), maximum(abs.(K6*y_in)))

φc = range(0, 2π, length = 200)
cx, cy = Xc .+ Rc*cos.(φc), Yc .+ Rc*sin.(φc)
la = 0.8                                         # largo de la flecha del rumbo inicial [m]
# ejes de tamaño fijo: cuadrados en el plano (círculo redondo), alineados en el tiempo
eje(w, hgt) = Dict(:subplot => Dict("scale only axis" => nothing, "width" => w, "height" => hgt))
p_cuarto = plot([0, Lc, Lc, 0, 0], [0, 0, Lc, Lc, 0]; lc = :black, lw = 1, label = "",
                xlabel = L"X\ [\si{\meter}]", ylabel = L"Y\ [\si{\meter}]",
                xlims = (-0.3, Lc + 0.3), ylims = (-0.3, Lc + 0.3), legend = false,
                estilo..., title = "Cuarto de 10 m × 10 m", extra_kwargs = eje("38mm", "38mm"))
plot!(p_cuarto, Xp, Yp; lc = :gray, ls = :dash, lw = 1.25)
plot!(p_cuarto, X, Y; lc = azul)
plot!(p_cuarto, cx, cy; lc = rojo, ls = :dash, lw = 1)
plot!(p_cuarto, [X0, X0 + la*cos(δ0)], [Y0, Y0 + la*sin(δ0)]; lc = rojo, lw = 1.5,
      arrow = arrow(:closed, :head, 7, 6))
scatter!(p_cuarto, [X0], [Y0]; mc = rojo, ms = 3, msw = 0)
annotate!(p_cuarto, X0 + 0.1, Y0 + 0.8, text(L"-135^\circ", 9, :left))
p_zoom = plot(cx, cy; lc = rojo, ls = :dash, lw = 1, label = "objetivo",
              xlabel = L"X\ [\si{\meter}]",
              xlims = (Xc - 1.4, Xc + 1.4), ylims = (Yc - 1.4, Yc + 1.4), legend = :topleft,
              estilo..., title = "Llegada", extra_kwargs = eje("38mm", "38mm"))
plot!(p_zoom, X, Y; lc = azul, label = "robot")
scatter!(p_zoom, [X[end]], [Y[end]]; mc = azul, ms = 3, msw = 0, label = "")
p_v = plot(tt, Rf[1, :]; lc = :gray, ls = :dash, lw = 1.25, label = "",
           ylabel = L"v\ [\si{\meter/\second}]", estilo...,
           xformatter = _ -> "", title = "Lazo interno", extra_kwargs = eje("42mm", "16mm"))
plot!(p_v, tt, y_in[1, :]; lc = azul, label = "")
p_ω = plot(tt, Rf[2, :]; lc = :gray, ls = :dash, lw = 1.25, label = "plan",
           xlabel = tlabel, ylabel = L"\omega_\delta\ [\si{\radian/\second}]", estilo..., legend = :right,
           extra_kwargs = eje("42mm", "16mm"))
plot!(p_ω, tt, y_in[5, :]; lc = azul, label = "")
lay = @layout [a b grid(2, 1)]
fig_circ = plot(p_cuarto, p_zoom, p_v, p_ω; layout = lay, link = :none, size = (660, 190))
savefig(fig_circ, path * "pir_circulo.pdf")

println("Figuras guardadas en: ", path)
