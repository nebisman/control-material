### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ b3000000-0000-0000-0000-000000000003
begin
    import Pkg
    Pkg.activate()
    using ControlSystems, Plots, PlutoUI
    gr()
    md"Paquetes cargados desde el entorno local."
end

# ╔═╡ b1000000-0000-0000-0000-000000000001
md"""
# Diseño interactivo de un controlador PI en el dominio de la frecuencia

Lazo de control con realimentación unitaria negativa:

$$C(s) = \dfrac{k_p\,s + k_i}{s} = k_p + \dfrac{k_i}{s}, \qquad
G(s) = \dfrac{b}{s+a}\,e^{-d\,s}, \qquad
L(s) = C(s)\,G(s)$$

El diseño calcula $k_p$ y $k_i$ para imponer una **frecuencia de cruce de ganancia** $\omega_{gc}$ y un **margen de fase** $\phi_m$ elegidos con los sliders.
"""

# ╔═╡ b2000000-0000-0000-0000-000000000002
md"""
## Solución algebraica del PI

Se imponen en $\omega=\omega_{gc}$ las condiciones de **magnitud** y **fase**:

$$|L(j\omega_{gc})| = 1, \qquad \angle L(j\omega_{gc}) = -180^\circ + \phi_m$$

Con
$\angle G(j\omega) = -\arctan(\omega/a) - \omega \cdot d$,
$\;|G(j\omega)| = b/\sqrt{\omega^2+a^2}$,
$\;\angle C(j\omega) = -\arctan\!\big(k_i/(\omega k_p)\big)$
y $|C(j\omega)| = \tfrac{1}{|G(j\omega)|}$.

La fase que debe aportar el controlador es

$$\varphi_c \equiv \angle C(j\omega_{gc}) = -180^\circ + \phi_m - \angle G(j\omega_{gc})
= -180^\circ + \phi_m + \arctan\!\tfrac{\omega_{gc}}{a} + \omega_{gc}\,d.$$

Como $|C(j\omega_{gc})| = 1/|G(j\omega_{gc})| = \sqrt{\omega_{gc}^2+a^2}\,/\,b$ y descomponiendo
$C=k_p - j\,k_i/\omega$ en parte real e imaginaria ($k_p=|C|\cos\varphi_c$, $\,k_i/\omega=-|C|\sin\varphi_c$):

$$\boxed{\;k_p = \frac{\sqrt{\omega_{gc}^2+a^2}}{b}\cos\varphi_c,\qquad
k_i = -\,\omega_{gc}\,\frac{\sqrt{\omega_{gc}^2+a^2}}{b}\sin\varphi_c\;}$$

**Factibilidad** ($k_p>0,\;k_i>0$): se requiere $-90^\circ < \varphi_c < 0^\circ$
(un PI sólo aporta fase entre $-90^\circ$ y $0^\circ$).
"""

# ╔═╡ 5cb111ad-5e3b-45d2-be76-b075cf9b384d
md"""
## Especificaciones de diseño (ajustables)

Frecuencia de cruce de ganancia  ``\omega_{gc}`` [rad/s] = $(@bind ωgc Slider(2:0.1:20.0, default=5.0, show_value=true))

Margen de fase  ``\phi_m`` [°] = $(@bind ϕm Slider(1.0:1.0:80.0, default=30.0, show_value=true))

Retardo ``d`` [milisegundos] = $(@bind d Slider(10.0:1.0:100.0, default=10, show_value=true))

"""

# ╔═╡ b4000000-0000-0000-0000-000000000004
begin
    b = 2310.3    # ganancia de la planta
    a = 3.13    # polo de la planta
    s   = tf("s")   
    Gnd = b/(s + a)  
    D = d/1000.0
    G   = Gnd*delay(D)   
    D
end

# ╔═╡ b6000000-0000-0000-0000-000000000006
begin
    ϕmrad  = deg2rad(ϕm)
    magG, phiG = bode(G, [ωgc])
    magG, phiG =  magG[1], phiG[1]
    φc     = -π + ϕmrad - deg2rad(phiG) 
    φc_deg = rad2deg(φc)                    # fase requerida del PI [rad]
    φc = clamp(φc,-π/2,0)
    kp     = cos(φc)/magG
    ki     = -ωgc*sin(φc)/magG
    φc_deg = rad2deg(φc)
    factible = (-90.0 < φc_deg < 0.0) && (kp > 0) && (ki > 0)  
end

# ╔═╡ b7000000-0000-0000-0000-000000000007
factible ?
    md"""✅ **Diseño factible.**  $\varphi_c$ = $(round(φc_deg, digits=2))°  ⟹  **kp = $(round(kp, digits=4))**, **ki = $(round(ki, digits=4))**.""" :
    md"""⚠️ **Combinación no factible con un PI de ganancias positivas.**
    La fase requerida es $\varphi_c$ = $(round(φc_deg, digits=2))°, fuera de (−90°, 0°).
    Aumente $\omega_{gc}$ y/o $\phi_m$.  (Cálculo: kp = $(round(kp, digits=4)), ki = $(round(ki, digits=4)).)"""

# ╔═╡ b8000000-0000-0000-0000-000000000008
begin
    
    C   = kp + ki/s                
    L   = C*G                       # lazo abierto exacto
    # Aproximación racional (Padé orden 4) del retardo SOLO para la simulación temporal
    Tpade = feedback(C*Gnd*pade(D, 4));
end

# ╔═╡ b9000000-0000-0000-0000-000000000009
begin
    # ---- Respuesta en frecuencia (retardo exacto vía freqresp/bode) ----
    w = exp10.(range(-1, 3, length=3000))

    bmL   = bode(L, w)                       # (mag, fase[°] desenrollada, w)
    magLdB = 20 .* log10.(vec(bmL[1]))
    phL    = vec(bmL[2])

    Lc     = vec(freqresp(L, w))             # respuesta compleja de L (retardo exacto)
    Tc     = Lc ./ (1 .+ Lc)                 # T = L/(1+L) sin construir el sistema con retardo
    magTdB = 20 .* log10.(abs.(Tc))

    # helper: primer cruce de una curva con un nivel (interpolación en log-frecuencia)
    function _cross(w, y, lvl)
        for i in 1:length(w)-1
            if (y[i]-lvl)*(y[i+1]-lvl) <= 0 && y[i] != y[i+1]
                t = (lvl - y[i])/(y[i+1]-y[i])
                return 10^(log10(w[i]) + t*(log10(w[i+1])-log10(w[i])))
            end
        end
        return NaN
    end

    # Margen de ganancia: frecuencia de cruce de fase (∠L = -180°)

    w180, gm, wgc,  ϕm_check = margin(L)
    ϕm_check = ϕm_check[1]
    ωpc  = w180[1]
    gm= gm[1]
    GMdB = 20*log10(gm)
    # Pico de resonancia Mt de T
    iMt  = argmax(magTdB)
    MtdB = magTdB[iMt];  ωMt = w[iMt];  Mt = 10^(MtdB/20)

    # Ancho de banda ωB (-3 dB de T; T(0)=1 ⇒ 0 dB en DC)
    ωB = _cross(w, magTdB, -3.0)
end

# ╔═╡ ba000000-0000-0000-0000-000000000010
begin
    # ---- Respuesta al escalón en lazo cerrado (Padé) + step_info ----
    tv      = 0.0:0.005:10
    stepres = ControlSystems.step(Tpade, tv)
    yv      = vec(stepres.y)
    si      = stepinfo(stepres)
    tr      = si.risetime
    ωB_tr   = ωB*tr
    ωB_gc   = ωB/ωgc
end

# ╔═╡ bb000000-0000-0000-0000-000000000011
let
    # ===== ARRIBA: Diagrama de Bode de L(s) (magnitud + fase, ancho completo) =====

    # ---------- Magnitud de L ----------
    pLm = plot(w, magLdB; xscale=:log10, lw=2, c="#00aad4", label="|L(jω)|",
               ylabel="Magnitud [dB]", ylim=(-60, 30), grid=true, legend=:bottomleft,
               title="Diagrama de Bode de L(s) = C(s)·G(s)")
    hline!(pLm, [0]; ls=:dash, c=:gray, label="")
    vline!(pLm, [ωgc]; ls=:dot, c=:orange, label="")
    scatter!(pLm, [ωgc], [0]; c=:orange, ms=6, label="ωgc = $(round(ωgc,digits=2)) rad/s")
    if isfinite(GMdB)
        vline!(pLm, [ωpc]; ls=:dot, c=:red, label="")
        plot!(pLm, [ωpc, ωpc], [-GMdB, 0]; c=:red, lw=2, label="")
        scatter!(pLm, [ωpc], [-GMdB]; c=:red, ms=6, label="GM = $(round(GMdB,digits=2)) dB")
        annotate!(pLm, ωpc, -GMdB/2, text("  GM=$(round(GMdB,digits=1))dB", 8, :left, :red))
    end

    # ---------- Fase de L ----------
    pLp = plot(w, phL; xscale=:log10, lw=2, c="#00aad4", label="∠L(jω)",
               ylims=(-300,90),
               xlabel="ω [rad/s]", ylabel="Fase [°]", grid=true, legend=:bottomleft)
    hline!(pLp, [-180]; ls=:dash, c=:gray, label="−180°")
    vline!(pLp, [ωgc]; ls=:dot, c=:orange, label="")
    plot!(pLp, [ωgc, ωgc], [-180, -180+ϕm]; c=:orange, lw=2, label="")
    scatter!(pLp, [ωgc], [-180+ϕm]; c=:orange, ms=6, label="ϕm = $(round(ϕm_check,digits=2))°")
    annotate!(pLp, ωgc, -180+ϕm/2, text("  ϕm=$(round(ϕm_check,digits=2))°", 8, :left, :orange))
    if isfinite(GMdB)
        vline!(pLp, [ωpc]; ls=:dot, c=:red, label="")
        scatter!(pLp, [ωpc], [-180]; c=:red, ms=6, label="")
    end

    # ===== ABAJO IZQUIERDA: Diagrama de Bode de T(s) =====
    ymaxT = isfinite(MtdB) ? max(6.0, MtdB + 4) : 6.0
    pT = plot(w, magTdB; xscale=:log10, lw=2, c=:purple, label="|T(jω)|",
              xlabel="ω [rad/s]", ylabel="Magnitud [dB]", ylim=(-40, ymaxT), grid=true,
              legend=:bottomleft, title="Bode de T(s) = Y(s)/R(s)")
    hline!(pT, [0]; ls=:dash, c=:gray, label="")
    hline!(pT, [-3]; ls=:dot, c=:gray, label="")
    scatter!(pT, [ωMt], [MtdB]; c=:green, ms=5, marker=:circle,
             label="Mt = $(round(MtdB,digits=2)) dB")
    annotate!(pT, ωMt, MtdB, text("Mt=$(round(MtdB,digits=1))dB ", 8, :right, :green))
    if isfinite(ωB)
        vline!(pT, [ωB]; ls=:dashdot, c=:cyan, label="")
        scatter!(pT, [ωB], [-3]; c=:cyan, ms=5, marker=:circle,
                 label="ωB = $(round(ωB,digits=2)) rad/s (−3dB)")
    end

    # ===== ABAJO DERECHA: Respuesta al escalón =====
    pStep = plot(tv, yv; lw=2, c="#00aad4", xlabel="Tiempo [s]", ylabel="y(t)",
                 grid=true, legend=:bottomright, title="Respuesta al escalón en lazo cerrado",
                 label="tr=$(round(tr,digits=3))s · Mp=$(round(si.overshoot,digits=1))% · ts=$(round(si.settlingtime,digits=2))s")
    hline!(pStep, [1]; ls=:dash, c=:gray, label="referencia")

    #  L(s) arriba (magnitud y fase) · abajo:  T(s) izquierda | escalón derecha
    plot(pLm, pLp,  pStep ,pT;
         layout=@layout([a; b; [c d]]), size=(1000, 1050), left_margin=6Plots.mm)
end

# ╔═╡ bc000000-0000-0000-0000-000000000012
md"""
## Resumen relaciones Tiempo -- Frecuencia

| Métrica | Símbolo | Valor |
|:--|:--:|:--:|
| Margen de fase | $\phi_m$ | $(round(ϕm_check, digits=2)) ° |
| Margen de ganancia | $GM$ | $(round(GMdB, digits=2)) dB |
| Frecuencia de cruce de ganancia | $\omega_{gc}$ | $(round(ωgc, digits=3)) rad/s |
| Ancho de banda (−3 dB) de $T$ | $\omega_{B}$ | $(round(ωB, digits=3)) rad/s |
| Tiempo de subida (10–90 %) | $t_r$ | $(round(tr, digits=3)) s |
| Producto | $\omega_{B}\!\cdot\! t_r$ | $(round(ωB_tr, digits=3)) |
| Relación | $\omega_{B}/\omega_{gc}$ | $(round(ωB_gc, digits=3)) |
"""

# ╔═╡ Cell order:
# ╠═b1000000-0000-0000-0000-000000000001
# ╠═b3000000-0000-0000-0000-000000000003
# ╠═b2000000-0000-0000-0000-000000000002
# ╟─5cb111ad-5e3b-45d2-be76-b075cf9b384d
# ╠═b4000000-0000-0000-0000-000000000004
# ╠═b6000000-0000-0000-0000-000000000006
# ╠═b7000000-0000-0000-0000-000000000007
# ╟─b8000000-0000-0000-0000-000000000008
# ╠═b9000000-0000-0000-0000-000000000009
# ╟─ba000000-0000-0000-0000-000000000010
# ╠═bb000000-0000-0000-0000-000000000011
# ╟─bc000000-0000-0000-0000-000000000012
