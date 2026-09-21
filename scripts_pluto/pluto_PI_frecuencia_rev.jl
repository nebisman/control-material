### A Pluto.jl notebook ###
# v1.0.3

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
Pkg.activate(joinpath(@__DIR__, ".."))
using ControlSystems, Plots, PlutoUI
using DCMotor    
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

# ╔═╡ 5cb111ad-5e3b-45d2-be76-b075cf9b384d
md"""
## Especificaciones de diseño (ajustables)

Frecuencia de cruce de ganancia  ``\omega_{gc}`` [rad/s] = $(@bind ωgc Slider(5:0.1:20.0, default=5.0, show_value=true))

Margen de fase  ``\phi_m`` [°] = $(@bind ϕm Slider(1.0:1.0:80.0, default=30.0, show_value=true))


"""

# ╔═╡ b4000000-0000-0000-0000-000000000004
begin
    sys = MotorSystem(port="/dev/ttyUSB0", bauds=460800);
    Gnd=transfer_function(sys, :speed)
    a = denvec(Gnd)[1][2]
    b = numvec(Gnd)[1][1]
    D = 0.02
    G   = Gnd*delay(D)   
    sys = MotorSystem(port="/dev/ttyUSB0");
    md""
end

# ╔═╡ b6000000-0000-0000-0000-000000000006
begin
    C, Kp, Ki, fig, CF = loopshapingPI(G, ωgc; rl=1,  phasemargin=ϕm, form=:parallel)
    L   = C*G                       # lazo abierto exacto
    Tpade = feedback(C*G,1);
md""
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


# ╔═╡ 42cec293-bf43-4724-9740-79983a052b3d
md"""
## Respuesta experimental
"""

# ╔═╡ 10361dd7-cbd8-4978-91f7-b90b6a35e943
begin
	set_pid(sys;  kp=Kp, ki=Ki, kd=0, beta=1, output=:speed, deadzone=0)	
	# respuesta del controlador
	result = step_closed(sys; r0 = 0, r1 = 400,  t0 = 0, t1 =2);
	sinf=stepinfo_exp(result; T=Tpade);
	md""
end

# ╔═╡ a14f8ad1-d283-4467-8f71-fc0127eb5132
screen_pluto()

# ╔═╡ db96c539-5557-4409-9773-574f84446210
md"""
## Resumen de parámetros Tiempo -- Frecuencia

| Métrica | Símbolo | Valor |
|:--|:--:|:--:|
| Margen de fase | $\phi_m$ | $(round(ϕm_check, digits=2)) ° |
| Margen de ganancia | $GM$ | $(round(GMdB, digits=2)) dB |
| Frecuencia de cruce de ganancia | $\omega_{gc}$ | $(round(ωgc, digits=3)) rad/s |
| Ancho de banda (−3 dB) de $T$ | $\omega_{B}$ | $(round(ωB, digits=3)) rad/s |
| Tiempo de establecimiento real (10–90 %) | $t_s$ | $(round(sinf.settlingtime, digits=3)) s 
| Tiempo de subida real (10–90 %) | $t_r$ | $(round(sinf.risetime, digits=3)) s 
| Sobrepico real  | $SP$ | $(round(sinf.overshoot, digits=3)) % 
"""

# ╔═╡ Cell order:
# ╟─b1000000-0000-0000-0000-000000000001
# ╟─b3000000-0000-0000-0000-000000000003
# ╟─5cb111ad-5e3b-45d2-be76-b075cf9b384d
# ╟─b4000000-0000-0000-0000-000000000004
# ╟─b6000000-0000-0000-0000-000000000006
# ╟─b9000000-0000-0000-0000-000000000009
# ╟─ba000000-0000-0000-0000-000000000010
# ╟─bb000000-0000-0000-0000-000000000011
# ╟─42cec293-bf43-4724-9740-79983a052b3d
# ╟─10361dd7-cbd8-4978-91f7-b90b6a35e943
# ╠═a14f8ad1-d283-4467-8f71-fc0127eb5132
# ╟─db96c539-5557-4409-9773-574f84446210
