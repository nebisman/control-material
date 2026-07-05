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

# ╔═╡ a2000000-0000-0000-0000-000000000002
begin
    import Pkg
    Pkg.activate()
    using ControlSystems
    using Plots
    using PlutoUI

    gr()   #  backend interactivo dentro de Pluto
    md"Paquetes cargados desde el entorno local."
end

# ╔═╡ a1000000-0000-0000-0000-000000000001
md"""
# Análisis interactivo de un sistema de control usando el teorema de Nyquist

La planta es  $G(s) = \dfrac{1}{(s+1)^6}$. La ganancia  del controlador es $k$.

$$L(s) = k\,G(s) = \dfrac{k}{(s+1)^6}, \qquad
T(s) = \dfrac{Y(s)}{R(s)} = \dfrac{k\,G(s)}{1 + k\,G(s)}$$

El slider ajusta $k \in [1.25,\,5
.0]$ y todos los paneles se recalculan automáticamente.
"""

# ╔═╡ a4000000-0000-0000-0000-000000000004
md"""
### Ganancia del lazo

``k`` = $(@bind k Slider(1.1:0.01:5, default=1.25, show_value=true))
"""

# ╔═╡ a5000000-0000-0000-0000-000000000005
begin
    s = tf("s")
    G = 1 / ((s + 1)^6)    # planta
    L = k * G              # lazo abierto  L(s) = k·G(s)
    T = feedback(L)        # lazo cerrado  T(s) = L/(1+L)
    estable = isstable(T)
end

# ╔═╡ 84c8fef6-7c4b-4ef9-90e9-7a4027760631
md"### Diagramas de Bode y Nyquist de $$L(s)$$"

# ╔═╡ a6000000-0000-0000-0000-000000000006
begin
    # margin devuelve, para el lazo abierto:
    #   wgm : frec. de cruce de fase   (∠L = -180°)  → ubica el margen de ganancia
    #   gm  : margen de ganancia (razón, no dB)
    #   wpm : frec. de cruce de ganancia (|L| = 1)   → ubica el margen de fase
    #   pm  : margen de fase (grados)
    w180, gm, wgc, pm = margin(L)
    GM    = gm[1]                 # razón
    PM    = pm[1]                 # grados
    ω180   = w180[1]                # rad/s (cruce de fase)
    ωgc   = wgc[1]                # rad/s (cruce de ganancia)
    GM_dB = 20 * log10(GM)        # margen de ganancia en dB
    #estable = isstable(T)         # estabilidad del lazo cerrado
end

# ╔═╡ a9000000-0000-0000-0000-000000000007
let
    cyan = "#00aad4"
    w = exp10.(range(-2, 1, length=600))

    mag, phase, _ = bode(L, w)
    mag   = vec(mag)
    phase = vec(phase)                 # en grados
    magdB = 20 .* log10.(mag)
    φrad  = deg2rad.(phase)
    re    = mag .* cos.(φrad)
    im    = mag .* sin.(φrad)          # Nyquist para ω > 0

    # --- Bode: magnitud ---
    p1 = plot(w, magdB; xscale=:log10, lw=2, c=cyan, label="",
              ylabel="Magnitud [dB]", title="Diagrama de Bode", ylims=(-80,20),
              legend=:bottomleft, grid=true)
  
    hline!(p1, [0]; ls=:dash, c=:gray, label="0 dB")
    vline!(p1, [ωgc]; ls=:dot, c=:orange, label="")
    vline!(p1, [ω180]; ls=:dot, c=:red, label="ω cruce fase")


  
    scatter!(p1, [ωgc], [0]; c=:orange, ms=5, label="wgc")
    scatter!(p1,[ω180], [-GM_dB] ; c=:red, ms=5, label="GM=$(round(GM_dB,digits=1)) dB")
    plot!(p1, [ω180, ω180],[0,-GM_dB]; c=:red,lw=2,label="")
    annotate!(p1, ω180, -GM_dB,
              text("  GM=$(round(GM_dB, digits=1))", 8, :left, :red))
    # --- Bode: fase ---
    p2 = plot(w, phase; xscale=:log10, lw=2, c=cyan, label="",
              xlabel="ω [rad/s]", ylabel="Fase [°]",
              legend=:bottomleft, grid=true)
    plot!(p2, [ωgc, ωgc],[-180,-180+PM]; c=:orange,lw=2)

    hline!(p2, [-180]; ls=:dash, c=:gray, label="-180°")
    vline!(p2, [ω180]; ls=:dot, c=:red, label="ω cruce fase")
    vline!(p2, [ωgc]; ls=:dot, c=:orange, label="")
    scatter!(p2, [ωgc], [-180 + PM]; c=:orange, ms=5, label="")   # margen de fase
    scatter!(p2,[ω180], [-180] ; c=:red, ms=5, label="")
    annotate!(p2, ωgc, -180 + PM,
              text("  PM=$(round(PM, digits=1))°", 8, :left, :orange))


    # --- Nyquist ---
    p3 = plot(re, im; lw=3, c=cyan, label="L(jω), ω>0",
              xlabel="Re", ylabel="Im", title="Diagrama de Nyquist",
              aspect_ratio=:equal, legend=:topright, grid=true)
    plot!(p3, re, -im; lw=2, ls=:dash, c=:green, label="ω<0")       # espejo
    # círculo unitario de referencia
    tt = range(0, 2π, length=200)

    plot!(p3, cos.(tt), sin.(tt); ls=:dot, c=:gray, alpha=0.5, label="", lw=2)
 
    # punto crítico -1
    scatter!(p3, [-1], [0]; c=:black, ms=6, marker=:xcross, label="-1")
    # margen de ganancia: cruce del eje real negativo en -1/GM
    scatter!(p3, [-1/GM], [0]; c=:red, ms=6, label="gm= $(round(GM,digits=1))")
    # margen de fase: cruce del círculo unitario
    θ = deg2rad(-180 + PM)
    scatter!(p3, [cos(θ)], [sin(θ)]; c=:orange, ms=6, label="PM=$(round(PM,digits=1))°")

    lay = @layout [[a; b] c]
    plot(p1, p2, p3; layout=lay, size=(1000, 520), left_margin=3Plots.mm)
end

# ╔═╡ a7000000-0000-0000-0000-000000000008
md"""
### Márgenes de estabilidad — lazo abierto $L(s)=k\,G(s)$

| Margen | Valor | Frecuencia |
|:--|:--:|:--:|
| **Margen de ganancia** (GM) | $(round(GM_dB, digits=2)) dB | ω180 = $(round(ω180, digits=3)) rad/s |
| **Margen de fase** (PM) | $(round(PM, digits=2)) ° | ωgc = $(round(ωgc, digits=3)) rad/s |

**Lazo cerrado (k = $(round(k, digits=2))):**  $(estable ? "🟢 Estable" : "🔴 Inestable")
"""


# ╔═╡ fa0cf430-7804-11f1-a9d7-83833e41913a
md"### Respuesta al escalón — lazo cerrado $T(s)=Y(s)/R(s)$"

# ╔═╡ ab000000-0000-0000-0000-000000000010
let
    tfinal = 40.0
    res = ControlSystems.step(T, tfinal)     
    plot(res.t, vec(res.y); lw=2, c="#00aad4", label="y(t)",
         xlabel="Tiempo [s]", ylabel="Salida",
         title="Respuesta al escalón  (k = $(round(k, digits=2)))",
         legend=:bottomright, size=(900, 340), grid=true)
    hline!([1]; ls=:dash, c=:gray, label="referencia")
end





# ╔═╡ Cell order:
# ╠═a1000000-0000-0000-0000-000000000001
# ╠═a2000000-0000-0000-0000-000000000002
# ╟─a4000000-0000-0000-0000-000000000004
# ╠═a5000000-0000-0000-0000-000000000005
# ╟─84c8fef6-7c4b-4ef9-90e9-7a4027760631
# ╟─a6000000-0000-0000-0000-000000000006
# ╠═a9000000-0000-0000-0000-000000000007
# ╟─a7000000-0000-0000-0000-000000000008
# ╟─fa0cf430-7804-11f1-a9d7-83833e41913a
# ╠═ab000000-0000-0000-0000-000000000010
