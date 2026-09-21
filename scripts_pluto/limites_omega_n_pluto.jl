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

# ╔═╡ 2ca9b8e8-b530-11f1-aae7-453547cca4ef
begin
	import Pkg
	Pkg.activate()
	using ControlSystems, Plots, PlutoUI
	md"Paquetes cargados desde el entorno local."
end


# ╔═╡ 2ca9b930-b530-11f1-9205-6dcad76f8414
md"""
# Límites de diseño de $\omega_n$ ante dinámica no modelada

Consideramos una planta de primer orden

$$G(s) = \dfrac{b}{s+a}$$

con $b = 40000$ y $a = 20$, en cascada con una dinámica rápida no modelada

$$D(s) = \dfrac{1}{\tau s + 1}$$

El controlador PI de dos grados de libertad se diseña **despreciando** la dinámica no
modelada, ajustando el polinomio característico nominal en la forma estándar
de segundo orden $s^2+2\zeta\omega_n s+\omega_n^2$:

$$k_p = \dfrac{2\zeta\omega_n - a}{b}, \qquad k_i = \dfrac{\omega_n^2}{b}$$

Incluyendo la dinámica no modelada, la función de transferencia real de
lazo cerrado está dada por

$$T(s) = \dfrac{\omega_n^2}{\tau s^3 + (1+a\tau)s^2 + 2\zeta\omega_n s + \omega_n^2}$$

Los deslizadores permiten explorar el efecto de $\zeta$, $\omega_n$ y
$\tau$ sobre el sistema completo con retrado $\tau$
"""


# ╔═╡ 2ca9b962-b530-11f1-9b2a-a16412aa90de
begin
	a = 15.0
	b = 4000.0
	s = tf("s")
	G = b / (s + a)
	md""
end


# ╔═╡ 2ca9b96a-b530-11f1-bca5-cb16495632e1
md"``\zeta`` = $(@bind ζ Slider(0.6:0.01:0.9, default=0.7, show_value=true))"


# ╔═╡ 2ca9b96a-b530-11f1-b3d6-df66985b2228
md"``\omega_n`` = $(@bind ωn Slider(1.0:0.5:100.0, default=15.0, show_value=true))"


# ╔═╡ 2ca9b974-b530-11f1-9140-db0b84fbf1ce
md"``\tau`` = $(@bind τ Slider(0.01:0.005:0.04
, default=0.01, show_value=true))"


# ╔═╡ a072c5e2-d2bf-40db-abfd-1e80e67b1663
begin
	kp = (2 * ζ * ωn - a) / b
	ki = ωn^2 / b	
	md"""
	** Controlador con: **   $(kp>0 ? "🟢 kp positivo" : "🔴 kp negativo")
	"""
end

# ╔═╡ 2ca9b990-b530-11f1-b51a-1122e44d41d6
begin


	# T(s) = ωn² / (τs³ + (1+aτ)s² + 2ζωn·s + ωn²)  [PI de dos grados de libertad]
	T = ωn^2 / (τ * s^3 + (1 + a * τ) * s^2 + 2 * ζ * ωn * s + ωn^2)

	Tnom = ωn^2 / ( s^2 + 2 * ζ * ωn * s + ωn^2)

	# Señal de control aproximada con el modelo nominal: U(s)/R(s) = T(s)/G(s)
	
	tfinal = clamp(10.0 / (ζ * ωn), 0.05, 30.0)
	t = range(0, tfinal, length = 800)

	referencia = 800
	ωn_min = a/(2*ζ)
	ωn_max = 0.5/τ
	
	
	yT = vec(step(T * referencia, t).y)


	p1 = plot(t, yT, xlabel = "tiempo (s)", ylabel = "salida",
		title = "Respuesta al escalón de T(s)", 
		linewidth = 3, color = (ωn < ωn_min) || (ωn> ωn_max)  ? :red : :green, label="modelo real")
	plot!(step(Tnom*referencia,t),  fillrange = 0, fillalpha = 0.05,
		  color=:blue,linestyle = :dash, label="modelo nominal de diseño")



	p2 = bodeplot(T, plotphase = false, xlims=[0.2*ωn_min, 10*ωn_max],
				  linewidth=3, title = "Bode del sistema |T(jω)",
				  label=" |T(jω)")
	
	vline!(p2[1], [ωn_min], color = :red, linestyle = :dash,
		label = "ωn min = $(round(ωn_min, digits=2))")
	vline!(p2[1], [ωn_max], color = :purple, linestyle = :dash,
		label = "ωn max = $(round(ωn_max, digits=2))")
	maxT = hinfnorm(T)
	scatter!([ωn], [maxT[1]], color=:red, label = "ωn = $(round(ωn, digits=2))")	
	plot(p1, p2, layout = (2, 1), size = (700, 600))
end


# ╔═╡ 5e56705d-6a59-480f-a854-bd6301da6213
md"""
## Limite impuesto por la señal de control u(t) 

Para este ejemplo tenemos una limitación de control dada por:

$$u_\max = \max |u(t)| \leq  5 \text{V}$$

Esta señal se calcula con la respuesta al escalón de 

$$U(s)=\dfrac{T(s)}{G(s)} \times \dfrac{a}{s}$$

donde $a$ es la amplitud del escalón
"""



# ╔═╡ 6f1e2a2c-b531-11f1-9f4d-3b6b6cf1a9a1
begin

	Gru = T / G
	yU = vec(step(Gru * referencia, t).y)

	umax, imax = findmax(abs.(yU))

	  u_plt = plot(t, yU, xlabel = "tiempo (s)", ylabel = "u(t)",
		title = "Señal de control u(t)",
		label = false, linewidth = 2, color = :red)
				  
	scatter!([t[imax]], [yU[imax]], color = :red, markersize = 6,
		label = "umax = $(round(umax, digits=4))")
	hline!( [5],  fillrange = 0, fillalpha = 0.05,
		  color=:darkorange,linestyle = :dash, label="límite de la señal de control")
 
	
end


# ╔═╡ Cell order:
# ╟─2ca9b8e8-b530-11f1-aae7-453547cca4ef
# ╟─2ca9b930-b530-11f1-9205-6dcad76f8414
# ╟─2ca9b962-b530-11f1-9b2a-a16412aa90de
# ╟─2ca9b96a-b530-11f1-bca5-cb16495632e1
# ╟─2ca9b96a-b530-11f1-b3d6-df66985b2228
# ╟─2ca9b974-b530-11f1-9140-db0b84fbf1ce
# ╟─a072c5e2-d2bf-40db-abfd-1e80e67b1663
# ╟─2ca9b990-b530-11f1-b51a-1122e44d41d6
# ╟─5e56705d-6a59-480f-a854-bd6301da6213
# ╟─6f1e2a2c-b531-11f1-9f4d-3b6b6cf1a9a1
