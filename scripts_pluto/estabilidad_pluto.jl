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

# ╔═╡ 99c4f65c-b087-11f1-bb6b-591a0fc0623e
begin
    import Pkg
    Pkg.activate()
	using ControlSystems, Plots, PlutoUI
	md"Paquetes cargados desde el entorno local."
end


# ╔═╡ 99c4ff12-b087-11f1-9275-5fd6747fab92
md"""
# Ejemplos interactivos de estabilidad

Este cuaderno permite explorar de forma interactiva dos ejemplos de estabilidad de un sistema realimentado.
"""


# ╔═╡ 99c4ff1c-b087-11f1-9947-5b5ed2878305
md"## Ejemplo 1: control proporcional

Consideremos una planta de tercer orden

$$G(s) = \dfrac{1}{s(s+1)(s+0.5)}$$

la cual se controla mediante un controlador proporcional $k_p$. El deslizador ajusta $k_p$ y actualiza simultáneamente la respuesta al escalón y la constelación d  polos de lazo cerrado."


# ╔═╡ 99c4ff26-b087-11f1-9106-87665a2e27de
begin
	s = tf("s")
	G1 = 1 / (s * (s + 1) * (s + 0.5))
	md""
end


# ╔═╡ 99c4ff30-b087-11f1-bc2c-3bb49b6af995
md"``k_p`` = $(@bind kp Slider(0.05:0.01:1.0, default=0.4, show_value=true))"


# ╔═╡ 99c4ff38-b087-11f1-b30d-b5882d09d65a
begin
	T1 = feedback(kp * G1, 1)
	poles1 = pole(T1)
	stable1 = all(real.(poles1) .< 0)

	resp1 = plot(step(T1, 50),
		xlabel = "tiempo (s)", ylabel = "salida",
		title = "Respuesta al escalón (k = $(round(kp, digits=3)))",
		label = false, linewidth = 2, color = stable1 ? :blue : :red)
	plot!(resp1, [0, 0, 50],[0, 1,1], color = :purple, label = false)

	rl1, _, _ = rlocus(G1, 3.0)
	xr1 = (-1.7, 0.1)
	yr1 = (-0.9, 0.9)
	pmap1 = plot(Shape([0, xr1[2], xr1[2], 0], [yr1[1], yr1[1], yr1[2], yr1[2]]),
		color = :red, alpha = 0.12, linewidth = 0, label = false,
		xlims = [xr1...], ylims = [yr1...],
		xlabel = "Re", ylabel = "Im", title = "Polos de lazo cerrado")
	hline!(pmap1, [0], color = :black, label = false)
	vline!(pmap1, [0], color = :black, label = false)
	plot!(pmap1, real.(rl1), imag.(rl1),
		color = :green, alpha = 0.35, label = false, lw=4)

	scatter!(pmap1, real.(poles1), imag.(poles1),
		markershape = :xcross, markersize = 8, markerstrokewidth = 2,
		color = [real(p) > 0 ? :red : :blue for p in poles1], label = false)

	plot(resp1, pmap1, layout = (1, 2), size = (800, 350))
end


# ╔═╡ 99c4ff44-b087-11f1-852d-bdf8a5828920
md"""## Ejemplo 2: control PI

Ahora la planta  $G(s) = \dfrac{10}{(s+1)^3}$ con un controlador PI $C(s) = k_p + k_i/s$. Los dos sliders ajustan, respectivamente $k_p$ y $k_i$.
"""

# ╔═╡ 99c4ff80-b087-11f1-a8cc-81c27049b1cc
begin
	G2 = 10 / ((s + 1)^3)

	kp_range = range(0, stop = 0.8, length = 1000)
	a_coef = -10 / 9
	b_coef = 7 / 9
	c_coef = 8.0 / 90
	ki_curve = a_coef .* kp_range .^ 2 .+ b_coef .* kp_range .+ c_coef
	md""
end


# ╔═╡ 99c4ff8a-b087-11f1-8786-0fdd9ecdc553
md"``k_p`` = $(@bind kp2 Slider(0.001:0.005:0.8, default=0.2, show_value=true))"


# ╔═╡ 99c4ff94-b087-11f1-845a-79f5a8cd6e35
md"``k_i`` = $(@bind ki2 Slider(0.01:0.002:0.25, default=0.05, show_value=true))"


# ╔═╡ 99c4ff9e-b087-11f1-b761-1fb276875783
begin
	C2 = kp2 + ki2 / s
	T2 = feedback(C2 * G2, 1)
	poles2 = pole(T2)
	stable2 = all(real.(poles2) .< 0)

	Pfam2 = (kp2 * s + ki2) / s * G2
	rl2, _, _ = rlocus(Pfam2, 2.5)

	xr2 = (-3.6, 0.2)
	yr2 = (-2, 2)
	pmap2 = plot(Shape([0, xr2[2], xr2[2], 0], [yr2[1], yr2[1], yr2[2], yr2[2]]),
		color = :red, alpha = 0.12, linewidth = 0, label = false,
		xlims = [xr2...], ylims = [yr2...],
		xlabel = "Re", ylabel = "Im", title = "Polos de lazo cerrado")
	hline!(pmap2, [0], color = :black, label = false)
	vline!(pmap2, [0], color = :black, label = false)

	plot!(pmap2, real.(rl2), imag.(rl2),
		color = :green, alpha = 0.35, lw=3, label = false)
	scatter!(pmap2, real.(poles2), imag.(poles2),
		markershape = :xcross, markersize = 8, markerstrokewidth = 2,
		color = [real(p) > 0 ? :red : :blue for p in poles2], label = false)


	region2 = plot(kp_range, ki_curve,
		xlabel = "k_p", ylabel = "k_i", title = "Región de estabilidad",
		label = "frontera", color = :green,
		fillrange = 0, fillalpha = 0.3, ylims=[0, .3], xlims=[0, 0.85],
		legend = :topright)
	scatter!(region2, [kp2], [ki2], color = :orange, markersize = 8,
		label = "(kp, ki) actual")

	resp2 = plot(step(T2, 50),
		xlabel = "tiempo (s)", ylabel = "salida",
		title = "Respuesta al escalón (kp = $(round(kp2, digits=3)), ki = $(round(ki2, digits=3)))",
		label = false, linewidth = 2, color = stable2 ? :blue : :red)
		plot!(resp2, [0, 0, 50],[0, 1,1], color = :purple, label = false)
τ

	plot(pmap2, region2, resp2, layout = @layout([a b; c]), size = (800, 650))
end


# ╔═╡ Cell order:
# ╠═99c4f65c-b087-11f1-bb6b-591a0fc0623e
# ╟─99c4ff12-b087-11f1-9275-5fd6747fab92
# ╟─99c4ff1c-b087-11f1-9947-5b5ed2878305
# ╟─99c4ff26-b087-11f1-9106-87665a2e27de
# ╟─99c4ff30-b087-11f1-bc2c-3bb49b6af995
# ╟─99c4ff38-b087-11f1-b30d-b5882d09d65a
# ╟─99c4ff44-b087-11f1-852d-bdf8a5828920
# ╟─99c4ff80-b087-11f1-a8cc-81c27049b1cc
# ╟─99c4ff8a-b087-11f1-8786-0fdd9ecdc553
# ╟─99c4ff94-b087-11f1-845a-79f5a8cd6e35
# ╟─99c4ff9e-b087-11f1-b761-1fb276875783
