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
	Pkg.activate(joinpath(@__DIR__, ".."))
	using ControlSystems, Plots, PlutoUI
	md"Paquetes cargados desde el entorno local."
end


# ╔═╡ 99c4ff12-b087-11f1-9275-5fd6747fab92
md"""
# Ejemplos interactivos de estabilidad

Este cuaderno acompaña a `scripts_julia/estabilidad.jl` y permite explorar de forma
interactiva los dos ejemplos allí desarrollados.
"""


# ╔═╡ 99c4ff1c-b087-11f1-9947-5b5ed2878305
md"## Ejemplo 1: control proporcional

Planta de tercer orden $G(s) = \dfrac{1}{s(s+1)(s+0.5)}$ realimentada con una
ganancia proporcional $k$. El deslizador controla $k$ y actualiza a la vez la
respuesta al escalón y la constelación de polos de lazo cerrado."


# ╔═╡ 99c4ff26-b087-11f1-9106-87665a2e27de
begin
	s = tf("s")
	G1 = 1 / (s * (s + 1) * (s + 0.5))
end


# ╔═╡ 99c4ff30-b087-11f1-bc2c-3bb49b6af995
@bind k Slider(0.0:0.01:2.0, default=0.4, show_value=true)


# ╔═╡ 99c4ff38-b087-11f1-b30d-b5882d09d65a
begin
	T1 = feedback(k * G1, 1)
	poles1 = pole(T1)

	resp1 = plot(step(T1, 50),
		xlabel = "tiempo (s)", ylabel = "salida",
		title = "Respuesta al escalón (k = $(round(k, digits=3)))",
		label = false, linewidth = 2)

	pmap1 = scatter(real.(poles1), imag.(poles1),
		xlabel = "Re", ylabel = "Im", title = "Polos de lazo cerrado",
		markershape = :xcross, markersize = 8, markerstrokewidth = 2,
		label = false)
	hline!(pmap1, [0], color = :black, label = false)
	vline!(pmap1, [0], color = :black, label = false)

	plot(resp1, pmap1, layout = (1, 2), size = (800, 350))
end


# ╔═╡ 99c4ff44-b087-11f1-852d-bdf8a5828920
md"## Ejemplo 2: control PI

Misma planta $G(s) = \dfrac{10}{(s+1)^3}$ realimentada ahora con un controlador
PI $C(s) = k_p + k_i/s$. Dos deslizadores controlan $k_p$ y $k_i$; el punto
actual se resalta sobre la región de estabilidad y se actualizan la constelación
de polos y la respuesta al escalón."


# ╔═╡ 99c4ff80-b087-11f1-a8cc-81c27049b1cc
begin
	G2 = 10 / ((s + 1)^3)

	kp_range = range(0, stop = 0.8, length = 1000)
	a_coef = -10 / 9
	b_coef = 7 / 9
	c_coef = 8.0 / 90
	ki_curve = a_coef .* kp_range .^ 2 .+ b_coef .* kp_range .+ c_coef
end


# ╔═╡ 99c4ff8a-b087-11f1-8786-0fdd9ecdc553
@bind kp2 Slider(0.0:0.005:0.8, default=0.2, show_value=true)


# ╔═╡ 99c4ff94-b087-11f1-845a-79f5a8cd6e35
@bind ki2 Slider(0.0:0.002:0.25, default=0.05, show_value=true)


# ╔═╡ 99c4ff9e-b087-11f1-b761-1fb276875783
begin
	C2 = kp2 + ki2 / s
	T2 = feedback(C2 * G2, 1)
	poles2 = pole(T2)

	pmap2 = scatter(real.(poles2), imag.(poles2),
		xlabel = "Re", ylabel = "Im", title = "Polos de lazo cerrado",
		markershape = :xcross, markersize = 8, markerstrokewidth = 2,
		label = false)
	hline!(pmap2, [0], color = :black, label = false)
	vline!(pmap2, [0], color = :black, label = false)

	region2 = plot(kp_range, ki_curve,
		xlabel = "k_p", ylabel = "k_i", title = "Región de estabilidad",
		label = "frontera", color = :green,
		fillrange = 0, fillalpha = 0.3, legend = :topright)
	scatter!(region2, [kp2], [ki2], color = :red, markersize = 8,
		label = "(kp, ki) actual")

	resp2 = plot(step(T2, 50),
		xlabel = "tiempo (s)", ylabel = "salida",
		title = "Respuesta al escalón (kp = $(round(kp2, digits=3)), ki = $(round(ki2, digits=3)))",
		label = false, linewidth = 2)

	plot(pmap2, region2, resp2, layout = @layout([a b; c]), size = (800, 650))
end


# ╔═╡ Cell order:
# ╠═99c4f65c-b087-11f1-bb6b-591a0fc0623e
# ╠═99c4ff12-b087-11f1-9275-5fd6747fab92
# ╠═99c4ff1c-b087-11f1-9947-5b5ed2878305
# ╠═99c4ff26-b087-11f1-9106-87665a2e27de
# ╠═99c4ff30-b087-11f1-bc2c-3bb49b6af995
# ╠═99c4ff38-b087-11f1-b30d-b5882d09d65a
# ╠═99c4ff44-b087-11f1-852d-bdf8a5828920
# ╠═99c4ff80-b087-11f1-a8cc-81c27049b1cc
# ╠═99c4ff8a-b087-11f1-8786-0fdd9ecdc553
# ╠═99c4ff94-b087-11f1-845a-79f5a8cd6e35
# ╠═99c4ff9e-b087-11f1-b761-1fb276875783
