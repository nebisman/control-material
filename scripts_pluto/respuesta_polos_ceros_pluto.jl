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

# ╔═╡ 714755d6-baa4-11f1-af5e-270487cfeb86
begin
	import Pkg
	Pkg.activate()
	using ControlSystems, Plots, PlutoUI
	md"Paquetes cargados desde el entorno local."
end


# ╔═╡ 7147561c-baa4-11f1-9d26-7be94a43d37b
md"""
# Región de diseño en el plano $s$ para un sistema de tercer orden

Consideramos un sistema de tercer orden formado por el par de polos
dominantes de segundo orden más un polo adicional $n\omega_n$ veces más
alejado del origen:

$$T(s) = \dfrac{n\,\omega_n^3}{(s^2+2\zeta\omega_n s+\omega_n^2)(s+n\omega_n)}$$

Los deslizadores permiten explorar el efecto de $\zeta$, $\omega_n$ y
$n$ sobre la respuesta al escalón y la ubicación de los polos respecto a
una región de diseño delimitada por:

- Una región angular con $\theta = \arccos(\zeta)$ (cono de amortiguamiento).
- Un círculo de radio $\omega_n$ (frecuencia natural).
- Una línea vertical en $-\sigma = -5/(\zeta\omega_n)$ (tiempo de establecimiento).
"""


# ╔═╡ 7147564e-baa4-11f1-9d0c-a74af98a91e0
md"``\zeta`` = $(@bind ζ Slider(0.1:0.01:0.99, default=0.5, show_value=true))"


# ╔═╡ 7c564a9b-9204-42d7-9951-ebdbc28f0f64


# ╔═╡ 71475658-baa4-11f1-82b9-838a82296316
md"``\omega_n`` = $(@bind ωn Slider(1.0:1.0:10.0, default=10.0, show_value=true))"


# ╔═╡ 71475662-baa4-11f1-b319-d1aeea6fa87e
md"``n`` = $(@bind n Slider(1.0:0.5:10.0, default=5.0, show_value=true))"


# ╔═╡ 3714fbc7-2085-4b61-8deb-91b5049f7627


# ╔═╡ 7147566e-baa4-11f1-88dd-6ff9dfeefc94
begin
	s = tf("s")
	T = n * ωn^3 / ((s^2 + 2 * ζ * ωn * s + ωn^2) * (s + n * ωn))

	tfinal = clamp(10.0 / (ζ * ωn), 0.05, 30.0)
	t = range(0, tfinal, length = 800)
	res = step(T, t)
	si = stepinfo(res; risetime_th = (0.0, 0.9))

	p1 = plot(si)

	T2 = ωn^2 / (s^2 + 2 * ζ * ωn * s + ωn^2)
	y2 = vec(step(T2, t).y)
	plot!(p1, t, y2, color = :red, linestyle = :dash, linewidth = 2,
		label = "2do orden (ζ, ωn)")

	poles = pole(T)

	θ = acos(ζ)
	σ =  (ζ * ωn)

	L = 10 * max(n * ωn, ωn)
	wedge = Shape([0.0, L * (-cos(θ)), L * (-cos(θ))], [0.0, L * sin(θ), -L * sin(θ)])

	xr = (-1.2 * max(n * ωn, ωn), 0.1 * ωn)
	yr = (-1.2 * ωn, 1.2 * ωn)

	p2 = plot(wedge, color = :green, alpha = 0.2, linewidth = 0,
		label = "θ = $(round(rad2deg(θ), digits=1))°",
		xlims = [xr...], ylims = [yr...],
		xlabel = "Re", ylabel = "Im", title = "Polos y región de diseño")

	φ = range(-π/2, 3*π/2,length = 200)
	plot!(p2, ωn .* cos.(φ), ωn .* sin.(φ), color = :blue, alpha = .5,
		linewidth = 2, label = "ωn = $(round(ωn, digits=2))")

	vline!(p2, [-σ], color = :purple, alpha = 0.6, linewidth = 2,
		linestyle = :dash, label = "σ = $(round(σ, digits=2))")

	hline!(p2, [0], color = :black, label = false)
	vline!(p2, [0], color = :black, label = false)

	scatter!(p2, real.(poles), imag.(poles), markershape = :xcross, markersize = 8,
		markerstrokewidth = 2, color = :red, label = "polos")

	plot(p1, p2, layout = (1, 2), size = (900, 400))
end


# ╔═╡ Cell order:
# ╠═714755d6-baa4-11f1-af5e-270487cfeb86
# ╠═7147561c-baa4-11f1-9d26-7be94a43d37b
# ╠═7147564e-baa4-11f1-9d0c-a74af98a91e0
# ╠═7c564a9b-9204-42d7-9951-ebdbc28f0f64
# ╠═71475658-baa4-11f1-82b9-838a82296316
# ╠═71475662-baa4-11f1-b319-d1aeea6fa87e
# ╠═3714fbc7-2085-4b61-8deb-91b5049f7627
# ╠═7147566e-baa4-11f1-88dd-6ff9dfeefc94
