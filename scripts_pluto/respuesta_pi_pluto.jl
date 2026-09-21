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

# ╔═╡ 2f5711d0-b39b-11f1-a6a3-ad6311e0d59b
begin
	import Pkg
	Pkg.activate()
	using ControlSystems, Plots, PlutoUI
	
	md"Paquetes cargados desde el entorno local."
end


# ╔═╡ 2f571216-b39b-11f1-8cf9-91738151e982
md"""
# Influencia de $$\omega_n$$ en la robustez del diseño de un controlador PI por asignación de polos

Consideramos una planta de primer orden

$$G(s) = \dfrac{b}{s+a}$$

con valores nominales $b = 40000$ y $a = 20$. El controlador PI se diseña
**siempre con los valores nominales de la planta**, fijando el polinomio
característico de lazo cerrado en la forma estándar de segundo orden
$s^2 + 2\zeta\omega_n s + \omega_n^2$:

$$k_p = \dfrac{2\zeta\omega_n - a}{b}, \qquad k_i = \dfrac{\omega_n^2}{b}$$

Los deslizadores permiten explorar el efecto de $\zeta$, $\omega_n$ y una
variación $\Delta a$ del parámetro real de la planta, mientras el
controlador permanece fijo (diseñado para el valor nominal $a=20$).
"""


# ╔═╡ 2f571222-b39b-11f1-a21b-81353aa8d1c9
begin
	a_nom = 20.0
	b_nom = 4000.0
	md""
end


# ╔═╡ 2f57122a-b39b-11f1-ad8f-5d86581e9cd5
md"``\zeta`` = $(@bind ζ Slider(0.3:0.01:0.9, default=0.6, show_value=true))"


# ╔═╡ 2f571234-b39b-11f1-b967-1d0c4adcb8be
md"``\omega_n`` = $(@bind ωn Slider(1.0:1.0:200.0, default=20.0, show_value=true))"


# ╔═╡ 2f57123e-b39b-11f1-99ec-59692cd4b430
md"``\Delta a`` = $(@bind Δa Slider(-10.0:0.5:10.0, default=0.0, show_value=true))"


# ╔═╡ 2f571266-b39b-11f1-bade-259e1fbe8233
begin
	kp = (2 * ζ * ωn - a_nom) / b_nom
	ki = ωn^2 / b_nom

	a_act = a_nom + Δa
	b_act = b_nom
    s=tf("s")
	
	# Lazo cerrado: T(s) = b·ki / (s² + (a + b·kp)·s + b·ki)
	Tnom = b_nom * ki/(s^2 + (a_nom + b_nom * kp)*s + b_nom*ki)
	Tact = b_act * ki/(s^2 + (a_act + b_act * kp)*s + b_act*ki)

	

	# Respuesta a la perturbación: Gvy(s) = b·s / (s² + (a + b·kp)·s + b·ki)
	Gvy_act = b_act * s/(s^2 + (a_act + b_act * kp)*s + b_act*ki)

	tfinal = clamp(10.0 / (ζ * ωn), 0.05, 50.0)
	t = range(0, tfinal, length = 800)

	reference = 360	
	ynom_step = step(reference * Tnom, t)
	yact_step = step(reference * Tact, t)
    srnom = stepinfo(ynom_step, risetime_th = (0.0, 0.9))
	  sract = stepinfo(yact_step, risetime_th = (0.0, 0.9))
	
	p1 = plot(ynom_step , label = "nominal", linewidth = 2, color = :blue,
		xlabel = "tiempo (s)", ylabel = "salida",
		   fillrange = 0, fillalpha = 0.1, linestyle = :dash, ylims=[0,700],
		title = "Respuesta al escalón")
	plot!(p1, yact_step, label = "actual (a = $(round(a_act, digits=2)))",	  
		  linewidth = 2, color = :red)
    scatter!([srnom.peaktime], [srnom.peak], color=:blue,
			label = "SP = $(round(srnom.overshoot, digits=2))")
	scatter!([sract.peaktime], [sract.peak], color=:red,
			label = "SP = $(round(sract.overshoot, digits=2))")
    plot!([sract.risetime, sract.risetime], [0, 0.9*reference], color=:red,		label = "tr = $(round(sract.risetime, digits=3))")

	plot!([srnom.risetime, srnom.risetime], [0, 0.9*reference], color=:blue,		label = "tr = $(round(srnom.risetime, digits=3))", linestyle=:dash)
	
    tp = range(0, 50, length = 2000)
	v = reshape(0.5*sin.(tp), 1, :)
	
	yact_step_p = vec(step(reference * Tact, tp).y)
	ynom_v = vec(lsim(Gvy_act, v, tp).y)
 
    
	p2 = plot(tp, ynom_v .+ yact_step_p , label = "nominal", linewidth = 2, color = :blue,
		xlabel = "tiempo (s)", ylabel = "salida",
		title = "Escalón + perturbación v(t) = 0.5*sen(t)")
	

	plot(p1, p2, layout = ( 2,1), size = (900, 900), bottom_margin=20*Plots.mm)
end


# ╔═╡ 792dde5f-aa99-4790-ab59-256909dd4069
begin
	Gact = b_act / (s + a_act)

	num_T = numvec(Tact)[1][1]
	den_T = denvec(Tact)[1]

	estable = isstable(Tact)

	amp_v = maximum(abs.(v))
	Vss = amp_v * abs(evalfr(Gvy_act, im * 1)[1])

	Gnom_tex = Markdown.LaTeX("\\dfrac{$(round(b_nom, digits=0))}{s + $(round(a_nom, digits=2))}")

	Gact_tex = Markdown.LaTeX("\\dfrac{$(round(b_act, digits=0))}{s + $(round(a_act, digits=2))}")
	Tact_tex = Markdown.LaTeX("\\dfrac{$(round(num_T, digits=2))}{s^2  $(estable ? "+" : "-")$(round(abs(den_T[2]), digits=2))\\,s + $(round(den_T[3], digits=2))}")

	estable = isstable(Tact)

	md"""
	## Valores actuales

	| Planta nominal ``G_n(s)`` | Planta real ``G(s)`` | Función de transferencia en lazo cerrado ``T(s)`` | Valor de estado estacionario de la perturbación |
	|:--:|:--:|:--:|:--:|
	|$(Gnom_tex) | $(Gact_tex) | $(Tact_tex) | $(round(Vss, digits=5)) |
	** Lazo cerrado: **  $(estable ? "🟢 Estable" : "🔴 Inestable")
	"""


end

# ╔═╡ Cell order:
# ╟─2f5711d0-b39b-11f1-a6a3-ad6311e0d59b
# ╟─2f571216-b39b-11f1-8cf9-91738151e982
# ╟─2f571222-b39b-11f1-a21b-81353aa8d1c9
# ╟─2f57122a-b39b-11f1-ad8f-5d86581e9cd5
# ╟─2f571234-b39b-11f1-b967-1d0c4adcb8be
# ╟─2f57123e-b39b-11f1-99ec-59692cd4b430
# ╟─2f571266-b39b-11f1-bade-259e1fbe8233
# ╟─792dde5f-aa99-4790-ab59-256909dd4069
