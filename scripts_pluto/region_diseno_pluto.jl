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

# ╔═╡ 23fce5ad-e993-4001-b4ff-742957cbd59e
begin
	import Pkg
	Pkg.activate()
	using ControlSystems, PlutoUI, PlutoPlotly, Printf
	md"Paquetes cargados desde el entorno local."
end

# ╔═╡ a5488db8-ade2-440c-849e-06ded69d9b5e
md"""
# Región de diseño a partir de especificaciones temporales

A partir de las especificaciones de desempeño (tiempo de establecimiento
$t_{ee}$, tiempo de subida $t_r$ y sobrepico $M_p$) se calcula una región
de diseño en el plano $s$:

$$\zeta_{min} = \dfrac{-\ln SP}{\sqrt{\pi^2 + \ln^2 SP}}, \qquad
\omega_{n,min} = \dfrac{2.23\,\zeta_{min}^2 + 0.036\,\zeta_{min} + 1.54}{t_r}, \qquad
\sigma_{min} = \dfrac{5}{t_{ee}}$$

donde $\zeta_{min}$ se obtiene invirtiendo la relación del sobrepico
$SP = e^{-\pi\zeta/\sqrt{1-\zeta^2}}$.

Se evalúa un sistema de tercer orden con un polo adicional

$$T(s)=\dfrac{n\,\omega_n^3}{(s^2+2\zeta\omega_n s+\omega_n^2)(s+n\omega_n)}$$


**Haga clic en el plano complejo (gráfico de la derecha)** para ubicar los
polos complejos de $T(s)$: se actualizan $\zeta$, $\omega_n$ y la
respuesta al escalón.
"""

# ╔═╡ 5d1b486e-535b-436d-aa61-9ba58cdee12e
md"``t_{ee}`` [s] = $(@bind t_ee Slider(0.1:0.01:0.5, default=0.5, show_value=true))"

# ╔═╡ 104b2ca8-79aa-4218-9006-64e14bf19c41
md"``t_r`` [s] = $(@bind t_r Slider(0.05:0.001:0.1, default=0.1, show_value=true))"

# ╔═╡ a7f6738b-fe1c-49b6-83a4-fc4c9c2f6978
md"``SP`` [%] = $(@bind SP Slider(0.2:0.5:10.0, default=5.0, show_value=true))"

# ╔═╡ 91a028da-5f54-4fb1-9109-a9fbffd23544
begin
	SP_frac = SP / 100
	ζ_min = -log(SP_frac) / sqrt(π^2 + log(SP_frac)^2)
	θ_max = acos(ζ_min)
	ωn_min = (2.23 * ζ_min^2 + 0.036 * ζ_min + 1.54) / t_r
	σ_min = 5 / t_ee
	md""
end

# ╔═╡ f82e1b85-b33f-43c6-8092-7991ec79133e
md"
Distancia del polo lejano 
``n`` = $(@bind n Slider(1.0:0.5:5.0, default=1.5, show_value=true))"

# ╔═╡ 9850a008-cf41-4def-a4d2-fc33a15fbeda
begin
	# Polo complejo superior seleccionado (x + jy). Inicial: s = -8 + j8.
	# Se guarda en un Ref para que el gráfico interactivo lo lea sin crear una
	# dependencia cíclica con el valor del clic.
	polo_actual = Ref((-13.0, 13.0))
	md""
end

# ╔═╡ 1f2715fa-1953-436d-9178-8b75cce17ffd
@bind click_polo let
	x0, y0 = polo_actual[]
	ωn0 = hypot(x0, y0)
	ζ0 = -x0 / ωn0

	s = tf("s")
	T2 = ωn0^2 / (s^2 + 2ζ0 * ωn0 * s + ωn0^2)
	T3 = n * ωn0^3 / ((s^2 + 2ζ0 * ωn0 * s + ωn0^2) * (s + n * ωn0))

	# Horizonte: que la respuesta se establezca y que se vean las líneas de t_r y t_ee
	tfinal = clamp(max(10 / (ζ0 * ωn0), 1.2 * max(t_ee, t_r)), 0.05, 30.0)
	t = collect(range(0, tfinal, length = 2000))
	y2 = vec(step(T2, t).y)
	res3 = step(T3, t)
	y3 = vec(res3.y)
	si = stepinfo(res3; risetime_th = (0.0, 0.9))

	S = max(σ_min, ωn_min, n * ωn0)
	Lax = 1.1 * n * ωn0   # mismo límite para ambos ejes del plano s
	L = 10 * S

	# Líneas de especificación en la respuesta al escalón (mismos colores que la región)
	SP_lim = 1 + SP / 100
	ytop = max(1.2, 1.1 * max(maximum(y2), maximum(y3), SP_lim))
	sp_txt, tr_txt, tee_txt = @sprintf("%.1f", SP), @sprintf("%.3f", t_r), @sprintf("%.2f", t_ee)
	lab_sp = "SP ≤ $(sp_txt) %  →  obtenido: $(@sprintf("%.2f", si.overshoot)) %"
	lab_tr = "tr ≤ $(tr_txt) s  →  obtenido: $(@sprintf("%.3f", si.risetime)) s"
	lab_tee = "tee ≤ $(tee_txt) s  →  obtenido: $(@sprintf("%.3f", si.settlingtime)) s"

	fig = make_subplots(rows = 1, cols = 2,
		subplot_titles = ["Respuesta al escalón" "Polos de T(s) y región de diseño"])

	add_trace!(fig, scatter(x = t, y = y2, mode = "lines", name = "2do orden (ζ, ωn)",
		legend = "legend2", line = attr(color = "red", width = 2)), row = 1, col = 1)
	idx_y2 = length(fig.data) - 1
	add_trace!(fig, scatter(x = t, y = y3, mode = "lines", name = "3er orden (n = $n)",
		legend = "legend2", line = attr(color = "blue", width = 2)), row = 1, col = 1)
	idx_y3 = length(fig.data) - 1

	add_trace!(fig, scatter(x = [0, tfinal], y = [SP_lim, SP_lim], mode = "lines",
		name = lab_sp, legend = "legend2", hoverinfo = "skip",
		line = attr(color = "rgba(0,160,0,0.6)", width = 2, dash = "dash")), row = 1, col = 1)
	idx_sp = length(fig.data) - 1
	add_trace!(fig, scatter(x = [t_r, t_r], y = [0, ytop], mode = "lines",
		name = lab_tr, legend = "legend2", hoverinfo = "skip",
		line = attr(color = "rgba(204,85,0,0.55)", width = 2, dash = "dash")), row = 1, col = 1)
	idx_tr = length(fig.data) - 1
	add_trace!(fig, scatter(x = [t_ee, t_ee], y = [0, ytop], mode = "lines",
		name = lab_tee, legend = "legend2", hoverinfo = "skip",
		line = attr(color = "rgba(148,0,211,0.55)", width = 2, dash = "dash")), row = 1, col = 1)
	idx_tee = length(fig.data) - 1

	add_trace!(fig, scatter(
		x = [0, -L * cos(θ_max), -L * cos(θ_max), 0],
		y = [0, L * sin(θ_max), -L * sin(θ_max), 0],
		mode = "lines", fill = "toself", fillcolor = "rgba(0,160,0,0.15)",
		line = attr(color = "rgba(0,160,0,0.6)", width = 2, dash = "dash"), hoverinfo = "skip",
		name = "θ_max = $(round(rad2deg(θ_max), digits=1))°"), row = 1, col = 2)

	φ = range(π / 2, 3π / 2, length = 100)
	add_trace!(fig, scatter(x = ωn_min .* cos.(φ), y = ωn_min .* sin.(φ),
		mode = "lines", line =  attr(color = "rgba(204,85,0,0.55)", width = 2, dash = "dash"),
		hoverinfo = "skip",
		name = "ωn_min = $(round(ωn_min, digits=2)) rad/s"), row = 1, col = 2)

	add_trace!(fig, scatter(x = [-σ_min, -σ_min], y = [-L, L],
		mode = "lines", line = attr(color = "rgba(148,0,211,0.55)", width = 2, dash = "dash"),
		hoverinfo = "skip",
		name = "σ_min = $(round(σ_min, digits=2))"), row = 1, col = 2)

	add_trace!(fig, scatter(x = [x0, x0, -n * ωn0], y = [y0, -y0, 0.0],
		mode = "markers", marker = attr(symbol = "x", size = 12, color = "red"),
		name = "polos: ζ = $(round(ζ0, digits=3)), ωn = $(round(ωn0, digits=3))"),
		row = 1, col = 2)
	idx_polos = length(fig.data) - 1   # índice (base 0) de la traza de polos en JS

	relayout!(fig,
		xaxis_title_text = "tiempo (s)", yaxis_title_text = "salida",
		xaxis2_title_text = "Re", yaxis2_title_text = "Im",
		xaxis2_range = [-1.1*Lax, 0], yaxis2_range = [-Lax, Lax],
		yaxis2_scaleanchor = "x2", yaxis2_scaleratio = 1,
		xaxis2_constrain = "domain", yaxis2_constrain = "domain",
		xaxis2_zerolinecolor = "black", yaxis2_zerolinecolor = "black",
		legend = attr(orientation = "v", x = 0.55, xanchor = "left", y = -0.2, yanchor = "top",
			title = attr(text = "Región de diseño")),
		legend2 = attr(orientation = "v", x = 0.0, xanchor = "left", y = -0.2, yanchor = "top",
			title = attr(text = "Especificaciones y stepinfo de T(s)")),
		height = 620, margin = attr(b = 210))

	# Clic en el subplot derecho: convierte píxeles a coordenadas del plano s,
	# actualiza polos y respuestas en el navegador (forma cerrada) y envía el
	# punto a Julia mediante @bind.
	js_click = """
	(evt) => {
		const fl = PLOT._fullLayout
		if (!fl || !fl.xaxis2 || !fl.yaxis2) return
		const xa = fl.xaxis2, ya = fl.yaxis2
		const rect = PLOT.getBoundingClientRect()
		const px = evt.clientX - rect.left - xa._offset
		const py = evt.clientY - rect.top - ya._offset
		if (px < 0 || px > xa._length || py < 0 || py > ya._length) return
		const x = xa.p2c(px)
		const y = Math.abs(ya.p2c(py))
		if (!(x < 0) || !(y > 0)) return

		const n = $(n)
		const sigma = -x, wd = y
		const wn = Math.hypot(x, y)
		const zeta = sigma / wn
		const a3 = n * wn

		const tee = $(t_ee), tr = $(t_r), SPlim = $(SP_lim)
		const tf = Math.min(Math.max(10 / sigma, 1.2 * Math.max(tee, tr), 0.05), 30)
		const N = 2000
		const t = Array.from({length: N}, (_, i) => tf * i / (N - 1))
		const y2 = t.map(tt => 1 - Math.exp(-sigma * tt) * (Math.cos(wd * tt) + (sigma / wd) * Math.sin(wd * tt)))

		const R3 = -wn * wn / (a3 * a3 - 2 * sigma * a3 + wn * wn)
		const cm = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
		const D = cm(cm([0, 2 * wd], [a3 - sigma, wd]), [-sigma, wd])
		const Nn = n * wn * wn * wn
		const den = D[0] * D[0] + D[1] * D[1]
		const R1 = [Nn * D[0] / den, -Nn * D[1] / den]
		const y3 = t.map(tt => 1 + R3 * Math.exp(-a3 * tt) + 2 * Math.exp(-sigma * tt) * (R1[0] * Math.cos(wd * tt) - R1[1] * Math.sin(wd * tt)))

		// Misma definición que ControlSystems.stepinfo (umbral 2 %, subida 0-90 %)
		const y0s = y3[0], yf = y3[N - 1], Ts = t[1] - t[0]
		const stepsize = Math.abs(yf - y0s)
		let peak = -Infinity, last = -1
		for (let i = 0; i < N; i++) {
			if (y3[i] > peak) peak = y3[i]
			if (Math.abs(y3[i] - yf) > 0.02 * stepsize) last = i
		}
		const overshoot = 100 * (peak - yf) / stepsize
		const settling = (last < 0 ? t[N - 1] : t[last]) + Ts
		const i10 = y3.findIndex(v => v > y0s)
		const i90 = y3.findIndex(v => v > y0s + 0.9 * stepsize)
		const rise = (i10 < 0 || i90 < 0) ? NaN : t[i90] - t[i10]

		let ymax = SPlim
		for (let i = 0; i < N; i++) ymax = Math.max(ymax, y2[i], y3[i])
		const ytop = Math.max(1.2, 1.1 * ymax)

		Plotly.restyle(PLOT, {
			x: [t, t, [0, tf], [tr, tr], [tee, tee]],
			y: [y2, y3, [SPlim, SPlim], [0, ytop], [0, ytop]]
		}, [$(idx_y2), $(idx_y3), $(idx_sp), $(idx_tr), $(idx_tee)])
		Plotly.restyle(PLOT, {name: [
			'SP ≤ $(sp_txt) %  →  obtenido: ' + overshoot.toFixed(2) + ' %',
			'tr ≤ $(tr_txt) s  →  obtenido (0–90%): ' + rise.toFixed(3) + ' s',
			'tee ≤ $(tee_txt) s  →  obtenido: ' + settling.toFixed(3) + ' s'
		]}, [$(idx_sp), $(idx_tr), $(idx_tee)])
		Plotly.restyle(PLOT, {
			x: [[x, x, -a3]], y: [[y, -y, 0]],
			name: ['polos: ζ = ' + zeta.toFixed(3) + ', ωn = ' + wn.toFixed(3)]
		}, [$(idx_polos)])

		const upd = {'xaxis.range': [0, tf], 'yaxis.autorange': true}
		const Lax = 1.1 * a3
		upd['xaxis2.range'] = [-2 * Lax, 0]
		upd['yaxis2.range'] = [-Lax, Lax]
		Plotly.relayout(PLOT, upd)

		PLOT.value = [x, y]
		PLOT.dispatchEvent(new CustomEvent('input'))
	}
	"""

	add_js_listener!(fig, "click", js_click)
	fig
end

# ╔═╡ 9a21116c-a73d-48a9-b0ae-ccae90cea436


# ╔═╡ 24242526-9406-4b86-ac62-f7227a5936cc
let
	if click_polo isa AbstractVector && length(click_polo) == 2
		xc, yc = Float64(click_polo[1]), abs(Float64(click_polo[2]))
		if xc < 0 && yc > 0
			polo_actual[] = (xc, yc)
		end
	end
	x_act, y_act = polo_actual[]
	ωn_act = hypot(x_act, y_act)
	ζ_act = -x_act / ωn_act
	cumple = ζ_act >= ζ_min && ωn_act >= ωn_min && -x_act >= σ_min
	md"""
	**Polos actuales:** ``\zeta`` = $(round(ζ_act, digits=3)), ``\omega_n`` = $(round(ωn_act, digits=3)) rad/s, polo adicional en $(round(-n * ωn_act, digits=3)) — $(cumple ? "🟢 dentro de la región de diseño" : "🔴 fuera de la región de diseño")
	"""
end

# ╔═╡ Cell order:
# ╟─23fce5ad-e993-4001-b4ff-742957cbd59e
# ╟─a5488db8-ade2-440c-849e-06ded69d9b5e
# ╟─5d1b486e-535b-436d-aa61-9ba58cdee12e
# ╟─104b2ca8-79aa-4218-9006-64e14bf19c41
# ╟─a7f6738b-fe1c-49b6-83a4-fc4c9c2f6978
# ╟─91a028da-5f54-4fb1-9109-a9fbffd23544
# ╟─f82e1b85-b33f-43c6-8092-7991ec79133e
# ╟─9850a008-cf41-4def-a4d2-fc33a15fbeda
# ╟─1f2715fa-1953-436d-9178-8b75cce17ffd
# ╠═9a21116c-a73d-48a9-b0ae-ccae90cea436
# ╟─24242526-9406-4b86-ac62-f7227a5936cc
