# ═══════════════════════════════════════════════════════════════════════════════
#  itae_model.jl – Funciones de transferencia ITAE de lazo cerrado.
#
#  Coeficientes tomados de las tablas de Dos_parametros.tex (criterio
#  ITAE = ∫ t|e(t)| dt), para error cero al escalón (:pos) o a la rampa (:speed).
# ═══════════════════════════════════════════════════════════════════════════════

using ControlSystemsBase

# Coeficientes normalizados del denominador (ω₀ = 1), de s^n a s^0.
const ITAE_POSICION = Dict(
    1 => [1, 1],
    2 => [1, 1.4, 1],
    3 => [1, 1.75, 2.15, 1],
    4 => [1, 2.1, 3.4, 2.7, 1],
    5 => [1, 2.8, 5.0, 5.5, 3.4, 1],
    6 => [1, 3.25, 6.6, 8.6, 7.45, 3.95, 1],
)

const ITAE_VELOCIDAD = Dict(
    2 => [1, 3.2, 1],
    3 => [1, 1.75, 3.25, 1],
    4 => [1, 2.41, 4.93, 5.14, 1],
    5 => [1, 2.19, 6.5, 6.3, 5.24, 1],
    6 => [1, 6.12, 13.42, 17.16, 14.14, 6.76, 1],
)

"""
    T = itaemodel(order=3, omega=1; type=:pos)

Retorna la función de transferencia ITAE de lazo cerrado de orden `order`.

# Argumentos
- `order::Int=3`: orden de `T(s)`. Con `type=:pos` admite 1 a 6; con `type=:speed`, 2 a 6.
- `omega::Real=1`: frecuencia ``\\omega_0`` (rad/s). Cambia la velocidad de la respuesta
  pero no su forma.

# Argumentos de palabra clave
- `type::Symbol=:pos`:
  + `:pos`: cero error de posición (sigue escalones sin error):
    ``T(s) = \\dfrac{\\omega_0^n}{D(s)}``
  + `:speed`: cero error de velocidad (sigue rampas sin error):
    ``T(s) = \\dfrac{a_1\\,\\omega_0^{n-1} s + \\omega_0^n}{D(s)}``

  donde ``D(s) = s^n + a_{n-1}\\,\\omega_0\\, s^{n-1} + \\cdots + a_1\\,\\omega_0^{n-1}\\, s + \\omega_0^n``.

# Retorna
- `T::ControlSystemsBase.TransferFunction`: función de transferencia continua.
"""
function itaemodel(order::Int = 3, omega::Real = 1; type=:pos)
    tabla = type == :pos ? ITAE_POSICION :
            type == :speed ? ITAE_VELOCIDAD :
            error("type debe ser :pos (posición) o :speed (velocidad)")
    haskey(tabla, order) ||
        error("order debe estar entre $(minimum(keys(tabla))) y $(maximum(keys(tabla))) para type=$(repr(type))")
    omega > 0 || error("omega debe ser positiva")

    n = order
    # el coeficiente de s^(n-k) se multiplica por ω₀^k
    den = tabla[n] .* omega .^ (0:n)
    num = type == :pos ? [omega^n] :
                        den[end-1:end]   # mismos dos últimos coeficientes que D(s) -> e₁ = 0
    return tf(num, den)
end


# ═══════════════════════════════════════════════════════════════════════════════
#  Ejemplo
# ═══════════════════════════════════════════════════════════════════════════════

## Función ITAE de orden 3 para error cero al escalón, con ω₀ = 1 rad/s
using Plots
T = itae_model(3, 1; type=:pos)

## Respuesta al escalón de las ITAE de posición (ω₀ = 1), órdenes 1 a 5
t = 0:0.005:15
fig_p = plot(title="ITAE de posición: respuesta al escalón",
             xlabel="ω₀ t", ylabel="y(t)", legend=:bottomright)
for n in 1:5
    r = step(itae_model(n, 1; type=:pos), t)
    plot!(fig_p, r.t, vec(r.y), label="n = $n")
end
display(fig_p)

## Respuesta a la rampa de las ITAE de velocidad (ω₀ = 1), órdenes 2 a 6
fig_v = plot(t, collect(t), label="rampa", color=:black, linestyle=:dash,
             title="ITAE de velocidad: respuesta a la rampa",
             xlabel="ω₀ t", ylabel="y(t)", legend=:topleft)
for n in 2:6
    r = lsim(itae_model(n, 1; type=:speed), reshape(collect(t), 1, :), t)
    plot!(fig_v, r.t, vec(r.y), label="n = $n")
end
display(fig_v)




