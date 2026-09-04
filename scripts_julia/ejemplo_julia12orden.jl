using ControlSystems
using Plots

## Simulacion de un sistema de primer orden
α = 2
τ = 0.25
s = tf("s")

H = α / (τ * s + 1)
tfinal = 10

sim = step(H, tfinal)
res = stepinfo(sim; risetime_th = (0.0, 0.9), settling_th = 0.01)   # tiempo de subida 0-90 %, asentamiento al 1 %
plot(res)

## Simulacion de un sistema de segundo orden
ζ  = 0.5
ωn = 1.0

G = ωn^2 / (s^2 + 2*ζ*ωn*s + ωn^2)
tfinal2 = 20
sim2 = step(G, tfinal2)
res2 = stepinfo(sim2; risetime_th = (0.0, 0.9), settling_th = 0.01)   # tiempo de subida 0-90 %, asentamiento al 1 %
plot(res2)

