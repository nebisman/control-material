using ControlSystems
using Plots

## ejemplo 1: estabilidad con un control proporcional en un sistema de tercer orden
s=tf("s")
G = 1/(s*(s+1)*(s+.5))
margin(G)

# caso 1 estable
T = feedback(0.4*G,1)
plot(step(T,50))

#caso2 inestable
T2 = feedback(0.8*G,1)
plot(step(T2,50))

## ejemplo 2: estabilidad con un controlador PI

G = 10/((s+1)^3)
kp = range(0, stop=0.8, length=1000)
ki = -10*kp.^2/9 +7*kp/9 .+ 8.0/90
plot(kp, ki, xlabel="k_p", ylabel="k_i", color =:green,title="Frontera de estabilidad",
     label="frontera de estabilidad", fillrange=0, fillalpha=0.3, legend=:topright)

## Aqui corremos un ejemplo donde el es sistema estable

C = 0.5 + 0.2/s
T = feedback(C*G,1)
plot(step(T,50))


## Resolver la cuadrática de la frontera para un valor dado de ki encontrar los valores posibles de kp

ki_valor = 0.2
a = -10/9
b = 7/9
c = 8/90 - ki_valor
r = b^2 - 4*a*c
kp1 = (-b + sqrt(r)) / (2*a)
kp2 = (-b - sqrt(r)) / (2*a)
println("Para ki = $ki_valor =>    $kp1 <= kp <= $kp2")
plot(kp, ki, xlabel="k_p", ylabel="k_i", color =:green,title="Frontera de estabilidad",
     label="frontera de estabilidad", fillrange=0, fillalpha=0.3, legend=:topright)
plot!( [kp1, kp2], [ki_valor, ki_valor], xlabel="k_p", ylabel="k_i", color =:red,
     label="Para ki = $ki_valor =>  $kp1 <= kp <= $kp2 ",  legend=:bottomleft)
