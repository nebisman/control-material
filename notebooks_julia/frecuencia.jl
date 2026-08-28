    using ControlSystems, Plots
    
# Diseño de PI en la frecuencia

b = 2310.3    # ganancia de la planta
a = 3.13    # polo de la planta
d = 0.02   # retardo [s]

s   = tf("s")
                # controlador PI = (kp·s + ki)/s
Gnd = b/(s + a)                 # planta sin retardo
G   = Gnd*delay(d)   

ωgc = 10
ϕm = 60
C, kp, ki, fig, CF = loopshapingPI(G, ωgc; rl=1,  phasemargin=ϕm, form=:parallel, doplot=true)

L  = C*G  
ω180, GM, ωgc, PM =  margin(L)
ω = exp10.(range(log10(.1), log10(100), length=200))
p1=marginplot(L,ω)
 
T = feedback(L)
p2= plot(step(T), title="Respuesta al escalón del lazo cerrado", xlabel="Tiempo [s]", ylabel="y(t)", lw=2, c=:red, label="")
plot(p1,p2,layout=(1,2))
fig


# diseño de PID por loopshaping 
# para el control de posición de un motor DC 

Gp = Gnd/s
Mt = 1.3            # Maximum magnitude of complementary sensitivity
ϕt = 60         # Angle of tangent point
ωt  = 50              # Frequency at which the specification holds
C1, kp1, ki1, kd1, fig1 = loopshapingPID(Gp, ωt; Mt, ϕt, doplot=true, form=:parallel)
L1 = C1*Gp
fig1


#marginplot(L1, ω, adjust_phase_start=true)
