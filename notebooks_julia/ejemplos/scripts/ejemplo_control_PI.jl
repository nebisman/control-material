
## definicion del sistema y parametros
using DCMotor
sys = MotorSystem();
G = tf(sys, output=:speed)

## Diseño por localización de polos
# Frecuencia natural y factos de amortiguamiento
ωn = 15
ζ = .7
s = tf("s")
T= ωn^2/(s^2 + 2*ζ*ωn*s + ωn^2)

# calculo de los coeficientes del PIas constantes
_, Kp, Ki = placePI(G, ωn, ζ; form=:parallel)
set_pid(sys;  kp=Kp, ki=Ki, kd=0, beta=0, output=:speed, deadzone=0)
result = step_closed(sys; r0 = 00, r1 =400,  t0 = 0.5, t1 = max(1.5, 12/(ζ * ωn))    );
stepinfo(result,T)


## Ahora diseñamos un PI para velocidad con Loopshaping, 

ωgc = 15
# Note que incluimos el retardo del muestreo digital, por lo cual se requiere 
# el paquete completo ControlSystems
using ControlSystems
Gd = G*delay(0.02)
C, Kp, Ki, fig, CF = loopshapingPI(Gd, ωgc; rl=1,  phasemargin=50, form=:parallel)

T1 = feedback(C*Gd, 1)
# y lo probamos
set_pid(sys;  kp=Kp, ki=Ki, kd=0, beta=1, output=:speed)
result = step_closed(sys; r0 = 0, r1 = 400,  t0 = .5, t1 =max(1.5, 16/ωgc)); 
stepinfo(result,T1)

