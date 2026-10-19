
## definicion del sistema
using DCMotor
sys = MotorSystem();
Gang = tf(sys; output=:angle)

b = numvec(Gang)[1][1]
a = denvec(Gang)[1][2]
## Calculo de las constantes del PID para el sistema de segundo orden
s=tf("s")
ωn =12
ζ = 0.7
n=2.5

T  = n*ωn^3/((s+n*ωn)*(s^2 + 2*ζ*ωn*s +ωn^2))
Kd = ((2ζ + n) * ωn - a) / b
Kp = (1 + 2n * ζ) * ωn^2 / b
Ki = n * ωn^3 / b
set_pid(sys;  kp=Kp, ki=Ki, kd=Kd, beta=0, Tf=1e-4, output=:angle)
result = step_closed(sys; r0 = 0, r1 = 100,  t0 = 1, t1 = 12/(ζ * ωn));
stepinfo(result, T)



#%% calculo de las constantes del PID con polinomio ITAE

ωn = 14
T = (ωn^3)/(s^3 + 1.75*s^2*ωn + 2.15*s*ωn^2 + ωn^3)
Kd = (1.78 * ωn - a) / b
Kp = 2.15 * ωn^2 / b
Ki = ωn^3/b
 
set_pid(sys;  kp=Kp, ki=Ki, kd=Kd, beta=0,  N=10, output=:angle)
result = step_closed(sys; r0 = 0, r1 = 100,  t0 = 1, t1 =2);
stepinfo(result, T)


