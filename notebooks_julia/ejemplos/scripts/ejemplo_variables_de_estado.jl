
## Control por VE simple sin accion integral
using DCMotor
sys = MotorSystem();
mot = ss(sys; output=:angle)

# diseñamos e implmentamos el controlador
K = place(mot, [-10+10im, -10-10im])
set_ss_controller(sys, K)

# ahora lo probamos
A, B, C ,D = ssdata(mot)
T = ss(A-B*K, B, C,0) 

beta=1/dcgain(T)[1]
T = T*beta
set_ss_controller(sys, K, beta=beta)
result = step_closed(sys; r0 = 0, r1 = 50,  t0 = 1.5, t1 =2) 
stepinfo(result,T)


## Ahora un controlador con RVE con accion integral
Aa = [A [0, 0]; -C 0]
Ba = [B; 0]

polos =  [-10+10im, -10-10im, -25]; 
Ka = place(Aa, Ba, polos)

set_ss_controller(sys, Ka)
result1 = step_closed(sys; r0 = 0, r1 = 100,  t0 = 1.5, t1 =2) 


T1 = ss(Aa-Ba*Ka, Ba, [C 0], [0])
den = denvec(tf(T1))[1] 
T1 = tf(den[end], den) 
stepinfo(result1,T1)

## ejemplo 3 diseño con filtro de Kalman y LQR

Q=[1.0 0 ; 0 1]
R=1
L1 = kalman(mot,Q,R)
QK = ([.00100 0 0; 0 1.0 0; 0 0  1000])
RK = 1000
Klqr = lqr(Continuous,Aa, Ba, QK, RK)
T2 = ss(Aa-Ba*Klqr, Ba, [C 0], [0])
den = denvec(tf(T2))[1] 
T2 = tf(den[end], den) 

set_ss_controller(sys, Klqr; L = L1)
result2 = step_closed(sys; r0 = 0, r1 = 100,  t0 = 1.5, t1 =3) 
stepinfo(result,T2)

