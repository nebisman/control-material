
## Ejemplo 1: controlador de  un grado de libertad (1-GDL)
# Supongamos que queremos diseñar e implementar un controlador de 1-GDL
# usando un método algebraico
# Para ello usamos el siguiente código:


## definiciones
using DCMotor
sys = MotorSystem();
G = tf(sys; output=:angle, numpar=3)


## diseño y carga del controlador
polos_lc = [-10+10im, -10-10im, -80, -70, -80]
C = cont1dof(G, polos_lc) 
set_controller(sys, C; output=:angle)


# Aqui vemos la respuesta al escalon en lazo cerrado

T = feedback(C*G, 1)
result = step_closed(sys; r0 = 0, r1 =100,  t0 = 0.5, t1 =2); 
stepinfo(result,T)
