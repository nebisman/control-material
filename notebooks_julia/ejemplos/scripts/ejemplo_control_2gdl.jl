
# Ejemplo: controlador de dos grados de libertad (2-GDL)
# Supongamos que queremos diseñar e implementar un controlador de dos
# grados de libertad por métodos algebraicos.
# Para ello usamos el siguiente código:


## definiciones
using DCMotor
sys = MotorSystem();
G_ang = tf(sys, output=:angle, numpar=3)

## Vamos a implementar la siguiente funcion itae
ω₀ = 20
s=tf("s")
T = itaemodel(3, ω₀; type=:pos)
C = cont2dof(G_ang, T, 2, [-80, -90])

# respuesta del controlador
set_controller(sys, C; output=:angle)
result = step_closed(sys; r0 = 0, r1 = 90,  t0 = 0.5, t1 =2); 
stepinfo(result, T)



## Ahora vamos a implementar una función cuadrática óptima
Tlq, Gur = lqmodel(G_ang, .005)
C = cont2dof(G_ang, Tlq, 2 , [-50,-55])
set_controller(sys, C; output=:angle)
result = step_closed(sys; r0 = 0, r1 = 90,  t0 = 0.5, t1 =2); 
stepinfo(result, Tlq)   