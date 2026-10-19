
## Ejemplo 1: controlador de  un grado de libertad (1-GDL)
# Supongamos que queremos diseñar e implementar un controlador de 1-GDL
# usando un método algebraico
# Para ello usamos el siguiente código:


## definiciones
using DCMotor
sys = MotorSystem();
G = tf(sys, numpar=3)
s=tf("s")



## diseño y carga del controlador
polos_lc = [-25+15im, -25-15im, -40,-50,-50]
C = cont1dof(G, polos_lc) 
set_controller(sys, C; output=:angle)


# Aqui vemos la respuesta al escalon en lazo cerrado

T = feedback(C*G, 1)
profile_closed(sys; timevalues =[0, 1, 2, 3, 4], refvalues=[0, 90, 90, 180, 0])

