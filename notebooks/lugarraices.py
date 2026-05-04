import matplotlib.pyplot as plt
import numpy as np
import control as ctrl
import mplcursors



def dibujarRegionDiseno(SP_max=0.05, tee_max=5, tr_max=1,  ax=None):
    global labSP, labWn, labTee

    zeta_min = abs(np.log(SP_max)) / np.sqrt(np.pi**2 + np.log(SP_max)**2)
    wn_min = (2.23 * zeta_min**2 + 0.036 * zeta_min + 1.54) / tr_max
    teta_max = np.arccos(zeta_min)
    np.degrees(teta_max)

    sigma_min = 5 / tee_max

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    r = 6

    # Líneas de sobrepico (ángulo zeta)
    labSP = f'$\\theta$ = {np.degrees(teta_max):.1f}° (SP={SP_max*100:.0f}%)'
    ax.plot([0, -r * np.cos(teta_max)], [0, -r * np.sin(teta_max)], 'g-', linewidth=1.5,
            label=labSP)
    
    ax.plot([0, -r * np.cos(teta_max)], [0,  r * np.sin(teta_max)], 'g-', linewidth=1.5)

   

    # Línea de tiempo de establecimiento (sigma)
    labTee = f'$\\sigma$ = {sigma_min:.3f} ($t_{{ee}}$ = {tee_max}s)'
    ax.axvline(x=-sigma_min, color='g', linestyle='--', linewidth=1.5,
               label=labTee)

    # Círculo de wn_min
    theta = np.linspace(np.pi/2,  3*np.pi/2, 200)
    labWn = f'$\\omega_n$ = {wn_min:.3f} ($t_r$ = {tr_max}s)'
    ax.plot(wn_min * np.cos(theta), wn_min * np.sin(theta), 'm--', linewidth=1,
            label=labWn)
    ax.plot([0,0], [-10, 10], 'k', linewidth=1.5)
    if standalone:
        ax.set_xlim([-1.5*np.max([wn_min, sigma_min]), 1])

        ax.set_ylim([-1.5*np.max([wn_min, sigma_min]),1.5*np.max([wn_min, sigma_min])])
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginario')
        ax.set_title('Región de diseño')
        ax.grid(True)
        ax.legend(loc="lower left")
        ax.set_aspect('equal')
        plt.show()
        
def hacerLugarRaices(G, SP_max=0.05, tee_max=5, tr_max=1):
    fig, (ax,ax2) = plt.subplots(1,2, figsize=(16, 8))
    rlist, klist = ctrl.root_locus(G, plot=True, ax=ax)
    
    def plotStep(k=0.1):
      
        T = ctrl.feedback(k * G, 1)


        t, y = ctrl.step_response(T)
        ax2.cla()
        ax2.plot(t, y, 'b', linewidth=2, label='Respuesta al escalón')

        # Valor final (estado estacionario)
        yss = 1

        # --- Especificación de sobrepico ---
        y_sp = yss * (1 + SP_max)
        ax2.axhline(y=y_sp, color='r', linestyle='--', linewidth=1.2,
                label=f'SP máx = {SP_max*100:.0f}%  (y = {y_sp:.3f})')

        # --- Especificación de tiempo de establecimiento ---
        ax2.axvline(x=tee_max, color='g', linestyle='--', linewidth=1.2,
                label=f'$t_{{ee}}$ máx = {tee_max} s')

        # Banda de establecimiento (±2% del valor final, criterio del 2%)
        banda = 0.02

        ax2.axhline(y=yss * (1 + banda), color='g', linestyle=':', linewidth=0.8, alpha=0.7)
        ax2.axhline(y=yss * (1 - banda), color='g', linestyle=':', linewidth=0.8, alpha=0.7)
        ax2.fill_between([tee_max, t[-1]], yss*(1 - banda), yss*(1 + banda),
                        color='green', alpha=0.08, label=f'Banda ±{banda*100:.0f}%')

        # --- Especificación de tiempo de subida ---
        ax2.axvline(x=tr_max, color='m', linestyle='--', linewidth=1.2,
                label=f'$t_r$ máx = {tr_max} s')

        # Línea del valor final
        ax2.axhline(y=yss, color='k', linestyle='-', linewidth=0.5, alpha=0.4)

        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Amplitud')
        ax2.set_title('Respuesta al escalón con especificaciones de diseño')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', fontsize=9)

    

    ax.set_xlim([-6, 1])
    ax.set_ylim([-4, 4])
    ax.set_title('Lugar de las raíces con región de diseño (clic para ver interactuar)')


    # Habilitar cursor interactivo en TODAS las líneas del gráfico

    lines = [line for line in ax.get_lines()]    
    cursor = mplcursors.cursor(lines, hover=False)

    # Superponer la región de diseño
    dibujarRegionDiseno(SP_max, tee_max, tr_max, ax=ax) 
 
    lines_esp =  [line for line in ax.get_lines() if line not in lines]  
    ax.grid(True)       
    ax.legend(lines_esp, ["", labSP, labTee,labWn],loc='lower left', fontsize=8)
    plotStep()

    @cursor.connect('add')
    def on_add(sel):

        x, y = sel.target
        s0 = x + y*1j         
        k =  1/np.abs(ctrl.evalfr(G, s0))
        if abs(y) < 1e-6:
                sel.annotation.set_text(f's = {x:.4f}; k={k:.4f}')
        elif y >= 0:
                sel.annotation.set_text(f's = {x:.4f} + j{y:.4f}; k={k:.4f}')
        else:
                sel.annotation.set_text(f's = {x:.4f} - j{abs(y):.4f}; k={k:.4f}')
        sel.annotation.get_bbox_patch().set(fc='lightyellow', alpha=0.9)

        plotStep(k)
        

    plt.show()



#s = ctrl.tf('s')
#G = 182.27 / (s * (s + 2.28) * (s + 3.664))
#hacerLugarRaices(G, SP_max=0.05, tee_max=5, tr_max=1)