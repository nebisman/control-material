import numpy as np
from PIL import Image
from scipy.signal import medfilt

# Cargar la imagen
img = Image.open('75.bmp').convert('RGB')
data = np.array(img)
height, width, _ = data.shape

# Corrección: Limitar la Región de Interés (ROI) a las dimensiones reales de la imagen
# Nos aseguramos de que roi_x_end nunca sea mayor que 'width' (800 en este caso)
roi_y_start = 30
roi_y_end = min(height, 450)
roi_x_start = 35
roi_x_end = min(width, 985)

def extract_clean_path(image_data, color_condition, x_range, y_range):
    points = []
    for x in range(x_range[0], x_range[1]):
        # Extraer columna en la ROI
        col_segment = image_data[y_range[0]:y_range[1], x]
        
        # Aplicar condición de color
        mask = color_condition(col_segment)
        y_indices = np.where(mask)[0]
        
        if len(y_indices) > 0:
            # Usar la mediana para ignorar ruido o etiquetas pequeñas
            y_val = np.median(y_indices) + y_range[0]
            points.append((x, y_val))
    
    if not points:
        return []

    # Extraer coordenadas X e Y
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # Aplicar filtro de mediana (ventana de 5) para suavizar inconsistencias
    y_smooth = medfilt(y_coords, kernel_size=5)
    
    return list(zip(x_coords, y_smooth))

# Condiciones de color refinadas
# Rojo: R predominante (Señal 1)
red_cond = lambda c: (c[:,0] > 160) & (c[:,1] < 100) & (c[:,2] < 100)
# Amarillo: R y G altos (Señal 2)
yellow_cond = lambda c: (c[:,0] > 160) & (c[:,1] > 160) & (c[:,2] < 100)

# Extraer las trayectorias con los límites corregidos
red_path = extract_clean_path(data, red_cond, (roi_x_start, roi_x_end), (roi_y_start, roi_y_end))
yellow_path = extract_clean_path(data, yellow_cond, (roi_x_start, roi_x_end), (roi_y_start, roi_y_end))

# Generar SVG plano y limpio
svg_content = [
    f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background-color: #000;">',
    '  '
]

if red_path:
    d_red = "M " + " L ".join([f"{x:.1f},{y:.1f}" for x, y in red_path])
    svg_content.append(f'  <path d="{d_red}" fill="none" stroke="#ff3333" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />')

if yellow_path:
    d_yellow = "M " + " L ".join([f"{x:.1f},{y:.1f}" for x, y in yellow_path])
    svg_content.append(f'  <path d="{d_yellow}" fill="none" stroke="#ffff33" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />')

svg_content.append('</svg>')

# Guardar resultado
output_path = 'osciloscopio_vectorizado_corregido.svg'
with open(output_path, 'w') as f:
    f.write("\n".join(svg_content))