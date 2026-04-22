import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------
# Parámetros configurables
# --------------------------------

NUM_ITER = 300                      # número de iteraciones
START_TEMP = 1                      # temperatura inicial
MIN_TEMP = 1e-4                     # temperatura mínima para detener
ALPHA = 0.995                       # factor de enfriamiento (0.995 = 0.5 en ~138 iteraciones)
NEIGH_STEP = 0.5                    # paso máximo para generar vecinos (ajustable)
PAUSA = 0.05                        # segundos entre actualizaciones (visualización)
RANGO = (-3.0, 3.0)                 # rango de búsqueda
GRID_RES = 160                      # resolución de la cuadrícula para visualización
TRAIL_LENGTH = 12                   # longitud de la trayectoria a visualizar
SEED = 2029                         # semilla para generación de números aleatorios
USE_GAUSSIAN_NEIGHBORHOOD = True    # utilizar vecindad gaussiana

# --------------------------------
# Función objetivo: "peaks"
# --------------------------------

def peaks(x, y):
    return 3*(1-x)**2*np.exp(-(x**2) - (y+1)**2) \
           - 10*(x/5 - x**3 - y**5)*np.exp(-x**2 - y**2) \
           - 1/3*np.exp(-(x+1)**2 - y**2)
           
# --------------------------------
# Inicialización de SA
# --------------------------------

rng = np.random.default_rng(SEED)
lo, hi = RANGO

#Solución inicial aleatoria
x_curr = rng.uniform(lo, hi)
y_curr = rng.uniform(lo, hi)
f_curr = peaks(x_curr, y_curr)

#Mejor Global
x_best, y_best, f_best = x_curr, y_curr, f_curr

#Historial para rastro 
history = [(x_curr, y_curr)]

# ---------------------------------
# Preparar la malla para visualización
# ---------------------------------

X = np.linspace(lo, hi, GRID_RES)
Y = np.linspace(lo, hi, GRID_RES)
Xg, Yg = np.meshgrid(X, Y)
Zg = peaks(Xg, Yg)

plt.ion()
fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# crear contorno inicial y colorbar 

cont0 = ax1.contourf(Xg, Yg, Zg, levels=40, cmap='viridis')
cbar = plt.colorbar(cont0, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('Valor de función')

def dibujar(iteracion, T, x_curr, y_curr, f_curr, x_best, y_best, f_best, history):
    ax1.clear()
    ax2.clear()
     
    # Contorno 2D
    ax1.contourf(Xg, Yg, Zg, levels=40, cmap='viridis')
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_title(f"Simulated Annealing - Iteración {iteracion} / {NUM_ITER} - T: {T:.4f}")
    
    # dibujar rastro con degradado de colores
    if TRAIL_LENGTH > 0:
        trail = np.array(history[-TRAIL_LENGTH:])
        L = trail.shape[0]
        if L > 1:
            for i in range(L-1):
                a = 0.15 + 0.85 * (i+1) / L  # alfa degradado
                ax1.plot(trail[i:i+2, 0], trail[i:i+2, 1], color='cyan', alpha=a, linewidth=1.2)
                
    #Punto actual
    ax1.scatter(x_curr, y_curr, color='blue', s=60, label='Actual')
    ax1.scatter(x_best, y_best, color='red', s=80, marker='*', label='Mejor Global')
    ax1.legend(loc='upper right', fontsize='small')
    
    #3D
    ax2.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.85, linewidth=0, antialiased=False)
    ax2.set_title(f"Simulated Annealing - Superficie 3D")
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.scatter(x_curr, y_curr, f_curr, color='blue', s=60)
    ax2.scatter(x_best, y_best, f_best, color='red', s=80, marker='*')
    ax2.view_init(elev=35, azim=60)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(PAUSA)
    time.sleep(PAUSA)
    
# ---------------------------------
# Simulated Annealing  Bucle Principal
# ---------------------------------

T = START_TEMP
iter_count = 0

while iter_count < NUM_ITER and T > MIN_TEMP:
    iter_count += 1
    
    # Generar vecino
    if USE_GAUSSIAN_NEIGHBORHOOD:
        x_new = x_curr + rng.normal(0, NEIGH_STEP)
        y_new = y_curr + rng.normal(0, NEIGH_STEP)
    else:
        x_new = x_curr + rng.uniform(-NEIGH_STEP, NEIGH_STEP)
        y_new = y_curr + rng.uniform(-NEIGH_STEP, NEIGH_STEP)
    
    #Asegurar Limites 
    
    xn = np.clip(xn, lo, hi)
    yn = np.clip(yn, lo, hi)
    
    f_new = peaks (xn, yn)
    delta = f_new - f_curr  # Positivo si mejora
    
    # Criterio de aceptación
    accept = False
    if delta > 0:
        accept = True
    else:
        prob = np.exp(delta / T)  # Probabilidad de aceptar peor solución
        if rng.uniform(0, 1) < prob:
            accept = True
    
    if accept:
        x_curr, y_curr, f_curr = xn, yn, f_new
        history.append((x_curr, y_curr))
        
        if f_curr > f_best:
            x_best, y_best, f_best = x_curr, y_curr, f_curr
    else:
        history.append((x_curr, y_curr))  # mantener rastro incluso si no se acepta
        
    # Enfriamiento
    T *= ALPHA
    
    # Dibujar estado actual
    
    dibujar(iter_count, T, x_curr, y_curr, f_curr, x_best, y_best, f_best, history)
    
# ----------------------------------
# Resultado final
# ----------------------------------
print("\nResultado final SA ->")
print(f"Mejor (x,y) = ({x_best:.6f}, {y_best:.6f}), f = {f_best:.6f}")
print(f"Ultima solución (x,y) = ({x_curr:.6f}, {y_curr:.6f}), f = {f_curr:.6f}")
print(f"Iteraciones Realizadas: {iter_count}, Temperatura Final: {T:.6f}")

plt.ioff() 
plt.show()
