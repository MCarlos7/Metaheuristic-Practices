import numpy as np 
import matplotlib.pyplot as plt 
import time
from mpl_toolkits.mplot3d import Axes3D

def peaks(x, y):
    return 3*(1 - x)**2 * np.exp(-(x**2) - (y + 1)**2) \
           - 10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
           - 1/3 * np.exp(-(x + 1)**2 - y**2)


#-------------------------------------------
# Parametros
#-------------------------------------------

NUM_ITERACAIONES = 60           #Numero de iteraciones (TAMAÑPO DEL ENJAMBRE)
SWARM_SIZE = 30                 #Numero de partículas en el enjambre
W_MAX = 0.9                     #Máximo coeficiente de inercia
W_MIN = 0.4                     #Mínimo coeficiente de inercia
C1 = 1.5                        #Coeficiente de cognición
C2 = 1.5                        #Coeficiente de socialización
RANGO = (-3.0, 3.0)             #Rango de búsqueda
PAUSA = 0.25                    #Tiempo de pausa entre iteraciones
SEED = 42                       #Semilla para generación de números aleatorios
V_MAX_RATIO = 0.2               #Proporción máxima de velocidad
TRAIL_LEGTH = 6                 #Longitud de la estela
GRID_RES = 160                  #Resolución de la cuadrícula para visualización

#-------------------------------------------
#INICIALIZACION DEL ENJAMBRE
#-------------------------------------------

rng = np.random.default_rng(SEED)
lo, hi = RANGO
vmax = V_MAX_RATIO * (hi - lo)

# Posiciones : warm_size x 2
pos = rng.uniform(lo, hi, size=(SWARM_SIZE, 2))
# Velocidades inician pequeñas
vel = rng.uniform(-vmax, vmax, size=(SWARM_SIZE, 2))

#Mejor personal (pbest) y sus valores
pbest_pos = pos.copy()
pbest_val = np.array([peaks(x,y) for x,y in pbest_pos])

#Mejor global (gbest)
gbest_idx = np.argmax(pbest_val)
gbest_pos = pbest_pos[gbest_idx].copy()
gbest_val = pbest_val[gbest_idx]

#Para dubujar rasto (opcional)
history = [[pos[i].copy()] for i in range(SWARM_SIZE)]

#-------------------------------------------
#Prepara malla y figuras
#-------------------------------------------

X = np.linspace(lo, hi, GRID_RES)
Y = np.linspace(lo, hi, GRID_RES)
Xg, Yg = np.meshgrid(X, Y)
Zg = peaks(Xg, Yg)

plt.ion()
fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

#Crear contprno inicial y una sola barra de color
cont = ax1.contourf(Xg, Yg, Zg, levels=40, cmap='viridis')
cbar = plt.colorbar(cont, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('Valor de f(x,y)', rotation=270, labelpad=15)
ax1.set_title('PSO - Contorno 2D')

#Funcion para dibujar el estado actual 
def dibujar (iteracion):
    ax1.clear()
    ax2.clear()
    
    #Redibujar contorno
    cont = ax1.contourf(Xg, Yg, Zg, levels=40, cmap='viridis')
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_title(f'PSO - Contorno 2D (Iteración {iteracion} / {NUM_ITERACAIONES})')
    
    #Dibujar rastros (si aplica)
    if TRAIL_LEGTH > 0:
        for i in range(SWARM_SIZE):
            trail = np.array(history[i][-TRAIL_LEGTH:])
            if trail.shape[0] > 1:
                ax1.plot(trail[:, 0], trail[:, 1], 'w-', alpha=0.5)
    
    ax1.scatter(pos[:, 0], pos[:, 1], color='blue', s=30, edgecolor='black', label='Partículas')
    ax1.scatter(pbest_pos[:, 0], pbest_pos[:, 1], color='green', s=20, marker='o', label='pbest')
    ax1.scatter(gbest_pos[0], gbest_pos[1], color='red', s=50, marker='*', label='gbest')
    ax1.legend(loc='upper right', fontsize='small')
    
    #Dibujar superficie 3D
    ax2.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.8, linewidth=0, antialiased=False)
    ax2.set_title('PSO - Superficie 3D')
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    itness_curr = np.array([peaks(x,y) for x,y in pos])
    ax2.scatter(pos[:, 0], pos[:, 1], itness_curr, color='blue', s=30, edgecolor='black')
    pbest_vals = np.array([peaks(x,y) for x,y in pbest_pos])
    ax2.scatter(pbest_pos[:,  0], pbest_pos[:, 1], pbest_vals, color='green', s=20)
    ax2.scatter(gbest_pos[:0], gbest_pos[:1], gbest_val, color='red', s=150, marker='*')
    ax2.view_init(elev=35, azim=-60)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(PAUSA)
    time.sleep(PAUSA)

#-------------------------------------------
#Ciclo principal de PSO
#-------------------------------------------

for t in range( 1, NUM_ITERACAIONES + 1):
    
    w = W_MAX - (W_MAX - W_MIN) * (t - 1) / max(1, NUM_ITERACAIONES - 1)  # Decreciente linealmente
    
    for i in range(SWARM_SIZE):
        r1 = rng.random(2)
        r2 = rng.random(2)
        
        cognitive = C1 * r1 * (pbest_pos[i] - pos[i])
        social = C2 * r2 * (gbest_pos - pos[i])
        vel[i] = w * vel[i] + cognitive + social
        
        # Limitar velocidad
        vel[i] = np.clip(vel[i], -vmax, vmax)
        
        #Actualizar posición
        pos[i] = pos[i] + vel[i]
        #Mantener en los limites de el dominio 
        pos[i, 0] = np.clip(pos[i, 0], lo, hi)
        pos[i, 1] = np.clip(pos[i, 1], lo, hi)
        
        #Guardar historia para el rastro
        history[i].append(pos[i].copy())
    
    fitness = np.array([peaks(x,y) for x,y in pos])
    improved_mask = fitness > pbest_val
    if np.any(improved_mask):
        pbest_pos[improved_mask] = pos[improved_mask]
        pbest_val[improved_mask] = fitness[improved_mask]   
    
    #Actualizar global
    idx = np.argmax(pbest_val)
    if pbest_val[idx] > gbest_val:
        gbest_val = pbest_val[idx]
        gbest_pos = pbest_pos[idx].copy()
    dibujar(t)
    
#-------------------------------------------
#Resultados finales 
#-------------------------------------------

print(f'Resultado final PSO -> gbest = ({gbest_pos[0]:.6f}, {gbest_pos[1]:.6f}), f= {gbest_val:.4f}')

plt.ioff()
plt.show()

