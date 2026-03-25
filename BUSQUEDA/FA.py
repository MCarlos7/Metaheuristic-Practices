import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# Parámetros configurables 
# -------------------------------

NUM_ITERACIONES = 80       # número de iteraciones principales
NUM_FIREFLIES = 30         # número de luciérnagas
BETA0 = 1.0                # intensidad de la luz base
GAMMA = 1.0                # coeficiente de absorción de luz
ALPHA = 0.2                # factor de aleatoriedad
ALPHA_DECAY = 0.98         # tasa de decaimiento de aleatoriedad por iteración
PAUSA = 0.18               # segundos entre actualizaciones (visualización)
RANGO = (-3.0, 3.0)        # dominio de búsqueda (x,y)
GRID_RES = 160             # resolución de malla para la superficie
TRIAL_LENGTH = 8           # longitud del rastro visual de cada luciérnaga (0 = sin rastro)
SEED = 2025                # semilla para reproducibilidad
TRAND_SCALE = 1            # escala para el término aleatorio (se multiplica por el rango de búsqueda)

# -------------------------------
# Función objetivo: "peaks"
# -------------------------------

def peaks(x, y):
    return 3*(1 - x)**2 * np.exp(-(x**2) - (y + 1)**2) \
           - 10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
           - 1/3 * np.exp(-(x + 1)**2 - y**2)
           
# -------------------------------
# Inicialización 
# -------------------------------

rng = np.random.default_rng(SEED)
lo, hi = RANGO
dim = 2
scale_noise = TRAND_SCALE * (hi - lo)

# posiciones de las luciérnagas (NUM_FIREFLIES x 2)
x = rng.uniform(lo, hi, size=(NUM_FIREFLIES, dim))
# valores y brillo 
vals = np.array([peaks(x, y) for x, y in x])

#historial para trazo (lista de listas)
history = [ [x[i].copy()] for i in range(NUM_FIREFLIES) ]

#mejor global
best_idx = np.argmax(vals)
best_pos = x[best_idx].copy()
best_val = vals[best_idx]

# -------------------------------
# Prepara malla y figura para visualización
# -------------------------------   

Xg = np.linspace(lo, hi, GRID_RES)
Yg = np.linspace(lo, hi, GRID_RES)
Xmg, Ymg = np.meshgrid(Xg, Yg)
Zmg = peaks(Xmg, Ymg)       

plt.ion()
fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(121)           # 2D contorno
ax2 = fig.add_subplot(122, projection='3d')  # 3D superficie

# contorno inicial y colorbar
cont0 = ax1.contourf(Xmg, Ymg, Zmg, levels=40, cmap='viridis')
cbar = plt.colorbar(cont0, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('Valor de la funcion')

def dibujar (iter, X, vals, best_pos, best_val, history, alphas_vis):
    ax1.clear()
    ax2.clear()
    
    # Redebujar contorno (la barra de color se mantiene)
    cont = ax1.contourf(Xmg, Ymg, Zmg, levels=40, cmap='viridis')
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_title(f'Firefly Algorithm - Contorno 2D (Iteración {iter+1}/{NUM_ITERACIONES})')
    
    #Dibujar rastros con degradado de alpha
    if TRIAL_LENGTH > 0:
        for i in range(NUM_FIREFLIES):
            trail = np.array(history[i][-TRIAL_LENGTH:])  # últimos TRIAL_LENGTH puntos
            if len(trail) > 1:
                alphas = np.linspace(0.1, 0.8, len(trail))  # degradado de alpha
                L = trail.shape[0]
                if L > 1:
                    # Dibujar segmentos con alpha variable
                    for k in range(L-1):
                        a = alphas_vis * (k+1) / L  # alpha proporcional a la posición en el trail
                        ax1.plot(trail[k:k+2, 0], trail[k:k+2, 1], linewidth=2, alpha=a, color='cyan')
    
    #posiciopnes actuales de las luciérnagas
    ax1.scatter(x[:, 0], x[:, 1], c='blue', label='Fireflies', s=36)
    
    #mejor global
    ax1.scatter(best_pos[0], best_pos[1], c='red', s=140, marker='*', label='Best Global')
    
    ax1.legend(loc='upper right', fontsize='small')
    
    #superficie 3D
    ax2.plot_surface(Xmg, Ymg, Zmg, cmap='viridis', alpha=0.85, linewidth=0, antialiased=False)
    ax2.set_title('Firefly Algorithm - Superficie 3D')
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    ax2.scatter(x[:, 0], x[:, 1], vals, c='blue', s=36)
    ax2.scatter(best_pos[0], best_pos[1], best_val, c='red', s=180, marker='*')
    ax2.view_init(elev=35, azim=-60)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(PAUSA)
    time.sleep(PAUSA)
    
# -------------------------------
# Bucle principal FA 
# -------------------------------

alpha_curr = ALPHA
for t in range(NUM_ITERACIONES + 1):
    # ordenar por brillo (valor de la función)
    order = np.argsort(-vals)  # orden descendente
    x_sorted = x[order]
    vals_sorted = vals[order]
    
    #para cada luciérnaga i (menos brillante) moverse hacia luciérnagas más brillantes j
    for idx_i in range(NUM_FIREFLIES):
        i = order[idx_i]
        xi = x[i].copy()
        fi = vals[i]
        
        #recorrer luciérnagas más brillantes
        for idx_j in range(idx_i):
            j = order[idx_j]
            xj = x[j]
            fj = vals[j]
            
            # si j es más brillante que i
            r = np.linalg.norm(xi - xj)
            # attracción euclídea 
            beta = BETA0 * np.exp(-GAMMA * r**2)
            # movimiento: xi <- xi + beta*(xj - xi) + alpha*rand_noise
            noise = rng.normal(0, 1, size=dim) # gaussiano 
            xi = xi + beta * (xj - xi) + alpha_curr * noise * (scale_noise/2)
            
            # asegura limites
            xi = np.clip(xi, lo, hi)
            
        #actualixar poción provitoria de i
        x[i] = xi
        vals[i] = peaks(xi[0], xi[1])
        history[i].append(xi.copy())
    
    # opcional: decaimiento de alpha para reducir aleatoriedad con el tiempo
    alpha_curr *= ALPHA_DECAY
    
    # actualizar mejor global
    best_idx = np.argmax(vals)
    best_pos = x[best_idx].copy()
    best_val = vals[best_idx]
    
    # dibujar estado actual
    dibujar(t, x, vals, best_pos, best_val, history, alphas_vis=0.9)
    
# -------------------------------
# Resultado final
# -------------------------------

print(f"Resultado final FA --> best = {best_pos[0]:.6f}, {best_pos[1]:.6f} f = {best_val:.4f}")

plt.ioff()
plt.show()
