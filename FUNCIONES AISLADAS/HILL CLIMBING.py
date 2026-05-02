import numpy as np 
import matplotlib.pyplot as plt 
import time

# ==========================================
# FUNCIONES OBJETIVO
# ==========================================
def peaks(x, y):
    return 3*(1 - x)**2 * np.exp(-(x**2) - (y + 1)**2) \
           - 10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
           - 1/3 * np.exp(-(x + 1)**2 - y**2)

def Ackley(x, y):
    A, B, C = 20, 0.2, 2 * np.pi
    term1 = -A * np.exp(-B * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(C * x) + np.cos(C * y)))
    return term1 + term2 + A + np.exp(1)

def Rastrigin(x, y):
    A = 10
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def Sphere(x, y):
    return x**2 + y**2

# ==========================================
# ALGORITMOS DE INICIALIZACIÓN
# ==========================================
def init_aleatoria(n, bounds):
    """Genera 'n' puntos de forma completamente aleatoria uniforme."""
    lo, hi = bounds
    return np.random.uniform(lo, hi, (n, 2))

def init_lhs(n, bounds):
    """Genera 'n' puntos utilizando Latin Hypercube Sampling (LHS) 2D."""
    lo, hi = bounds
    pts = np.empty((n, 2))
    for d in range(2):
        # Dividir el espacio en 'n' intervalos y elegir un punto en cada uno
        intervals = np.linspace(lo, hi, n + 1)
        rnds = np.random.uniform(intervals[:-1], intervals[1:])
        np.random.shuffle(rnds)
        pts[:, d] = rnds
    return pts

def init_maxdistance(n, bounds, num_candidates=50):
    """Genera 'n' puntos maximizando la distancia mínima entre ellos (Farthest Point)."""
    lo, hi = bounds
    # El primer punto es aleatorio
    pts = [np.random.uniform(lo, hi, 2)]
    for _ in range(1, n):
        # Generar candidatos aleatorios
        cands = np.random.uniform(lo, hi, (num_candidates, 2))
        # Para cada candidato, calcular la distancia al punto ya elegido más cercano
        dists = np.array([min(np.linalg.norm(c - p) for p in pts) for c in cands])
        # Elegir el candidato que maximice esta distancia mínima
        pts.append(cands[np.argmax(dists)])
    return np.array(pts)

# ==========================================
# BÚSQUEDA HILL CLIMBING
# ==========================================
def hill_climbing(func_objetivo, init_method='aleatoria', n_init=10, bounds=(-3,3), step_size=0.3, step_decay=0.5, min_step=1e-3, pause=0.1):
    lo, hi = bounds
    
    # 1. Generar puntos iniciales según el algoritmo seleccionado
    if init_method == 'aleatoria':
        pts = init_aleatoria(n_init, bounds)
    elif init_method == 'lhs':
        pts = init_lhs(n_init, bounds)
    elif init_method == 'maxdistance':
        pts = init_maxdistance(n_init, bounds)
    else:
        pts = init_aleatoria(n_init, bounds) # Fallback
        
    # 2. Evaluar todos los puntos generados para empezar desde el mejor candidato
    best_pt = None
    best_val = -np.inf # Nota: Tu algoritmo original busca maximizar
    for p in pts:
        val = func_objetivo(p[0], p[1])
        if val > best_val:
            best_val = val
            best_pt = p
            
    x, y = best_pt
    fx = best_val
    
    dirs = [(dx, dy) for dx in (-1,0,1) for dy in (-1,0,1) if not (dx==0 and dy==0)]
    
    # 3. Preparar los gráficos
    X = np.linspace(-3, 3, 200)
    Y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(X, Y)
    z = func_objetivo(X, Y)
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    cont = ax1.contourf(X, Y, z, levels=40, cmap='viridis')
    plt.colorbar(cont, ax=ax1)
    ax1.set_title(f'Hill Climbing ({init_method.upper()}) - Contorno 2D')
    
    # Dibujar los candidatos generados por la inicialización en negro
    ax1.scatter(pts[:, 0], pts[:, 1], c='black', marker='x', label='Muestra Inicial')
    ax1.legend()
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, z, cmap='viridis', alpha=0.8)
    ax2.set_title('Superficie 3D')
    
    path = [(x, y, fx)]
    
    # 4. Iniciar el ciclo de optimización
    while step_size > min_step:
        best_neighbor = None
        current_best = fx
        
        for dx, dy in dirs:
            xn = np.clip(x + dx*step_size, lo, hi)
            yn = np.clip(y + dy*step_size, lo, hi)
            fn = func_objetivo(xn, yn)
            
            # Buscar el máximo local
            if fn > current_best:
                current_best = fn
                best_neighbor = (xn, yn)
                
        if best_neighbor:
            x, y = best_neighbor
            fx = current_best
            path.append((x, y, fx))
            
            # DIBUJAR PASO
            ax1.plot([p[0] for p in path], [p[1] for p in path], "w.-")
            ax2.plot([p[0] for p in path], [p[1] for p in path], [p[2] for p in path], "r.-")
            
            plt.draw()
            plt.pause(pause)
            time.sleep(pause) 
        else:
            # Si no hay mejora, reducir el tamaño del paso
            step_size *= step_decay
            
    print(f'Mejor solución encontrada: ({x:.4f}, {y:.4f}), f(x,y) = {fx:.4f}')
    plt.ioff()
    plt.show()

# ==========================================
# MENÚ PRINCIPAL
# ==========================================
print("Selecciona tu función objetivo:")
print("1. Peaks")
print("2. Ackley")
print("3. Rastrigin")
print("4. Sphere")
input_func = input("Ingresa el número de la función: ")

print("\nSelecciona el algoritmo de inicialización:")
print("1. Aleatoria (Random)")
print("2. LHS (Latin Hypercube Sampling)")
print("3. MaxDistance (Farthest Point)")
input_init = input("Ingresa el número del método: ")

map_func = {'1': peaks, '2': Ackley, '3': Rastrigin, '4': Sphere}
map_init = {'1': 'aleatoria', '2': 'lhs', '3': 'maxdistance'}

if input_func in map_func and input_init in map_init:
    func_elegida = map_func[input_func]
    metodo_init = map_init[input_init]
    
    print(f"\nIniciando optimización buscando el MÁXIMO con inicialización '{metodo_init}'...")
    hill_climbing(func_elegida, init_method=metodo_init, n_init=15, step_size=1, step_decay=0.7, min_step=1e-4, pause=0.1)
else:
    print("Opción no válida. Por favor, reinicia el programa e ingresa números válidos.")