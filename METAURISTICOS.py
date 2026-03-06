import numpy as np 
import matplotlib.pyplot as plt 
import time

# ==========================================
# 1. FUNCIONES OBJETIVO
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
# 2. ALGORITMOS DE INICIALIZACIÓN
# ==========================================
def init_aleatoria(n, bounds):
    lo, hi = bounds
    return np.random.uniform(lo, hi, (n, 2))

def init_lhs(n, bounds):
    lo, hi = bounds
    pts = np.empty((n, 2))
    for d in range(2):
        intervals = np.linspace(lo, hi, n + 1)
        rnds = np.random.uniform(intervals[:-1], intervals[1:])
        np.random.shuffle(rnds)
        pts[:, d] = rnds
    return pts

def init_maxdistance(n, bounds, num_candidates=50):
    lo, hi = bounds
    pts = [np.random.uniform(lo, hi, 2)]
    for _ in range(1, n):
        cands = np.random.uniform(lo, hi, (num_candidates, 2))
        dists = np.array([min(np.linalg.norm(c - p) for p in pts) for c in cands])
        pts.append(cands[np.argmax(dists)])
    return np.array(pts)

# ==========================================
# 3. ALGORITMOS DE BÚSQUEDA
# ==========================================

def hill_climbing(func_objetivo, init_method='aleatoria', n_init=15, bounds=(-3,3), step_size=0.3, step_decay=0.5, min_step=1e-3, pause=0.1, mode='max'):
    lo, hi = bounds
    
    # Inicialización
    if init_method == 'aleatoria': pts = init_aleatoria(n_init, bounds)
    elif init_method == 'lhs': pts = init_lhs(n_init, bounds)
    elif init_method == 'maxdistance': pts = init_maxdistance(n_init, bounds)
    else: pts = init_aleatoria(n_init, bounds)
        
    # Evaluar puntos iniciales
    best_pt = None
    best_val = -np.inf if mode == 'max' else np.inf
    
    for p in pts:
        val = func_objetivo(p[0], p[1])
        if (mode == 'max' and val > best_val) or (mode == 'min' and val < best_val):
            best_val = val
            best_pt = p
            
    x, y = best_pt
    fx = best_val
    dirs = [(dx, dy) for dx in (-1,0,1) for dy in (-1,0,1) if not (dx==0 and dy==0)]
    
    # Preparar gráficos
    X, Y = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    z = func_objetivo(X, Y)
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    cont = ax1.contourf(X, Y, z, levels=40, cmap='viridis')
    plt.colorbar(cont, ax=ax1)
    ax1.set_title(f'Hill Climbing ({mode.upper()}) - Init: {init_method.upper()}')
    ax1.scatter(pts[:, 0], pts[:, 1], c='black', marker='x', label='Muestra Inicial')
    ax1.legend()
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, z, cmap='viridis', alpha=0.8)
    
    path = [(x, y, fx)]
    
    # Ciclo
    while step_size > min_step:
        best_neighbor = None
        current_best = fx
        
        for dx, dy in dirs:
            xn = np.clip(x + dx*step_size, lo, hi)
            yn = np.clip(y + dy*step_size, lo, hi)
            fn = func_objetivo(xn, yn)
            
            if (mode == 'max' and fn > current_best) or (mode == 'min' and fn < current_best):
                current_best = fn
                best_neighbor = (xn, yn)
                
        if best_neighbor:
            x, y = best_neighbor
            fx = current_best
            path.append((x, y, fx))
            
            ax1.plot([p[0] for p in path], [p[1] for p in path], "w.-")
            ax2.plot([p[0] for p in path], [p[1] for p in path], [p[2] for p in path], "r.-")
            
            plt.draw()
            plt.pause(pause)
        else:
            step_size *= step_decay
            
    print(f'Mejor solución HC: ({x:.4f}, {y:.4f}), f(x,y) = {fx:.4f}')
    plt.ioff()
    plt.show()

def random_search(func_objetivo, iterations=100, bounds=(-3,3), pause=0.05, mode='max'):
    lo, hi = bounds
    
    # Variables para guardar el mejor encontrado
    best_pt = None
    best_val = -np.inf if mode == 'max' else np.inf
    path = [] # Guardaremos el historial de los mejores hallazgos
    
    # Preparar gráficos
    X, Y = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    z = func_objetivo(X, Y)
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    cont = ax1.contourf(X, Y, z, levels=40, cmap='viridis')
    plt.colorbar(cont, ax=ax1)
    ax1.set_title(f'Random Search ({mode.upper()})')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, z, cmap='viridis', alpha=0.8)
    
    # Ciclo de búsqueda aleatoria
    for i in range(iterations):
        x, y = np.random.uniform(lo, hi), np.random.uniform(lo, hi)
        val = func_objetivo(x, y)
        
        # Dibujar punto actual evaluado (suave)
        ax1.scatter(x, y, c='grey', s=10, alpha=0.5)
        
        # Comprobar si es el mejor hasta ahora
        is_better = (val > best_val) if mode == 'max' else (val < best_val)
        
        if is_better or best_pt is None:
            best_val = val
            best_pt = (x, y)
            path.append((x, y, best_val))
            
            # Trazar línea conectando los mejores hallazgos
            ax1.plot([p[0] for p in path], [p[1] for p in path], "w.-", linewidth=1.5)
            ax2.plot([p[0] for p in path], [p[1] for p in path], [p[2] for p in path], "r.-", linewidth=2)
            
            plt.draw()
            plt.pause(pause)
            
    print(f'Mejor solución Búsqueda Aleatoria: ({best_pt[0]:.4f}, {best_pt[1]:.4f}), f(x,y) = {best_val:.4f}')
    plt.ioff()
    plt.show()


# ==========================================
# 4. MENÚ PRINCIPAL E INTERFAZ
# ==========================================
print("=== CONFIGURACIÓN DE OPTIMIZACIÓN ===")

# 1. Función
print("\nSelecciona tu función objetivo:")
print("1. Peaks\n2. Ackley\n3. Rastrigin\n4. Sphere")
input_func = input("Ingresa el número: ")

# 2. Objetivo (Maximizar o Minimizar)
print("\n¿Qué deseas hacer con la función?")
print("1. Maximizar (Buscar el pico más alto)")
print("2. Minimizar (Buscar el valle más bajo - Ideal para Ackley/Rastrigin/Sphere)")
input_mode = 'min' if input("Ingresa el número: ") == '2' else 'max'

# 3. Algoritmo de Búsqueda
print("\nSelecciona el algoritmo de búsqueda:")
print("1. Hill Climbing (Búsqueda Local)")
print("2. Random Search (Búsqueda Aleatoria Pura)")
input_algo = input("Ingresa el número: ")

map_func = {'1': peaks, '2': Ackley, '3': Rastrigin, '4': Sphere}

if input_func in map_func:
    func_elegida = map_func[input_func]
    
    if input_algo == '1':
        # Menú extra solo si es Hill Climbing
        print("\nSelecciona el algoritmo de inicialización para Hill Climbing:")
        print("1. Aleatoria\n2. LHS\n3. MaxDistance")
        input_init = input("Ingresa el número: ")
        map_init = {'1': 'aleatoria', '2': 'lhs', '3': 'maxdistance'}
        metodo_init = map_init.get(input_init, 'aleatoria')
        
        print(f"\nIniciando Hill Climbing ({input_mode.upper()})...")
        hill_climbing(func_elegida, init_method=metodo_init, mode=input_mode, step_size=1, step_decay=0.7)
        
    elif input_algo == '2':
        print(f"\nIniciando Búsqueda Aleatoria ({input_mode.upper()})...")
        # 150 iteraciones para darle tiempo a encontrar algo bueno
        random_search(func_elegida, iterations=150, pause=0.05, mode=input_mode) 
        
    else:
        print("Algoritmo no válido.")
else:
    print("Función no válida.")
    