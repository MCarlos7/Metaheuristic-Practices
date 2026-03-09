# ==========================================
# 3. ALGORITMOS DE BÚSQUEDA
# ========================================
import numpy as np
import matplotlib.pyplot as plt
from MAIN.FUNCIONES_OBJETIVO import peaks, Ackley, Rastrigin, Sphere
from MAIN.INICIALIZACION import init_aleatoria, init_lhs, init_maxdistance

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

def genetic_algorithm(func_objetivo, bounds=(-3, 3), num_generaciones=30, tam_poblacion=20, prob_cruce=0.8, prob_mutacion=0.2, pause=0.5, mode='max'):
    lo, hi = bounds
    
    # Población Inicial 
    poblacion = np.random.uniform(lo, hi, size=(tam_poblacion, 2))
    
    # Preparar Visualización 
    X = np.linspace(lo, hi, 200) 
    Y = np.linspace(lo, hi, 200)
    X, Y = np.meshgrid(X, Y)
    Z = func_objetivo(X, Y)
    
    plt.ion()
    # CORRECCIÓN AQUÍ: Inicializar explicitamente ax2 como 3D
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    
    mejor_sol = None
    mejor_val = -np.inf if mode == 'max' else np.inf
    
    for gen in range(num_generaciones):
        # Evaluar Población 
        fitness = np.array([func_objetivo(ind[0], ind[1]) for ind in poblacion])
        
        # Seleccionar Mejor Solución
        if mode == 'max':
            idx_best = np.argmax(fitness)
            if fitness[idx_best] > mejor_val:
                mejor_val = fitness[idx_best]
                mejor_sol = poblacion[idx_best]
        else:
            idx_best = np.argmin(fitness)
            if fitness[idx_best] < mejor_val:
                mejor_val = fitness[idx_best]
                mejor_sol = poblacion[idx_best]
            
        # Visualizar Población
        ax1.clear()
        cont = ax1.contourf(X, Y, Z, levels=40, cmap='viridis')
        ax1.set_title(f'Algoritmo Genético ({mode.upper()}) - 2D - Generación {gen+1}')
        ax1.scatter(poblacion[:,0], poblacion[:,1], color='blue', s=30)
        ax1.scatter(mejor_sol[0], mejor_sol[1], color='red', s=80, marker='*')
        
        ax2.clear()
        ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax2.set_title(f'Superficie 3D - Generación {gen+1}')
        ax2.scatter(poblacion[:,0], poblacion[:,1], fitness, color='blue', s=30)
        ax2.scatter(mejor_sol[0], mejor_sol[1], mejor_val, color='red', s=80, marker='*')
        
        plt.draw()
        plt.pause(pause)
        
        # Selección por Ruleta Proporcional
        if mode == 'max':
            fitness_pos = fitness - fitness.min() + 1e-6
        else:
            fitness_pos = fitness.max() - fitness + 1e-6
            
        probs = fitness_pos / fitness_pos.sum()
        padres_idx = np.random.choice(len(poblacion), size=len(poblacion), p=probs)
        padres = poblacion[padres_idx]
        
        # Cruce de 1 Punto
        descendencia = []   
        for i in range(0, len(padres), 2):
            p1, p2 = padres[i], padres[(i+1) % len(padres)]
            if np.random.rand() < prob_cruce:
                alpha = np.random.rand()
                h1 = alpha * p1 + (1 - alpha) * p2
                h2 = alpha * p2 + (1 - alpha) * p1
                descendencia.extend([h1, h2])
            else:
                descendencia.extend([p1, p2])
                
        poblacion = np.array(descendencia)
        
        # Mutación
        for ind in poblacion:
            if np.random.rand() < prob_mutacion:
                ind += np.random.uniform(-0.3, 0.3, size=2)
                ind[0] = np.clip(ind[0], lo, hi)
                ind[1] = np.clip(ind[1], lo, hi)
                
    print(f'Mejor solución encontrada (Genético): ({mejor_sol[0]:.4f}, {mejor_sol[1]:.4f}), f(x,y): {mejor_val:.4f}')
    plt.ioff()
    plt.show()
