# ==========================================
# 3. ALGORITMOS DE BÚSQUEDA
# ========================================
import numpy as np
import matplotlib.pyplot as plt
from FUNCIONES_OBJETIVO import peaks, Ackley, Rastrigin, Sphere
from INICIALIZACION import init_aleatoria, init_lhs, init_maxdistance

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


def pso(func_objetivo, bounds=(-3.0, 3.0), num_iteraciones=60, swarm_size=30, pause=0.25, mode='max'):
    # Parámetros PSO adaptados de tu script original
    w_max, w_min = 0.9, 0.4
    c1, c2 = 1.5, 1.5
    v_max_ratio = 0.2
    trail_length = 6
    grid_res = 160
    
    lo, hi = bounds
    vmax = v_max_ratio * (hi - lo)
    rng = np.random.default_rng(42)

    # Inicialización del enjambre
    pos = rng.uniform(lo, hi, size=(swarm_size, 2))
    vel = rng.uniform(-vmax, vmax, size=(swarm_size, 2))

    pbest_pos = pos.copy()
    pbest_val = np.array([func_objetivo(x, y) for x, y in pbest_pos])

    # Mejor global (gbest) dinámico según maximización/minimización
    if mode == 'max':
        gbest_idx = np.argmax(pbest_val)
    else:
        gbest_idx = np.argmin(pbest_val)

    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    history = [[pos[i].copy()] for i in range(swarm_size)]

    # Preparar malla y figuras
    X = np.linspace(lo, hi, grid_res)
    Y = np.linspace(lo, hi, grid_res)
    Xg, Yg = np.meshgrid(X, Y)
    Zg = func_objetivo(Xg, Yg)

    plt.ion()
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    def dibujar(iteracion):
        ax1.clear()
        ax2.clear()

        # Redibujar contorno 2D
        cont = ax1.contourf(Xg, Yg, Zg, levels=40, cmap='viridis')
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(lo, hi)
        ax1.set_title(f'PSO ({mode.upper()}) - Iter {iteracion}/{num_iteraciones}')

        if trail_length > 0:
            for i in range(swarm_size):
                trail = np.array(history[i][-trail_length:])
                if trail.shape[0] > 1:
                    ax1.plot(trail[:, 0], trail[:, 1], 'w-', alpha=0.5)

        ax1.scatter(pos[:, 0], pos[:, 1], color='blue', s=30, edgecolor='black', label='Partículas')
        ax1.scatter(pbest_pos[:, 0], pbest_pos[:, 1], color='green', s=20, marker='o', label='pbest')
        ax1.scatter(gbest_pos[0], gbest_pos[1], color='red', s=50, marker='*', label='gbest')
        ax1.legend(loc='upper right', fontsize='small')

        # Redibujar Superficie 3D
        ax2.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.8, linewidth=0, antialiased=False)
        ax2.set_title('PSO - Superficie 3D')
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)

        fitness_curr = np.array([func_objetivo(x, y) for x, y in pos])
        ax2.scatter(pos[:, 0], pos[:, 1], fitness_curr, color='blue', s=30, edgecolor='black')
        pbest_vals = np.array([func_objetivo(x, y) for x, y in pbest_pos])
        ax2.scatter(pbest_pos[:, 0], pbest_pos[:, 1], pbest_vals, color='green', s=20)
        ax2.scatter(gbest_pos[0], gbest_pos[1], gbest_val, color='red', s=150, marker='*')
        ax2.view_init(elev=35, azim=-60)

        plt.draw()
        plt.pause(pause)

    # Ciclo principal de PSO
    for t in range(1, num_iteraciones + 1):
        w = w_max - (w_max - w_min) * (t - 1) / max(1, num_iteraciones - 1)

        for i in range(swarm_size):
            r1 = rng.random(2)
            r2 = rng.random(2)

            cognitive = c1 * r1 * (pbest_pos[i] - pos[i])
            social = c2 * r2 * (gbest_pos - pos[i])
            vel[i] = w * vel[i] + cognitive + social
            
            # Limitar velocidad y actualizar posición
            vel[i] = np.clip(vel[i], -vmax, vmax)
            pos[i] = pos[i] + vel[i]
            pos[i, 0] = np.clip(pos[i, 0], lo, hi)
            pos[i, 1] = np.clip(pos[i, 1], lo, hi)

            history[i].append(pos[i].copy())

        fitness = np.array([func_objetivo(x, y) for x, y in pos])

        # Actualizar mejores personales (pbest)
        if mode == 'max':
            improved_mask = fitness > pbest_val
        else:
            improved_mask = fitness < pbest_val

        if np.any(improved_mask):
            pbest_pos[improved_mask] = pos[improved_mask]
            pbest_val[improved_mask] = fitness[improved_mask]

        # Actualizar mejor global (gbest)
        if mode == 'max':
            idx = np.argmax(pbest_val)
            if pbest_val[idx] > gbest_val:
                gbest_val = pbest_val[idx]
                gbest_pos = pbest_pos[idx].copy()
        else:
            idx = np.argmin(pbest_val)
            if pbest_val[idx] < gbest_val:
                gbest_val = pbest_val[idx]
                gbest_pos = pbest_pos[idx].copy()

        dibujar(t)

    print(f'Mejor solución encontrada (PSO): ({gbest_pos[0]:.4f}, {gbest_pos[1]:.4f}), f(x,y) = {gbest_val:.4f}')
    plt.ioff()
    plt.show()
    
    
def abc_algorithm(func_objetivo, bounds=(-3.0, 3.0), num_iteraciones=80, num_fuentes=20, limit=15, pause=0.25, mode='max'):
    lo, hi = bounds
    rng = np.random.default_rng(123)
    
    # Inicialización de fuentes
    sources = rng.uniform(lo, hi, size=(num_fuentes, 2))
    values = np.array([func_objetivo(x, y) for x, y in sources])
    trials = np.zeros(num_fuentes, dtype=int)
    history = [ [sources[i].copy()] for i in range(num_fuentes) ]
    
    # Identificar el mejor inicial según el modo
    best_idx = np.argmax(values) if mode == 'max' else np.argmin(values)
    best_pos = sources[best_idx].copy()
    best_val = values[best_idx]
    
    # Preparar malla y figuras
    grid_res = 160
    X = np.linspace(lo, hi, grid_res)
    Y = np.linspace(lo, hi, grid_res)
    Xg, Yg = np.meshgrid(X, Y)
    Zg = func_objetivo(Xg, Yg)
    
    plt.ion()
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Ajuste para probabilidades dinámico
    def fitness_to_prob(vals):
        if mode == 'max':
            shifted = vals - vals.min() + 1e-9
        else:
            shifted = vals.max() - vals + 1e-9
        return shifted / shifted.sum()

    def dibujar(iteracion):
        ax1.clear()
        ax2.clear()
        
        # Redibujar contorno 2D
        cont = ax1.contourf(Xg, Yg, Zg, levels=40, cmap="viridis")
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(lo, hi)
        ax1.set_title(f"ABC ({mode.upper()}) - Iter {iteracion}/{num_iteraciones}")
        
        trail_length = 6
        if trail_length > 0:
            for i in range(num_fuentes):
                trail = np.array(history[i][-trail_length:])
                if trail.shape[0] > 1:
                    ax1.plot(trail[:,0], trail[:,1], 'w-', alpha=0.5)
                    
        ax1.scatter(sources[:,0], sources[:,1], c='blue', s=35, label='Fuentes')
        ax1.scatter(best_pos[0], best_pos[1], c='red', s=140, marker='*', label='Mejor global')
        ax1.legend(loc='upper right', fontsize='small')
        
        # Superfície 3D
        ax2.plot_surface(Xg, Yg, Zg, cmap="viridis", alpha=0.85, linewidth=0, antialiased=False)
        ax2.set_title("ABC - Superficie 3D")
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)
        
        current_vals = np.array([func_objetivo(x,y) for x,y in sources])
        ax2.scatter(sources[:,0], sources[:,1], current_vals, c='blue', s=35, edgecolor='black')
        ax2.scatter(best_pos[0], best_pos[1], best_val, c='red', s=180, marker='*')
        ax2.view_init(elev=35, azim=-60)
        
        plt.draw()
        plt.pause(pause)

    # Bucle principal ABC
    for t in range(1, num_iteraciones + 1):
        # 1) Employed bees
        for i in range(num_fuentes):
            k = rng.integers(0, num_fuentes-1)
            if k >= i: k += 1
            phi = rng.uniform(-1.0, 1.0, size=2)
            v = sources[i] + phi * (sources[i] - sources[k])
            v = np.clip(v, lo, hi)
            fv = func_objetivo(v[0], v[1])
            
            is_better = (fv > values[i]) if mode == 'max' else (fv < values[i])
            if is_better:
                sources[i] = v
                values[i] = fv
                trials[i] = 0
            else:
                trials[i] += 1
            history[i].append(sources[i].copy())

        # 2) Onlooker bees
        probs = fitness_to_prob(values)
        count = 0
        while count < num_fuentes:
            sel = rng.choice(np.arange(num_fuentes), p=probs)
            k = rng.integers(0, num_fuentes-1)
            if k >= sel: k += 1
            phi = rng.uniform(-1.0, 1.0, size=2)
            v = sources[sel] + phi * (sources[sel] - sources[k])
            v = np.clip(v, lo, hi)
            fv = func_objetivo(v[0], v[1])
            
            is_better = (fv > values[sel]) if mode == 'max' else (fv < values[sel])
            if is_better:
                sources[sel] = v
                values[sel] = fv
                trials[sel] = 0
            else:
                trials[sel] += 1
                
            history[sel].append(sources[sel].copy())
            count += 1

        # 3) Scout bees
        for i in range(num_fuentes):
            if trials[i] >= limit:
                sources[i] = rng.uniform(lo, hi, size=2)
                values[i] = func_objetivo(sources[i][0], sources[i][1])
                trials[i] = 0
                history[i].append(sources[i].copy())

        # Actualizar mejor global final de la iteración
        idx = np.argmax(values) if mode == 'max' else np.argmin(values)
        is_new_best = (values[idx] > best_val) if mode == 'max' else (values[idx] < best_val)
        if is_new_best:
            best_val = values[idx]
            best_pos = sources[idx].copy()

        dibujar(t)

    print(f"\nResultado final ABC -> mejor = ({best_pos[0]:.6f}, {best_pos[1]:.6f}), f(x,y) = {best_val:.6f}")
    plt.ioff()
    plt.show()
    
def firefly_algorithm(func_objetivo, bounds=(-3.0, 3.0), num_iteraciones=80, num_fireflies=30, pause=0.18, mode='max'):
    lo, hi = bounds
    dim = 2
    beta0 = 1.0
    gamma = 1.0
    alpha_curr = 0.2
    alpha_decay = 0.98
    seed = 2025
    grid_res = 160
    trail_length = 8
    scale_noise = 1.0 * (hi - lo)
    
    rng = np.random.default_rng(seed)
    
    # Inicialización de posiciones de las luciérnagas
    x = rng.uniform(lo, hi, size=(num_fireflies, dim))
    vals = np.array([func_objetivo(p[0], p[1]) for p in x])
    
    history = [ [x[i].copy()] for i in range(num_fireflies) ]
    
    # Evaluar mejor global inicial dependiendo del modo (max/min)
    best_idx = np.argmax(vals) if mode == 'max' else np.argmin(vals)
    best_pos = x[best_idx].copy()
    best_val = vals[best_idx]
    
    # Preparar malla para la visualización
    Xg = np.linspace(lo, hi, grid_res)
    Yg = np.linspace(lo, hi, grid_res)
    Xmg, Ymg = np.meshgrid(Xg, Yg)
    Zmg = func_objetivo(Xmg, Ymg)
    
    plt.ion()
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    
    def dibujar(iteracion, alphas_vis=0.9):
        ax1.clear()
        ax2.clear()
        
        # Redibujar contorno 2D
        cont = ax1.contourf(Xmg, Ymg, Zmg, levels=40, cmap='viridis')
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(lo, hi)
        ax1.set_title(f'Firefly Algorithm ({mode.upper()}) - Iter {iteracion}/{num_iteraciones}')
        
        # Dibujar rastro
        if trail_length > 0:
            for i in range(num_fireflies):
                trail = np.array(history[i][-trail_length:])
                if len(trail) > 1:
                    alphas = np.linspace(0.1, 0.8, len(trail))
                    L = trail.shape[0]
                    for k in range(L-1):
                        a = alphas_vis * (k+1) / L
                        ax1.plot(trail[k:k+2, 0], trail[k:k+2, 1], linewidth=2, alpha=a, color='cyan')
                        
        ax1.scatter(x[:, 0], x[:, 1], c='blue', label='Fireflies', s=36)
        ax1.scatter(best_pos[0], best_pos[1], c='red', s=140, marker='*', label='Best Global')
        ax1.legend(loc='upper right', fontsize='small')
        
        # Superficie 3D
        ax2.plot_surface(Xmg, Ymg, Zmg, cmap='viridis', alpha=0.85, linewidth=0, antialiased=False)
        ax2.set_title('Firefly Algorithm - Superficie 3D')
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)
        ax2.scatter(x[:, 0], x[:, 1], vals, c='blue', s=36, edgecolor='black')
        ax2.scatter(best_pos[0], best_pos[1], best_val, c='red', s=180, marker='*')
        ax2.view_init(elev=35, azim=-60)
        
        plt.draw()
        plt.pause(pause)

    # Ciclo principal FA
    for t in range(1, num_iteraciones + 1):
        # Ordenar por brillo según el modo (descendente para max, ascendente para min)
        if mode == 'max':
            order = np.argsort(-vals)
        else:
            order = np.argsort(vals)
            
        for idx_i in range(num_fireflies):
            i = order[idx_i]
            xi = x[i].copy()
            
            # Recorrer luciérnagas más brillantes
            for idx_j in range(idx_i):
                j = order[idx_j]
                xj = x[j]
                
                r = np.linalg.norm(xi - xj)
                beta = beta0 * np.exp(-gamma * r**2)
                noise = rng.normal(0, 1, size=dim)
                xi = xi + beta * (xj - xi) + alpha_curr * noise * (scale_noise/2)
                xi = np.clip(xi, lo, hi)
                
            x[i] = xi
            vals[i] = func_objetivo(xi[0], xi[1])
            history[i].append(xi.copy())
            
        # Decaimiento de aleatoriedad
        alpha_curr *= alpha_decay
        
        # Actualizar mejor global
        if mode == 'max':
            idx = np.argmax(vals)
            is_new_best = (vals[idx] > best_val)
        else:
            idx = np.argmin(vals)
            is_new_best = (vals[idx] < best_val)
            
        if is_new_best:
            best_val = vals[idx]
            best_pos = x[idx].copy()
                
        dibujar(t)
        
    print(f"\nResultado final FA -> mejor = ({best_pos[0]:.6f}, {best_pos[1]:.6f}), f(x,y) = {best_val:.6f}")
    plt.ioff()
    plt.show()
    
    
# ----------------------------------
# ACO
# ----------------------------------

def aco_algorithm(func_objetivo, bounds=(-3.0, 3.0), num_iteraciones=70, num_ants=30, archive_size=25, q=0.5, xi=0.85, pause=0.25, mode='max'):
    lo, hi = bounds
    dim = 2
    rng = np.random.default_rng(42)
    grid_res = 160
    trail_length = 6

    # 1. Crear archivo de feromonas
    archive_X = rng.uniform(lo, hi, size=(archive_size, dim))
    archive_vals = np.array([func_objetivo(x, y) for x, y in archive_X])

    # Ordenar por calidad (según el modo)
    if mode == 'max':
        order = np.argsort(-archive_vals)
    else:
        order = np.argsort(archive_vals)
        
    archive_X = archive_X[order]
    archive_vals = archive_vals[order]

    history = [ [archive_X[i].copy()] for i in range(archive_size) ]

    best_pos = archive_X[0].copy()
    best_val = archive_vals[0]

    # Preparar malla para visualización
    Xg = np.linspace(lo, hi, grid_res)
    Yg = np.linspace(lo, hi, grid_res)
    Xmg, Ymg = np.meshgrid(Xg, Yg)
    Zmg = func_objetivo(Xmg, Ymg)

    plt.ion()
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    def dibujar(iteracion, sols, vals, history):
        ax1.clear()
        ax2.clear()
        
        # Contorno 2D
        cont0 = ax1.contourf(Xmg, Ymg, Zmg, levels=40, cmap='viridis')
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(lo, hi)
        ax1.set_title(f'ACOR ({mode.upper()}) - Iter {iteracion+1}/{num_iteraciones}')
        
        if trail_length > 0:
            for i in range(len(history)):
                trail = np.array(history[i][-trail_length:])
                L = trail.shape[0]
                if L > 1:
                    for j in range(L-1):
                        a = 0.15 + 0.85 * (j + 1) / L
                        ax1.plot(trail[j:j+2, 0], trail[j:j+2, 1], linewidth=1, alpha=a, color='cyan')
                        
        ax1.scatter(sols[:, 0], sols[:, 1], c='blue', s=36, label='Muestras')
        ax1.scatter(best_pos[0], best_pos[1], c='red', s=140, marker='*', label='Mejor global')
        ax1.legend(loc='upper right', fontsize='small')
        
        # 3D
        ax2.plot_surface(Xmg, Ymg, Zmg, cmap='viridis', alpha=0.85, linewidth=0, antialiased=False)
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)
        ax2.scatter(sols[:, 0], sols[:, 1], vals, c='blue', s=36, edgecolor='black')
        ax2.scatter(best_pos[0], best_pos[1], best_val, c='red', s=180, marker='*')
        ax2.view_init(elev=35, azim=-60)
        
        plt.draw()
        plt.pause(pause)

    # Funciones auxiliares para ACO
    def compute_weights(m, q_val):
        k = np.arange(1, m+1)
        denom = q_val * m * np.sqrt(2 * np.pi)
        numer = np.exp(-(k - 1)**2 / (2 * (q_val * m)**2))
        w = numer / denom
        return w / np.sum(w)

    def compute_sigma(archive):
        m_arc, d = archive.shape
        sigmas = np.zeros_like(archive)
        for i in range(m_arc):
            diffs = np.abs(archive[i] - archive)  
            denom = max(1, m_arc-1)
            sigmas[i] = xi * diffs.sum(axis=0) / denom
            sigmas[i] = np.maximum(sigmas[i], 1e-6)
        return sigmas

    # Bucle principal ACO
    for it in range(num_iteraciones):
        m = archive_size
        weights = compute_weights(m, q)
        sigmas = compute_sigma(archive_X)
         
        new_sols = np.zeros((num_ants, dim))
        new_vals = np.zeros(num_ants)
        
        cum_weights = np.cumsum(weights)
        for a in range(num_ants):
            r = rng.random()
            k = np.searchsorted(cum_weights, r, side='right')
            if k >= m: k = m - 1
            
            sample = rng.normal(loc=archive_X[k], scale=sigmas[k]) 
            sample = np.clip(sample, lo, hi)
            new_sols[a] = sample
            new_vals[a] = func_objetivo(sample[0], sample[1])
        
        # Actualizar archivo
        combined_X = np.vstack((archive_X, new_sols))
        combined_vals = np.concatenate((archive_vals, new_vals))
        
        if mode == 'max':
            order = np.argsort(-combined_vals)
        else:
            order = np.argsort(combined_vals)
            
        combined_X = combined_X[order][:archive_size]
        combined_vals = combined_vals[order][:archive_size]
        
        for idx in range(archive_size):
            history[idx].append(combined_X[idx].copy())
        
        # Actualizar mejor global (ya está en el índice 0 tras el ordenamiento)
        best_pos = combined_X[0].copy()
        best_val = combined_vals[0]
            
        dibujar(it, np.vstack((archive_X, new_sols)), np.concatenate((archive_vals, new_vals)), history)
        
        archive_X = combined_X.copy()
        archive_vals = combined_vals.copy()
        
    print(f'\nResultado final ACO ---> best = ({best_pos[0]:.6f}, {best_pos[1]:.6f}), f(x,y) = {best_val:.6f}')
    plt.ioff()
    plt.show()
    
def simulated_annealing(funcion_objetivo, limites, mode='min', t_inicial=100.0, t_minima=1e-4, alpha=0.99, step_size=0.5, iteraciones=60, animar=True, pausa=0.01):
    num_variables = len(limites)
    limite_inf = np.array([l[0] for l in limites])
    limite_sup = np.array([l[1] for l in limites])

    x_curr = np.random.uniform(limite_inf, limite_sup, num_variables)
    f_curr = funcion_objetivo(*x_curr)

    x_best = np.copy(x_curr)
    f_best = f_curr

    T = t_inicial
    iter_count = 0
    history = [(x_curr[0], x_curr[1])] if num_variables >= 2 else []

    # --- CONFIGURACIÓN DE LA MALLA VISUAL (SOLO 2D) ---
    hacer_animacion = animar and num_variables == 2
    if hacer_animacion:
        grid_res = 80 # Resolución reducida ligeramente para mayor fluidez
        X = np.linspace(limite_inf[0], limite_sup[0], grid_res)
        Y = np.linspace(limite_inf[1], limite_sup[1], grid_res)
        Xg, Yg = np.meshgrid(X, Y)
        Zg = np.zeros_like(Xg)
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):
                Zg[i,j] = funcion_objetivo(Xg[i,j], Yg[i,j])
                
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')

    # --- BUCLE PRINCIPAL ---
    while T > t_minima and iter_count < iteraciones:
        iter_count += 1
        x_new = x_curr + np.random.normal(0, step_size, num_variables)
        x_new = np.clip(x_new, limite_inf, limite_sup)
        
        f_new = funcion_objetivo(*x_new)
        delta = f_new - f_curr

        if mode == 'min':
            es_mejora = delta < 0
            prob_aceptacion = np.exp(-delta / T)
        else: 
            es_mejora = delta > 0
            prob_aceptacion = np.exp(delta / T)

        if es_mejora:
            x_curr = x_new
            f_curr = f_new
            if (mode == 'min' and f_curr < f_best) or (mode == 'max' and f_curr > f_best):
                x_best = np.copy(x_curr)
                f_best = f_curr
        else:
            if np.random.rand() < prob_aceptacion:
                x_curr = x_new
                f_curr = f_new

        T *= alpha
        
        # --- LÓGICA DE DIBUJO EN VIVO ---
        if hacer_animacion:
            history.append((x_curr[0], x_curr[1]))
            ax1.clear()
            ax2.clear()
            
            # 2D Contour
            ax1.contourf(Xg, Yg, Zg, levels=40, cmap='viridis')
            ax1.set_xlim(limite_inf[0], limite_sup[0])
            ax1.set_ylim(limite_inf[1], limite_sup[1])
            ax1.set_title(f"SA - Iter: {iter_count} | T: {T:.4f} | F: {f_curr:.2f}")
            
            # Rastro
            trail_length = 15
            trail = np.array(history[-trail_length:])
            if trail.shape[0] > 1:
                ax1.plot(trail[:, 0], trail[:, 1], color='cyan', alpha=0.8, linewidth=1.5)
                
            ax1.scatter(x_curr[0], x_curr[1], color='blue', s=60, label='Actual')
            ax1.scatter(x_best[0], x_best[1], color='red', s=80, marker='*', label='Mejor Global')
            ax1.legend(loc='upper right', fontsize='small')
            
            # 3D Surface
            ax2.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.7, linewidth=0, antialiased=False)
            ax2.scatter(x_curr[0], x_curr[1], f_curr, color='blue', s=60)
            ax2.scatter(x_best[0], x_best[1], f_best, color='red', s=100, marker='*')
            ax2.set_xlim(limite_inf[0], limite_sup[0])
            ax2.set_ylim(limite_inf[1], limite_sup[1])
            ax2.view_init(elev=35, azim=60)
            
            plt.draw()
            plt.pause(pausa)

    if hacer_animacion:
        plt.ioff()
        plt.show()

    return x_best, f_best

def harmony_search(funcion_objetivo, limites, mode='min', hms=20, hmcr=0.95, par=0.3, bw=0.1, iteraciones=80, animar=True, pausa=0.25, trail_length=8):
    num_variables = len(limites)
    limite_inf = np.array([l[0] for l in limites])
    limite_sup = np.array([l[1] for l in limites])

    # Inicializar Harmony Memory (HM)
    hm = np.random.uniform(limite_inf, limite_sup, (hms, num_variables))
    hm_fitness = np.array([funcion_objetivo(*ind) for ind in hm])
    
    historial_fitness = []

    # --- CONFIGURACIÓN DE LA MALLA VISUAL (SOLO 2D) ---
    hacer_animacion = animar and num_variables == 2
    if hacer_animacion:
        grid_res = 80 
        X = np.linspace(limite_inf[0], limite_sup[0], grid_res)
        Y = np.linspace(limite_inf[1], limite_sup[1], grid_res)
        Xg, Yg = np.meshgrid(X, Y)
        Zg = np.zeros_like(Xg)
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):
                Zg[i,j] = funcion_objetivo(Xg[i,j], Yg[i,j])
                
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Crear colorbar UNA sola vez
        cont0 = ax1.contourf(Xg, Yg, Zg, levels=40, cmap='viridis')
        cbar = plt.colorbar(cont0, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Valor de la función')
        
        # Historial para rastro (lista de listas para cada armonía)
        history = [[hm[i].copy()] for i in range(hms)]

    # --- BUCLE PRINCIPAL ---
    for iteracion in range(1, iteraciones + 1):
        nueva_armonia = np.zeros(num_variables)

        for i in range(num_variables):
            if np.random.rand() < hmcr:
                indice_aleatorio = np.random.randint(0, hms)
                nueva_armonia[i] = hm[indice_aleatorio, i]

                if np.random.rand() < par:
                    # Ajuste proporcional al rango de la variable
                    delta = bw * (limite_sup[i] - limite_inf[i])
                    ajuste = np.random.uniform(-delta, delta)
                    nueva_armonia[i] += ajuste
                    nueva_armonia[i] = np.clip(nueva_armonia[i], limite_inf[i], limite_sup[i])
            else:
                nueva_armonia[i] = np.random.uniform(limite_inf[i], limite_sup[i])

        nuevo_fitness = funcion_objetivo(*nueva_armonia)

        # Lógica de reemplazo dependiendo del modo
        if mode == 'min':
            peor_indice = np.argmax(hm_fitness)
            es_mejor = nuevo_fitness < hm_fitness[peor_indice]
        else: 
            peor_indice = np.argmin(hm_fitness)
            es_mejor = nuevo_fitness > hm_fitness[peor_indice]

        if es_mejor:
            hm[peor_indice] = nueva_armonia
            hm_fitness[peor_indice] = nuevo_fitness
            if hacer_animacion:
                # Reemplazar historial de la peor armonía con la nueva
                history[peor_indice].append(nueva_armonia.copy())
        else:
            if hacer_animacion:
                # Mostrar actividad visual en una armonía aleatoria
                r_idx = np.random.randint(0, hms)
                history[r_idx].append(hm[r_idx].copy())
                
        # Guardar el mejor valor para la gráfica de convergencia final
        if mode == 'min':
            mejor_actual = np.min(hm_fitness)
            mejor_indice = np.argmin(hm_fitness)
        else:
            mejor_actual = np.max(hm_fitness)
            mejor_indice = np.argmax(hm_fitness)
            
        historial_fitness.append(mejor_actual)

        # --- LÓGICA DE DIBUJO EN VIVO ---
        if hacer_animacion:
            mejor_pos = hm[mejor_indice]
            mejor_val = hm_fitness[mejor_indice]
            
            ax1.clear()
            ax2.clear()
            
            # Contorno 2D
            ax1.contourf(Xg, Yg, Zg, levels=40, cmap='viridis')
            ax1.set_xlim(limite_inf[0], limite_sup[0])
            ax1.set_ylim(limite_inf[1], limite_sup[1])
            ax1.set_title(f"Harmony Search ({mode.upper()}) - Iter: {iteracion}/{iteraciones}")
            
            # Dibujar rastros con degradado alpha
            if trail_length > 0:
                for idx_hist in range(len(history)):
                    trail = np.array(history[idx_hist][-trail_length:])
                    L = trail.shape[0]
                    if L > 1:
                        for k in range(L-1):
                            a = 0.15 + 0.85 * (k+1)/L
                            ax1.plot(trail[k:k+2, 0], trail[k:k+2, 1], linewidth=1.2, alpha=a, color='cyan')

            # Dibujar armonías y la mejor
            ax1.scatter(hm[:, 0], hm[:, 1], color='blue', s=36, label='HM (armonías)')
            ax1.scatter(mejor_pos[0], mejor_pos[1], color='red', s=140, marker='*', label='Mejor Global')
            ax1.legend(loc='upper right', fontsize='small')
            
            # Superficie 3D
            ax2.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.86, linewidth=0, antialiased=False)
            ax2.set_title("Superficie 3D")
            ax2.set_xlim(limite_inf[0], limite_sup[0])
            ax2.set_ylim(limite_inf[1], limite_sup[1])
            ax2.scatter(hm[:, 0], hm[:, 1], hm_fitness, color='blue', s=36)
            ax2.scatter(mejor_pos[0], mejor_pos[1], mejor_val, color='red', s=180, marker='*')
            ax2.view_init(elev=35, azim=-60)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(pausa)

    if hacer_animacion:
        plt.ioff()
        plt.close() # Cierra la animación para dar paso a la tabla final

    # Extraer el mejor resultado final
    if mode == 'min':
        mejor_indice = np.argmin(hm_fitness)
    else:
        mejor_indice = np.argmax(hm_fitness)
        
    mejor_solucion = hm[mejor_indice]
    mejor_fitness = hm_fitness[mejor_indice]

    # --- GRÁFICA DE CONVERGENCIA FINAL ---
    plt.figure(figsize=(10, 6))
    plt.plot(historial_fitness, label=f'Mejor Global HS ({mode.upper()})', color='green', linewidth=2)
    plt.title('Curva de Convergencia - Harmony Search')
    plt.xlabel('Iteración')
    plt.ylabel('Valor de la Función Objetivo (Fitness)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
    return mejor_solucion, mejor_fitness