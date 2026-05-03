from INICIALIZACION import _get_init_points, init_aleatoria, init_lhs, init_maxdistance
import numpy as np
import itertools

# ==========================================
# 3. ALGORITMOS DE BÚSQUEDA (N-Dimensiones, Optimizados)
# ==========================================

def hill_climbing(func_objetivo, init_method='aleatoria', dim=2, n_init=15, bounds=(-3,3), step_size=0.3, step_decay=0.5, min_step=1e-3, mode='max'):
    lo, hi = bounds
    pts = _get_init_points(init_method, n_init, bounds, dim)
        
    vals = np.array([func_objetivo(*p) for p in pts])
    best_idx = np.argmax(vals) if mode == 'max' else np.argmin(vals)
    x = pts[best_idx].copy()
    fx = vals[best_idx]
    
    dirs = []
    for i in range(dim):
        dir_pos = np.zeros(dim); dir_pos[i] = 1
        dir_neg = np.zeros(dim); dir_neg[i] = -1
        dirs.extend([dir_pos, dir_neg])
    dirs = np.array(dirs)
    
    while step_size > min_step:
        vecinos = np.clip(x + dirs * step_size, lo, hi)
        f_vecinos = np.array([func_objetivo(*v) for v in vecinos])
        
        idx_mejor_vecino = np.argmax(f_vecinos) if mode == 'max' else np.argmin(f_vecinos)
        mejor_f_vecino = f_vecinos[idx_mejor_vecino]
        
        if (mode == 'max' and mejor_f_vecino > fx) or (mode == 'min' and mejor_f_vecino < fx):
            x = vecinos[idx_mejor_vecino]
            fx = mejor_f_vecino
        else:
            step_size *= step_decay
            
    return x, fx


def random_search(func_objetivo, init_method='aleatoria', dim=2, iterations=100, bounds=(-3,3), mode='max'):
    pts = _get_init_points(init_method, iterations, bounds, dim)
    vals = np.array([func_objetivo(*p) for p in pts])
    
    best_idx = np.argmax(vals) if mode == 'max' else np.argmin(vals)
    return pts[best_idx], vals[best_idx]


def genetic_algorithm(func_objetivo, init_method='aleatoria', dim=2, bounds=(-3, 3), num_generaciones=30, tam_poblacion=20, prob_cruce=0.8, prob_mutacion=0.2, mode='max'):
    lo, hi = bounds
    poblacion = _get_init_points(init_method, tam_poblacion, bounds, dim)
    
    mejor_sol = None
    mejor_val = -np.inf if mode == 'max' else np.inf
    
    for gen in range(num_generaciones):
        fitness = np.array([func_objetivo(*ind) for ind in poblacion])
        
        idx_best = np.argmax(fitness) if mode == 'max' else np.argmin(fitness)
        if (mode == 'max' and fitness[idx_best] > mejor_val) or (mode == 'min' and fitness[idx_best] < mejor_val):
            mejor_val = fitness[idx_best]
            mejor_sol = poblacion[idx_best].copy()
        
        if mode == 'max':
            fitness_pos = fitness - fitness.min() + 1e-6
        else:
            fitness_pos = fitness.max() - fitness + 1e-6
            
        probs = fitness_pos / fitness_pos.sum()
        padres_idx = np.random.choice(tam_poblacion, size=tam_poblacion, p=probs)
        padres = poblacion[padres_idx]
        
        descendencia = np.empty_like(poblacion)
        for i in range(0, tam_poblacion, 2):
            p1 = padres[i]
            p2 = padres[(i+1) % tam_poblacion]
            
            if np.random.rand() < prob_cruce:
                alpha = np.random.rand(dim)
                descendencia[i] = alpha * p1 + (1 - alpha) * p2
                if i + 1 < tam_poblacion:
                    descendencia[i+1] = alpha * p2 + (1 - alpha) * p1
            else:
                descendencia[i] = p1
                if i + 1 < tam_poblacion:
                    descendencia[i+1] = p2
                
        poblacion = descendencia
        
        mask_mutacion = np.random.rand(tam_poblacion) < prob_mutacion
        poblacion[mask_mutacion] += np.random.uniform(-0.3, 0.3, size=(np.sum(mask_mutacion), dim))
        np.clip(poblacion, lo, hi, out=poblacion)
                
    return mejor_sol, mejor_val


def pso(func_objetivo, init_method='aleatoria', dim=2, bounds=(-3.0, 3.0), num_iteraciones=60, swarm_size=30, mode='max'):
    w_max, w_min = 0.9, 0.4
    c1, c2 = 1.5, 1.5
    vmax = 0.2 * (bounds[1] - bounds[0])
    lo, hi = bounds
    rng = np.random.default_rng(42)

    pos = _get_init_points(init_method, swarm_size, bounds, dim)
    vel = rng.uniform(-vmax, vmax, size=(swarm_size, dim))

    pbest_pos = pos.copy()
    pbest_val = np.array([func_objetivo(*p) for p in pbest_pos])

    gbest_idx = np.argmax(pbest_val) if mode == 'max' else np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    for t in range(num_iteraciones):
        w = w_max - (w_max - w_min) * (t / max(1, num_iteraciones - 1))
        
        r1 = rng.random((swarm_size, dim))
        r2 = rng.random((swarm_size, dim))

        cognitive = c1 * r1 * (pbest_pos - pos)
        social = c2 * r2 * (gbest_pos - pos)
        
        vel = w * vel + cognitive + social
        np.clip(vel, -vmax, vmax, out=vel)
        
        pos += vel
        np.clip(pos, lo, hi, out=pos)

        fitness = np.array([func_objetivo(*p) for p in pos])

        improved_mask = (fitness > pbest_val) if mode == 'max' else (fitness < pbest_val)
        pbest_pos[improved_mask] = pos[improved_mask]
        pbest_val[improved_mask] = fitness[improved_mask]

        idx = np.argmax(pbest_val) if mode == 'max' else np.argmin(pbest_val)
        if (mode == 'max' and pbest_val[idx] > gbest_val) or (mode == 'min' and pbest_val[idx] < gbest_val):
            gbest_val = pbest_val[idx]
            gbest_pos = pbest_pos[idx].copy()

    return gbest_pos, gbest_val

#NUEVOOOOOOOOOOOOOOOOOOOOOOOOO <-------------------------
def differential_evolution(func_objetivo, init_method='aleatoria', dim=2, bounds=(-3.0, 3.0), num_iteraciones=60, pop_size=30, F=0.8, CR=0.9, mode='max'):
    lo, hi = bounds
    pop = _get_init_points(init_method, pop_size, bounds, dim)
    fitness = np.array([func_objetivo(*ind) for ind in pop])

    best_idx = np.argmax(fitness) if mode == 'max' else np.argmin(fitness)
    best_pos = pop[best_idx].copy()
    best_val = fitness[best_idx]

    for _ in range(num_iteraciones):
        for i in range(pop_size):
            # Selección de 3 agentes aleatorios distintos a 'i'
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

            # Mutación
            mutant = np.clip(a + F * (b - c), lo, hi)

            # Cruce Binomial
            cross_points = np.random.rand(dim) <= CR
            if not np.any(cross_points): 
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, pop[i])
            f_trial = func_objetivo(*trial)

            # Selección In-Place
            if (mode == 'max' and f_trial > fitness[i]) or (mode == 'min' and f_trial < fitness[i]):
                pop[i] = trial
                fitness[i] = f_trial

                if (mode == 'max' and f_trial > best_val) or (mode == 'min' and f_trial < best_val):
                    best_val = f_trial
                    best_pos = trial.copy()

    return best_pos, best_val


def abc_algorithm(func_objetivo, init_method='aleatoria', dim=2, bounds=(-3.0, 3.0), num_iteraciones=80, num_fuentes=20, limit=15, mode='max'):
    lo, hi = bounds
    rng = np.random.default_rng(123)
    
    sources = _get_init_points(init_method, num_fuentes, bounds, dim)
    values = np.array([func_objetivo(*p) for p in sources])
    trials = np.zeros(num_fuentes, dtype=int)
    
    best_idx = np.argmax(values) if mode == 'max' else np.argmin(values)
    best_pos = sources[best_idx].copy()
    best_val = values[best_idx]

    for t in range(num_iteraciones):
        # Empleadas
        for i in range(num_fuentes):
            k = rng.integers(0, num_fuentes)
            while k == i: k = rng.integers(0, num_fuentes)
            
            phi = rng.uniform(-1.0, 1.0, size=dim)
            v = np.clip(sources[i] + phi * (sources[i] - sources[k]), lo, hi)
            fv = func_objetivo(*v)
            
            if (mode == 'max' and fv > values[i]) or (mode == 'min' and fv < values[i]):
                sources[i] = v
                values[i] = fv
                trials[i] = 0
            else:
                trials[i] += 1

        # Observadoras
        shifted = (values - values.min() + 1e-9) if mode == 'max' else (values.max() - values + 1e-9)
        probs = shifted / shifted.sum()
        
        for _ in range(num_fuentes):
            sel = rng.choice(num_fuentes, p=probs)
            k = rng.integers(0, num_fuentes)
            while k == sel: k = rng.integers(0, num_fuentes)
            
            phi = rng.uniform(-1.0, 1.0, size=dim)
            v = np.clip(sources[sel] + phi * (sources[sel] - sources[k]), lo, hi)
            fv = func_objetivo(*v)
            
            if (mode == 'max' and fv > values[sel]) or (mode == 'min' and fv < values[sel]):
                sources[sel] = v
                values[sel] = fv
                trials[sel] = 0
            else:
                trials[sel] += 1

        # Exploradoras
        scout_mask = trials >= limit
        if np.any(scout_mask):
            num_scouts = np.sum(scout_mask)
            sources[scout_mask] = _get_init_points(init_method, num_scouts, bounds, dim)
            values[scout_mask] = [func_objetivo(*p) for p in sources[scout_mask]]
            trials[scout_mask] = 0

        idx = np.argmax(values) if mode == 'max' else np.argmin(values)
        if (mode == 'max' and values[idx] > best_val) or (mode == 'min' and values[idx] < best_val):
            best_val = values[idx]
            best_pos = sources[idx].copy()

    return best_pos, best_val


def firefly_algorithm(func_objetivo, init_method='aleatoria', dim=2, bounds=(-3.0, 3.0), num_iteraciones=80, num_fireflies=30, mode='max'):
    lo, hi = bounds
    beta0, gamma, alpha_curr, alpha_decay = 1.0, 1.0, 0.2, 0.98
    scale_noise = 0.5 * (hi - lo)
    rng = np.random.default_rng(2025)
    
    x = _get_init_points(init_method, num_fireflies, bounds, dim)
    vals = np.array([func_objetivo(*p) for p in x])
    
    best_idx = np.argmax(vals) if mode == 'max' else np.argmin(vals)
    best_pos = x[best_idx].copy()
    best_val = vals[best_idx]
    
    for t in range(num_iteraciones):
        order = np.argsort(-vals) if mode == 'max' else np.argsort(vals)
            
        for idx_i in range(num_fireflies):
            i = order[idx_i]
            for idx_j in range(idx_i):
                j = order[idx_j]
                
                r2 = np.sum((x[i] - x[j])**2)
                beta = beta0 * np.exp(-gamma * r2)
                noise = rng.normal(0, 1, size=dim)
                
                x[i] = np.clip(x[i] + beta * (x[j] - x[i]) + alpha_curr * noise * scale_noise, lo, hi)
            
            vals[i] = func_objetivo(*x[i])
            
        alpha_curr *= alpha_decay
        
        idx = np.argmax(vals) if mode == 'max' else np.argmin(vals)
        if (mode == 'max' and vals[idx] > best_val) or (mode == 'min' and vals[idx] < best_val):
            best_val = vals[idx]
            best_pos = x[idx].copy()
                
    return best_pos, best_val


def aco_algorithm(func_objetivo, init_method='aleatoria', dim=2, bounds=(-3.0, 3.0), num_iteraciones=70, num_ants=30, archive_size=25, q=0.5, xi=0.85, mode='max'):
    lo, hi = bounds
    rng = np.random.default_rng(42)

    archive_X = _get_init_points(init_method, archive_size, bounds, dim)
    archive_vals = np.array([func_objetivo(*p) for p in archive_X])

    order = np.argsort(-archive_vals) if mode == 'max' else np.argsort(archive_vals)
    archive_X, archive_vals = archive_X[order], archive_vals[order]

    k_arr = np.arange(1, archive_size + 1)
    weights = np.exp(-(k_arr - 1)**2 / (2 * (q * archive_size)**2))
    weights /= np.sum(weights)
    cum_weights = np.cumsum(weights)

    for it in range(num_iteraciones):
        # Sigma vectorizado
        diffs = np.abs(archive_X[:, np.newaxis, :] - archive_X[np.newaxis, :, :])
        sigmas = np.maximum(xi * np.sum(diffs, axis=1) / max(1, archive_size - 1), 1e-6)
         
        r_vals = rng.random(num_ants)
        k_indices = np.searchsorted(cum_weights, r_vals, side='right')
        k_indices = np.clip(k_indices, 0, archive_size - 1)
        
        new_sols = rng.normal(loc=archive_X[k_indices], scale=sigmas[k_indices])
        np.clip(new_sols, lo, hi, out=new_sols)
        new_vals = np.array([func_objetivo(*sample) for sample in new_sols])
        
        combined_X = np.vstack((archive_X, new_sols))
        combined_vals = np.concatenate((archive_vals, new_vals))
        
        order = np.argsort(-combined_vals) if mode == 'max' else np.argsort(combined_vals)
        archive_X = combined_X[order][:archive_size]
        archive_vals = combined_vals[order][:archive_size]
        
    return archive_X[0].copy(), archive_vals[0]


def simulated_annealing(func_objetivo, init_method='aleatoria', dim=2, bounds=(-3.0, 3.0), mode='min', t_inicial=100.0, t_minima=1e-4, alpha=0.99, step_size=0.5, iteraciones=60):
    lo, hi = bounds
    x_curr = _get_init_points(init_method, 1, bounds, dim)[0]
    f_curr = func_objetivo(*x_curr)
    x_best, f_best = np.copy(x_curr), f_curr
    T = t_inicial

    for _ in range(iteraciones):
        if T <= t_minima: break
            
        x_new = np.clip(x_curr + np.random.normal(0, step_size, dim), lo, hi)
        f_new = func_objetivo(*x_new)
        delta = f_new - f_curr

        es_mejora = (delta < 0) if mode == 'min' else (delta > 0)
        prob_aceptacion = np.exp(np.clip(-delta / T if mode == 'min' else delta / T, -700, 700))

        if es_mejora or np.random.rand() < prob_aceptacion:
            x_curr, f_curr = x_new, f_new
            if (mode == 'min' and f_curr < f_best) or (mode == 'max' and f_curr > f_best):
                x_best, f_best = np.copy(x_curr), f_curr

        T *= alpha

    return x_best, f_best


def harmony_search(func_objetivo, init_method='aleatoria', dim=2, bounds=(-3.0, 3.0), mode='min', hms=20, hmcr=0.95, par=0.3, bw=0.1, iteraciones=80):
    lo, hi = bounds
    hm = _get_init_points(init_method, hms, bounds, dim)
    hm_fitness = np.array([func_objetivo(*ind) for ind in hm])
    delta_rango = bw * (hi - lo)
    
    for _ in range(iteraciones):
        nueva_armonia = np.empty(dim)
        rands = np.random.rand(dim)
        
        mask_hmcr = rands < hmcr
        mask_par = rands < (hmcr * par)

        # Toma valores de la memoria
        indices_aleatorios = np.random.randint(0, hms, size=dim)
        nueva_armonia[mask_hmcr] = hm[indices_aleatorios[mask_hmcr], np.where(mask_hmcr)[0]]
        
        # Mutación
        ajustes = np.random.uniform(-delta_rango, delta_rango, size=dim)
        nueva_armonia[mask_par] += ajustes[mask_par]
        
        # Aleatorios nuevos
        nueva_armonia[~mask_hmcr] = np.random.uniform(lo, hi, size=np.sum(~mask_hmcr))
        
        np.clip(nueva_armonia, lo, hi, out=nueva_armonia)
        nuevo_fitness = func_objetivo(*nueva_armonia)

        peor_indice = np.argmax(hm_fitness) if mode == 'min' else np.argmin(hm_fitness)
        es_mejor = (nuevo_fitness < hm_fitness[peor_indice]) if mode == 'min' else (nuevo_fitness > hm_fitness[peor_indice])

        if es_mejor:
            hm[peor_indice] = nueva_armonia
            hm_fitness[peor_indice] = nuevo_fitness

    mejor_indice = np.argmin(hm_fitness) if mode == 'min' else np.argmax(hm_fitness)
    return hm[mejor_indice], hm_fitness[mejor_indice]