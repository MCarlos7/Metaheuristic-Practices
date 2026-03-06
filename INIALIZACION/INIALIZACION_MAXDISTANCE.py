# Inializacion dispercion 
import numpy as np
import matplotlib.pyplot as plt

def latin_Hipercube (n_samples, dim, rng):
    #Generar LHS Simple en [0,1]^dim
    cut = np.linspace(0, 1, n_samples + 1)
    u = np.random.rand(n_samples, dim)
    samples = np.zeros((n_samples, dim))
    for j in range (dim):
        idx = rng.permutation(n_samples)
        samples[:, j] = cut[idx] + u[:, j] * (1.0 / n_samples)
    return np.clip(samples, 0, 1)

def maxmin_selecction(candidatos, m, rng = None):
    N = candidatos.shape[0]
    assert m <= N, "m debe ser menor o igual al número de candidatos"
    if rng is None:
        rng = np.random.default_rng()
    selected_idx = []
    first = rng.integers(0, N)
    selected_idx.append(first)
    
    while len(selected_idx) < m:
        sel = np.array(selected_idx)
        chosen = -1
        best_min_dist = -1
        for i in range(N):
            if i in sel:
                continue
            dist = np.linalg.norm(candidatos[sel] - candidatos[i], axis=1)
            min_dist = np.min(dist)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                chosen = i
        selected_idx.append(chosen)
    return candidatos[selected_idx, :]

# Parámetros
rng = np.random.default_rng(2025)
numPuntosDeseados = 50
dim = 2
pool_size = 500

# Generar candidatos LHS
candidatos = latin_Hipercube(pool_size, dim, rng)
# Seleccionar puntos LHS o LHS MaxMin
seleccion_lhs = candidatos[:numPuntosDeseados, :]
seleccion_maxmin = maxmin_selecction(candidatos, numPuntosDeseados, rng = rng)

# Graficar resultados
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(candidatos[:, 0], candidatos[:, 1], s=8, alpha=0.35, label='Candidatos LHS')
plt.scatter(seleccion_lhs[:, 0], seleccion_lhs[:, 1], s=50, c = 'r', marker='o', edgecolors='k', label='Selección LHS (primeros)')
plt.title('Selección LHS Simple(Selccion Directa)')
plt.xlim([0, 1]); plt.ylim([0, 1]); plt.grid(True); plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(candidatos[:, 0], candidatos[:, 1], s=8, alpha=0.35, label='Candidatos LHS')
plt.scatter(seleccion_maxmin[:, 0], seleccion_maxmin[:, 1], s=50, c = 'g', marker='s', edgecolors='k', label='Selección LHS MaxMin')
plt.title('Selección LHS MaxMin')
plt.xlim([0, 1]); plt.ylim([0, 1]); plt.grid(True); plt.legend()

plt.tight_layout()
plt.show()

