# ==========================================
# 2. ALGORITMOS DE INICIALIZACIÓN (N-Dimensiones)
# ==========================================
import numpy as np
import itertools

def init_aleatoria(n, bounds, dim=2):
    lo, hi = bounds
    return np.random.uniform(lo, hi, (n, dim))

def init_lhs(n, bounds, dim=2):
    lo, hi = bounds
    pts = np.empty((n, dim))
    for d in range(dim):
        intervals = np.linspace(lo, hi, n + 1)
        rnds = np.random.uniform(intervals[:-1], intervals[1:])
        np.random.shuffle(rnds)
        pts[:, d] = rnds
    return pts

def init_maxdistance(n, bounds, dim=2, num_candidates=50):
    lo, hi = bounds
    pts = [np.random.uniform(lo, hi, dim)]
    for _ in range(1, n):
        cands = np.random.uniform(lo, hi, (num_candidates, dim))
        dists = np.array([min(np.linalg.norm(c - p) for p in pts) for c in cands])
        pts.append(cands[np.argmax(dists)])
    return np.array(pts)

def _get_init_points(method, n, bounds, dim):
    if method == 'lhs':
        return init_lhs(n, bounds, dim)
    elif method == 'maxdistance':
        return init_maxdistance(n, bounds, dim)
    else:
        return init_aleatoria(n, bounds, dim)
