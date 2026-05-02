# ==========================================
# 2. ALGORITMOS DE INICIALIZACIÓN
# ==========================================
import numpy as np

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
