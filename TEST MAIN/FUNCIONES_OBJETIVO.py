# ==========================================
# 1. FUNCIONES OBJETIVO
# ==========================================
import numpy as np

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

