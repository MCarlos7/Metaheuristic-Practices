# ==========================================
# 1. FUNCIONES OBJETIVO
# ==========================================
import numpy as np

def Sphere(*X):
    """
    Función Sphere (Esfera).
    Mínimo global en f(0, ..., 0) = 0
    Rango habitual de búsqueda: [-5.12, 5.12]
    """
    x = np.array(X)
    return np.sum(x**2)

def Rastrigin(*X):
    """
    Función Rastrigin. Altamente multimodal.
    Mínimo global en f(0, ..., 0) = 0
    Rango habitual de búsqueda: [-5.12, 5.12]
    """
    x = np.array(X)
    n = len(x)
    A = 10
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def Ackley(*X):
    """
    Función Ackley.
    Mínimo global en f(0, ..., 0) = 0
    Rango habitual de búsqueda: [-32.768, 32.768]
    """
    x = np.array(X)
    n = len(x)
    A, B, C = 20, 0.2, 2 * np.pi
    term1 = -A * np.exp(-B * np.sqrt((1/n) * np.sum(x**2)))
    term2 = -np.exp((1/n) * np.sum(np.cos(C * x)))
    return term1 + term2 + A + np.exp(1)

def peaks(*X):
    """
    La función Peaks es estrictamente 2D. 
    Si se le pasan más de 2 dimensiones, solo evaluará las primeras dos.
    """
    x = X[0]
    y = X[1]
    return 3*(1 - x)**2 * np.exp(-(x**2) - (y + 1)**2) \
           - 10*(x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
           - 1/3 * np.exp(-(x + 1)**2 - y**2)

def Rosenbrock(*X):
    """
    Función Rosenbrock. Unimodal pero con un valle plano muy engañoso.
    Mínimo global en f(1, ..., 1) = 0
    Rango habitual de búsqueda: [-5.0, 10.0] o [-2.048, 2.048]
    """
    x = np.array(X)
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def Griewank(*X):
    """
    Función Griewank. Altamente multimodal con ruido regular.
    Mínimo global en f(0, ..., 0) = 0
    Rango habitual de búsqueda: [-600, 600]
    """
    x = np.array(X)
    n = len(x)
    
    sum_term = np.sum(x**2) / 4000.0
    
    i_indices = np.arange(1, n + 1)
    prod_term = np.prod(np.cos(x / np.sqrt(i_indices)))
    
    return sum_term - prod_term + 1.0