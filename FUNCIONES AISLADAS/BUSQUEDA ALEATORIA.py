import numpy as np 
import matplotlib.pyplot as plt 
import time

# FUNCIONES 
def peaks(x,y):
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
           
# BUSQUEDA ALEATORIA CON VISUALIZACION 

def busqueda_aleatoria(func_objetivo, z_mesh, num_iter=50, rango=(-3,3), pausa=0.01):
    x_range = np.linspace(rango[0], rango[1], 200)
    y_range = np.linspace(rango[0], rango[1], 200)
    X, Y = np.meshgrid(x_range, y_range)
    
    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Contorno
    cont = ax1.contourf(X, Y, z_mesh, levels=40, cmap='viridis')
    plt.colorbar(cont, ax=ax1)
    ax1.set_title('Búsqueda Aleatoria - 2D')
    
    # Superficie 3D
    ax2.plot_surface(X, Y, z_mesh, cmap='viridis', alpha=0.5)
    ax2.set_title('Búsqueda Aleatoria - 3D')
    
    mejor_val = -np.inf
    mejor_punto = None
    
    for i in range(num_iter):    
        px = np.random.uniform(rango[0], rango[1])
        py = np.random.uniform(rango[0], rango[1])
        val = func_objetivo(px, py)
        
        if val > mejor_val:
            mejor_val = val
            mejor_punto = (px, py)    
        
        ax1.scatter(px, py, color='blue', s=10)
        ax2.scatter(px, py, val, color='blue', s=10)
        
        best_plot1 = ax1.scatter(mejor_punto[0], mejor_punto[1], color='red', s=40, edgecolor='black')
        best_plot2 = ax2.scatter(mejor_punto[0], mejor_punto[1], mejor_val, color='red', s=50, edgecolor='black')
        
        plt.draw()
        plt.pause(pausa)
    
    print(f'Mejor solución: {mejor_punto}, f(x,y)={mejor_val}')
    plt.ioff()
    plt.show()

# EJECUCIÓN 
x_vec = np.linspace(-3, 3, 200)
y_vec = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x_vec, y_vec)

print("Selecciona tu función objetivo:\n1. Peaks\n2. Ackley\n3. Rastrigin\n4. Sphere\n")
input_func = input("Ingresa el número de la función: ")

mapping = {'1': peaks, '2': Ackley, '3': Rastrigin, '4': Sphere}

if input_func in mapping:
    f_sel = mapping[input_func]
    z_mesh = f_sel(X, Y)
    busqueda_aleatoria(f_sel, z_mesh)
else:
    print("Función no válida.")
    
    
