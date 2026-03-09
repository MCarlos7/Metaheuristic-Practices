import numpy as np 
import matplotlib.pyplot as plt 
import time
from MAIN.FUNCIONES_OBJETIVO import peaks, Ackley, Rastrigin, Sphere
from MAIN.INICIALIZACION import init_aleatoria, init_lhs, init_maxdistance
from MAIN.ALGORITMOS_BÚSQUEDA import hill_climbing, random_search, genetic_algorithm

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
print("3. Algoritmo Genético (Búsqueda Poblacional)")
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
        
    elif input_algo == '3':
        print(f"\nIniciando Algoritmo Genético ({input_mode.upper()})...")
        genetic_algorithm(func_elegida, pause=0.3, mode=input_mode)

    else:
        print("Algoritmo no válido.")
else:
    print("Función no válida.")