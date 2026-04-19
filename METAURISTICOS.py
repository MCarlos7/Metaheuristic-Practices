import numpy as np 
import matplotlib.pyplot as plt 
import time
from MAIN.FUNCIONES_OBJETIVO import peaks, Ackley, Rastrigin, Sphere
from MAIN.INICIALIZACION import init_aleatoria, init_lhs, init_maxdistance
from MAIN.ALGORITMOS_BÚSQUEDA import hill_climbing, random_search, genetic_algorithm, pso, abc_algorithm, firefly_algorithm, aco_algorithm

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
print("4. Particle Swarm Optimization - PSO (Enjambre de Partículas)")
print("5. Artificial Bee Colony - ABC (Colonia de Abejas)")
print("6. Firefly Algorithm (Algoritmo de Luciérnagas)")
print("7. Ant Colony Optimization - ACO (Colonia de Hormigas)") # <- Nueva opción agregada
input_algo = input("Ingresa el número: ")

map_func = {'1': peaks, '2': Ackley, '3': Rastrigin, '4': Sphere}

if input_func in map_func:
    func_elegida = map_func[input_func]
    
    if input_algo == '1':
        print("\nSelecciona el algoritmo de inicialización para Hill Climbing:")
        print("1. Aleatoria\n2. LHS\n3. MaxDistance")
        input_init = input("Ingresa el número: ")
        map_init = {'1': 'aleatoria', '2': 'lhs', '3': 'maxdistance'}
        metodo_init = map_init.get(input_init, 'aleatoria')
        
        print(f"\nIniciando Hill Climbing ({input_mode.upper()})...")
        hill_climbing(func_elegida, init_method=metodo_init, mode=input_mode, step_size=1, step_decay=0.7)
        
    elif input_algo == '2':
        print(f"\nIniciando Búsqueda Aleatoria ({input_mode.upper()})...")
        random_search(func_elegida, iterations=150, pause=0.05, mode=input_mode) 
        
    elif input_algo == '3':
        print(f"\nIniciando Algoritmo Genético ({input_mode.upper()})...")
        genetic_algorithm(func_elegida, pause=0.3, mode=input_mode)
        
    elif input_algo == '4':
        print(f"\nIniciando PSO - Particle Swarm Optimization ({input_mode.upper()})...")
        pso(func_elegida, mode=input_mode)
        
    elif input_algo == '5':
        print(f"\nIniciando ABC - Artificial Bee Colony ({input_mode.upper()})...")
        abc_algorithm(func_elegida, mode=input_mode)
        
    elif input_algo == '6':
        print(f"\nIniciando Firefly Algorithm ({input_mode.upper()})...")
        firefly_algorithm(func_elegida, mode=input_mode)
        
    elif input_algo == '7':
        print(f"\nIniciando ACO - Ant Colony Optimization ({input_mode.upper()})...")
        aco_algorithm(func_elegida, mode=input_mode)

    else:
        print("Algoritmo no válido.")
else:
    print("Función no válida.")