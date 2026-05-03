import numpy as np 
import time
import pandas as pd
import concurrent.futures
import os
from FUNCIONES_OBJETIVO import Griewank, Rosenbrock, Ackley, Rastrigin
from ALGORITMOS_BÚSQUEDA import (
    hill_climbing,
    random_search, 
    genetic_algorithm,
    pso,
    abc_algorithm,
    firefly_algorithm,
    aco_algorithm,
    simulated_annealing,
    harmony_search,
    differential_evolution
)

def ejecucion_individual(algoritmo, func_objetivo, init_method, dim, bounds, mode):
    _, best_val = algoritmo(
        func_objetivo=func_objetivo, 
        init_method=init_method, 
        dim=dim, 
        bounds=bounds, 
        mode=mode
    )
    return best_val

def run_silent_benchmark(num_ejecuciones=30, dim=3, func_objetivo=Rastrigin, bounds=(-5.12, 5.12), init_method='lhs', mode='min'):
    algoritmos = {
        "Hill Climbing": hill_climbing,
        "Random Search": random_search,
        "Alg. Genético": genetic_algorithm,
        "PSO": pso,
        "ABC": abc_algorithm,
        "Firefly": firefly_algorithm,
        "ACO": aco_algorithm,
        "Sim. Annealing": simulated_annealing,
        "Harmony Search": harmony_search,
        "Differential Evolution": differential_evolution
    }
    
    resultados_tabla = []

    for nombre, algoritmo in algoritmos.items():
        start_time = time.time()
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futuros = [
                executor.submit(ejecucion_individual, algoritmo, func_objetivo, init_method, dim, bounds, mode) 
                for _ in range(num_ejecuciones)
            ]
            
            fitness_history = [f.result() for f in concurrent.futures.as_completed(futuros)]
            
        end_time = time.time()
        
        fitness_array = np.array(fitness_history)
        ad = np.mean(fitness_array)
        md = np.median(fitness_array)
        sd = np.std(fitness_array)
        mejor = np.min(fitness_array) if mode == 'min' else np.max(fitness_array)
        peor = np.max(fitness_array) if mode == 'min' else np.min(fitness_array)
        tiempo_total = end_time - start_time
        
        resultados_tabla.append({
            "Función": func_objetivo.__name__.upper(),
            "Dimensión": dim,
            "Inicialización": init_method.upper(),
            "Algoritmo": nombre,
            "Mejor": mejor,
            "Peor": peor,
            "AD (Media)": ad,
            "MD (Mediana)": md,
            "SD (Desv. Est)": sd,
            "Tiempo (s)": round(tiempo_total, 2)
        })

    df_resultados = pd.DataFrame(resultados_tabla)
    ascending_sort = True if mode == 'min' else False
    df_resultados = df_resultados.sort_values(by="AD (Media)", ascending=ascending_sort).reset_index(drop=True)
    
    return df_resultados

def ejecutar_y_mostrar_resultados(dim, init_method, configuraciones_funciones):
    print("=" * 85)
    print(f" RESULTADOS: {dim} DIMENSIONES | INICIALIZACIÓN: {init_method.upper()} ".center(85, "="))
    print("=" * 85)
    
    # Formato científico
    pd.options.display.float_format = '{:.4e}'.format 
    dfs_recopilados = []

    for config in configuraciones_funciones:
        func_name = config["func"].__name__.upper()
        print(f"\n--- Función: {func_name} ---")
        
        df = run_silent_benchmark(
            num_ejecuciones=30,
            dim=dim,
            func_objetivo=config["func"],
            bounds=config["bounds"],
            init_method=init_method,
            mode='min'
        )
        
        # Agregamos al recolector para el CSV final
        dfs_recopilados.append(df)
        
        # Imprimimos en consola ocultando las columnas de contexto para que no se deforme la tabla
        columnas_ocultas = ["Función", "Dimensión", "Inicialización"]
        print(df.drop(columns=columnas_ocultas).to_string())
        
    print("\n")
    return pd.concat(dfs_recopilados, ignore_index=True)

if __name__ == "__main__":
    # Diccionario de funciones objetivo
    configuraciones_funciones = [
        {"func": Rastrigin, "bounds": (-5.12, 5.12)},
        {"func": Ackley, "bounds": (-32.768, 32.768)},
        {"func": Rosenbrock, "bounds": (-5.0, 10.0)},
        {"func": Griewank, "bounds": (-600.0, 600.0)}
    ]

    print("Iniciando pruebas exhaustivas en paralelo... Por favor espera.\n")
    
    lista_dfs_globales = []

    # 1. Tabla: 3 Dimensiones, Inicialización LHS
    df1 = ejecutar_y_mostrar_resultados(dim=3, init_method='lhs', configuraciones_funciones=configuraciones_funciones)
    lista_dfs_globales.append(df1)

    # 2. Tabla: 3 Dimensiones, Inicialización Aleatoria
    df2 = ejecutar_y_mostrar_resultados(dim=3, init_method='aleatoria', configuraciones_funciones=configuraciones_funciones)
    lista_dfs_globales.append(df2)

    # 3. Tabla: 10 Dimensiones, Inicialización LHS
    df3 = ejecutar_y_mostrar_resultados(dim=10, init_method='lhs', configuraciones_funciones=configuraciones_funciones)
    lista_dfs_globales.append(df3)

    # 4. Tabla: 10 Dimensiones, Inicialización Aleatoria
    df4 = ejecutar_y_mostrar_resultados(dim=10, init_method='aleatoria', configuraciones_funciones=configuraciones_funciones)
    lista_dfs_globales.append(df4)

    # ==========================================
    # EXPORTACIÓN A CSV
    # ==========================================
    df_maestro = pd.concat(lista_dfs_globales, ignore_index=True)
    nombre_archivo = "benchmark_resultados_completos.csv"
    
    # Usamos utf-8-sig para que Excel lea correctamente cualquier carácter especial
    df_maestro.to_csv(nombre_archivo, index=False, encoding='utf-8-sig')
    
    print("=" * 85)
    print(" ¡TODAS LAS SIMULACIONES HAN FINALIZADO! ".center(85, "="))
    print(f" Los resultados completos se han guardado en: {os.path.abspath(nombre_archivo)}")
    print("=" * 85)